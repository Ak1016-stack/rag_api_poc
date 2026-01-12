from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import logging
import time
import json
import re
from .config import TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_COLLECTION
from .schemas.schemas_openai import ChatCompletionsRequest, IngestTextRequest
from .chunking import chunk_text
from .embeddings import embed_texts
from .llm import chat as llama_chat, chat_stream as llama_chat_stream, route_action
from .qdrant_store import (
    ensure_collection,
    get_collection_vector_size,
    upsert_chunks,
    search,
    client as qdrant_client,
)
from .schemas.schemas_llm import ToolCall, FinalAnswer
from .tools.calc import run as calc_tool

app = FastAPI(title="RAG POC (OpenAI-compatible)")
logger = logging.getLogger("uvicorn.error")

# Tune later
SCORE_THRESHOLD = 0.45  # ignore weak matches

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "rag-proxy", "object": "model", "owned_by": "local"}],
    }

@app.post("/admin/ingest_text")
def ingest_text(req: IngestTextRequest):
    start = time.time()
    logger.info("Ingest: source=%s text_len=%d", req.source, len(req.text or ""))
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    chunks = chunk_text(req.text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from text")
    logger.info("Ingest: chunks=%d chunk_size=%d overlap=%d", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    # Embed first chunk to know vector dim
    first_vec = embed_texts([chunks[0]])[0]
    ensure_collection(len(first_vec))

    # Embed all chunks (batching)
    vectors = []
    batch_size = 4
    for i in range(0, len(chunks), batch_size):
        vectors.extend(embed_texts(chunks[i:i + batch_size]))

    n = upsert_chunks(req.source, chunks, vectors)
    logger.info("Ingest: indexed=%d elapsed_ms=%.1f", n, (time.time() - start) * 1000)
    return {"ok": True, "collection": QDRANT_COLLECTION, "chunks_indexed": n}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    start = time.time()
    logger.info("Chat: model=%s max_tokens=%s temperature=%s", req.model, req.max_tokens, req.temperature)
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found")
    question = user_msgs[-1].strip()
    logger.info("Chat: question_len=%d", len(question))
    is_math = bool(re.search(r"[0-9]\s*[\+\-\*/\(\)%\.]\s*[0-9]", question))

    # ---- Pre-router (tool vs RAG) ----
    try:
        action = route_action(question)
    except HTTPException:
        raise
    except Exception:
        action = FinalAnswer(type="final", answer="use_rag")

    # If router failed but it's clearly math, run calc anyway.
    if isinstance(action, FinalAnswer) and is_math:
        action = ToolCall(type="tool_call", tool="calc", args={"expression": question})

    if isinstance(action, ToolCall) and is_math:
        tool_result = calc_tool(action.args)
        content = (
            f"The result is {tool_result.get('result')}"
            if "result" in tool_result
            else f"Error: {tool_result.get('error', 'Tool error')}"
        )
        if req.stream:
            def tool_stream():
                payload = {"choices": [{"delta": {"content": content}}]}
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_stream(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-poc",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "model": req.model or "rag-proxy",
        }

    # ---- Retrieve context (only if collection exists) ----
    context_block = ""
    citations = []

    # Safely check collection existence
    collection_exists = False
    try:
        # Works on most qdrant-client versions
        collection_exists = qdrant_client.collection_exists(QDRANT_COLLECTION)
    except Exception:
        # Fallback: if this fails, assume it exists and let search fail gracefully
        collection_exists = True

    if collection_exists:
        # Embed query
        q_vec = embed_texts([question])[0]

        # Ensure vector dimensionality matches the collection
        collection_size = get_collection_vector_size()
        if collection_size and collection_size != len(q_vec):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Vector dimension mismatch: "
                    f"collection expects {collection_size}, got {len(q_vec)}. "
                    "Recreate the collection or switch QDRANT_COLLECTION."
                ),
            )

        # Retrieve
        hits = search(q_vec, limit=TOP_K)
        logger.info("Chat: hits=%d threshold=%.2f", len(hits), SCORE_THRESHOLD)

        # Filter by score threshold
        good_hits = [h for h in hits if h.get("score", 0.0) >= SCORE_THRESHOLD]

        # If no good hits: strict RAG behaviour (no hallucination, no llama call)
        if not good_hits:
            logger.info("Chat: no_hits elapsed_ms=%.1f", (time.time() - start) * 1000)
            if req.stream:
                def no_context_stream():
                    msg = "I couldn’t find this information in the provided document."
                    yield f"data: {{\"choices\":[{{\"delta\":{{\"content\":\"{msg}\"}}}}]}}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(no_context_stream(), media_type="text/event-stream")
            return {
                "id": "chatcmpl-poc",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I could not find this information in the provided document."
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": req.model or "rag-proxy",
            }

        # Build context + citations
        contexts = []
        for h in good_hits:
            contexts.append(h.get("text", ""))
            citations.append(
                f'{h.get("source", "unknown")}#chunk{h.get("chunk_index", -1)} (score {h.get("score", 0.0):.3f})'
            )

        context_block = "\n\n---\n\n".join([c for c in contexts if c.strip()])

    # ---- Prompt ----
    system = (
        "You are a helpful internal assistant. "
        "If the user's question requires a tool to be called, call the tool and report its results. "
        "Answer ONLY using the provided context. Make sure to give the most useful answer, the most relevant and key piece of information you can find about the user's query and prioritize returning that."
        "If the context does not contain the answer, say: "
        "\"I could not find this information in the provided document.\""
    )

    augmented_user = (
        f"Context:\n{context_block}\n\nQuestion: {question}"
        if context_block
        else f"Question: {question}"
    )

    # ---- LLM call ----
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": augmented_user},
    ]
    max_tokens = req.max_tokens or 300
    temperature = req.temperature or 0.2

    if req.stream:
        final_suffix = ""
        if citations:
            final_suffix = "\n\nSources:\n- " + "\n- ".join(citations)
        return StreamingResponse(
            llama_chat_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                final_suffix=final_suffix,
            ),
            media_type="text/event-stream",
        )

    answer = llama_chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    logger.info("Chat: answered elapsed_ms=%.1f", (time.time() - start) * 1000)

    # Append citations only if we actually used context
    if citations:
        answer = answer.rstrip() + "\n\nSources:\n- " + "\n- ".join(citations)

    return {
        "id": "chatcmpl-poc",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "model": req.model or "rag-proxy",
    }
