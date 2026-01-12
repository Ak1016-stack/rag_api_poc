from typing import Any, Dict, Iterator, List
import logging
import json
import requests
from fastapi import HTTPException
from .config import LLAMA_BASE, ROUTER_GRAMMAR_PATH
from .schemas.schemas_llm import normalize_router_output, RouterOutput, FinalAnswer

logger = logging.getLogger("uvicorn.error")

def chat(messages: List[Dict[str, Any]], max_tokens: int = 300, temperature: float = 0.2) -> str:
    payload = {
        "model": "llama3.2:3b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info("Ollama: model=%s max_tokens=%d temperature=%.2f messages=%d chars=%d", payload["model"], max_tokens, temperature, len(messages), total_chars)
        logger.info("Ollama prompt payload: %s", payload)
        r = requests.post(f"{LLAMA_BASE}/v1/chat/completions", json=payload, timeout=900)
        r.raise_for_status()
        j = r.json()
        logger.info("Ollama response: %s", j)

        try:
            return j["choices"][0]["message"]["content"]
        except Exception as parse_err:
            # show actual response shape when parsing fails
            raise HTTPException(
                status_code=502,
                detail=f"Could not parse llama.cpp response ({parse_err}): {j}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat call failed: {e}")

def route_action(user_message: str) -> RouterOutput:
    system = (
        "You are a strict router. You must return ONLY a JSON object.\n"
        "You must output EXACTLY one of these shapes:\n"
        '1) {"type":"tool_call","tool":"calc","args":{"expression":"<EXPR>"}}\n'
        '2) {"type":"final","answer":"use_rag"}\n'
        "Rules for calling calc:\n"
        "- ONLY call calc if the user's message CONTAINS a math expression made of digits and operators (+ - * / ( ) .).\n"
        "- If the user asks a definition, policy, reporting, procedures, or anything not explicitly math, return type=final.\n"
        "- Do NOT invent expressions.\n"
        '- Do NOT reuse examples like "4+3" unless the user asked "4+3".\n'
        "Return JSON only. No extra keys. No extra text."
    )
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    if ROUTER_GRAMMAR_PATH:
        try:
            grammar = open(ROUTER_GRAMMAR_PATH, "r", encoding="utf-8").read()
            payload["grammar"] = grammar
        except Exception as e:
            logger.warning("Router grammar load failed: %s", e)
    try:
        logger.info("Router: payload=%s", payload)
        r = requests.post(f"{LLAMA_BASE}/v1/chat/completions", json=payload, timeout=900)
        r.raise_for_status()
        j = r.json()
        raw_output = j["choices"][0]["message"]["content"]
        logger.info("Router: raw_output=%s", raw_output)
        try:
            return normalize_router_output(json.loads(raw_output))
        except Exception as e:
            repair_system = (
                "Return ONLY valid JSON in one of these shapes:\n"
                '{"type":"tool_call","tool":"calc","args":{"expression":"<EXPR>"}}\n'
                '{"type":"final","answer":"use_rag"}\n'
                "No other keys. No extra text."
            )
            repair_user = (
                "Fix this invalid output to match the schema.\n\n"
                f"Invalid output:\n{raw_output}\n\n"
                f"User message:\n{user_message}"
            )
            repair_payload = {
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": repair_user},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            }
            logger.info("Router repair: payload=%s", repair_payload)
            r2 = requests.post(f"{LLAMA_BASE}/v1/chat/completions", json=repair_payload, timeout=900)
            r2.raise_for_status()
            j2 = r2.json()
            raw_output2 = j2["choices"][0]["message"]["content"]
            logger.info("Router repair: raw_output=%s", raw_output2)
            try:
                return normalize_router_output(json.loads(raw_output2))
            except Exception as e2:
                logger.warning("Router repair failed, defaulting to RAG: %s", e2)
                return FinalAnswer(type="final", answer="use_rag")
    except Exception as e:
        logger.warning("Router failed, defaulting to RAG: %s", e)
        return FinalAnswer(type="final", answer="use_rag")

def chat_stream(
    messages: List[Dict[str, Any]],
    max_tokens: int = 300,
    temperature: float = 0.2,
    final_suffix: str = "",
) -> Iterator[str]:
    payload = {
        "model": "llama3.2:3b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    try:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "Ollama stream: model=%s max_tokens=%d temperature=%.2f messages=%d chars=%d",
            payload["model"],
            max_tokens,
            temperature,
            len(messages),
            total_chars,
        )
        logger.info("Ollama stream payload: %s", payload)

        r = requests.post(
            f"{LLAMA_BASE}/v1/chat/completions",
            json=payload,
            timeout=900,
            stream=True,
        )
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[len("data: "):]
                if data == "[DONE]":
                    if final_suffix:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': final_suffix}}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                yield f"{line}\n\n"
            else:
                yield f"{line}\n\n"

        if final_suffix:
            yield f"data: {json.dumps({'choices': [{'delta': {'content': final_suffix}}]})}\n\n"
        yield "data: [DONE]\n\n"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat stream failed: {e}")
