from typing import List, Optional
import logging
import requests
from fastapi import HTTPException
from .config import LLAMA_BASE, EMBEDDINGS_BASE, EMBEDDINGS_MODEL, EMBEDDINGS_ENDPOINT

logger = logging.getLogger("uvicorn.error")

def embed_texts(texts: List[str]) -> List[List[float]]:
    base_url = EMBEDDINGS_BASE or LLAMA_BASE
    logger.info("Embeddings: count=%d model=%s base=%s endpoint=%s", len(texts), EMBEDDINGS_MODEL, base_url, EMBEDDINGS_ENDPOINT or "auto")

    def openai_embeddings() -> Optional[List[List[float]]]:
        r = requests.post(
            f"{base_url}/v1/embeddings",
            json={"model": EMBEDDINGS_MODEL, "input": texts},
            timeout=1200,
        )
        if r.status_code == 404:
            if "model" in r.text.lower():
                r.raise_for_status()
            return None
        r.raise_for_status()
        data = r.json()["data"]
        return [item["embedding"] for item in data]

    def ollama_embeddings() -> Optional[List[List[float]]]:
        vectors: List[List[float]] = []
        for text in texts:
            r = requests.post(
                f"{base_url}/api/embeddings",
                json={"model": EMBEDDINGS_MODEL, "prompt": text},
                timeout=1200,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            vectors.append(r.json()["embedding"])
        return vectors

    try:
        errors: List[str] = []

        # Allow explicit override if needed.
        if EMBEDDINGS_ENDPOINT:
            if EMBEDDINGS_ENDPOINT.endswith("/api/embeddings"):
                vectors = ollama_embeddings()
                if vectors is not None:
                    return vectors
                errors.append(f"{base_url}{EMBEDDINGS_ENDPOINT} -> 404")
            r = requests.post(
                f"{base_url}{EMBEDDINGS_ENDPOINT}",
                json={"model": EMBEDDINGS_MODEL, "input": texts},
                timeout=1200,
            )
            if r.status_code == 404:
                errors.append(f"{base_url}{EMBEDDINGS_ENDPOINT} -> 404")
            else:
                r.raise_for_status()
                data = r.json()["data"]
                return [item["embedding"] for item in data]

        vectors = openai_embeddings()
        if vectors is not None:
            return vectors
        errors.append(f"{base_url}/v1/embeddings -> 404")

        vectors = ollama_embeddings()
        if vectors is not None:
            return vectors
        errors.append(f"{base_url}/api/embeddings -> 404")

        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding call failed: no embeddings endpoint found. "
                f"Check EMBEDDINGS_BASE/LLAMA_BASE. Tried: {', '.join(errors)}"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding call failed: {e}")

    
