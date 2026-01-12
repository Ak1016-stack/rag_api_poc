import uuid
from typing import Dict, List, Optional, Tuple
import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from .config import QDRANT_URL, QDRANT_COLLECTION

client = QdrantClient(url=QDRANT_URL)
logger = logging.getLogger("uvicorn.error")

def get_collection_vector_size() -> Optional[int]:
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return vectors.size
        if isinstance(vectors, dict):
            for v in vectors.values():
                if hasattr(v, "size"):
                    return v.size
    except Exception:
        return None
    return None

def ensure_collection(vector_size: int) -> None:
    if not client.collection_exists(QDRANT_COLLECTION):
        logger.info("Qdrant: creating collection=%s size=%d", QDRANT_COLLECTION, vector_size)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def upsert_chunks(source: str, chunks: List[str], vectors: List[List[float]]) -> int:
    points: List[PointStruct] = []
    logger.info("Qdrant: upsert collection=%s points=%d source=%s", QDRANT_COLLECTION, len(chunks), source)
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"source": source, "chunk_index": idx, "text": chunk},
            )
        )
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)

def search(query_vector: List[float], limit: int) -> List[Dict]:
    logger.info("Qdrant: search collection=%s limit=%d", QDRANT_COLLECTION, limit)
    res = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    out: List[Dict] = []
    for p in res.points:
        payload = p.payload or {}
        out.append(
            {
                "score": p.score,
                "source": payload.get("source", "unknown"),
                "chunk_index": payload.get("chunk_index", -1),
                "text": payload.get("text", ""),
            }
        )
    return out
