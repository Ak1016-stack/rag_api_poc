import os
from dotenv import load_dotenv

load_dotenv()

LLAMA_BASE = os.getenv("LLAMA_BASE", "http://localhost:11434").rstrip("/")
EMBEDDINGS_BASE = os.getenv("EMBEDDINGS_BASE", "").strip().rstrip("/")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
# Optional override for embeddings endpoint path (e.g. "/v1/embeddings" or "/api/embeddings")
EMBEDDINGS_ENDPOINT = os.getenv("EMBEDDINGS_ENDPOINT", "").strip()
ROUTER_GRAMMAR_PATH = os.getenv("ROUTER_GRAMMAR_PATH", "").strip()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "it_poc")

TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
