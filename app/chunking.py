from typing import List

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:

    text = " ".join(text.split())
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(i + chunk_size, len(text))
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks
