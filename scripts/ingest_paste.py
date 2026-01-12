import os
import json
import requests
from pathlib import Path

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SOURCE = os.getenv("SOURCE", "my_pdf_poc")
TEXT_PATH = Path(os.getenv("TEXT_PATH", "data/pasted_text.txt"))

def main():
    if not TEXT_PATH.exists():
        raise SystemExit(f"File not found: {TEXT_PATH.resolve()}")

    text = TEXT_PATH.read_text(encoding="utf-8", errors="ignore")

    payload = {"source": SOURCE, "text": text}
    r = requests.post(f"{API_BASE}/admin/ingest_text", json=payload, timeout=600)
    if not r.ok:
        print("STATUS:", r.status_code)
        print("BODY:", r.text[:2000])
        r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()