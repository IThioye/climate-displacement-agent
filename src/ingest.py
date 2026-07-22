"""Extract, sanitize, and parent/child chunk the source documents."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

try:
    from .guardrails import sanitise_external_content
except ImportError:  # python src/ingest.py
    from guardrails import sanitise_external_content


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "sources.json"
CACHE_PATH = DATA_DIR / "processed" / "chunks.json"


def split_words(text: str, size: int, overlap: int) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    return [
        " ".join(words[start:start + size])
        for start in range(0, len(words), size - overlap)
        if words[start:start + size]
    ]


def extract_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned, suspicious = sanitise_external_content(text, max_chars=100_000)
        if cleaned:
            pages.append((page_number, cleaned))
        if suspicious:
            print(f"[SECURITY] Suspicious instructions marked in {path.name}, page {page_number}")
    return pages


def build_chunks(
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = CACHE_PATH,
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for source in manifest["sources"]:
        pdf_path = DATA_DIR / "raw" / source["file"]
        if not pdf_path.exists():
            print(f"[WARN] Missing {pdf_path.name}; run python data/download_documents.py")
            continue
        for page_number, page_text in extract_pdf(pdf_path):
            for parent_index, parent_text in enumerate(split_words(page_text, 450, 70)):
                parent_id = f"{source['id']}:p{page_number}:b{parent_index}"
                for child_index, child_text in enumerate(split_words(parent_text, 110, 20)):
                    chunks.append({
                        "id": f"{parent_id}:c{child_index}",
                        "parent_id": parent_id,
                        "text": child_text,
                        "parent_text": parent_text,
                        "metadata": {
                            "source_id": source["id"],
                            "title": source["title"],
                            "publisher": source["publisher"],
                            "year": source["year"],
                            "region": source["region"],
                            "country": source.get("country", "Global"),
                            "url": source["url"],
                            "page": page_number,
                        },
                    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} child chunks to {output_path}")
    return chunks


if __name__ == "__main__":
    build_chunks()

