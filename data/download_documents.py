"""Download the authoritative PDFs listed in sources.json."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"


def download_all() -> None:
    manifest = json.loads((DATA_DIR / "sources.json").read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for source in manifest["sources"]:
        target = RAW_DIR / source["file"]
        if target.exists() and target.stat().st_size > 1_000:
            print(f"exists   {target.name} ({target.stat().st_size / 1_000_000:.1f} MB)")
            continue
        parsed = urllib.parse.urlparse(source["url"])
        request = urllib.request.Request(source["url"], headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "Referer": f"{parsed.scheme}://{parsed.netloc}/",
        })
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
            if not content.startswith(b"%PDF"):
                raise ValueError("server response is not a PDF")
            target.write_bytes(content)
            print(f"download {target.name} ({len(content) / 1_000_000:.1f} MB)")
        except Exception as exc:
            print(f"FAILED   {target.name}: {exc}")


if __name__ == "__main__":
    download_all()
