# Document corpus

The corpus contains authoritative reports from IDMC, IOM, UNHCR/OHCHR, the
World Bank, and the Asian Development Bank. Their URLs and retrieval metadata
are recorded in `sources.json`.

To populate and process the corpus:

```bash
python data/download_documents.py
python src/ingest.py
```

PDF text is treated as untrusted input. During ingestion it is Unicode-normalized,
stripped of active HTML/control characters, scanned for prompt-injection patterns,
and divided into overlapping parent/child chunks. The generated
`processed/chunks.json` records publisher, year, region, country, page, and URL.

The selected documents deliberately overlap on global figures, Asia-Pacific risk,
human-rights framing, and policy responses. That overlap supplies realistic
distractors for measuring retrieval precision.

