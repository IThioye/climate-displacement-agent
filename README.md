# Climate Displacement Evidence Agent

An evidence-first research agent for humanitarian analysts comparing disaster
displacement risks and policy responses. It retrieves from a fixed corpus of
authoritative reports, produces a cited regional brief, runs self-consistency
`k=3`, and sends the result through an independent critic.

The agent is advisory. It does **not** predict individual movement, determine
eligibility, or automate aid-allocation decisions.

## Quick start

```bash
git clone <your-repository-url>
cd climate-displacement-agent
cp .env.example .env
pip install -r requirements.txt
python data/download_documents.py
python src/ingest.py
python src/agent.py "Compare flood displacement evidence for Bangladesh and Southeast Asia"
```

The first run may download two small Hugging Face models. Set
`RAG_USE_LOCAL_MODELS=0` for deterministic no-download fallbacks.

## Flask test interface

Set a non-default `ADMIN_PASSWORD` in `.env`, then start the local interface:

```bash
python app.py
```

Open `http://127.0.0.1:5000`. The main page provides prepared factual,
comparative, methodological, and security-test questions. Each run displays a
live text-only trace using plain-language stages. Repeated updates replace the
existing stage instead of adding an indefinitely growing log. The user page does
not expose model backends, retrieval terminology, filters, cost accounting, prompt
hashes, or private chain-of-thought.

The administration page is at `http://127.0.0.1:5000/admin` and uses HTTP Basic
authentication (`ADMIN_USERNAME` / `ADMIN_PASSWORD`). It retains the technical
events and reports run counts,
blocked/failed outcomes, average latency and cost, tool and publisher distribution,
recent runs, detailed event logs, and a raw JSON export. Logs are stored locally in
`data/processed/agent_runs.db` and are excluded from Git.

## What was adapted from the course labs

- Lab B1: parent-child chunking, BM25 + dense retrieval, reciprocal-rank fusion,
  cross-encoder reranking, metadata filtering, and baseline/final evaluation.
- Lab B2: Unicode-normalized L1 filtering, sanitization of untrusted document/tool
  text, the L4 action matrix, token-cost caps, and per-tool quotas.
- Production lab: prompt hashing, latency/cost/tool measurements, Langfuse spans,
  and a visible critic verdict.

## Architecture

1. `data/download_documents.py` retrieves the PDFs recorded in `data/sources.json`.
2. `src/ingest.py` extracts and sanitizes every page, then creates parent/child chunks.
3. `src/retrieval.py` fuses BM25 and dense rankings with RRF, resolves child hits to
   parent context, and applies a cross-encoder before returning cited evidence.
4. `src/agent.py` applies L1/L4/budget checks, retrieves evidence, creates three
   structured syntheses, and asks a critic agent to select or correct the answer.
5. `src/mcp_server.py` exposes four documented, error-safe tools over MCP.

See [docs/architecture.md](docs/architecture.md) for the diagram and trust boundaries.

## MCP server

```bash
python src/mcp_server.py
```

Tools:

- `search_evidence`
- `compare_regions`
- `get_source`
- `store_finding`

## Tests and evaluation

```bash
python -m pytest tests/test_security.py
python -m pytest tests/test_interface.py
python evaluation/evaluate_retrieval.py
python evaluation/evaluate_ragas.py --limit 10
```

The evaluation set contains ten questions. Save the baseline and final retrieval
results before running the RAGAS generation metrics. Do not invent report numbers;
paste the measurements from your actual runs into `REPORT.md`.

## Data provenance

The starter corpus contains reports from IDMC, IOM, UNHCR/OHCHR, the World Bank,
and the Asian Development Bank. Each retrieved passage retains publisher, title,
year, page, region/country, and source URL. Review each publisher's reuse terms
before making the repository public.

## Security boundary

PDFs and MCP tool results are untrusted. Text is normalized, active markup and
control characters are removed, instruction-like passages are visibly marked, and
the synthesis prompt states that evidence can be quoted but never obeyed.
