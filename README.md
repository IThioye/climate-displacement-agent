# Climate Displacement Evidence Agent

An evidence-first research agent for humanitarian analysts comparing disaster
displacement risks and policy responses. It retrieves from a fixed corpus of
authoritative reports, produces a cited regional brief, runs self-consistency
`k=3`, and sends the result through an independent critic.

The agent is advisory. It does **not** predict individual movement, determine
eligibility, or automate aid-allocation decisions.

## Quick start

```bash
git clone https://github.com/IThioye/climate-displacement-agent.git
cd climate-displacement-agent
cp .env.example .env
pip install -r requirements.txt
python data/download_documents.py
python src/ingest.py
python src/agent.py "Compare flood displacement evidence for Bangladesh and Southeast Asia"
```

The first run may download two small Hugging Face models. Retrieval runs on CPU
by default (`RAG_DEVICE=cpu`) so it does not compete with another local model for
GPU memory. Set `RAG_USE_LOCAL_MODELS=0` for deterministic no-download fallbacks,
or set `RAG_DEVICE=cuda` only when sufficient dedicated VRAM is available.

The CLI prints the same user-friendly stages as the web interface while it runs:
request check, safety authorization, evidence search, analysis, critic review, and
completion.

At completion it prints the cited source list, run metrics, and Langfuse trace ID.
Add `LANGFUSE_PROJECT_ID` to `.env` if you also want a directly clickable trace
URL. The tool span records query/filter metadata but deliberately excludes the
in-memory RAG models and indexes from serialization.

### Hosted Mistral configuration

To use Mistral instead of local Ollama, create an API key in Mistral Studio and set:

```env
LLM_PROVIDER=mistral
LLM_MODEL=mistral-small-latest
MISTRAL_API_KEY=replace-with-your-key
MISTRAL_BASE_URL=https://api.mistral.ai/v1
```

Restart Flask after changing `.env`. The synthesis candidates, critic, RAGAS judge,
token accounting and Langfuse model metadata then use Mistral. Retrieval embeddings
and the cross-encoder remain local; changing the LLM does not rebuild the document
index.

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

## Langfuse traces

Langfuse is separate from the Flask administration page. Configure the three
values from the Langfuse project settings in `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Then run the agent normally through Flask or the CLI. No separate Langfuse function
needs to be called:

```bash
python app.py
# ask a question at http://127.0.0.1:5000

# or run one short-lived CLI trace (the CLI flushes before exit)
python src/agent.py "What does the evidence say about flood displacement in Bangladesh?"
```

Open the Langfuse project and select **Tracing**. One run contains the root agent
observation, retrieval tool, self-consistency chain, three synthesis generations,
and one critic generation. Set `LANGFUSE_DEBUG=True` temporarily if traces do not
appear. The implementation is in `src/observability.py`, `src/agent.py`,
`src/reasoning.py`, and `src/mcp_server.py`.


## Architecture

1. `data/download_documents.py` retrieves the PDFs recorded in `data/sources.json`.
2. `src/ingest.py` extracts and sanitizes every page, then creates parent/child chunks.
3. `src/retrieval.py` fuses BM25 and dense rankings with RRF, resolves child hits to
   parent context, and applies a cross-encoder before returning cited evidence.
4. `src/agent.py` applies L1/L4/budget checks through the course tool-registry
   pattern, retrieves evidence, creates three structured syntheses, takes a Lab B3
   stance majority, and asks a critic agent to verify or correct the answer.
5. `src/mcp_server.py` exposes four documented, error-safe tools over MCP.


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



## Data provenance

The starter corpus contains reports from IDMC, IOM, UNHCR/OHCHR, the World Bank,
and the Asian Development Bank. Each retrieved passage retains publisher, title,
year, page, region/country, and source URL.

## Security boundary

PDFs and MCP tool results are untrusted. Text is normalized, active markup and
control characters are removed, instruction-like passages are visibly marked, and
the synthesis prompt states that evidence can be quoted but never obeyed.
