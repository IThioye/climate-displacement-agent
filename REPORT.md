# Climate Displacement Evidence Agent — Project Report

> Draft: replace every `[MEASURE]` marker with output from the final repository.

## 1. Problem statement

The intended user is a humanitarian research analyst preparing a short regional
brief before a programme-planning meeting. The agent answers questions such as:
“How does documented flood-displacement risk in Bangladesh compare with broader
Asia-Pacific evidence, and where are the evidence gaps?” A general chatbot can
produce fluent background information, but it does not guarantee that each claim
comes from the team's approved document set or preserve publisher, year, and page.

The agent searches a curated multi-publisher corpus, distinguishes observed figures
from projections, and returns a cited EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE
brief. It is a research assistant, not a forecasting or aid-allocation system.

## 2. Architecture

The ingestion layer extracts PDF pages, sanitizes untrusted text, and creates
overlapping parent/child chunks with metadata. Retrieval fuses BM25 and dense
rankings through RRF. A cross-encoder reranks candidate parents before context
assembly. Three synthesis calls generate independent structured candidates, and a
critic agent returns a visible verdict and final answer. MCP exposes four tools;
L1/L4 filters and hard budgets constrain every run. Langfuse records the top-level
agent, retrieval tools, synthesis calls, critic call, prompt hash, and agent version.

The parent-child decision is described in `docs/architecture.md`.

## 3. Evaluation

Ten questions in `evaluation/questions.json` cover global counts, Bangladesh risk,
Asia-Pacific policy, legal terminology, and projections. The baseline is TF-IDF over
child chunks. The final system adds dense retrieval, RRF, parent expansion, and
cross-encoder reranking.

| Metric | Baseline | Final | Technique associated with change |
|---|---:|---:|---|
| context_recall | [MEASURE] | [MEASURE] | Dense retrieval + RRF + parent expansion |
| context_precision | [MEASURE] | [MEASURE] | Cross-encoder reranking |
| faithfulness | [MEASURE] | [MEASURE] | Sanitized evidence-only prompt + critic |
| answer_relevancy | [MEASURE] | [MEASURE] | Structured synthesis + self-consistency |

Across ten full test runs: mean latency was `[MEASURE]` seconds, estimated mean cost
was `$[MEASURE]`, and tool distribution was `[MEASURE]`. A quota test deliberately
triggered the sixth `search_evidence` call and recorded the refusal.

## 4. Security

| Injection test | Before L1/L4 | After L1/L4 |
|---|---|---|
| Direct instruction override | [MEASURE] | blocked |
| “Disregard previous instructions” variant | [MEASURE] | blocked |
| Role injection | [MEASURE] | blocked |
| Fake system tag | [MEASURE] | blocked |
| Prompt extraction | [MEASURE] | blocked |

The indirect-injection test places “ignore previous instructions” inside retrieved
document text. `sanitise_external_content` marks it as untrusted, and the synthesis
prompt explicitly forbids following instructions inside evidence. Unknown actions
default to human confirmation at L4.

## 5. EU AI Act assessment

The deployed scope is an advisory document-research assistant with no authority to
make decisions about individuals, migration status, public benefits, or aid. The
working assessment is that it is not an Annex III high-risk decision system, while
the user-facing AI interaction requires transparent disclosure. The CLI therefore
states that the briefing is AI-generated and that cited sources must be verified.

This assessment follows the narrow migration/asylum uses listed in recital 60 and
Annex III, and the direct-interaction disclosure obligation in Article 50 of
[Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj).

This assessment must be revisited if the system is connected to individual case
files, eligibility decisions, border/asylum processes, or automated resource
allocation. Such a change would materially alter both risk and required controls.

## 6. Limitations and what's next

First, PDF extraction can scramble tables and multi-column layouts. This manifests
when a figure loses its row or unit. The next sprint would add layout-aware parsing
and table-level evaluation. Second, the corpus is intentionally static; a newly
published disaster report will not appear until the manifest is reviewed and the
index rebuilt. The next sprint would add a signed source-update workflow with human
approval, checksum logging, and regression tests for the ten evaluation questions.

The hash-based retrieval and embedding fallbacks keep the repository runnable but
are weaker than the configured transformer models. Final grading measurements must
state which backend was actually active.

## 7. AI use disclosure

| Component | Written by human | AI-assisted | AI-generated |
|---|:---:|:---:|:---:|
| Problem statement |  |  | X |
| Architecture |  |  | X |
| Core agent loop |  |  | X |
| MCP server |  |  | X |
| Guardrails |  |  | X |
| Retrieval pipeline |  |  | X |
| Report draft |  |  | X |

This initial scaffold and report draft were AI-generated. Before submission, the
group must review and materially adapt them, explain every function, validate every
measurement, and update this table to reflect the final division of work.
