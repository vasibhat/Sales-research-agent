# Sales Research Agent

Type a company name. The agent researches it and gives back a sales meeting briefing: company summary, latest news, lead score, sales opportunities, talking points and matching case studies.

Built from scratch with the OpenAI SDK. There is no LangChain, LangGraph or CrewAI, so every part of the agent is plain Python you can read.

## Setup (one time)

```bash
cd ~/sales-research-agent
uv sync                      # installs everything into .venv
cp .env.example .env         # then put your OPENAI_API_KEY in .env
uv run sales-agent ingest    # builds the search index from data/docs
```

## Use it

```bash
uv run sales-agent research "Research Infosys for Monday's meeting"
uv run sales-agent research "Is Delhivery a good lead?" --json
uv run sales-agent prefs add "Keep talking points short"
uv run sales-agent serve     # API at http://127.0.0.1:8080/docs
```

## Check it

```bash
uv run pytest                    # 37 tests, uses a fake model, free, no key needed
uv run python evals/run_evals.py # real model on 5 test questions, scores quality
```

## How it works

```
question -> input guardrail -> PLAN -> RESEARCH LOOP -> WRITE BRIEFING -> output guardrail -> memory
                                        |
                       model asks for a tool -> our code runs it -> result goes back
                       (repeat until the model stops asking, or MAX_STEPS is reached)
```

## Where each phase lives

| Phase | What it teaches | File |
|---|---|---|
| 1. Basic loop | The while-loop, tool_calls, tool_call_id | `agent.py` -> `_research` |
| 2. Multiple tools | Tool schemas, checking arguments, running tools | `tools/__init__.py`, `tools/*.py` |
| 3. State | What the agent knows during one run | `state.py` |
| 4. Planning | Decide if and how to research before starting | `planner.py` |
| 5. Memory | Remember companies and preferences between runs | `memory.py` |
| 6. RAG as a tool | Chunk, embed, search our own documents | `rag/*.py`, `tools/internal_docs.py` |
| 7. Guardrails | Block bad input, unsafe links and personal data in output | `guardrails.py`, `briefing.py` |
| 8. Observability | Trace every call: time, tokens, cost | `tracing.py`, `logs/traces/*.jsonl` |
| 9. Evaluation | Rule checks + LLM judge on fixed questions | `evals/` |
| 10. Production | HTTP API, API key, Docker | `api.py`, `Dockerfile` |

Start reading at `agent.py`. Everything else plugs into it.

## Tools

| Tool | Source | Needs a key? |
|---|---|---|
| `search_company` | Wikipedia | No |
| `get_news` | Google News RSS (India edition) | No |
| `fetch_website` | The company's own website | No |
| `search_internal_docs` | `data/docs/` (RAG) | OpenAI (embeddings) |
| `score_lead` | Rules in `data/icp.json` | No |

## Make it yours

- Replace `data/docs/*.md` with your company's real product sheets and case studies, then run `ingest` again.
- Edit `data/icp.json` to set your target industries, company size and buying signals.
- Set `SELLER_NAME` in `.env`.

LedgerLeaf Technologies and its documents are made up, for demo use only.

---

