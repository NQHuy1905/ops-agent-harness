# Ops Agent Harness

An agent harness for an **operations assistant**. It is written from scratch in Python and uses no agent framework. A user objective arrives through the REST API or the CLI. The harness then:
- runs an LLM ↔ tool loop,
- validates every tool call and every tool result,
- persists run state and history to SQLite,
- requires human approval before `create_incident`,
- enforces step, time and loop limits,
- writes JSON logs and a span trace for every run.

Tools (mock implementations): `search_knowledge_base(query)`, `get_service_status(service_name)`, `create_incident(title, description, severity)`.

📄 Design report: [docs/report.md](docs/report.md) · 📮 Postman: [postman/](postman/) · 🗂 Mock dataset: [data/](data/) (public copy: _Google Drive link, see report §15_)

---

## Quick start

Requires Python ≥ 3.10. The Docker image uses 3.12.

```bash
make install                 # creates .venv and installs app + dev deps
cp .env.example .env         # optional; defaults run fully offline
make run                     # API on http://localhost:8000  (OpenAPI UI: /docs)
```

There are two LLM providers:

| `LLM_PROVIDER` | What it does |
|---|---|
| `scripted` (default) | Offline demo. No API key needed. It picks a canned scenario from keywords in the objective, so all Postman flows work without a key. Its answers are labelled `[Scripted demo output]`. |
| `deepseek` | Real LLM: `deepseek-v4-flash` through the OpenAI-compatible API, using native tool calling. Set `DEEPSEEK_API_KEY`. |

```bash
LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... make run
```

Docker:
```bash
docker build -t ops-agent-harness .
docker run -p 8000:8000 -e LLM_PROVIDER=scripted ops-agent-harness
```
If `docker build` fails with an iptables or bridge-network error, the problem is the host, not the Dockerfile. Add `--network=host` to `docker build` and `docker run`.

## Using the API

```bash
# 1. status check -> 200, status=completed
curl -s localhost:8000/runs -H 'content-type: application/json' \
  -d '{"objective":"Check the status of payments-api"}'

# 2. incident -> 202, status=awaiting_approval, pending_approval.tool_call_id
curl -s localhost:8000/runs -H 'content-type: application/json' \
  -d '{"objective":"checkout-web is down, create an incident"}'

# 3. approve (optionally edit the args) / reject
curl -s localhost:8000/runs/<run_id>/approval -H 'content-type: application/json' \
  -d '{"tool_call_id":"<id>","decision":"approve","edited_args":{"title":"checkout-web outage","description":"Checkout returns 5xx for all users","severity":"critical"}}'
curl -s localhost:8000/runs/<run_id>/approval -H 'content-type: application/json' \
  -d '{"tool_call_id":"<id>","decision":"reject","reason":"already tracked"}'

# 4. inspect
curl -s localhost:8000/runs/<run_id>            # status, final answer / error, pending approval
curl -s localhost:8000/runs/<run_id>/history    # ordered steps, tool calls, approvals
curl -s localhost:8000/runs/<run_id>/trace      # span tree (invoke_agent -> chat / execute_tool)
curl -s localhost:8000/incidents                # the mock "external" incident system
```

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | provider + model |
| GET | `/tools` | tool names, JSON schemas, `requires_approval` |
| POST | `/runs` | `{objective, max_steps?, max_wall_time_s?}` → **200** completed / failed / limit_exceeded, **202** awaiting_approval |
| GET | `/runs`, `/runs/{id}` | run views |
| GET | `/runs/{id}/history` | execution history |
| GET | `/runs/{id}/trace` | trace |
| POST | `/runs/{id}/approval` | `{tool_call_id, decision: approve\|reject, edited_args?, reason?}`. Errors: **409** when the id doesn't match the pending call or the call was already decided, **422** when edited args are invalid |
| GET | `/incidents?run_id=` | created incidents |

Errors are returned as `{"error": {"code": "...", "message": "..."}}`.

> **Prototype note:** a run executes **synchronously inside the HTTP request**. The request returns when the run completes, fails, hits a limit, or pauses for approval. The step and time limits keep requests short. The report discusses a background-worker design as future work.

**Failure demos.** Some service names in the dataset inject faults:
- `flaky-service` returns a transient error twice, then the retry succeeds.
- `slow-service` times out on every attempt.
- `broken-service` returns output with an invalid schema.

The objective `"loop on auth-service status"` triggers loop detection. Sending `max_steps: 1` triggers the step limit.

## Using the CLI

```bash
.venv/bin/harness run "Check the status of payments-api"
.venv/bin/harness run "checkout-web is down, create an incident"     # prompts: approve / reject / edit
.venv/bin/harness run "..." --no-input                               # leave it awaiting approval
.venv/bin/harness approve <run_id> <tool_call_id> [--edit-json '{...}']
.venv/bin/harness reject  <run_id> <tool_call_id> --reason "..."
.venv/bin/harness show <run_id>        # history
.venv/bin/harness trace <run_id>       # span tree JSON
```
Add `-v` to print JSON logs to stderr.

## Testing

```bash
make test        # offline pytest suite (scripted LLM, temp SQLite, fake clock)
make coverage    # with coverage report
make lint        # ruff
make test-live   # optional DeepSeek smoke test (needs DEEPSEEK_API_KEY)
```

Postman / Newman: import `postman/ops-agent-harness.postman_collection.json` and `postman/ops-agent-harness.postman_environment.json`, start the server, then run the folders in order. From the command line:
```bash
npx newman run postman/ops-agent-harness.postman_collection.json -e postman/ops-agent-harness.postman_environment.json
```
To regenerate the collection, run `python scripts/build_postman_collection.py`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `scripted` | `scripted` (offline demo) or `deepseek` |
| `DEEPSEEK_API_KEY` | – | required for `deepseek` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `deepseek-v4-flash` | model name |
| `DEEPSEEK_THINKING` | `false` | turns DeepSeek thinking mode on |
| `LLM_TIMEOUT_S` | `30` | per LLM request |
| `LLM_MAX_RETRIES` | `2` | retries for transient LLM errors (timeout, 429, 5xx) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./harness.db` | SQLAlchemy async URL |
| `MOCK_DATA_DIR` | `./data` | knowledge base, service statuses, demo scenarios |
| `MAX_STEPS` | `10` | maximum LLM decisions per run |
| `MAX_WALL_TIME_S` | `60` | maximum active execution time per run (approval wait excluded) |
| `MAX_TOOL_CALLS_PER_STEP` | `5` | maximum tool calls in one LLM response |
| `MAX_MALFORMED_LLM_RESPONSES` | `2` | consecutive invalid responses tolerated |
| `LOOP_DETECT_THRESHOLD` | `3` | the Nth identical tool call aborts the run |
| `TOOL_TIMEOUT_S` | `5` | timeout per tool attempt |
| `TOOL_MAX_RETRIES` | `2` | retries on tool timeout or transient error |
| `TOOL_RETRY_BASE_DELAY_S` | `0.2` | base delay for exponential backoff |
| `LOG_LEVEL` | `INFO` | log level |
| `LOG_FILE` | – | also write JSON logs to this file |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | – | optional OTLP span export (`pip install .[otel]`) |

## Project layout

```
app/
  main.py, api/          FastAPI app, routes, schemas
  cli.py                 Typer CLI
  container.py           composition root shared by API and CLI
  config.py              settings (env vars)
  core/                  runner (loop), executor, guards, retry, history builder, errors, state
  llm/                   LLMClient protocol, DeepSeek client, parser, scripted/demo LLMs, prompt
  tools/                 Tool base, registry, 3 mock tools
  storage/               SQLAlchemy models, repository
  observability/         JSON logging, span tracer
data/                    mock dataset (KB, service statuses, demo scenarios)
tests/                   unit + integration (+ optional live) tests
postman/                 collection + environment
docs/report.md           design report
```
