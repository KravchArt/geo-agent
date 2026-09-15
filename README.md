# Geo-Agent

An LLM agent for travel and location questions. It answers by calling geo and web
tools, behind gates that decide what it may work on, and every place or source it
cites is traceable back to a real tool result.

- **How it works:** [`docs/architecture.md`](docs/architecture.md)
- **Where each piece runs (GPU, local, hosted):** [`docs/deployment.md`](docs/deployment.md)

---

## Quickstart

Zero config — starts ui + api + redis + postgres with the mock model:

```bash
docker compose up --build
```

UI on `http://localhost:8501`. Check the backend:

```bash
curl -s localhost:8000/health | jq
# {"status":"ok","checks":{"postgres":true,"redis":true},"llm_mode":"mock"}
```

`/health` really connects to Postgres and Redis and returns **503** if either is
down. The api container applies migrations on start.

```bash
curl -s -X POST localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: 11111111-1111-4111-8111-111111111111' \
  -d '{"session_id":"22222222-2222-4222-8222-222222222222","message":"Plan one day in Prague"}' \
  | jq

curl -s localhost:8000/api/v1/conversations \
  -H 'X-Client-ID: 11111111-1111-4111-8111-111111111111' | jq
```

The Streamlit UI generates its own anonymous client UUID; the fixed values above
are only reproducible curl examples. PostgreSQL is the durable source for the
conversation list and messages. Redis contains disposable cache/scratch state
and the records behind short-lived `plc_`/`src_` refs.

`docker compose down` stops the stack without removing the PostgreSQL volume.
`docker compose down -v` also deletes that volume and therefore permanently
wipes conversation history and observability data.

### Conversation history

The browser stores an anonymous ID in the `geoagent_client_id` localStorage key
and sends it as `X-Client-ID`. The active conversation UUID is kept separately
in the `conversation` URL query parameter. Consequently, F5 or reopening the URL
reloads the same transcript from the API rather than creating a new session.

The sidebar can create, select and delete conversations through:

- `GET /api/v1/conversations?limit=50&offset=0`
- `GET /api/v1/conversations/{session_id}`
- `DELETE /api/v1/conversations/{session_id}`

List/detail/delete require `X-Client-ID`; the UI also includes it on chat and
stream requests. This is anonymous browser scoping, not authentication: a client
ID is a bearer-like identifier, provides no verified identity or cross-device
sync, and must be replaced or bound to real login/authorization in a multi-user
deployment. Delete removes the user-visible transcript; the separate request and
trace observability records remain subject to their own retention policy.

---

## Run with a real model (OpenRouter)

The mock proves the plumbing; a hosted model makes the agent actually call tools.
OpenRouter is OpenAI-compatible, so this is configuration only.

**1. Put your key and models in `.env`** (copy `.env.example` first):

```bash
LLM_MODE=vllm                                   # any OpenAI-compatible endpoint
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3-30b-a3b-instruct-2507      # the agent
LLM_API_KEY=sk-or-...
LLM_TIMEOUT=300                                 # a tool loop is several calls

SCOPE_PROVIDER=llm                              # classify scope with a model
SCOPE_BASE_URL=https://openrouter.ai/api/v1
SCOPE_API_KEY=sk-or-...
SCOPE_MODEL=qwen/qwen3-8b                       # small and cheap is enough

WEB_SEARCH_PROVIDERS=["exa"]                    # exa | tavily | firecrawl
EXA_API_KEY=...
```

**2. Behind a proxy?** Set it explicitly — the app ignores `HTTP_PROXY`/`ALL_PROXY`
from the shell on purpose, so nothing reroutes silently:

```bash
LLM_HTTP_PROXY=http://127.0.0.1:12334
TOOLS_HTTP_PROXY=http://127.0.0.1:12334
```

**3. Run.** Locally the API needs migrations applied once — only the Docker image
does that on start:

```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run uvicorn backend.app.main:app --port 8080

API_URL=http://localhost:8080 uv run --with streamlit==1.53.0 streamlit run ui/app.py
```

Streamlit is intentionally not a project dependency — it ships in its own image.

**4. Confirm the loop really ran**, rather than the model answering from memory:

```sql
SELECT num_model_calls, num_tool_calls, extra->'grounding'
FROM metrics ORDER BY created_at DESC LIMIT 5;
```

`num_model_calls > 1` with `num_tool_calls > 0` means it called tools and came
back with the results.

> Enabling a provider without its key **fails at startup** by design: a
> half-configured tool that errors at runtime is worse than not booting.
> `places_search` and `routing_tool` need their configured provider settings;
> textual cities, anchors, and routing points additionally use shared place
> resolution. `TEXT_PLACE_RESOLUTION_PROVIDERS` defines one named-place chain
> shared by nearby-search anchors and textual routing points (normally 2GIS,
> then TomTom, then the shared TomTom geocoder). Its behavior does not change
> when either model-facing tool is disabled.
> With no
> providers configured, the tools are simply absent and the prompt is built
> from the tools that do exist. With GraphHopper and OSRM enabled, routing tries
> GraphHopper first. Textual points are resolved to `plc_` refs once before
> provider attempts, so a routing fallback reuses their hidden coordinates.
> With `ROUTING_PROVIDERS=["twogis","graphhopper"]`, 2GIS Routing/Distance
> Matrix is primary for both route construction and ranked candidates. Missing
> routes, transient/provider failures, and provider-local capability gaps fall
> back to GraphHopper using the same resolved `plc_` refs. Authentication,
> configuration, and internal contract errors remain visible. GraphHopper's
> routing HTTP exchange retains its 2-second deadline.
> With `PLACES_SEARCH_PROVIDERS=["twogis","tomtom"]`, 2GIS checks locality
> coverage first and is the primary POI source in covered regions. An uncovered,
> empty, or transiently failed 2GIS search falls back to TomTom. Their result sets
> are not mixed. The coordinator resolves a textual
> city or nearby anchor once before provider attempts and passes the resulting
> `plc_` ref to every fallback provider. A nearby anchor with `city` uses the
> shared named-place chain first: 2GIS in a covered city or settlement, then
> TomTom, then TomTom geocoding. City/area boundaries remain geocoder-backed. Explicit
> object qualifiers such as `парк`, `метро`, `территория`, or `причал` are
> preserved; genuinely separate branches return a bounded clarification instead
> of being selected locally.

---

## Local development

```bash
uv sync --frozen                    # .venv with pinned deps
```

```bash
uv run ruff check . && uv run ruff format .
uv run mypy common backend tools
```

Tests default to `localhost`, so with `docker compose up` running they exercise
real Postgres and Redis. Integration tests **skip** when a service is unreachable
locally but **fail** in CI (`REQUIRE_INTEGRATION=1`), so they can never silently
no-op. The suite is hermetic: proxy variables and gate providers are pinned in
`conftest.py`, so it behaves the same on every machine.

### Run database tests safely

Use a dedicated, disposable database. Do **not** point the complete suite at the
development or production database: `backend/tests/test_migrations.py`
deliberately runs `alembic downgrade base` before rebuilding the schema, which
deletes every table and row in the selected database.

Start the dependencies and create a test-only database (choose another fresh
name if `geoagent_test` already contains anything you need):

```bash
docker compose up -d postgres redis
docker compose exec -T postgres sh -lc 'createdb -U "$POSTGRES_USER" geoagent_test'
```

Export overrides for that database and a separate Redis logical DB:

```bash
export POSTGRES_HOST=localhost \
  POSTGRES_DB=geoagent_test \
  REDIS_HOST=localhost \
  REDIS_DB=15 \
  APP_ENV=dev \
  LLM_MODE=mock \
  SCOPE_PROVIDER=rule_based \
  CENSORSHIP_PROVIDER=rule_based \
  IP_GEOLOCATION_ENABLED=true \
  IP_GEOLOCATION_SELF_LOOKUP_ENABLED=true \
  IP_GEOLOCATION_BASE_URL=http://127.0.0.1:9 \
  IP_GEOLOCATION_TIMEOUT=0.1 \
  PLACES_SEARCH_PROVIDERS='[]' \
  ROUTING_PROVIDERS='[]' \
  WEB_SEARCH_PROVIDERS='[]' \
  REQUIRE_INTEGRATION=1

uv run pytest -q                     # tests only
make check                            # ruff + mypy + pytest
```

This verifies the from-scratch migration, durable conversation endpoints,
pipeline integration and Redis behavior without calling a real model or
external provider. Both test commands must be run only while these disposable
database overrides are present in the environment.

For the browser acceptance check, create two conversations, refresh the page,
switch between them, delete one, then restart the UI/API and repeat. To prove the
transcript is not coming only from Redis, delete just the active conversation's
legacy/snapshot cache keys and refresh it:

```bash
docker compose exec -T redis redis-cli DEL \
  'session:<conversation-uuid>:history' \
  'session:<conversation-uuid>:last_user_message' \
  'session:<conversation-uuid>:last_gate_results' \
  'session:<conversation-uuid>:last_pipeline_status' \
  'session:<conversation-uuid>:last_answer'
```

Its list entry and messages must remain available from PostgreSQL, and a new
follow-up must still receive the prior turns as context. A new conversation must
remain isolated from them.

Hooks (ruff, mypy, gitleaks): `uv run pre-commit install`.

Migrations:

```bash
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "msg"
```

---

## Deployment

[`docs/deployment.md`](docs/deployment.md) — where each model runs and how to size
them. [`docs/runbook-a100.md`](docs/runbook-a100.md) — a step-by-step procedure for
bringing the whole stack up on a single shared A100, gates and UI included.

## Configuration

All config is environment variables, typed in
[`backend/app/config.py`](backend/app/config.py). [`.env.example`](.env.example)
lists every variable; `.env` is git-ignored — **never commit real secrets**.

| Variable | Purpose |
|---|---|
| `LLM_MODE` | `mock` (default, CI) or `vllm` for any OpenAI-compatible server |
| `SCOPE_PROVIDER` | `rule_based` (regex), `llm`, `classifier` (e5 encoder + LLM fallback), or `no_scoper` |
| `CENSORSHIP_PROVIDER` | `rule_based` (regex) or `model` (+ `CENSORSHIP_BASE_URL/MODEL`) |
| `*_PROVIDERS` | tool providers as JSON arrays; empty disables the tool |
| `*_HTTP_PROXY` | explicit outbound proxy; shell proxy variables are ignored |

`APP_ENV=prod` fails fast on `APP_DEBUG=true` or `LLM_MODE=vllm` without a key.

CI needs **no secrets**: it never calls external APIs and always runs
`LLM_MODE=mock`. For later use, `gh secret set LLM_API_KEY --repo yaGeoAgent/GeoAgent`.

---

## CI

`.github/workflows/ci.yml` runs on push and PR to `main` and `develop`:
`ruff` · `mypy` (strict) · `pytest` (with redis + postgres services) · `gitleaks`
· `pip-audit` · `docker-build`.

**Hard rule:** CI never reaches a real external API and never needs inference.

---

## Manual setup (not doable from the CLI)

**Branch protection on `main` is NOT enforced.** The repo is private on the free
plan, where GitHub blocks both classic protection and rulesets:

> `403: Upgrade to GitHub Pro or make this repository public to enable this feature.`

So `main` accepts direct pushes and CI is advisory. To enforce it, upgrade the org
to GitHub Team (keeps the repo private) or make the repo public, then apply a
ready-made payload:

```bash
gh api -X PUT repos/yaGeoAgent/GeoAgent/branches/main/protection \
  -H "Accept: application/vnd.github+json" --input .github/branch-protection.json
# or the equivalent ruleset:
gh api -X POST repos/yaGeoAgent/GeoAgent/rulesets \
  -H "Accept: application/vnd.github+json" --input .github/main-ruleset.json
```

Also human-only: team→repo roles, real secret values, and enabling secret
scanning / push protection / Dependabot in *Settings → Code security*.


LLM-as-a-Judge setup, SSH tunnel, scoring commands, rubric, and weights are
documented in [`eval/README.md`](eval/README.md).
