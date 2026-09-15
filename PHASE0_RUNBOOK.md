# Phase 0 runbook

## What the pipeline proves

A message follows this path:

`Streamlit UI -> POST /api/v1/chat -> Redis session state -> mock LLM -> Postgres audit log -> UI response`

Phase 0 intentionally does not include real geo tools, maps, a scope gate, validators,
or a production LLM.

## Start

```bash
cp .env.example .env
docker compose up --build
```

Open:

- UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Smoke checks

```bash
curl -s http://localhost:8000/health | jq
```

Expected: HTTP 200, both `postgres` and `redis` are `true`, and `llm_mode` is `mock`.

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"phase0-smoke","message":"Plan one day in Prague"}' | jq
```

Expected fields: `request_id`, `answer`, `llm.mode = mock`, and non-empty
`llm.trace.steps`.

## Check Redis

```bash
docker compose exec redis redis-cli GET session:phase0-smoke:last_user_message
docker compose exec redis redis-cli GET session:phase0-smoke:last_answer
```

## Check Postgres

```bash
docker compose exec postgres psql -U geoagent -d geoagent -c \
  "SELECT id, session_id, status, user_query FROM request ORDER BY created_at DESC LIMIT 5;"

docker compose exec postgres psql -U geoagent -d geoagent -c \
  "SELECT model, mode, content FROM model_response ORDER BY created_at DESC LIMIT 5;"
```

## Automated checks

With the stack running:

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy common backend tools
uv run pytest -v
```

Or:

```bash
make check
```

## Stop

```bash
docker compose down
```

To remove Postgres data too:

```bash
docker compose down -v
```
