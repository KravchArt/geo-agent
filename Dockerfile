# syntax=docker/dockerfile:1
# API image for Geo-Agent. One shape: lean, no ML stack.
#
# Both classifier gates (scope and censorship) talk to an OpenAI-compatible
# endpoint over HTTP, so the image never carries model weights or torch. The
# models themselves stay separate services (docker-compose `inference` profile).

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# 1) Install pinned runtime deps first (cached unless lock changes).
#    --no-install-project: we run from source via PYTHONPATH, so we never build
#    the project wheel (keeps eval/ui out of the image cleanly).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Copy only what the service needs at runtime.
COPY common ./common
COPY tools ./tools
COPY backend ./backend
COPY alembic.ini ./

EXPOSE 8000

# Apply migrations, then serve. (Skeleton convenience; real deploys run
# migrations as a separate step.)
CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
