.PHONY: help install up down logs test lint format typecheck check migrate revision precommit build

help:
	@echo "Targets:"
	@echo "  install    - uv sync (install pinned deps)"
	@echo "  up         - docker compose up --build (api + redis + postgres)"
	@echo "  down       - docker compose down -v"
	@echo "  logs       - tail api logs"
	@echo "  test       - run pytest"
	@echo "  lint       - ruff check"
	@echo "  format     - ruff format"
	@echo "  typecheck  - mypy"
	@echo "  check      - lint + typecheck + test"
	@echo "  migrate    - alembic upgrade head"
	@echo "  revision   - alembic autogenerate (m=\"message\")"
	@echo "  precommit  - run all pre-commit hooks"
	@echo "  build      - docker build the api image"

install:
	uv sync --frozen

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f api

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy common backend tools

check: lint typecheck test

migrate:
	uv run alembic upgrade head

revision:
	uv run alembic revision --autogenerate -m "$(m)"

precommit:
	uv run pre-commit run --all-files

build:
	docker build -t geoagent-api .
