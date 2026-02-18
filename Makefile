.PHONY: dev test lint fmt db-up db-down migrate download train

# Local development
dev:
	uvicorn infernis.main:app --reload --port 8000

# Testing
test:
	pytest --tb=short -q

test-v:
	pytest -v --tb=short

test-cov:
	pytest --cov=infernis --cov-report=term-missing --tb=short

# Linting
lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Docker dev stack
db-up:
	docker compose up -d db redis

db-down:
	docker compose down

stack:
	docker compose up --build

# Database migrations
migrate:
	alembic upgrade head

migration:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

# Data pipeline
download:
	python scripts/download/00_orchestrator.py

train:
	python scripts/train.py all
