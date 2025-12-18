.PHONY: help install dev db-up db-down run test lint format clean

# Default target
help:
	@echo "ML Agent Platform - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo ""
	@echo "Database:"
	@echo "  make db-up       Start PostgreSQL container"
	@echo "  make db-down     Stop PostgreSQL container"
	@echo "  make db-reset    Reset database (destroy and recreate)"
	@echo ""
	@echo "Development:"
	@echo "  make run         Start the API server"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       Remove cache files"

# Setup
install:
	pip install -r requirements.txt

dev: install
	pip install black isort mypy pytest pytest-asyncio

# Database
db-up:
	docker-compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@echo "PostgreSQL is ready"

db-down:
	docker-compose down

db-reset:
	docker-compose down -v
	docker-compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@echo "Database reset complete"

# Development
run:
	python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Code quality
lint:
	python -m black --check .
	python -m isort --check-only .
	python -m mypy .

format:
	python -m black .
	python -m isort .

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

