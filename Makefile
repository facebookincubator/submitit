# Modern Python development with uv
# For development: uv sync --all-groups && source .venv/bin/activate
# Or use: uv run <command>

.PHONY: help which test test_coverage format check_format mypy pylint lint pre_commit sync clean

help:
	@echo Available tasks:
	@echo "  which              - Show Python version"
	@echo "  test               - Run pytest"
	@echo "  test_coverage      - Run pytest with coverage"
	@echo "  format             - Format code with ruff"
	@echo "  check_format       - Check formatting without changes"
	@echo "  mypy               - Run type checking"
	@echo "  pylint             - Run linting"
	@echo "  lint               - Run mypy and pylint"
	@echo "  pre_commit         - Format and lint code"
	@echo "  sync               - Sync dependencies with uv"
	@echo "  clean              - Remove .venv"

which:
	uv run python --version

test:
	uv run pytest -n auto submitthem

test_coverage:
	uv run pytest -v -n auto --cov=submitthem --cov-report=html --cov-report=term --durations=10 submitthem

format:
	uv run ruff format submitthem docs integration
	uv run ruff check --fix submitthem

check_format:
	uv run ruff format --check submitthem docs integration
	uv run ruff check submitthem

mypy:
	uv run mypy submitthem

pylint:
	uv run pylint submitthem

lint: mypy pylint

pre_commit: format lint

sync:
	uv sync --all-groups

clean:
	rm -rf .venv
