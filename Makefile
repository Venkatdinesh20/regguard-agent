.PHONY: install demo test lint format typecheck check api docker

install:            ## create the virtualenv and install everything
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

demo:               ## run the four demo investigations (no API key needed)
	.venv/bin/python -m app.main

test:               ## run the test suite with coverage
	.venv/bin/python -m pytest --cov=app --cov-report=term-missing

lint:               ## style and correctness checks
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:             ## apply formatting
	.venv/bin/ruff format .
	.venv/bin/ruff check . --fix

typecheck:          ## static types
	.venv/bin/mypy app tests

check: lint typecheck test   ## everything CI runs

api:                ## serve the HTTP API on :8000
	.venv/bin/uvicorn app.api:app --reload --port 8000

docker:             ## build and run the container
	docker build -t regguard:latest .
	docker run --rm -p 8000:8000 --env-file .env regguard:latest
