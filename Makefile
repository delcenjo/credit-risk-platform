.PHONY: install train serve test lint up down clean

install:
	pip install -e ".[dev]"

train:
	credit-scoring train

serve:
	credit-scoring serve --reload

test:
	pytest

lint:
	ruff check src tests

up:
	docker compose up --build

down:
	docker compose down -v

clean:
	rm -rf artifacts .pytest_cache
