.PHONY: install install-dev clean test lint format check ingest build-dataset

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

test:
	pytest tests/ -v

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

check: lint test

ingest:
	python -m tehran_house_price.data.ingest_kaggle

build-dataset:
	python -m tehran_house_price.data.build_dataset
