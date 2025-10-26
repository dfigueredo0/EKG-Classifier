PY=poetry run
PKG=ekgclf

.PHONY: setup lint format test train eval explain serve docker-build docker-run mlflow

setup:
	poetry install --no-interaction
	pre-commit install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run isort .

test:
	$(PY) pytest -q --maxfail=1 --disable-warnings

train:
	$(PY) python -m $(PKG).train

eval:
	$(PY) python -m $(PKG).eval

explain:
	$(PY) python -m $(PKG).explain

mlflow:
	$(PY) mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
