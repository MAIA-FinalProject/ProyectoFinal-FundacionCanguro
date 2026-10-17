# Atajos del proyecto (ticket 2.3). Requieren `make` y `uv`.
# En Windows sin make, usa directamente los comandos `uv run ...` de cada regla.

PYTHON ?= uv run python

.DEFAULT_GOAL := help
.PHONY: help build-features list-steps dry-run ingest test lint

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

build-features: ## Pipeline de datos de extremo a extremo (ingesta -> HW -> target)
	$(PYTHON) -m src.pipeline

list-steps: ## Lista los pasos del pipeline y si ya estan disponibles
	$(PYTHON) -m src.pipeline --list

dry-run: ## Muestra que correria el pipeline, sin ejecutarlo
	$(PYTHON) -m src.pipeline --dry-run

ingest: ## Solo la ingesta de cohortes (data/raw -> data/interim)
	$(PYTHON) -m src.pipeline --only ingesta

test: ## Corre las pruebas
	uv run pytest

lint: ## Linter (lo mismo que valida el CI)
	uv run ruff check .
