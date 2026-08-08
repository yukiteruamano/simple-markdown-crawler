# simple-markdown-crawler — check, lint, test, build y publish con uv
RUFF  := uv run ruff
PYTEST := uv run pytest

.PHONY: help install sync check format lint lint-fix test build publish clean clean-dist

.DEFAULT_GOAL := help

help:  ## Muestra los targets disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependencias (incluye las de desarrollo)
	uv sync --dev

sync:  ## Actualiza el lock y el entorno
	uv lock --upgrade
	uv sync --dev

check:  ## Lint estático (ruff check)
	$(RUFF) check .

format:  ## Formatea el código (ruff format)
	$(RUFF) format .

lint: check format  ## Lint + verificación de formato

lint-fix:  ## Auto-corrige lint y formato
	$(RUFF) check . --fix
	$(RUFF) format .

test:  ## Ejecuta los tests
	$(PYTEST)

build: test  ## Construye sdist + wheel en dist/ (tras pasar los tests)
	uv build

publish: build  ## Publica en PyPI (requiere UV_PUBLISH_TOKEN o credenciales)
	uv publish

clean: clean-dist  ## Elimina artefactos de build
	rm -rf build *.egg-info

clean-dist:
	rm -rf dist
