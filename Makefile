# ============================================
# Makefile — Estructura Topológica
# ============================================
# USO:
#   make install     Instalar todo (prod + dev)
#   make test        Correr tests con pytest
#   make test-cov    Tests + cobertura
#   make lint        Verificar linting (ruff check)
#   make format      Formatear código (ruff format)
#   make check       Lint + tests (pre-commit check)
# ============================================

.PHONY: install install-dev test test-cov lint format check

VENV_DIR ?= .venv
VENV_PY  = $(VENV_DIR)/bin/python
VENV_RUF = $(VENV_DIR)/bin/ruff

# ───── Instalación ─────

install: $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -e .

install-dev: $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -e .
	$(VENV_DIR)/bin/pip install -r requirements-dev.txt

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

# ───── Tests ─────

test:
	$(VENV_PY) -m pytest -v --tb=short

test-cov:
	$(VENV_PY) -m pytest -v --tb=short --cov=src/tda --cov-report=term-missing

# ───── Linting & formateo ─────

lint:
	$(VENV_RUF) check src/tda tests/

format:
	$(VENV_RUF) format src/tda tests/

# ───── Check completo ─────

check: lint test
