PYTHON    := .venv/bin/python
PIP       := .venv/bin/pip
UVICORN   := .venv/bin/uvicorn
STREAMLIT := .venv/bin/streamlit
COMPOSE   := docker compose

# WSL2: Python vertraut dem System-CA-Bundle nicht ohne diesen Fix
SSL_ENV := SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
           REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

.DEFAULT_GOAL := help

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Multimodal RAG — available commands:"
	@echo ""
	@echo "  make setup        Create virtualenv and install dependencies"
	@echo "  make api          Start FastAPI backend  (localhost:8000)"
	@echo "  make ui           Start Streamlit frontend  (localhost:8501)"
	@echo "  make test         Run test suite"
	@echo "  make docker-up    Build and start full stack via Docker Compose"
	@echo "  make docker-down  Stop Docker Compose stack"
	@echo "  make clean        Remove generated artifacts"

# ─── Setup ────────────────────────────────────────────────────────────────────

.PHONY: setup
setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ─── Run (lokal) ──────────────────────────────────────────────────────────────

.PHONY: api
api:
	$(SSL_ENV) PYTHONPATH=. $(UVICORN) src.api.main:app \
	  --host 127.0.0.1 --port 8000 --reload

.PHONY: ui
ui:
	$(SSL_ENV) PYTHONPATH=. $(STREAMLIT) run src/ui/streamlit_app.py \
	  --server.port 8501 --server.address 127.0.0.1

# ─── Tests ────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(SSL_ENV) PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

# ─── Docker ───────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	@$(COMPOSE) version
	$(COMPOSE) up --build

.PHONY: docker-down
docker-down:
	@$(COMPOSE) version
	$(COMPOSE) down

# ─── Cleanup ──────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf logs/*.log data/images/* data/vectorstore/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
