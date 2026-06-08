# Multimodal RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system built with Python and LangChain that understands text, images, and tables from PDF documents. Ask questions about your documents — including their figures and tables — and get grounded, cited answers via a chat interface.

---

## Features

- **Multimodal Ingestion** — PDFs are parsed at high resolution (Tesseract OCR + Poppler). Text, embedded images (described by GPT-4o Vision), and tables (Markdown + AI-generated summary) are all indexed.
- **Semantic Retrieval + Reranking** — Cosine-similarity search over ChromaDB, followed by a CrossEncoder (or Cohere) reranker for precision.
- **Streaming Chat UI** — Streamlit frontend with conversation memory (last 20 messages).
- **FastAPI Backend** — RESTful API with streaming support, health endpoint, and OpenAPI docs.
- **Provider Agnostic** — swap LLM, embedding model, and vector store via a single config file. No code changes required.
- **Observability** — Prometheus metrics + pre-built Grafana dashboard out of the box.
- **RAGAS Evaluation** — built-in evaluator for faithfulness, answer relevancy, context recall, and context precision.
- **Docker Compose** — one command spins up the full stack (API, UI, ChromaDB, Prometheus, Grafana).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (port 8501)                    │
│           File upload · Chat interface · Source display             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP / SSE
┌───────────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend  (port 8000)                   │
│   /api/v1/rag/query   /api/v1/documents   /api/v1/health            │
└──────────┬───────────────────────────────────────┬──────────────────┘
           │                                       │
┌──────────▼──────────┐                 ┌──────────▼──────────────────┐
│  IngestionPipeline  │                 │     MultimodalRAGChain      │
│                     │                 │                             │
│  PDFLoader          │                 │  SemanticRetriever          │
│  ├─ hi_res OCR      │                 │  └─ VectorStore.search()    │
│  ├─ image extract   │                 │                             │
│  └─ table detect    │                 │  CrossEncoderReranker       │
│                     │                 │  └─ sentence-transformers   │
│  TextProcessor      │                 │                             │
│  ImageProcessor     │                 │  LLM (GPT-4o default)       │
│  └─ GPT-4o Vision   │                 │  └─ multimodal messages     │
│  TableProcessor     │                 │     (text + base64 images)  │
│  └─ Summary + MD    │                 │                             │
└──────────┬──────────┘                 └──────────┬──────────────────┘
           │                                       │
┌──────────▼───────────────────────────────────────▼──────────────────┐
│                    ChromaDB VectorStore  (local / Docker)           │
│                    OpenAI text-embedding-3-large                    │
└─────────────────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                   Monitoring (optional)                             │
│   Prometheus (port 8001/9090)  ·  Grafana Dashboard (port 3000)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Overview

| Package | Description |
|---|---|
| `src/api/` | FastAPI app, routers (`rag`, `documents`, `health`), Pydantic schemas |
| `src/ingestion/` | Loaders (PDF, text, image) and processors (text chunking, image captioning, table summarisation) |
| `src/rag/` | `MultimodalRAGChain`, conversation memory, prompt templates |
| `src/retrieval/` | `SemanticRetriever`, `CrossEncoderReranker`, `CohereReranker` |
| `src/vectorstore/` | `ChromaVectorStore` (default), `QdrantVectorStore` (ready) |
| `src/embeddings/` | OpenAI and HuggingFace embedding providers |
| `src/llm/` | OpenAI, Azure OpenAI, Ollama, HuggingFace LLM providers |
| `src/evaluation/` | RAGAS-based evaluator |
| `src/monitoring/` | Prometheus metrics |
| `src/core/` | Abstract base classes (`interfaces.py`), custom exceptions |
| `src/ui/` | Streamlit app |

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| `tesseract-ocr` + `poppler-utils` | `sudo apt install tesseract-ocr poppler-utils` |
| OpenAI API key | Required for embeddings and default LLM |

### 1 — Clone & install

```bash
git clone https://github.com/<your-username>/llm_multimodal_rag.git
cd llm_multimodal_rag

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env and set at minimum:
#   OPENAI_API_KEY=sk-...
```

All other settings have sensible defaults. See [Configuration](#configuration) for the full reference.

### 3 — Start

```bash
# Terminal 1 — FastAPI backend
./start_api.sh

# Terminal 2 — Streamlit UI
./start_ui.sh
```

Open **http://localhost:8501** in your browser, upload a PDF, and start chatting.

API docs (Swagger UI) are available at **http://localhost:8000/docs**.

---

## Docker Compose (full stack)

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Streamlit UI | http://localhost:8501 |
| ChromaDB | http://localhost:8002 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

---

## Configuration

All settings live in `config/config.yaml` and can be overridden via environment variables (or `.env`).

### LLM Provider

```yaml
# config/config.yaml
llm:
  provider: openai          # openai | azure | ollama | huggingface
  model: gpt-4o
  temperature: 0.0
  max_tokens: 2048
```

| Provider | Required env vars |
|---|---|
| `openai` (default) | `OPENAI_API_KEY` |
| `azure` | `AZURE_OPENAI_API_KEY`, `LLM_AZURE_ENDPOINT`, `LLM_AZURE_DEPLOYMENT` |
| `ollama` | `LLM_OLLAMA_BASE_URL` (default: `http://localhost:11434`) |
| `huggingface` | `HUGGINGFACE_API_KEY` |

### Embedding Provider

```yaml
embedding:
  provider: openai          # openai | huggingface
  model: text-embedding-3-large
```

### Vector Store

```yaml
vectorstore:
  provider: chroma          # chroma | qdrant
  persist_directory: ./data/vectorstore
```

### Reranker

```yaml
reranker:
  enabled: true
  provider: crossencoder    # crossencoder | cohere
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 5
```

---

## API Reference

### `POST /api/v1/rag/query`

Query the RAG system.

```json
{
  "query": "What is the attention mechanism?",
  "chat_history": []
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "type": "text",
      "content": "...",
      "source_file": "paper.pdf",
      "page": 3,
      "score": 0.92
    }
  ],
  "model": "gpt-4o",
  "latency_ms": 1240
}
```

### `POST /api/v1/documents/ingest`

Upload and ingest a document.

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@paper.pdf"
```

### `GET /api/v1/documents/`

List all ingested documents.

### `DELETE /api/v1/documents/{source_file}`

Remove a document and all its chunks from the vector store.

### `GET /api/v1/health`

Health check.

---

## PDF Ingestion Pipeline

Documents are processed in three parallel tracks:

```
PDF
 ├─ TextProcessor     → chunk (512 tokens, 50 overlap) → embed → store
 ├─ ImageProcessor    → GPT-4o Vision description → embed → store (+ raw image)
 └─ TableProcessor    → GPT-4o summary (embedded) + Markdown (context) → store
```

The `hi_res` strategy (Tesseract + Poppler) is used by default. It correctly detects and separates text blocks, images, and tables. A `fast` fallback is available for plain-text PDFs.

---

## Evaluation (RAGAS)

The built-in evaluator measures four RAGAS metrics against a gold-standard question set:

| Metric | What it measures |
|---|---|
| Faithfulness | Are all claims in the answer supported by the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Recall | Were the relevant chunks actually retrieved? |
| Context Precision | Are the retrieved chunks relevant (no noise)? |

```python
from src.evaluation.evaluator import EvaluationSample, RAGEvaluator

samples = [
    EvaluationSample(
        question="What is multi-head attention?",
        ground_truth="Multi-head attention runs attention in parallel across h heads...",
    ),
]

evaluator = RAGEvaluator(rag_chain)
result = evaluator.evaluate(samples)
print(result)
# EvaluationResult(faithfulness=0.94, answer_relevancy=0.91, ...)
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

The test suite contains 33 tests across unit, integration, and evaluation layers. All external API calls are mocked — no real API key is needed to run the tests.

---

## Project Structure

```
llm_multimodal_rag/
├── config/
│   ├── config.yaml          # Main configuration
│   └── logging.yaml         # Logging configuration
├── docker/
│   ├── docker-compose.yml
│   ├── grafana/dashboards/  # Pre-built Grafana dashboard
│   └── prometheus/
├── src/
│   ├── api/                 # FastAPI application
│   ├── core/                # Abstract base classes & exceptions
│   ├── embeddings/          # Embedding providers
│   ├── evaluation/          # RAGAS evaluator
│   ├── ingestion/           # Loaders & processors
│   ├── llm/                 # LLM providers
│   ├── monitoring/          # Prometheus metrics
│   ├── rag/                 # RAG chain & memory
│   ├── retrieval/           # Retriever & reranker
│   ├── ui/                  # Streamlit app
│   ├── utils/               # Logging setup
│   └── vectorstore/         # Vector store implementations
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── .env.example
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── start_api.sh
└── start_ui.sh
```

---

## WSL2 Notes

If you are running on WSL2 and encounter SSL certificate errors (`CERTIFICATE_VERIFY_FAILED`) with OpenAI API calls, add the following to your `.env`:

```
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

The `start_api.sh` and `start_ui.sh` scripts set these automatically.

---

## Roadmap / Possible Next Steps

### Retrieval Quality
- [ ] **Hybrid Search** — combine dense (semantic) and sparse (BM25/keyword) retrieval with Reciprocal Rank Fusion for better recall on exact terms and proper nouns
- [ ] **Contextual Chunking** — prepend a document-level summary to each chunk before embedding (reduces context loss at chunk boundaries)
- [ ] **HyDE** (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it, and use it as the query vector
- [ ] **MMR retrieval** — Maximal Marginal Relevance to reduce redundant chunks in the context window

### Multimodal
- [ ] **Layout-aware chunking** — respect document layout (columns, headers, captions) during chunking
- [ ] **Multi-vector image indexing** — store both the raw image and its GPT-4o description as separate vectors
- [ ] **OCR language support** — configure Tesseract language packs for non-English PDFs

### Infrastructure
- [ ] **Qdrant** as production vector store (client already implemented, needs Docker integration)
- [ ] **Redis semantic cache** — cache embeddings and LLM responses for repeated queries (skeleton in `.env`)
- [ ] **Async ingestion queue** — background task queue (Celery / ARQ) for large document batches
- [ ] **Authentication** — API key middleware or OAuth2 for the FastAPI backend
- [ ] **Rate limiting** — per-user request throttling

### Observability & Evaluation
- [ ] **LangSmith tracing** — already wired in `main.py`, just needs `LANGSMITH_ENABLED=true`
- [ ] **Automated RAGAS CI** — run the evaluator on every PR against a fixed golden dataset to catch regressions
- [ ] **Chunk-level provenance** — surface exact character offsets so the UI can highlight the cited passage in the original PDF

### UX
- [ ] **Multi-document sessions** — let users organise documents into named collections
- [ ] **Source highlighting** — render the cited PDF page and highlight the exact passage in the UI
- [ ] **Export** — download conversation as PDF or Markdown

---

## License

MIT — see [LICENSE](LICENSE).
