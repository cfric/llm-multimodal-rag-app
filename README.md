# Multimodal RAG System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![LangChain](https://img.shields.io/badge/LangChain-1.3-yellow)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

This is a portfolio project for a local, single-user multimodal Retrieval-Augmented Generation
pipeline built with Python and LangChain. It demonstrates a production-inspired architecture:
PDFs, images and text documents are ingested, stored in a vector store and queried through a
FastAPI and Streamlit interface.

Multimodal means that not only text is processed — images are described via GPT-4o Vision and
stored as semantically searchable chunks; tables are stored as a summary (embedded) and as
Markdown (context). LLM and embedding providers are configurable; ChromaDB is the supported
vector store.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Document Upload (PDF · Image · Text)       │
│              POST /api/v1/documents/upload               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Ingestion Pipeline (Unstructured hi_res)   │
│  ┌─────────────┐  ┌──────────────────────┐  ┌────────┐ │
│  │ Text Chunk  │  │ Image → GPT-4o Vision│  │ Table  │ │
│  │ + Splitter  │  │ → Description        │  │Summary │ │
│  └─────────────┘  └──────────────────────┘  └────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │   Embeddings                     │
          │   OpenAI text-embedding-3-large  │
          │   (alternative: HuggingFace)     │
          └──────────────┬───────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
           │   ChromaDB VectorStore            │
          └──────────────┬───────────────────┘
                         │  Query
                         ▼
     ┌───────────────────────────────────────────┐
     │   SemanticRetriever (Top-K)               │
     │   → CrossEncoder Reranker                 │
     │   → ConversationMemory (max. 20 msgs.)    │
     └─────────────────────┬─────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────┐
          │   LLM                              │
          │   GPT-4o · Azure · Ollama          │
          │   HuggingFace                      │
          └────────────────┬───────────────────┘
                           │
                           ▼
     ┌─────────────────────────────────────────────┐
     │           FastAPI Backend  (Port 8000)      │
     │  POST /query · POST /stream · /health       │
     │  Prometheus Metrics  (Port 8001)            │
     └─────────────────────┬───────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Streamlit Frontend   │
              │   Chat Interface       │
              │   Source Display       │
              └────────────────────────┘
```

---

## Two Ways to Run the Project

This project can be started in two different ways. Both options are listed in the table below.

| | Option A — Makefile (local) | Option B — Docker |
|---|---|---|
| **Requires** | Python 3.12 + API key | Docker + API key |
| **Best for** | Development, fast iteration | Consistent local environment |
| **Effort** | One-time setup required | Runs immediately, no Python setup |
| **Command** | `make setup && make api` | `make docker-up` |

---

## Prerequisite — API Key

The system requires an **OpenAI API key** (default LLM and embeddings).
Create a `.env` file in the project directory:

```bash
cp .env.example .env
# Then add to .env:
OPENAI_API_KEY=sk-...
```

Alternatively, Ollama (local, no API key required) or HuggingFace can be configured as the
LLM provider — see `config/config.yaml`.

> **Scope and privacy:** This is a local single-user demo, not a public deployment template.
> Do not expose its API to the internet. With the default OpenAI configuration, document content,
> extracted images, tables and chat context are sent to OpenAI.

---

## Option A — Local Development with Makefile

> **What is a Makefile?**
> A Makefile is a file that stores shortcuts for frequently used commands.
> Instead of typing a long command, you just type `make <name>`. All available
> shortcuts can be displayed with `make help`.

### Step 1 — Check Prerequisites

Open a terminal and verify the required tools are installed:

```bash
python3 --version   # should output 3.12.x
git --version       # any version
make --version      # any version
```

> **Windows users:** `make` is not available by default on Windows.
> Install [Git Bash](https://gitforwindows.org/) or use WSL2 — all commands
> will then work identically.

For hi_res PDF processing (detecting images and tables), the following are also required:

```bash
sudo apt-get install tesseract-ocr poppler-utils   # Ubuntu / Debian / WSL2
brew install tesseract poppler                      # macOS
```

### Step 2 — Clone the Repository

```bash
git clone https://github.com/<your-username>/llm_multimodal_rag.git
cd llm_multimodal_rag
```

### Step 3 — Set Up the Environment

```bash
make setup
```

This command automatically:
- Creates a Python virtual environment (`.venv/`) inside the project folder
- Installs all required Python packages from `requirements.txt`

### Step 4 — Start the Services

The application consists of two independent services. Each requires its **own terminal window**.

**Terminal 1 — Start the API:**
```bash
make api
```
Wait until you see: `Application startup complete.`
The API is now available at: **http://localhost:8000**
It only accepts connections from the local machine.

**Terminal 2 — Start the frontend** *(open a new window)*:
```bash
make ui
```
Wait until you see: `You can now view your Streamlit app in your browser.`
The dashboard is now available at: **http://localhost:8501**

### Step 5 — Open in the Browser

| Address | What you will see |
|---|---|
| http://localhost:8501 | Streamlit chat — upload documents and ask questions |
| http://localhost:8000/docs | Automatic API documentation (Swagger UI) |
| http://localhost:8000/api/v1/health | API status as JSON |

### Stopping the Application

Press **Ctrl + C** in each terminal window to stop the respective service.

### All Makefile Commands

```bash
make help          # show all available commands
make setup         # set up environment (run once)
make api           # start FastAPI  →  http://localhost:8000
make ui            # start Streamlit  →  http://localhost:8501
make test          # run test suite
make clean         # delete generated files (logs, vector data, images)
```

---

## Option B — Docker

> **What is Docker?**
> Docker packages the entire application — including Python, all packages and
> configuration — into isolated containers. These run identically on any
> machine, regardless of the installed operating system or Python version.
> You do **not need Python** installed on your machine for this option.

### Step 1 — Check Prerequisites

```bash
docker --version           # should output a version number
docker compose version     # should output a version number
```

### Step 2 — Clone the Repository

```bash
git clone https://github.com/<your-username>/llm_multimodal_rag.git
cd llm_multimodal_rag
```

### Step 3 — Create .env

```bash
cp .env.example .env
# Add OPENAI_API_KEY and a unique GRAFANA_ADMIN_PASSWORD to .env
```

### Step 4 — Start the Application

```bash
make docker-up
```

Docker starts the following containers simultaneously:
- **ChromaDB** — vector store, starts first (with health check)
- **API container** — starts only after ChromaDB is ready
- **Streamlit container** — starts after the API
- **Prometheus** — metrics server (port 9090)
- **Grafana** — monitoring dashboard (port 3000)

Wait until you see in the terminal:
```
rag_api  | INFO:     Application startup complete.
rag_ui   | You can now view your Streamlit app in your browser.
```

The first run takes slightly longer because Docker needs to build all images.

### Step 5 — Open in the Browser

| Address | What you will see |
|---|---|
| http://localhost:8501 | Streamlit chat — upload documents and ask questions |
| http://localhost:8000/docs | Automatic API documentation (Swagger UI) |
| http://localhost:9090 | Prometheus — metrics and query latencies |
| http://localhost:3000 | Grafana — monitoring dashboard |

### Stopping the Application

```bash
make docker-down
```

Or press **Ctrl + C** in the terminal, then run:

```bash
make docker-down
```

### All Docker Commands

```bash
make docker-up      # build and start the full stack
make docker-down    # stop everything
```

---

## Troubleshooting

### `make: command not found`
`make` is not installed. On Ubuntu/Debian/WSL2:
```bash
sudo apt-get install build-essential
```
On Windows: install [Git Bash](https://gitforwindows.org/).

### `docker compose version` fails
The Docker Compose plugin is missing:
```bash
sudo apt-get update && sudo apt-get install -y docker-compose-plugin
```

### `429 — You exceeded your current quota`
The OpenAI credit balance is depleted. Top up at
[platform.openai.com/account/billing](https://platform.openai.com/account/billing)
or switch to `provider: ollama` in `config/config.yaml` (no API key required).

### `partition_pdf() is not available`
The Unstructured PDF package is missing:
```bash
.venv/bin/pip install "unstructured[pdf]" pdfminer.six
```

### Port already in use — `address already in use`
Another process is using port 8000 or 8501. Check which one:
```bash
lsof -i :8000
lsof -i :8501
```

---

## API Reference

### `POST /api/v1/rag/query`

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main contributions of the Transformer architecture?",
    "session_id": "my-session"
  }'
```

**Response:**

```json
{
  "answer": "The Transformer architecture dispenses with recurrence entirely ...",
  "sources": [
    {
      "source_file": "attention-is-all-you-need.pdf",
      "type": "text",
      "page": 2
    }
  ],
  "session_id": "my-session"
}
```

### All Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/rag/query` | POST | RAG query with conversation memory |
| `/api/v1/rag/stream` | POST | Streaming response (Server-Sent Events) |
| `/api/v1/rag/session/{id}` | DELETE | Clear conversation history |
| `/api/v1/documents/upload` | POST | Upload and ingest a document |
| `/api/v1/documents` | GET | List all ingested documents |
| `/api/v1/documents/{source}` | DELETE | Remove a document from the vector store |
| `/api/v1/health` | GET | API health status |

Interactive documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

Uploads are limited to 25 MB by default. Adjust `API_MAX_UPLOAD_SIZE_MB` only for trusted local use.

---

## Configuration

All components are configurable in `config/config.yaml` and can be overridden via
environment variables (`.env`):

| Section | Env Prefix | Key Options |
|---|---|---|
| `llm` | `LLM_` | `provider`: openai · azure · ollama · huggingface |
| `embedding` | `EMBEDDING_` | `provider`: openai · huggingface |
| `vectorstore` | `VECTORSTORE_` | `provider`: chroma |
| `reranker` | `RERANKER_` | `enabled`: true · false |
| `retrieval` | `RETRIEVAL_` | `k`: number of chunks to retrieve |

---

## Project Structure

```
llm_multimodal_rag/
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI app + startup
│   │   ├── routers/           # documents.py · rag.py · health.py
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── dependencies.py    # Dependency injection
│   ├── core/
│   │   ├── interfaces.py      # Abstract base classes
│   │   └── exceptions.py      # Domain exceptions
│   ├── ingestion/
│   │   ├── pipeline.py        # Ingestion orchestration
│   │   ├── loaders/           # pdf_loader · image_loader · text_loader
│   │   └── processors/        # text · image · table
│   ├── retrieval/
│   │   ├── semantic_retriever.py
│   │   └── reranker.py        # CrossEncoder / Cohere
│   ├── rag/
│   │   ├── chain.py           # RAG chain (LangChain)
│   │   ├── memory.py          # ConversationMemory
│   │   └── prompt_templates.py
│   ├── vectorstore/
│   │   ├── chroma_store.py    # ChromaDB (embedded + HTTP)
│   │   └── factory.py
│   ├── llm/                   # openai · azure · ollama · huggingface + factory
│   ├── embeddings/            # openai · huggingface + factory
│   ├── monitoring/            # Prometheus metrics
│   ├── evaluation/            # RAGAS evaluation
│   └── ui/
│       └── streamlit_app.py   # Streamlit frontend
├── tests/
│   ├── unit/                  # Loaders, processors, chain, vector store
│   └── integration/           # Ingestion pipeline, RAG pipeline
├── config/
│   ├── config.yaml            # Central configuration
│   └── logging.yaml
├── docker/
│   ├── prometheus/
│   └── grafana/
├── data/
│   ├── documents/             # Uploaded documents
│   ├── images/                # Extracted images
│   └── vectorstore/           # ChromaDB (local)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Ingestion | Unstructured (hi_res), Tesseract OCR, Poppler |
| Multimodal | GPT-4o Vision (images), GPT-4o (tables) |
| Embeddings | OpenAI text-embedding-3-large, HuggingFace (local) |
| Vector Store | ChromaDB |
| RAG / LLM | LangChain, OpenAI GPT-4o, Azure OpenAI, Ollama, HuggingFace |
| Reranking | CrossEncoder (sentence-transformers), Cohere |
| API | FastAPI, Pydantic, uvicorn |
| Frontend | Streamlit |
| Monitoring | Prometheus, Grafana |
| Evaluation | RAGAS |
| Testing | pytest |
| Infra | Docker, Docker Compose |

---

## Next Steps

The following improvements are planned for future implementation:

1. **Streaming UI** — Streamlit frontend displays the answer token-by-token via the `/stream` endpoint
2. **Document deduplication** — detect on upload whether a document already exists in the vector store
3. **Authentication** — API key middleware before adapting the project for public deployment
4. **Extended evaluation** — RAGAS metrics (Faithfulness, Answer Relevancy) as an automated report after each deployment

---

## License

MIT License — see [LICENSE](LICENSE) for details.
