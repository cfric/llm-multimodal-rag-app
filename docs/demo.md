# Demo Guide

This guide documents a reproducible local demonstration of the multimodal RAG
pipeline. It uses a small, copyright-free document and a generated diagram.

## Start

Create the environment and configure a provider:

```bash
cp .env.example .env
# Set OPENAI_API_KEY or configure a compatible local provider.
make setup
```

Start the API and UI in separate terminals:

```bash
make api
make ui
```

Open `http://localhost:8501` and upload the demo document. The generated image
is available after running:

```bash
make demo-assets
```

## Demo Questions

Use these questions to demonstrate the three content types:

| Content | Question |
| --- | --- |
| Text | Welche Komponenten bildet die Demo-Anwendung ab? |
| Table | Welche Aufgabe übernimmt der Reranker laut Tabelle? |
| Image | Welche Stationen zeigt das Demo-Diagramm? |

The UI should display the answer together with the source file, content type,
page information where available, and relevance score.

## Evaluation

Index the generated demo data before running the evaluation:

```bash
make demo-index
make evaluate
```

The evaluation requires configured model access and writes the report to
`reports/evaluation/latest.json`. The dataset contains separate text, table,
and image questions.

The evaluation distinguishes deterministic source/modality metrics from model-based RAGAS metrics.
See [`evaluation.md`](evaluation.md) for metric definitions, reproducibility metadata and the
current result status. See [`design-decisions.md`](design-decisions.md) for the rationale behind the
multimodal and retrieval design.

## Portfolio Capture Checklist

Capture the following views for the portfolio README or project presentation:

- Streamlit chat after uploading `demo_document.md`
- Text answer with a visible source section
- Table answer with the original table content
- Image answer with the generated diagram
- Grafana dashboard after at least one query and one ingestion
- Terminal output showing the test and evaluation commands
- FastAPI OpenAPI page showing the documented API surface

The expected screenshot filenames and current capture status are documented in
[`assets/screenshots/README.md`](assets/screenshots/README.md).

Do not include API keys, local absolute paths, or private document content in
the captured material.
