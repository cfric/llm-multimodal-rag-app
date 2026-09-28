# Screenshot Evidence

Runtime screenshots are part of the portfolio evidence, not application data. They must be captured
from the local demo with generated, copyright-free assets only.

## Capture Set

| File | Evidence |
| --- | --- |
| `streamlit-text-answer.png` | Text answer with visible source section |
| `streamlit-table-answer.png` | Table answer with original table content |
| `streamlit-image-answer.png` | Image answer with the generated diagram |
| `grafana-rag-metrics.png` | Query and ingestion metrics |
| `verification-terminal.png` | Test and evaluation commands |
| `fastapi-openapi.png` | Public API surface in Swagger UI |

The FastAPI OpenAPI screenshot is available in this directory. Streamlit answer, Grafana and
terminal screenshots are not populated yet; the evaluation now succeeds, but the headless browser
only captured the Streamlit loading skeleton rather than the connected application state. The
existing architecture visual is available at
[`../multimodal-flow.svg`](../multimodal-flow.svg).

Before committing screenshots, remove API keys, absolute local paths, private document names and
provider account details.
