#!/bin/bash
cd /home/cf82550/python_projects/llm_multimodal_rag

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
PYTHONPATH=. \
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
