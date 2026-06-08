#!/bin/bash
cd /home/cf82550/python_projects/llm_multimodal_rag

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
PYTHONPATH=. \
.venv/bin/streamlit run src/ui/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
