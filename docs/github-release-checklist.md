# GitHub Release Checklist

This checklist is for publishing the project as a portfolio repository. It deliberately separates
public source code from local runtime data and provider credentials.

## Before Staging

- Create local credentials from `.env.example`; never commit `.env`.
- Revoke and rotate any provider key that was ever exposed outside the local machine.
- Inspect `git status --short` before staging.
- Confirm that `git ls-files .env` returns no result.
- Confirm that `git check-ignore -v data/documents data/images data/vectorstore logs` shows the intended ignore rules.
- Remove private documents, extracted images and local logs from any manually selected staging set.
- Use only `data/demo/demo_document.md` and the generated demo asset for public demonstrations.

## Secret Checks

Run a repository search before publishing:

```bash
git grep -n -I -E 'sk-[A-Za-z0-9_-]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY' -- ':!*.lock'
```

Also review API-key variable assignments in `.env` files, untracked files, ignored files that might
be copied into an archive, screenshots and evaluation logs. A `.gitignore` rule prevents normal staging but does not remove a secret from a
backup, archive or previously shared working directory.

## Quality Gate

Run the same checks documented in the README:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
PYTHONPATH=. .venv/bin/python -m mypy src
.venv/bin/python -m pip check
GRAFANA_ADMIN_PASSWORD=release-check-only docker compose config --quiet
```

The current verified baseline is 41 passing tests. The existing Starlette/httpx deprecation
warning is known and does not fail the suite.

## Portfolio Claims

Use this positioning:

> Advanced, production-inspired multimodal RAG prototype for local single-user use.

Do not claim public production readiness. The documented limitations include in-memory sessions,
non-persistent background ingestion jobs, absent authentication and rate limiting, provider-specific
vision capabilities, and external-model evaluation costs.

## Public Evidence

- Link `docs/evaluation.md` for the evaluation protocol and actual results.
- Link `docs/design-decisions.md` for architecture trade-offs.
- Include only screenshots without secrets, private paths or private document content.
- Keep the Swagger screenshot and architecture diagram as public visual evidence.
- Clearly label missing Streamlit/Grafana screenshots as follow-up evidence rather than implying they
  were captured.

## Final Git Review

- Check the complete staged diff, not only the latest file changes.
- Verify that no `.env`, private data, logs or generated vector stores are staged.
- Verify that README links resolve and the repository URL is correct.
- Verify that the license and author/copyright information are intentional.
- Push only after the final staged diff has been reviewed.
