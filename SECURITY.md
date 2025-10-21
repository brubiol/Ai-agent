# Security Overview

This project serves a text rephrasing API with a React frontend. The following
measures are in place to reduce common threats:

## Backend (FastAPI)

- **CORS allow list** – origins are controlled via `ALLOWED_ORIGINS` so that only
  trusted sites can call the API from a browser.
- **Bearer token gate** – write endpoints (`POST /rephrase` and
  `/rephrase/stream`) require the `Authorization: Bearer <token>` header when
  `AUTH_BEARER_TOKEN` is configured.
- **Rate limiting & body limits** – SlowAPI throttles requests per IP and large
  payloads receive HTTP 413 responses.
- **Structured logging** – logs are JSON with request ids and omit secrets
  (OpenAI / Anthropic keys, bearer tokens) to prevent accidental leakage.

### Threat Model

| Threat | Mitigation |
| --- | --- |
| API key exposure | Keys are only read from environment variables; logs redact any secret values before emission. |
| Prompt injection / misuse | Requests remain server-side; the optional bearer token lets you restrict who can trigger real provider calls. |
| Abuse / flooding | Rate limiting (30 req/min/IP) and request size limits reduce brute-force or scraping attempts. |

## Frontend (React)

- The browser bundle only receives `VITE_API_BASE_URL`; no provider keys or
  secrets are embedded client-side.
- Environment-specific configuration happens during the Docker build
  (`VITE_API_BASE_URL` build arg) and can be changed without touching source
  control.

## Operations Checklist

1. Populate `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and `AUTH_BEARER_TOKEN` in a
   server-side `.env` (never in the client).
2. Update `ALLOWED_ORIGINS` to the production hostname.
3. Monitor JSON logs for request ids, errors, and rate-limit events.
4. Rotate tokens regularly and treat the bearer token like a password.
