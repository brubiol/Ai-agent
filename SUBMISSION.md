## Submission Notes

### Assumptions
- The reviewer has Docker installed; `.env` values are supplied before `docker compose up`.
- Bearer auth is optional (only enforced when `AUTH_BEARER_TOKEN` is set).
- OpenAI/Anthropic API keys are real in production but omitted during tests to use the mock client.

### Trade-offs
- Structured logging uses stdout with JSON format—simple but requires downstream log aggregation.
- Backend bind mount in `docker-compose.yml` is kept for developer convenience (hot reload) even though production images would omit it.
- Frontend build arg `VITE_API_BASE_URL` is injected at build time; changing it requires a rebuild.
- Tests rely on in-memory ASGI clients; no end-to-end browser tests are included.
