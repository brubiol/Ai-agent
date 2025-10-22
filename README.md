## Rephraser Workbench

FastAPI + React application that rewrites user text across multiple tones using
pluggable LLM providers (OpenAI, Anthropic, or an in-app mock). The frontend
streams incremental deltas so users see each style fill in live.

---

### Features
- 🎛️ Select and stream multiple tones (professional, casual, polite, social)
- 🔌 Provider auto-selection with mock fallback for offline development
- 🚦 Rate limiting, body size guard, bearer auth, and CORS allow list
- 📜 Structured JSON logging with request IDs & secret redaction
- 🧪 Backend pytest coverage + Jest/RTL streaming tests
- 🐳 Docker & Compose setup for one-command local runs

---

### Architecture
```
┌────────────────────────────────────────────┐
│                  Browser                   │
│        React (Vite) + useRephraser         │
└──────────────────────┬─────────────────────┘
                       │ fetch / SSE
┌──────────────────────▼─────────────────────┐
│             FastAPI (backend)              │
│  ┌──────────────────────────────────────┐  │
│  │ /rephrase       /rephrase/stream     │  │
│  │ auth, rate limit, task manager       │  │
│  └────────────────┬─────────────────────┘  │
│                   │ async LLM client       │
└───────────────────▼────────────────────────┘
            Provider abstraction
        ┌────────────┴────────────┐
        │  OpenAI   │  Anthropic  │
        └────────────┴────────────┘
              Mock fallback
```

---

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker (optional but recommended for Mission 10/12 runs)
- OpenAI / Anthropic API keys (optional; mock runs without them)

---

### Local Development (no Docker)
```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="sk-..."   # optional
export DEFAULT_PROVIDER=openai   # optional
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```
Visit `http://localhost:5173`.

---

### Docker & Compose
```bash
# place provider keys in .env (next to docker-compose.yml)
cat <<'ENV' > .env
OPENAI_API_KEY=sk-...
DEFAULT_PROVIDER=openai
AUTH_BEARER_TOKEN=my-dev-token
ENV

docker compose up --build
```
- Frontend → `http://localhost:5173`
- Backend docs → `http://localhost:8000/docs`

---

### Environment Variables
| Key | Description |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI credential (optional) |
| `ANTHROPIC_API_KEY` | Anthropic credential (optional) |
| `DEFAULT_PROVIDER` | `auto`, `openai`, `anthropic`, `mock` |
| `AUTH_BEARER_TOKEN` | Optional bearer required for write endpoints |
| `ALLOWED_ORIGINS` | Comma-separated origin list for CORS |
| `VITE_API_BASE_URL` | Frontend API target (Docker build arg) |

---

### Testing
```bash
# backend
cd backend
PYTHONPATH=. pytest

# frontend
cd frontend
npm test -- --runInBand
```

---

### Continuous Integration
The project is ready for CI (e.g., GitHub Actions) by running:
```bash
PYTHONPATH=. pytest
npm ci && npm test -- --runInBand
docker compose build
```
Add caching for `.venv`/`node_modules` or rely on containers for isolation.

---

### Troubleshooting
| Symptom | Fix |
| --- | --- |
| Mock responses `[Professional] ...` | Set `OPENAI_API_KEY` or `DEFAULT_PROVIDER=mock` intentionally |
| 401 Unauthorized | Supply `Authorization: Bearer <token>` matching `AUTH_BEARER_TOKEN` |
| CORS errors | Adjust `ALLOWED_ORIGINS` |
| SSE not streaming | Ensure backend reachable at `VITE_API_BASE_URL` |
| Docker build fails | Remove bind mount on backend for production runs |

---

### Demo GIF (optional)
Capture an 8-second terminal/browser demo with ffmpeg:
```bash
./scripts/record_demo.sh
```
This script records the `:0` display (edit for your OS) and saves `demo.gif`.
