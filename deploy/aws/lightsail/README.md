# AWS Lightsail (Cheapest Demo) Deployment

This path runs **both backend + frontend on one small instance** using Docker
Compose. It is the cheapest AWS path and good for a demo.

## 1) Create the Lightsail Instance
1. Create a Linux instance (Ubuntu 22.04 or Amazon Linux 2023).
2. Smallest plan is fine for a demo (1 vCPU / 512MB or 1GB).
3. Open ports:
   - `80` (HTTP)
   - `443` (optional, if you add HTTPS later)
   - `22` (SSH)

## 2) SSH In and Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
newgrp docker
```

## 3) Pull the Repo and Configure Env
```bash
git clone <your-repo-url>
cd Ai\\ Agent

# create env file (only on the server)
cat <<'ENV' > .env
OPENAI_API_KEY=sk-...
DEFAULT_PROVIDER=openai
AUTH_BEARER_TOKEN=change-me
ENV
```

Notes:
- If you do not want to spend on LLM APIs, set `DEFAULT_PROVIDER=mock`.
- If you want public access without auth, remove `AUTH_BEARER_TOKEN`.

## 4) Build + Run Production Compose
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Visit:
- `http://<your-instance-public-ip>/`

## 5) (Optional) Basic Health Check
```bash
curl http://localhost/api/health
```

## 6) Updating
```bash
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

## Notes / Limits
- Single instance only; no autoscaling.
- Rate limiting is per instance (in-memory).
- Logs are local to the instance.
