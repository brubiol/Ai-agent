FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application source (includes pyproject.toml)
COPY app /app/app

RUN set -eux; \
    pip install --no-cache-dir --upgrade pip; \
    python - <<'PY' > /tmp/requirements.txt
import tomllib
with open('/app/app/pyproject.toml', 'rb') as fh:
    project = tomllib.load(fh)['project']
for dep in project.get('dependencies', []):
    print(dep)
PY

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN rm -rf /app/app/tests /app/app/rephraser_backend.egg-info
RUN addgroup --system app && adduser --system --ingroup app app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
