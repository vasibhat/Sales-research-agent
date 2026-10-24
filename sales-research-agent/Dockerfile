# Build:  docker build -t sales-agent .
# Run:    docker run -p 8080:8080 --env-file .env sales-agent
# Run `sales-agent ingest` before building, so data/index.json is copied in.
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data

ENV DATA_DIR=/app/data \
    TRACES_DIR=/app/logs/traces \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["uvicorn", "sales_agent.api:app", "--host", "0.0.0.0", "--port", "8080"]
