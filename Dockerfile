FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# REST API (8000) and MCP streamable-http (8001).
EXPOSE 8000 8001

# Default command serves the REST API; the MCP service overrides this in compose.
CMD ["uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
