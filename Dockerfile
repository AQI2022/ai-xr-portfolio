FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY ai_xr ./ai_xr
RUN pip install --no-cache-dir . && useradd -m appuser && mkdir /app/data /app/models && chown -R appuser /app
COPY examples ./examples
COPY scripts/seed_demo.py ./scripts/seed_demo.py
USER appuser
EXPOSE 8000
CMD ["uvicorn", "ai_xr.api:app", "--host", "0.0.0.0", "--port", "8000"]
