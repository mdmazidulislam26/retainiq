# Serving image for the API. Needs artifacts/model.joblib, so run train.py first. Port comes from $PORT (hosting platforms) or 8000.
FROM python:3.12-slim
WORKDIR /app
ENV RETAINIQ_ROOT=/app PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY artifacts/model.joblib artifacts/model_meta.json artifacts/
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
