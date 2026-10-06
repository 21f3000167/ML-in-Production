# Simple Dockerfile for the FastAPI ML API
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model
COPY main.py .
COPY model.pkl .

# Render uses PORT (default 10000)
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]