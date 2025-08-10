FROM python:3.10-slim

# Dependencies di sistema minimali
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copia requirements e installa
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia codice
COPY app ./app
COPY training ./training

EXPOSE 8000

# comando di avvio (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
