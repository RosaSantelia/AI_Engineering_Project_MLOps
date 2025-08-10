# Dockerfile
FROM python:3.10-slim

# Aggiorna apt e installa dipendenze di sistema minime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Setta la directory di lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice sorgente
COPY app ./app
COPY tests ./tests

# Espone la porta per uvicorn
EXPOSE 8000

# Comando di avvio dell'app FastAPI con uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
