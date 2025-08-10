# Usa una base Python 3.10 ufficiale
FROM python:3.10-slim

# Imposta la working directory
WORKDIR /app

# Copia i file requirements e il resto del progetto
COPY requirements.txt ./
COPY . .

# Aggiorna pip e installa le dipendenze
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta per FastAPI
EXPOSE 8000

# Comando di default per partire con uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
