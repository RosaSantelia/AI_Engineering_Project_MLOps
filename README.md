# AI_Engineering_Projects_MLOps
My AI Engineering Master's Projects - MLOps

# Documentazione Tecnica del Progetto di Sentiment Analysis

Questo progetto implementa un sistema di analisi del sentiment utilizzando un modello pre-addestrato cardiffnlp/twitter-roberta-base-sentiment-latest di HuggingFace.
Il sistema è organizzato in un flusso MLOps che comprende:

- Addestramento e valutazione del modello

- Deploy su HuggingFace Hub

- Pipeline di monitoraggio automatico per valutare periodicamente le performance.


📂 Struttura del progetto

La struttura del progetto è organizzata in modo modulare per separare le diverse fasi di sviluppo e deployment.

app/: Contiene il codice dell'API (FastAPI) e la logica del modello.

training/: Script per l'addestramento e la valutazione del modello, inclusa la logica per il push su Hugging Face.

monitoring/: Script per il monitoraggio continuo delle performance del modello.

tests/: Test unitari e di integrazione per l'API e il modello.

.github/workflows/: File di configurazione per le pipeline di GitHub Actions (CI/CD e Monitoraggio).

Dockerfile: Definizione dell'ambiente containerizzato per l'applicazione.

requirements.txt: Elenco delle dipendenze Python del progetto.

⚙️ Installazione e configurazione
1️⃣ Clonare il repository
bash
git clone https://github.com/RosaSantelia/AI_Engineering_Projects_MLOps_rivisto.git
cd AI_Engineering_Projects_MLOps_rivisto
2️⃣ Creare e attivare un ambiente Conda
bash
conda create -n sentiment python=3.10
conda activate sentiment
3️⃣ Installare le dipendenze
bash
pip install -r requirements.txt
🚀 Esecuzione in locale
Addestrare e valutare il modello
bash
python training/train.py
Monitoraggio locale
bash
python monitoring/monitoring.py
🐳 Esecuzione con Docker
bash
docker build -t sentiment-analysis .
docker run --rm sentiment-analysis

🔄 Pipeline CI/CD
Il file .github/workflows/ci-cd.yml gestisce:

- Test unitari con pytest

- Addestramento del modello

- Deploy automatico su HuggingFace Hub

📈 Pipeline di monitoraggio
Il file .github/workflows/monitoring.yml esegue automaticamente:

- Ogni commit su main

- Ogni giorno alle 2:00 UTC

- Su richiesta manuale

📖 Guida rapida per utenti finali
Inserisci il testo che vuoi analizzare nel modello

Ottieni la classificazione: positive, neutral, negative

Consulta i log in data/sentiment_log.csv per vedere risultati e confronto con etichette reali

📊 Esempio di output
Esempio di predizione sentiment:

plaintext
Input: "I love working with this team!"
Predicted: positive
True label: positive

❓ FAQ
1. Che modello viene usato?
cardiffnlp/twitter-roberta-base-sentiment-latest di HuggingFace.

2. Come viene testato?
Con dataset pubblico TweetEval.

3. Qual è l'ambiente di sviluppo e testing?
La fase di sviluppo è stata testata utilizzando Conda come ambiente virtuale.
Successivamente, il progetto è stato containerizzato con Docker per garantire portabilità.

4. Come posso avviare il monitoraggio manualmente?
Dalla sezione Actions di GitHub, selezionare il workflow Monitoring TweetEval e cliccare su Run workflow.
