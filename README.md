MLOps per l'Analisi del Sentiment su Twitter

🏆 Panoramica del Progetto

Questo repository presenta una soluzione end-to-end di MLOps per l'analisi del sentiment su Twitter. Il progetto è stato costruito per dimostrare le pratiche di ingegneria del machine learning, includendo la containerizzazione con Docker, l'automazione del testing e del deployment con GitHub Actions e il monitoraggio continuo del modello.

L'obiettivo è classificare il sentiment di un testo in una delle tre categorie: positivo, neutrale o negativo, utilizzando un modello pre-addestrato basato su RoBERTa.

🚀 Caratteristiche Principali

Analisi del Sentiment: Utilizzo del modello cardiffnlp/twitter-roberta-base-sentiment-latest per predizioni accurate.

- API (FastAPI): Un'API RESTful robusta per l'inferenza del modello in tempo reale.

- Containerizzazione (Docker): Un ambiente isolato e riproducibile per l'applicazione e le sue dipendenze.

- Automazione CI/CD: Pipeline di GitHub Actions per l'integrazione e il deployment continuo.

- Monitoraggio Continuo: Un sistema di monitoraggio automatizzato per valutare periodicamente le performance del modello e prevenire la deriva (model drift).

📂 Struttura del Progetto

La struttura del progetto è organizzata in modo modulare per separare le diverse fasi del ciclo di vita MLOps.

.
├── .github/
│   └── workflows/
│       ├── ci-cd.yml
│       └── monitoring.yml
├── app/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   └── schema.py
├── data/
├── docs/
│   └── TROUBLESHOOTING.md
├── monitoring/
│   ├── reports/
│   └── monitoring.py
├── rosa-twitter-sentiment/
├── tests/
│   ├── pytest.ini
│   ├── test_api.py
│   └── test_model.py
├── training/
│   ├── push_to_hub.py
│   └── train.py
├── .gitignore
├── Dockerfile
├── environment.yml
├── README.md
├── requirements.txt
├── run_tests.sh
└── setup_conda.sh

🛠 Guida Rapida per gli Sviluppatori

Prerequisiti

Assicurati di avere installati i seguenti strumenti:

- Git

- Docker

Avvio dell'API (con Docker)

Segui questi passaggi per avviare l'API di sentiment analysis in locale.

Clona il repository:

git clone https://github.com/RosaSantelia/AI_Engineering_Projects_MLOps_rivisto.git
cd AI_Engineering_Projects_MLOps_rivisto

Costruisci l'immagine Docker:

docker build -t sentiment-analysis .

Avvia l'API:

docker run -p 8000:8000 sentiment-analysis

L'API sarà disponibile all'indirizzo a cui sarai dirottato dal popup a monitor e potrai testare gli endopoint (ricordati di configurare che la porta sia aperta e non privata).

Test degli Endpoint dell'API

Per testare l'endpoint di predizione /predict dell'API, puoi utilizzare lo strumento curl con il seguente comando:

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text":"I love this new project!"}'

Questo comando invierà una richiesta POST all'API con un testo di esempio e riceverai una risposta con il sentiment predetto.

📈 Pipeline CI/CD e Monitoraggio

Il progetto integra due pipeline di GitHub Actions per automatizzare il ciclo di vita del modello.

CI_CD PIPELINE:

Esegue i test, addestra il modello, e lo deploysu Hugging Face Spaces se tutti i passaggi sono superati.

MONITORING PIPELINE:

Esegue lo script monitoring.py periodicamente per valutare le performance del modello in produzione.

Lo script monitoring/monitoring.py esegue predizioni batch, salva i risultati in CSV, genera matrici di confusione e un report HTML.

La pipeline è automatizzata tramite GitHub Actions (.github/workflows/monitoring.yml), eseguita:

- Su ogni push al branch main

- Ogni giorno alle 2:00 UTC

- Manualmente tramite trigger manuale

I report sono caricati come artifact scaricabili dall’interfaccia Actions di GitHub.

Avvio manuale del monitoraggio:

Dal tab Actions su GitHub, seleziona il workflow Monitoring TweetEval e clicca su Run workflow.

❓ FAQ Utenti Finali

1. Che modello viene usato?
Viene utilizzato il modello cardiffnlp/twitter-roberta-base-sentiment-latest di HuggingFace, specificamente addestrato su dati di Twitter.

2. Come viene testato?
Il modello è testato automaticamente utilizzando un sottoinsieme del dataset pubblico TweetEval per verificare le sue performance in modo continuo.

3. Qual è l'ambiente di sviluppo e testing?
La fase di sviluppo è stata testata inizialmente con Conda. Per garantire la portabilità e la riproducibilità, l'intero progetto è stato poi containerizzato con Docker.

4. Come posso avviare il monitoraggio manualmente?
Dalla sezione Actions del repository su GitHub, devi selezionare il workflow Monitoring TweetEval e cliccare sul pulsante Run workflow.