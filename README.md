# AI_Engineering_Projects_MLOps_rivisto
My AI Engineering Master's Projects - MLOps

# Documentazione Tecnica del Progetto di Sentiment Analysis

1. Panoramica del Progetto
Questo progetto ha l'obiettivo di implementare e testare una soluzione MLOps per l'analisi del sentiment. Il cuore del sistema è un modello di analisi del sentiment basato su RoBERTa, in grado di classificare testi provenienti dai social media. Il progetto segue una metodologia MLOps completa, che include l'implementazione del modello, la sua validazione automatizzata e la preparazione per il deploy e il monitoraggio continuo.

2. Architettura del Sistema
2.1. Componenti Principali
API (FastAPI): L'applicazione esposta tramite un server Uvicorn che gestisce le richieste HTTP. L'API ha un endpoint /predict che prende in input un testo e restituisce il sentiment predetto dal modello (positivo, neutro o negativo).

Modello di Sentiment Analysis (RoBERTa): Viene utilizzato il modello pre-addestrato cardiffnlp/twitter-roberta-base-sentiment-latest. Questo modello è stato specificamente addestrato su dati di Twitter, rendendolo ideale per l'analisi del sentiment sui social media. Le etichette di output del modello sono mappate in negative (0), neutral (1) e positive (2).

Test di Valutazione: Lo script evaluate.py, situato nella cartella app/, esegue una valutazione automatizzata del modello. Carica il dataset pubblico tweet_eval, esegue le predizioni in batch e genera un classification_report dettagliato che mostra metriche come precisione, recall e F1-score. I test effettivi dell'API e del modello sono gestiti dai file test_api.py e test_train_smoke.py nella cartella tests/.

2.2. Struttura delle Cartelle
Il progetto è organizzato per separare il codice dell'applicazione, i test e la configurazione, seguendo le best practice dell'MLOps.

.
├── .github/                # Cartella per le configurazioni di GitHub Actions
│   └── workflows/          # Definizione dei workflow CI/CD
│       └── ci-cd.yml       # Workflow per il Continuous Integration e Continuous Deployment
├── .pytest_cache/          # Cache generata da pytest
├── app/                    # Codice dell'applicazione (API, modello, logica di valutazione)
│   ├── evaluate.py         # Script per la valutazione del modello
│   ├── main.py             # Endpoint dell'API FastAPI
│   ├── model.py            # Logica del modello di sentiment analysis
│   └── schema.py           # Definizione degli schemi di dati (Pydantic)
├── docs/                   # Documentazione tecnica
│   └── TROUBLESHOOTING.md  # Guida alla risoluzione dei problemi comuni
├── prometheus/             # File di configurazione per il monitoraggio con Prometheus e Grafana
│   ├── grafana-dashboard.json # Configurazione della dashboard di Grafana
│   └── prometheus.yml      # Configurazione di Prometheus per la raccolta delle metriche
├── tests/                  # Codice per i test
│   ├── pytest.ini          # File di configurazione per pytest
│   ├── test_api.py         # Test degli endpoint API
├── .gitignore              # Elenco dei file e delle cartelle da ignorare in Git
├── Dockerfile              # Definizione del container Docker per l'applicazione
├── environment.yml         # Configurazione dell'ambiente Conda
├── README.md               # Panoramica del progetto e guida rapida
├── requirements.txt        # Elenco delle dipendenze Python (alternativa a environment.yml)
├── run_tests.sh            # Script di shell per l'esecuzione dei test
└── setup_conda.sh          # Script di shell per la configurazione dell'ambiente Conda

3. Guida per gli Sviluppatori
3.1. Prerequisiti
Per eseguire e testare il progetto, è necessario avere installato :

Docker Desktop (per Windows/macOS) o Docker Engine (per Linux).

Git (per il controllo versione).

3.2. Configurazione dell'Ambiente di Sviluppo
Clone del repository: Clona il repository del progetto.

Build dell'Immagine Docker: Dalla directory principale del progetto, esegui il seguente comando per costruire l'immagine Docker.


3.3. Esecuzione del Test
Per eseguire i test, usa lo script run_tests.sh. Questo script è il modo più affidabile per lanciare tutti i test del progetto all'interno del container Docker.

docker run --rm sentiment-project ./run_tests.sh

3.4. Esecuzione dell'API
Per avviare l'API, utilizza il comando docker run per eseguire il container, mappando la porta 8000.

docker run -p 8000:8000 sentiment-project

4. Pipeline CI/CD
Questo progetto è progettato per essere integrato in una pipeline CI/CD (ad esempio con GitHub Actions). Il workflow tipico è il seguente:

Trigger: La pipeline si attiva ad ogni push o pull request sul branch principale.

Fase di Build: L'immagine Docker viene costruita.

Fase di Test: Viene eseguito lo script di valutazione del modello. Se l'accuracy scende al di sotto di una soglia predefinita (es. 70%), il test fallisce e la pipeline si interrompe.

Fase di Deploy: Se tutti i test passano, il modello e l'API vengono deployati in produzione.

5. Monitoraggio e Manutenzione
Una volta in produzione, il modello verrà monitorato utilizzando strumenti come :

Prometheus: Per la raccolta di metriche come la latenza delle richieste, il numero di errori e le performance del modello nel tempo.

Grafana: Per visualizzare i dati di Prometheus tramite dashboard personalizzate, consentendo una facile analisi e l'impostazione di alert.