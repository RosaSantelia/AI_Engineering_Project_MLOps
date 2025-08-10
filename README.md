# AI_Engineering_Projects_MLOps_rivisto
My AI Engineering Master's Projects - MLOps

# Sentiment MLOps Project

Progetto MLOps per analisi del sentiment di testi provenienti dai social media, utilizzando il modello `cardiffnlp/twitter-roberta-base-sentiment-latest`.

## 🚀 Funzionalità
- API REST con **FastAPI**
- Modello HuggingFace pre-addestrato per sentiment (positivo, neutro, negativo)
- Training / fine-tuning su dataset **TweetEval**
- Pipeline CI/CD con GitHub Actions
- Containerizzazione con Docker + Monitoraggio con Prometheus & Grafana
- Test automatici con pytest

## 📂 Struttura

AI_Engineering_Projects_MLOps_rivisto/
│
├── README.md
├── environment.yml
├── conda-lock.yml
├── Dockerfile
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── app/
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   ├── __init__.py
│   └── tests/
│       ├── __init__.py
│       └── test_app.py
├── data/
│   └── sample_dataset.csv
└── docs/
    └── troubleshooting.md
