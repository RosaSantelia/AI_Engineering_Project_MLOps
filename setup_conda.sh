#!/bin/bash

# Nome ambiente Conda
ENV_NAME=sentiment-mlops

# File environment Conda
ENV_FILE=environment.yml

echo "=== Setup ambiente Conda per il progetto ==="

# Verifica se conda è installato
if ! command -v conda &> /dev/null; then
    echo "Errore: conda non trovato. Installa Anaconda o Miniconda prima di procedere."
    exit 1
fi

# Controlla se l'ambiente esiste già
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Ambiente $ENV_NAME trovato. Aggiorno le dipendenze..."
    conda env update -n $ENV_NAME -f $ENV_FILE
else
    echo "Creo l'ambiente $ENV_NAME da $ENV_FILE..."
    conda env create -n $ENV_NAME -f $ENV_FILE
fi

echo "Attiva l'ambiente con: conda activate $ENV_NAME"
echo "Setup completato!"
