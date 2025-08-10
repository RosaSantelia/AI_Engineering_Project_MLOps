#!/bin/bash
# run_tests.sh

# Avvia i test con un filtro per i warning specifici
echo "Avvio dei test con filtro warning specifico..."

# Imposta PYTHONPATH per includere la root del progetto
export PYTHONPATH="$PYTHONPATH:/app"

pytest tests -W ignore::UserWarning

echo "Test completati."