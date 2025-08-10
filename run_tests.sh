#!/bin/bash

echo "Avvio dei test con filtro warning specifico..."

pytest -v tests/ -W ignore::FutureWarning

echo "Test completati."
