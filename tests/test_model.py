# tests/test_model.py
import unittest
import os
import sys

# Aggiunge la directory principale del progetto al PYTHONPATH per l'importazione
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from app.model import SentimentModel

class TestSentimentModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Metodo che viene eseguito una sola volta prima di tutti i test.
        Carica il modello di sentiment analysis.
        """
        cls.sentiment_model = SentimentModel()

    def test_sentiment_prediction(self):
        """
        Testa che il modello sia in grado di predire correttamente il sentiment
        per frasi positive e negative.
        """
        # Definisce una lista di testi di esempio da testare
        texts = ["I love this!", "This is terrible."]
        
        # Esegue la predizione utilizzando il modello
        predictions = self.sentiment_model.predict(texts)

        # Asserisce che la lista di predizioni abbia la stessa lunghezza della lista di input
        self.assertEqual(len(predictions), len(texts))
        
        # Asserisce che il sentiment predetto per il primo testo sia 'positive'
        self.assertEqual(predictions[0]["label"], 'positive')
        
        # Asserisce che il sentiment predetto per il secondo testo sia 'negative'
        self.assertEqual(predictions[1]["label"], 'negative')