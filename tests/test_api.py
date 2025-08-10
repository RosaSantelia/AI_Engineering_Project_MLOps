# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    payload = {"text": "I love this!"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "predictions" in r.json()
    # controlliamo struttura minima
    preds = r.json()["predictions"]
    assert isinstance(preds, list)
    assert "label" in preds[0]
