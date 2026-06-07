from fastapi.testclient import TestClient

from src.api import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json={"age": 25.0, "fare": 30.0})
        assert response.status_code == 200
        body = response.json()
        assert body["age"] == 25.0
        assert body["fare"] == 30.0
        assert "survived" in body
        assert "probability" in body
