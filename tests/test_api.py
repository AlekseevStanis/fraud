from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_healthcheck():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["model_loaded"] == True

def test_predict_api():
    sample = [{
        "Time": 10000,
        "Amount": 120.0,
        **{f"V{i}": 0 for i in range(1, 29)}
    }]

    res = client.post("/predict", json=sample)
    assert res.status_code == 200
    result = res.json()["results"][0]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["prediction"] in [0, 1]
