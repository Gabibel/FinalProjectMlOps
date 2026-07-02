import time
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_then_predict_flow():
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    with patch("backend.main.model") as mock_model:
        mock_model.predict.return_value = [87.5]
        start = time.time()
        response = client.post(
            "/predict",
            json={"features": [10, 1.95, 0.047, 32.8, 4.19, 3.2, 3600, 3000, 32.8]},
        )
        elapsed = time.time() - start

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], list)
    assert elapsed < 2.0