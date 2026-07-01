from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    """Verifies /health responds with 200 and the expected status payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}