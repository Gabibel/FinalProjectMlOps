import pytest
from fastapi.testclient import TestClient
from backend.main import app, validate_features

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_validate_features_valid_input():
    data = {"features": [10, 1.95, 0.047, 32.8, 4.19, 3.2, 3600, 3000, 32.8]}
    result = validate_features(data)
    assert result == [10.0, 1.95, 0.047, 32.8, 4.19, 3.2, 3600.0, 3000.0, 32.8]


def test_validate_features_wrong_length_raises():
    data = {"features": [1, 2, 3]}
    with pytest.raises(ValueError):
        validate_features(data)