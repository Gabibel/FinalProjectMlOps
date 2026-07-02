from unittest.mock import patch
import numpy as np


def test_predict_returns_valid_prediction(client):
    with patch("backend.main.model") as mock_model:
        mock_model.predict.return_value = np.array([87.5])
        response = client.post(
            "/predict",
            json={"features": [10, 1.95, 0.047, 32.8, 4.19, 3.2, 3600, 3000, 32.8]},
        )
        assert response.status_code == 200
        assert "prediction" in response.json()