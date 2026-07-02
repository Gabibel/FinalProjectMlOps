from unittest.mock import patch
import numpy as np
import pandas as pd
import mlflow
from ml.train import train


def test_predict_returns_valid_prediction(client):
    with patch("backend.main.model") as mock_model:
        mock_model.predict.return_value = np.array([87.5])
        response = client.post(
            "/predict",
            json={"features": [10, 1.95, 0.047, 32.8, 4.19, 3.2, 3600, 3000, 32.8]},
        )
        assert response.status_code == 200
        assert "prediction" in response.json()


def test_training_logs_run_in_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    rows = []
    for battery_id in range(8):
        for cycle in range(5):
            rows.append({
                "Battery_ID": battery_id,
                "Cycle": cycle,
                "Capacity_Ah": 2.0 - cycle * 0.01,
                "Internal_Resistance_Ohm": 0.05,
                "Temperature_C": 25.0,
                "Voltage_Max_V": 4.2,
                "Voltage_Min_V": 3.0,
                "Charge_Time_s": 3600,
                "Discharge_Time_s": 3000,
                "Ambient_Temp_C": 22.0,
                "SOH": 100 - cycle,
            })

    csv_path = tmp_path / "data.csv"
    dvc_path = tmp_path / "data.csv.dvc"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    dvc_path.write_text("md5: testhash123\n")

    run_id = train(
        str(csv_path),
        str(dvc_path),
        experiment_name="test-experiment",
        registered_model_name="TestModel",
    )

    run = mlflow.get_run(run_id)
    assert "mae" in run.data.metrics
    assert run.data.params["git_commit"]
    assert run.data.params["dvc_data_version"] == "testhash123"