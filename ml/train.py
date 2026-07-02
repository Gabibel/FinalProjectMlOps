import os
import subprocess
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "Cycle", "Capacity_Ah", "Internal_Resistance_Ohm", "Temperature_C",
    "Voltage_Max_V", "Voltage_Min_V", "Charge_Time_s", "Discharge_Time_s",
    "Ambient_Temp_C",
]
TARGET = "SOH"


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def get_dvc_data_version(dvc_file_path):
    with open(dvc_file_path) as f:
        for line in f:
            if "md5" in line:
                return line.strip().split(": ")[1]
    return None


def train(data_path, dvc_file_path, experiment_name="battery-soh-prediction", registered_model_name="BatteryHealthModel"):
    mlflow.set_experiment(experiment_name)

    git_hash = get_git_commit_hash()
    data_version = get_dvc_data_version(dvc_file_path)

    df = pd.read_csv(data_path)

    batteries = df["Battery_ID"].unique()
    train_ids, test_ids = train_test_split(batteries, test_size=0.25, random_state=42)
    train_df = df[df["Battery_ID"].isin(train_ids)]
    test_df = df[df["Battery_ID"].isin(test_ids)]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    with mlflow.start_run() as run:
        mlflow.log_param("git_commit", git_hash)
        mlflow.log_param("dvc_data_version", data_version)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        return run.info.run_id


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    train("ml/data/battery_data.csv", "ml/data/battery_data.csv.dvc")