import mlflow
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

mlflow.set_experiment("sanity-check")
with mlflow.start_run():
    mlflow.log_param("check", "connexion")
    mlflow.log_metric("ok", 1)

print("OK - run loggé avec succès")