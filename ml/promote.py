import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "BatteryHealthModel")
SOURCE_STAGE = os.environ.get("SOURCE_STAGE", "Staging")
TARGET_STAGE = os.environ.get("TARGET_STAGE", "Production")
R2_THRESHOLD = float(os.environ.get("R2_THRESHOLD", "0.7"))
ENFORCE_GATE = os.environ.get("ENFORCE_GATE", "true").lower() == "true"


def get_latest_version(client, model_name, stage):
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No model version found for '{model_name}' in stage '{stage}'")
    return versions[0]


def passes_quality_gate(client, version):
    run = client.get_run(version.run_id)
    r2 = run.data.metrics.get("r2")
    if r2 is None:
        raise RuntimeError("Run has no 'r2' metric logged, cannot evaluate quality gate")
    return r2 >= R2_THRESHOLD, r2


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    version = get_latest_version(client, MODEL_NAME, SOURCE_STAGE)

    if ENFORCE_GATE:
        passed, r2 = passes_quality_gate(client, version)
        print(f"Model '{MODEL_NAME}' v{version.version} ({SOURCE_STAGE}): r2={r2:.4f}, threshold={R2_THRESHOLD}")
        if not passed:
            print(f"Quality gate FAILED (r2 {r2:.4f} < {R2_THRESHOLD}). Model stays in '{SOURCE_STAGE}'.")
            sys.exit(1)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage=TARGET_STAGE,
        archive_existing_versions=(TARGET_STAGE == "Production"),
    )
    print(f"Model '{MODEL_NAME}' v{version.version} promoted: {SOURCE_STAGE} -> {TARGET_STAGE}")


if __name__ == "__main__":
    main()