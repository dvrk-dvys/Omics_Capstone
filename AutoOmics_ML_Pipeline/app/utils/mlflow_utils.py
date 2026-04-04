import os
import mlflow
import pandas as pd


def setup_mlflow(config: dict) -> None:
    uri = config["mlflow"]["tracking_uri"]
    # Resolve relative file paths the same way io_utils resolves config paths
    if not uri.startswith("http") and not os.path.isabs(uri):
        here = os.path.dirname(os.path.abspath(__file__))          # app/utils/
        project_root = os.path.abspath(os.path.join(here, "..", ".."))  # AutoOmics_ML_Pipeline/
        uri = os.path.join(project_root, uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    print(f"MLflow tracking: {uri}")
    print(f"Experiment: {config['mlflow']['experiment_name']}")


def log_dataframe_artifact(df: pd.DataFrame, filename: str, tmp_dir: str = "/tmp") -> None:
    """Save a DataFrame to CSV and log it as an MLflow artifact."""
    import os
    path = os.path.join(tmp_dir, filename)
    df.to_csv(path, index=True)
    mlflow.log_artifact(path)
