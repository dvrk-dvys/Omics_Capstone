import os
import yaml
import pandas as pd


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve all relative paths against the config file's directory."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    # config lives at app/config/pipeline.yaml — two levels up is omics_ml_pipeline/
    project_root = os.path.abspath(os.path.join(config_dir, "..", ".."))
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, val in config.get("paths", {}).items():
        if val and not os.path.isabs(val):
            config["paths"][key] = os.path.join(project_root, val)
    return config


def ensure_dirs(config: dict) -> None:
    """Create all output directories defined in config paths if they don't exist."""
    for key, path in config["paths"].items():
        if path.endswith("/") or "." not in os.path.basename(path):
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"Saved: {path}")
