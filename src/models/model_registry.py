import json
import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import RestException


# ------------------------------------------------------------
# Load env + MLflow setup
# ------------------------------------------------------------
# Loads .env from the current working directory (usually repo root)
# This avoids fragile parents[2] path logic.
load_dotenv(".env")

dagshub_user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
token = (
    os.getenv("DAGSHUB_PAT")
    or os.getenv("DAGSHUB_TOKEN")
    or os.getenv("DAGSHUB_USER_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
)

if not dagshub_user:
    raise EnvironmentError("DAGSHUB_USERNAME (or MLFLOW_TRACKING_USERNAME) is not set")

if not token:
    raise EnvironmentError("DAGSHUB_PAT (or DAGSHUB_TOKEN / DAGSHUB_USER_TOKEN / MLFLOW_TRACKING_PASSWORD) is not set")

# MLflow uses these for HTTP basic auth against DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Prefer env var if set. Else use your DagsHub repo MLflow endpoint.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "https://dagshub.com/jimit-code/mlops-mini-project.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

# Files
MODEL_INFO = Path("reports/model_info.json")


# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------
def get_logger(name: str = "model_registry") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler("model_reg.log")
        fh.setLevel(logging.DEBUG)

        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)

        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


logger = get_logger()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_model_info(file_path: Path = MODEL_INFO) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"model_info file not found at: {p.resolve()}")

    try:
        with p.open("r", encoding="utf-8") as f:
            info = json.load(f)
            logger.debug("model_info loaded safely from %s", p.resolve())
            return info
    except Exception as e:
        logger.exception("Couldn't load model_info file. Error. %s", e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """
    Registers a model version into MLflow Model Registry
    using run_id + artifact path written in reports/model_info.json.
    """
    try:
        run_id = model_info["run_id"]
        artifact_path = model_info["model_path"]

        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.debug("Registering model from URI %s as '%s'", model_uri, model_name)

        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info(
            "Model '%s' version %s registered successfully",
            model_name,
            model_version.version,
        )

    except RestException as e:
        msg = str(e)
        if "unsupported endpoint" in msg.lower():
            logger.warning(
                "MLflow model registry endpoint is not supported by this backend. "
                "Skipping registry step. Error. %s",
                msg,
            )
            return
        logger.error("MLflow REST error during model registration. %s", e)
        raise

    except KeyError as e:
        logger.error("Missing key in model_info.json: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during model registration. %s", e)
        raise


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    try:
        model_info = load_model_info(MODEL_INFO)
        model_name = os.getenv("MODEL_NAME", "Sentiment_XGB_BOW")
        register_model(model_name, model_info)

    except Exception as e:
        logger.error("Failed to complete the model registration process. %s", e)
        print(f"Error. {e}")


if __name__ == "__main__":
    main()