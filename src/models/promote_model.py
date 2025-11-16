import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# -------------------------------------------------------------------
# Config . env . MLflow setup
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

dagshub_user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
dagshub_token = (
    os.getenv("DAGSHUB_PAT")
    or os.getenv("DAGSHUB_USER_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
)

if not dagshub_user:
    raise EnvironmentError("DAGSHUB_USERNAME is not set")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT (or DAGSHUB_USER_TOKEN) is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "jimit-code"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

MODEL_NAME = "Sentiment_XGB_BOW"   # same as in model_registration
TARGET_STAGE = "Production"    # or "Staging" . "Archived"


# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------

def get_logger(name: str = "model_promotion") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler("model_promotion.log")
        fh.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)

        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


logger = get_logger()


# -------------------------------------------------------------------
# Promotion logic
# -------------------------------------------------------------------

def get_latest_model_version(client: MlflowClient, model_name: str) -> str:
    """Return the version string of the latest model version."""
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    # sort by version number . newest last
    versions_sorted = sorted(
        versions,
        key=lambda v: int(v.version),
    )
    latest = versions_sorted[-1]
    logger.debug(
        "Latest version of model '%s' is %s with current stage '%s'",
        model_name,
        latest.version,
        latest.current_stage,
    )
    return latest.version


def promote_model(model_name: str, target_stage: str = "Production") -> None:
    client = MlflowClient()

    try:
        version = get_latest_model_version(client, model_name)

        logger.info(
            "Promoting model '%s' version %s to stage '%s'",
            model_name,
            version,
            target_stage,
        )

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage,
        )

        logger.info(
            "Model '%s' version %s successfully moved to stage '%s'",
            model_name,
            version,
            target_stage,
        )

    except RestException as e:
        msg = str(e)
        if "unsupported endpoint" in msg.lower():
            logger.warning(
                "Model registry stage transitions are not supported "
                "by this MLflow backend. Skipping promotion. Error. %s",
                msg,
            )
        else:
            logger.error("MLflow REST error during promotion. %s", e)
            raise
    except Exception as e:
        logger.error("Unexpected error during model promotion. %s", e)
        raise


def main():
    try:
        promote_model(MODEL_NAME, TARGET_STAGE)
    except Exception as e:
        logger.error("Failed to promote model. %s", e)
        print(f"Error. {e}")


if __name__ == "__main__":
    main()
