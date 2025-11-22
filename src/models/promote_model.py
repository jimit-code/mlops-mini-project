import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


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

MODEL_NAME = "Sentiment_XGB_BOW"


# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------

def get_logger(name: str = "model_promote_mlflow") -> logging.Logger:
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
# Helpers
# -------------------------------------------------------------------

def get_latest_version(client: MlflowClient, model_name: str) -> str:
    """
    Return the version string of the latest registered model version.
    """
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    versions_sorted = sorted(versions, key=lambda v: int(v.version))
    latest = versions_sorted[-1]

    logger.info(
        "Latest version for model '%s' is %s (current_stage=%s)",
        model_name,
        latest.version,
        latest.current_stage,
    )
    return latest.version


def archive_current_production(client: MlflowClient, model_name: str) -> None:
    """
    Move all current Production versions to Archived.
    On DagsHub, filtering by current_stage may not be supported.
    In that case, we log a warning and skip archiving.
    """
    try:
        prod_versions = list(
            client.search_model_versions(
                f"name='{model_name}' and current_stage='Production'"
            )
        )
    except RestException as e:
        msg = str(e)
        if "INVALID_PARAMETER_VALUE" in msg or "unsupported endpoint" in msg.lower():
            logger.warning(
                "Backend does not support filtering by current_stage. "
                "Skipping archive step. Error: %s",
                msg,
            )
            return
        raise

    if not prod_versions:
        logger.info("No existing Production versions to archive for '%s'", model_name)
        return

    for v in prod_versions:
        logger.info(
            "Archiving model '%s' version %s (was Production)",
            model_name,
            v.version,
        )
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived",
            )
        except RestException as e:
            msg = str(e)
            if "unsupported endpoint" in msg.lower():
                logger.warning(
                    "Stage transition to Archived not supported by backend. "
                    "Error: %s",
                    msg,
                )
                return
            raise


def promote_to_production(client: MlflowClient, model_name: str, version: str) -> None:
    """
    Promote the given version to Production.
    """
    logger.info(
        "Promoting model '%s' version %s to Production",
        model_name,
        version,
    )
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
    except RestException as e:
        msg = str(e)
        if "unsupported endpoint" in msg.lower():
            logger.warning(
                "Stage transition to Production not supported by backend. "
                "The model is still the latest version but registry stage "
                "will remain unchanged. Error: %s",
                msg,
            )
            return
        raise


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    client = MlflowClient()

    try:
        # 1. Find latest version
        latest_version = get_latest_version(client, MODEL_NAME)

        # 2. Archive any current Production versions (best-effort)
        archive_current_production(client, MODEL_NAME)

        # 3. Promote the latest version to Production (best-effort)
        promote_to_production(client, MODEL_NAME, latest_version)

        logger.info(
            "Promotion script finished. Model '%s' version %s is intended for Production.",
            MODEL_NAME,
            latest_version,
        )

    except RestException as e:
        # only unexpected REST errors reach here
        logger.error("MLflow REST error during promotion. %s", e)
        print(f"Error: {e}")
        # raise  # you can uncomment this if you WANT CI to fail on unexpected errors
    except Exception as e:
        logger.error("Unexpected error during model promotion. %s", e)
        print(f"Error: {e}")
        # raise


if __name__ == "__main__":
    main()