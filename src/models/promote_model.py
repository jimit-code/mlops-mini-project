import json
import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

# -------------------------------------------------------------------
# Config, env, MLflow setup
# -------------------------------------------------------------------

# Adjust this if your script location differs
ROOT = Path(__file__).resolve().parents[2]

env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

dagshub_user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
dagshub_token = (
    os.getenv("DAGSHUB_PAT")
    or os.getenv("DAGSHUB_USER_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
)

if not dagshub_user:
    raise EnvironmentError("DAGSHUB_USERNAME (or MLFLOW_TRACKING_USERNAME) is not set")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT (or DAGSHUB_USER_TOKEN or MLFLOW_TRACKING_PASSWORD) is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Prefer env MLFLOW_TRACKING_URI if present. Else build it.
dagshub_url = "https://dagshub.com"
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "jimit-code")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "mlops-mini-project")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

MODEL_NAME = os.getenv("MODEL_NAME", "Sentiment_XGB_BOW")
STAGING_INFO = Path(os.getenv("STAGING_INFO_PATH", "reports/staging_model.json"))

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

        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

def _search_versions(client: MlflowClient, model_name: str):
    # IMPORTANT: filter must be "name = '...'"
    return list(client.search_model_versions(f"name = '{model_name}'"))

def get_latest_version(client: MlflowClient, model_name: str) -> str:
    """Return the version string of the latest registered model version."""
    versions = _search_versions(client, model_name)
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Latest version for model '%s' is %s (current_stage=%s, run_id=%s)",
        model_name,
        latest.version,
        getattr(latest, "current_stage", None),
        getattr(latest, "run_id", None),
    )
    return latest.version


def get_version_for_run_id(client: MlflowClient, model_name: str, run_id: str) -> str:
    """Find the model version whose run_id matches the staged candidate."""
    versions = _search_versions(client, model_name)
    matches = [v for v in versions if getattr(v, "run_id", None) == run_id]
    if not matches:
        raise ValueError(f"No model version found for model='{model_name}' with run_id='{run_id}'")

    chosen = max(matches, key=lambda v: int(v.version))
    logger.info(
        "Matched staged run_id=%s to model '%s' version %s (current_stage=%s)",
        run_id,
        model_name,
        chosen.version,
        getattr(chosen, "current_stage", None),
    )
    return chosen.version


def archive_current_production(client: MlflowClient, model_name: str, keep_version: str) -> None:
    """Archive other Production versions, except keep_version."""
    try:
        versions = _search_versions(client, model_name)
    except RestException as e:
        logger.warning("Could not search model versions, skipping archive step. Error. %s", e)
        return

    prod_versions = [
        v for v in versions
        if getattr(v, "current_stage", None) == "Production" and v.version != keep_version
    ]

    if not prod_versions:
        logger.info("No existing Production versions to archive for '%s' (other than v%s)", model_name, keep_version)
        return

    for v in prod_versions:
        logger.info("Archiving model '%s' version %s", model_name, v.version)
        try:
            client.transition_model_version_stage(name=model_name, version=v.version, stage="Archived")
        except RestException as e:
            logger.warning("Archive stage transition failed or unsupported. Skipping. Error. %s", e)
            return


def promote_to_production(client: MlflowClient, model_name: str, version: str) -> None:
    logger.info("Promoting model '%s' version %s to Production", model_name, version)

    # 1) stage promotion (optional, may be unsupported in some backends)
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
    except RestException as e:
        logger.warning("Stage transition failed or unsupported. Error. %s", e)

    # 2) alias pointer (recommended)
    client.set_registered_model_alias(model_name, "prod", str(version))


def pick_version_to_promote(client: MlflowClient, model_name: str) -> str:
    """
    Prefer staging pointer file if present. Otherwise fall back to latest version.
    """
    if STAGING_INFO.exists():
        info = json.loads(STAGING_INFO.read_text(encoding="utf-8"))
        run_id = info.get("run_id")
        if run_id:
            return get_version_for_run_id(client, model_name, run_id)
        logger.warning("Staging file exists but run_id missing. Falling back to latest version.")
    else:
        logger.warning("Staging pointer file not found at %s. Falling back to latest version.", STAGING_INFO.resolve())

    return get_latest_version(client, model_name)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    client = MlflowClient()

    try:
        version = pick_version_to_promote(client, MODEL_NAME)

        promote_to_production(client, MODEL_NAME, version)
        archive_current_production(client, MODEL_NAME, keep_version=version)

        logger.info("Promotion finished. Model '%s' version %s is the prod pointer.", MODEL_NAME, version)

    except RestException as e:
        logger.error("MLflow REST error during promotion. %s", e)
        print(f"Error: {e}")
    except Exception as e:
        logger.error("Unexpected error during model promotion. %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()