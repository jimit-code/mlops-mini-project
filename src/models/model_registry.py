import json
import logging
import os
from pathlib import Path
import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import RestException

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT/".env")

dagshub_user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
token = (
    os.getenv("DAGSHUB_PAT")
    or os.getenv("DAGSHUB_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
)

if not dagshub_user:
    raise EnvironmentError("Dagshub username is not set")

if not token:
    raise EnvironmentError("Dagshub Token is not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

dagshub_url = "https://dagshub.com"
repo_owner = "jimit-code"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

MODEL_PATH = Path("model/xgb.joblib")
MODEL_INFO = Path("reports/model_info.json")

def get_logger(name:str = __name__) -> logging.Logger:
    
    if not isinstance(name, str) or not name:
        name = __name__
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler("model_reg.log")
    fh.setLevel(logging.ERROR)
    
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

logger = get_logger()

def load_model_info(file_path: Path = MODEL_INFO) -> dict:
    
    p = Path(file_path)
    try:
        with p.open("r", encoding="utf-8") as f:
            MODEL_INFO = json.load(f)
            logger.debug("model info loaded safely %s", p.resolve())
            return MODEL_INFO
        
    except Exception as e:
        logger.exception("Couldn't load the model_info file %s", e)
        
def register_model(model_name: str, model_info: str) -> None:
    
    try:
        run_id = model_info['run_id']
        artifact_path = model_info['model_path']
        
        model_uri = f"run:/{run_id}/{artifact_path}"
        logger.debug("Registering model from URI %s as '%s'", model_uri, model_name)
        
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage = "stagging"
        )
        
        logger.info(
            "Model '%s' version %s registered and moved to 'Staging'",
            model_name,
            model_version.version,
        )

    except RestException as e:
        msg = str(e)
        # DagsHub currently does not support some registry endpoints
        if "unsupported endpoint" in msg.lower():
            logger.warning(
                "MLflow model registry endpoint is not supported by this "
                "DagsHub MLflow server. Skipping registry step. Error. %s",
                msg,
            )
        else:
            logger.error("MLflow REST error during model registration. %s", e)
            raise
    except KeyError as e:
        logger.error("Missing key in model_info. %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during model registration. %s", e)
        raise


def main():
    try:
        model_info = load_model_info(MODEL_INFO)
        model_name = "Sentiment_XGB_BOW"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error("Failed to complete the model registration process. %s", e)
        print(f"Error. {e}")


if __name__ == "__main__":
    main()