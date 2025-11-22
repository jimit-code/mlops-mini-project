import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn
import dagshub


# -------------------------------------------------------------------
# Config and MLflow / DagsHub setup
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
token = (
    os.getenv("DAGSHUB_PAT")
    or os.getenv("DAGSHUB_USER_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
)

if not user:
    raise EnvironmentError("DAGSHUB_USERNAME is not set")

if not token:
    raise EnvironmentError("DAGSHUB_PAT (or DAGSHUB_USER_TOKEN) is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = user
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

dagshub_url = "https://dagshub.com"
repo_owner = "jimit-code"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------

def get_logger(name: str = __name__) -> logging.Logger:
    if not isinstance(name, str) or not name:
        name = __name__

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler("model_eval.log")
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


logger = get_logger()

MODEL_PATH = "model/xgb.joblib"
TEST_PATH = "data/features/bow/test.bow.csv"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH):
    p = Path(model_path)

    if not p.exists():
        logger.error("Model file not found: %s", p.resolve())
        raise FileNotFoundError(p)

    if p.is_dir():
        q = p / "xgb.joblib"
        if not q.exists():
            raise IsADirectoryError(f"No model file inside {p.resolve()}")
        p = q

    if not p.is_file():
        raise IsADirectoryError(f"Path is not a file: {p.resolve()}")

    model = load(p)
    logger.debug("Model loaded from %s", p)
    logger.debug("Loaded model type: %s", type(model))

    if model is None:
        logger.error("Loaded model from %s is None. Check model_building stage", p)
        raise ValueError("Loaded model is None. Check model_building stage")

    return model


def load_data(file_path: str = TEST_PATH) -> pd.DataFrame:
    p = Path(file_path)

    if not p.exists() or p.is_dir():
        raise FileNotFoundError(f"Test data not found at {p.resolve()}")

    try:
        df = pd.read_csv(p, engine="pyarrow")
        logger.debug(
            "Test data loaded with pyarrow rows=%s, columns=%s",
            len(df),
            len(df.columns),
        )
        return df
    except Exception:
        df = pd.read_csv(p)
        logger.debug(
            "Test data loaded with default engine rows=%s, columns=%s",
            len(df),
            len(df.columns),
        )
        return df


def eval_metrics(clf, X_test: np.ndarray, y_test) -> dict:
    try:
        y_test = np.asarray(y_test)
        y_pred = clf.predict(X_test)

        n_classes = np.unique(y_test).size
        avg = "binary" if n_classes == 2 else "macro"

        scores = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            scores = proba[:, 1] if n_classes == 2 else proba
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average=avg,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            average=avg,
            zero_division=0,
        )

        if scores is not None:
            if n_classes == 2:
                auc = roc_auc_score(y_test, scores)
            else:
                auc = roc_auc_score(
                    y_test,
                    scores,
                    multi_class="ovr",
                    average="macro",
                )
        else:
            auc = float("nan")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }

        if np.isnan(auc):
            metrics["auc"] = None
        else:
            metrics["auc"] = float(auc)

        logger.debug("Model evaluation metrics calculated: %s", metrics)
        return metrics
    except Exception as e:
        logger.exception("Error during calculating evaluation metrics %s", e)
        raise


def save_metric(metric: dict, file_path: str | Path) -> Path:
    p = Path(file_path)

    if p.suffix == "":
        p = p.with_suffix(".json")

    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        temp = p.with_suffix(p.suffix + ".temp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(metric, f, ensure_ascii=False, indent=2)

        temp.replace(p)
        logger.debug("Metrics saved to %s", p)
        return p
    except Exception as e:
        logger.exception("Error occurred while saving the metrics %s", e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", p)
    except Exception as e:
        logger.error("Error occurred while saving the model info: %s", e)
        raise


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    mlflow.set_experiment("mlflow-with-dvcpipeline")

    with mlflow.start_run(run_name="XGB_BOW_Pipeline") as run:
        try:
            logger.debug("Starting model evaluation pipeline")

            clf = load_model(MODEL_PATH)
            test = load_data(TEST_PATH)

            X_test = test.iloc[:, :-1].values
            y_test = test.iloc[:, -1].values

            metrics = eval_metrics(clf, X_test, y_test)

            metrics_path = save_metric(metrics, "reports/metrics.json")

            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)

            if hasattr(clf, "get_params"):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                    
            try: 
                mlflow.sklearn.log_model(
                    clf,
                    artifact_path= "model",
                    registered_model_name= "Sentiment_XGB_BOW"
                )
            except Exception as e:
                logger.warning("Skipping model logging to MLFLOW got error %s", e)

            save_model_info(
                run.info.run_id,
                "model/xgb.joblib",
                "reports/model_info.json",
            )

            # Log artifacts
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact("reports/model_info.json")
            mlflow.log_artifact("model_eval.log")

            logger.info(
                "Model evaluation completed successfully. "
                "Metrics written to %s",
                metrics_path,
            )
        except Exception as e:
            logger.exception(
                "Failed to complete the model evaluation process: %s",
                e,
            )
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    main()