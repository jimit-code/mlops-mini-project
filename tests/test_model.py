import os
import unittest
from pathlib import Path
import pickle

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Resolve project root and load .env (useful for local runs)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class TestModelLoading(unittest.TestCase):
    # Paths
    MODEL_PATH = ROOT / "model" / "xgb.joblib"
    VEC_PATH = ROOT / "model" / "vectorizer.pkl"
    TEST_PATH = ROOT / "data" / "features" / "bow" / "test.bow.csv"

    # Minimum performance thresholds
    MIN_ACCU = 0.40
    MIN_PREC = 0.40
    MIN_REC = 0.40
    MIN_F1 = 0.40

    # DagsHub . MLflow config
    DAGSHUB_URL = "https://dagshub.com"
    REPO_OWNER = "jimit-code"
    REPO = "mlops-mini-project"  # repo name only
    MLFLOW_EXP = "mlflow-with-dvcpipeline"

    @classmethod
    def setUpClass(cls):
        """
        One time setup.
        - Load local model joblib
        - Load holdout data
        - Optionally configure MLflow if creds are present
        """
        # Check model exists
        if not cls.MODEL_PATH.exists():
            raise unittest.SkipTest(
                f"Model file not found at {cls.MODEL_PATH.resolve()}. "
                "Run the model_building stage again."
            )

        # Check test data exists
        if not cls.TEST_PATH.exists():
            raise unittest.SkipTest(
                f"Test file not found at {cls.TEST_PATH.resolve()}. "
                "Run the feature engineering stage again."
            )

        # Load model and data
        cls.model = joblib.load(cls.MODEL_PATH)
        cls.holdout = pd.read_csv(cls.TEST_PATH)
        cls.X = cls.holdout.iloc[:, :-1]
        cls.y = cls.holdout.iloc[:, -1]

        # Load vectorizer if available
        cls.vectorizer = None
        if cls.VEC_PATH.exists():
            with cls.VEC_PATH.open("rb") as f:
                cls.vectorizer = pickle.load(f)

        # Configure MLflow / DagsHub if creds exist
        user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
        token = (
            os.getenv("DAGSHUB_PAT")
            or os.getenv("DAGSHUB_USER_TOKEN")
            or os.getenv("MLFLOW_TRACKING_PASSWORD")
        )

        if user and token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token

            tracking_uri = f"{cls.DAGSHUB_URL}/{cls.REPO_OWNER}/{cls.REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cls.MLFLOW_EXP)
            cls.mlflow_enabled = True
        else:
            cls.mlflow_enabled = False

    def test_model_loaded(self):
        self.assertIsNotNone(self.model, "Model is None after loading")
        self.assertTrue(
            hasattr(self.model, "predict"),
            "Loaded model does not have a 'predict' method",
        )

    def test_signature_with_vec(self):
        if self.vectorizer is None:
            self.skipTest("Vectorizer not found. Skipping signature test.")

        text = "we are starting the test here and I like the movie"
        # Vectorizer expects an iterable of texts
        vec = self.vectorizer.transform([text])
        X_df = pd.DataFrame(vec.toarray())

        pred = self.model.predict(X_df)

        # One prediction per row
        self.assertEqual(
            len(pred),
            X_df.shape[0],
            "Number of predictions does not match number of rows",
        )
        # 1D output
        self.assertEqual(
            pred.ndim,
            1,
            "Prediction is expected to be 1D for this classifier",
        )

    def test_performance_holdout(self):
        """
        Evaluate model on holdout data and assert minimum performance.
        Optionally log metrics to MLflow/DagsHub.
        """
        preds = self.model.predict(self.X)

        acc = accuracy_score(self.y, preds)
        prec = precision_score(self.y, preds, zero_division=0)
        rec = recall_score(self.y, preds, zero_division=0)
        f1 = f1_score(self.y, preds, zero_division=0)

        msg = (
            "Got metrics "
            f"accuracy={acc:.3f}, precision={prec:.3f}, "
            f"recall={rec:.3f}, f1={f1:.3f}"
        )

        self.assertGreaterEqual(acc, self.MIN_ACCU, f"Accuracy too low. {msg}")
        self.assertGreaterEqual(prec, self.MIN_PREC, f"Precision too low. {msg}")
        self.assertGreaterEqual(rec, self.MIN_REC, f"Recall too low. {msg}")
        self.assertGreaterEqual(f1, self.MIN_F1, f"F1 score too low. {msg}")

        # Optional logging to MLflow
        if getattr(self.__class__, "mlflow_enabled", False):
            with mlflow.start_run(run_name="unittest-xgb", nested=True):
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_precision", prec)
                mlflow.log_metric("test_recall", rec)
                mlflow.log_metric("test_f1", f1)


if __name__ == "__main__":
    unittest.main()