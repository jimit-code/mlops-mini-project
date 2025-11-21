import unittest
import mlflow
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import pandas as pd
import pickle

class TestModelLoading(unittest.TestCase):
    
    MODEL_PATH = Path("model/xgb.joblib")
    VEC_PATH = Path("model/vectorizer.pkl")
    TEST_PATH = Path("data/features/bow/test.bow.csv")
    
    MIN_ACCU = 0.40
    MIN_PREC = 0.40
    MIN_REC = 0.40
    MIN_F1 = 0.40
    
    DAGSHUB_URL = "https://dagshub.com"
    REPO_OWNER = "jimit-code"
    REPO = "mlops-mini-project.mlflow"
    MLFLOW_EXP = "mlflow-with-dvcpipeline"
    
    @classmethod
    def setUpClass(cls):
        """
        one time set up:
        - load local model joblib
        - load data 
        """
        
        if not cls.MODEL_PATH.exists():
            raise unittest.SkipTest(
                f"model file doesn't found {cls.MODEL_PATH.resolve()}. "
                "run the model building stage again"
            )
        
        if not cls.TEST_PATH.exists():
            raise unittest.SkipTest(
                f"test file doesn't exist {cls.TEST_PATH.resolve()}"
                "run feature engineering file again and make sure in correct path"
            ) 
            
        cls.model = joblib.load(cls.MODEL_PATH)
        cls.holdout = pd.read_csv(cls.TEST_PATH)
        cls.X = cls.holdout.iloc[:, :-1]
        cls.y = cls.holdout.iloc[:, -1]
        
        cls.vectorizer = None
        if cls.VEC_PATH.exists():
            cls.vectorizer = pickle.load(cls.VEC_PATH)
        
        user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
        token = os.getenv("DAGSHUB_PAT") or os.getenv("MLFLOW_TRACKING_PASSWORD")
        
        if user and token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = user
            os.environ['MLFLOW_TRACKING_PASSWORD'] = token
            
            tracking_uri = f"{cls.DAGSHUB_URL}/{cls.REPO_OWNER}/{cls.REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cls.MLFLOW_EXP)
            cls.mlflow_enable = True
        else:
            cls.mlflow_enable = False
            
    def test_model_loaded(self):
        
        self.assertIsNotNone(self.model, "Model is None after loading..")
        self.assertTrue(
            hasattr(self.model, "predict"),
            "Loaded model doesn't have predict attribute"
        )
    def test_signature_with_vec(self):
        
        if self.vectorizer is None:
            self.skipTest("Vectorizer cannot find skipping the signature step")
            
        text = "we are starting the test here and I like the movie"
        vec = self.vectorizer.transform(text)
        X_df = pd.DataFrame(vec.toarray()
                            )
        pred = self.model.predict(X_df)
        
        self.assertEqual(len(preds), X_df.shape[0], "number of prediction doesn't match")
        self.assertEqual(
            pred.ndim, 1,
            "Prediction expected to be 1D for this classifier"
            
        )
        
    def test_performance_holdout(self):
        """
        Evaluate model on holdout data and assert min performance.
        Optionally log metrics to mlflow/Dagshub
        """
        preds = self.model.predict(self.X)
        acc = accuracy_score(self.y, preds)
        prec = precision_score(self.y, preds, zero_division=0)
        recall = recall_score(self.y, preds)
        f1 = f1_score(self.y, preds)
        
        msg = (
            "Got metrics"
            f"accuracy: {acc: .3f}, precision: {prec: .3f}"
            f"recall: {recall: .3f}, f1_score: {f1: .3f}"
        )
        
        self.assertGreaterEqual(acc, self.MIN_ACCU, f"accuracy too low.{msg}")
        self.assertGreaterEqual(prec, self.MIN_PREC, f"precision too low{msg}")
        self.assertGreaterEqual(recall, self.MIN_REC, f"recall is too low{msg}")
        self.assertGreaterEqual(f1, self.MIN_F1, f"f1_score is too low{msg}")
        
        if self.mlflow_enable:
            with mlflow.start_run(run_name="unittest-xgb", nested=True):
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_precision", prec)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1", f1)

if __name__ == "__main__":
    unittest.main()