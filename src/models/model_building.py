import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import yaml
import logging
from pathlib import Path
from joblib import dump

def get_logger(name: str = __name__) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler("model_building.log")
    fh.setLevel(logging.DEBUG)
    
    
    if not logger.handlers:
        logger.addHandler(ch); logger.addHandler(fh)
    logger.propagate = False
    return logger

logger = get_logger()

TRAIN_IN = Path("data/features/bow")
OUT_DIR = Path("model")
PARAMS = Path("params.yaml")

def load_data_fast(file_path: Path = TRAIN_IN) -> pd.DataFrame:
    
    try:
        return pd.read_csv(file_path, engine="pyarrow")
    
    except Exception:
        return pd.read_csv(file_path)    
    
def load_params(params_path:Path = PARAMS):
    
    with open(params_path, "r") as f:
        try:
            params = yaml.safe_load(f)
            logger.debug("params file found and loaded safely")
        
        except FileNotFoundError:
            params = {}
            logger.debug("params file not found and putting empty file")
        
        cfg = params.get("xgboost")
        n_estimators = cfg.get("n_estimators", 400)
        lr = cfg.get("learning_rate", 0.1)
        max_depth = cfg.get("max_depth", 8)
        subsample = cfg.get("sub_sample", 0.8)
        colsample_bytree = cfg.get("colsample_bytree", 0.8)
        reg_lam = cfg.get("reg_lambda", 1.0)
        tree_method = cfg.get("tree_method", "hist")
        eval_metrics = cfg.get("eval_metrics", "logloss")
        random_state = cfg.get("random_state", 42)
        
        return n_estimators, lr, max_depth, subsample, colsample_bytree, reg_lam, tree_method, eval_metrics, random_state

def model_fit(X_train: np.ndarray, y_train:np.ndarray) -> XGBClassifier:
    n_estimators, lr, max_depth, subsample, colsample_bytree, reg_lam, tree_method, eval_metrics, random_state = load_params()
    
    try:
        clf = XGBClassifier(
            n_estimators= n_estimators,
            learning_rate = lr,
            max_depth = max_depth,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            reg_lambda = reg_lam,
            tree_method = tree_method,
            eval_metrics = eval_metrics,
            random_state = random_state,
        )
        clf.fit(X_train, y_train)
        logger.info("the model training is successful")
        return clf
    except Exception as e:
        logger.debug("error during model training %s", e)
        raise

def save_model(model, file_path: Path = OUT_DIR, compress: int = 3) -> Path:
    
    path = Path(file_path)
    
    if path.suffix == "":
        path = path/"model.joblib"
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    temp = path.with_suffix(path.suffix + ".temp")
    dump(model, temp, compress=compress)
    temp.replace(path)
    
    try:
        size = path.stat().st_size
        logger.info("Model saved to %s (%d bytes)", path, size)
    except Exception as e:
        logger.info("model saved to %s", path)
        
    return path

def main():
    
    try:
        train_df = load_data_fast(TRAIN_IN/"train_bow.csv")
        X_train = train_df.iloc[: , :-1].values
        y_train = train_df.iloc[:, -1].values
        
        clf = model_fit(X_train, y_train)
        
        if clf is None:
            raise ValueError("model_fit returned None. Model Not Trained")
        
        save_model(clf, OUT_DIR/"xgb.joblib")
        
    except Exception as e:
        logger.error("Unable to complete model building %s", e)
        raise

if __name__ == "__main__":
    main()