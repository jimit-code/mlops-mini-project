import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import logging
from pathlib import Path
import yaml
import pickle

def get_logger(name: str | None = None) -> logging.Logger:
    if not isinstance(name, str) or not name:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler("my_app.log", encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(ch); logger.addHandler(fh)
    logger.propagate = False
    return logger

logger = get_logger()

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("data/features/bow")
TRAIN_IN = PROCESSED_DIR/"train.csv"
TEST_IN = PROCESSED_DIR/ "test.csv"
USE_COLS = ['content', 'sentiment']

def load_params(path : Path = Path("params.yaml")) -> dict:
    
    with open("params.yaml", 'r') as f:
        
        try:
        
            params = yaml.safe_load(f)
            logger.debug("yaml file loaded safely")
        except FileNotFoundError:
            params = {}
            logger.exception("error in loading the file and getting empty dictionary")
            
        cfg = params.get("bow", {})
        max_feature = int(cfg.get("max_features", 50_000))
        ngram_range = (int(cfg.get("ngram_min", 1)), int(cfg.get("ngram_max",1)))
        
        return max_feature, ngram_range
    
def read_fast(csv_path: Path) -> pd.DataFrame:
    
    try:
        return pd.read_csv(csv_path, usecols=USE_COLS, engine='pyarrow')
    except Exception:
        return pd.read_csv(csv_path, usecols=USE_COLS)   
    
def main():
    
    max_features, ngram_range = load_params()
    
    try:
        
        vec = CountVectorizer(
            max_features=max_features,
            ngram_range= ngram_range
        )
        
        train_df = read_fast(TRAIN_IN)
        test_df = read_fast(TEST_IN)
        logger.info("Data loaded successfully from train=%s test %s", TRAIN_IN, TEST_IN)
        
        X_tr = train_df['content'].fillna("").values
        X_te = test_df['content'].fillna("").values
        y_tr = train_df['sentiment'].values
        y_te = test_df['sentiment'].values
        
        logger.info("fittng BOW max_feature=%d ngrams=%s", max_features, ngram_range)
        X_train_bow = vec.fit_transform(X_tr)
        X_test_bow = vec.transform(X_te)
        logger.info("Shapes X_train=%s X_test=%s", X_test_bow.shape, X_test_bow.shape)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_tr
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_te
        
        path = Path("model/vectorizer.pkl")
        path.parent.mkdir(parents=True, exist_ok= True)
        
        with path.open("wb") as f:
            pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.debug("BOW applied to the data frame, transformed and vec saved")
        
        OUT_DIR.mkdir(parents=True, exist_ok= True)
        train_df.to_csv(OUT_DIR/"train_bow.csv", index=False)
        test_df.to_csv(OUT_DIR/"test.bow.csv", index= False)
        logger.info("New Data stored in %s", OUT_DIR)
        
    except Exception as e:
        logger.error("Error durring bow %s", e)
        raise
    
if __name__ == "__main__":
    main()
    