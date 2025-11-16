import pandas as pd
import numpy as np
import logging
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

def get_logger(name:str = __name__) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler("my_app.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    
    #logger.handlers provide the list of all the handlers it has like Streamhandler, Filehandler
        
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    
    logger.propagate = False
    
    return logger

logger = get_logger()

URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
PARAMS_PATH = Path("params.yaml")
OUT_DIR = Path("src/data/raw")
USECOLS = ["tweet_id", "sentiment", "content"]

def load_params(path: Path = PARAMS_PATH) -> dict:
    
    """
    Loading the file here
    """
    
    if not path.exists():
        logger.info("params.yaml not found using built-in default")
        return {}
    
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f) or {}
        logger.debug("parameters loaded from %s", path)
        return params
        
    except yaml.YAMLError as e:
        logger.exception("Yaml parse error %s", e)
        raise
    
def read_csv_fast(url: str) -> pd.DataFrame:
    
    """
    reading the file with the columns required and using pyArrow engine
    """
    
    dtypes = {
        "tweet_id" : "string",
        "sentiment" : "string",
        "content" : "string",
    }
    
    try:
        
        df = pd.read_csv(url, usecols=USECOLS, dtype= dtypes, engine = "pyarrow")
        logger.debug("loaded the csv with pyArrow engine")
        
    except Exception:
        df = pd.read_csv(url, usecols=USECOLS, dtype=dtypes)
        logger.debug("loaded the data with default C engine")
        
    logger.info("loaded the data. rows=%d columns=%d", len(df), len(df.columns))
    
    return df

def peprocess(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
    df['sentiment'] = df['sentiment'].map({
        "happiness" : 1, "sadness" : 0 
    }).astype("int8")
    
    if "tweet_id" in df.columns:
       df= df.drop(columns=['tweet_id'])
    logger.info("after filter, rows=%d class_count=%s",
                len(df), df['sentiment'].value_counts().to_dict())
    
    return df.reset_index(drop=True)

def save_path(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # FIX: standardize filenames to match typical usage and your DVC outs
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # FIX: log sizes to verify writes
    logger.info(
        "save the train and test file to %s  train_bytes=%d  test_bytes=%d",
        out_dir.resolve(),
        train_path.stat().st_size,
        test_path.stat().st_size,)
    
def main() -> None:
    
    try:
        params = load_params()
        test_size = float(params.get("data_ingestion", {}).get("test_size", 0.2))
        random_state = int(params.get("data_ingestion", {}).get("random_state", 42))
        stratify = bool(params.get("data_ingestion", {}).get("stratify", True))
        
        logger.info("source URL %s", URL)
        
        df = read_csv_fast(URL)
        df = peprocess(df)
        
        strat_val = df['sentiment'] if stratify else None
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=strat_val,
            random_state=random_state,
            shuffle= True,
        )
        
        logger.info("splitting don train=%s test=%s", train_df.shape, test_df.shape)
        
        save_path(train_df, test_df, OUT_DIR)
    
    except Exception:
        logger.exception("data ingestion failed")
        raise

if __name__ == "__main__":
    main()
        
    
