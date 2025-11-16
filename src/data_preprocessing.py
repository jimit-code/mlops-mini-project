import pandas as pd
import numpy as np
import logging
import nltk, string
from pathlib import Path
import re
from functools import lru_cache
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

# Ensure the resource is available
try:
    STOPWORDS = set(nltk_stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(nltk_stopwords.words("english"))
    
# Build a set for O(1) membership checks
# STOPWORDS = set(stopwords.words("english"))

    
OUT_DIR = Path("data/processed")
    
def get_logger(name: str = __name__) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler("data_preprocessing_error.log")
    fh.setLevel(logging.WARNING)
    
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    logger.propagate = False
    
    return logger

logger = get_logger()    
    
LEMMANTIZE = WordNetLemmatizer()

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_WS = re.compile(r"\s+")

TRANS_TBL = str.maketrans({C: "" for C in (string.punctuation + string.digits)})

@lru_cache(maxsize=100_000)
def lemma(text: str) -> str:
    return LEMMANTIZE.lemmatize(text, pos="v")

def vector_clearn(series: pd.Series) -> pd.Series:
    
    s = series.astype(str).str.lower()
    s = s.str.replace(RE_URL, "", regex=True)
    s = s.apply(lambda x: x.translate(TRANS_TBL))
    s = s.str.replace(RE_WS, "", regex=True).str.strip()
    
    return s

def tokenize(text: str) -> str:
    
    tok = []
    
    for w in text.split():
        
        if w in STOPWORDS:
            continue
        
        if not w.isalpha():
            continue
        
        tok.append(lemma(w))
    
    return " ".join(tok)

def normalize_text(df: pd.DataFrame, content_col: str = "content") -> pd.DataFrame:
    
    if content_col not in df.columns:
        logger.error("the content columns is not there")
    
    try:
        
        out = df.copy()
        out[content_col] = vector_clearn(out[content_col])
        out[content_col] = out[content_col].apply(tokenize)
        logger.debug("clearned the string and converted into the main form")
    except Exception as e:
        logger.exception("error during text normalisation %s", e)
        raise
    
    return out

def save_path(train_df: pd.DataFrame, test_df:pd.DataFrame, out_dir:Path = OUT_DIR) -> None:
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir/ "train.csv"
    test_path = out_dir / "test.csv"
    
    train_df.to_csv(train_path, index= False)
    test_df.to_csv(test_path, index=False)
    
    logger.info("saved train and test file to %s train_bytes= %d test_bytes = %d", 
                out_dir.resolve(), train_path.stat().st_size, test_path.stat().st_size)
    
def main():
    try:
        train_data = pd.read_csv("./src/data/raw/train.csv")
        test_data = pd.read_csv("./src/data/raw/test.csv")
        logger.debug("Train and Test Data loaded successfully")
        
        train_data_processed = normalize_text(train_data)
        test_data_processed = normalize_text(test_data)
        
        logger.debug("Train and Test data content columns cleaned")
        
        save_path(train_data_processed, test_data_processed, OUT_DIR)
        
        logger.info("file saved to the path successfully")
    
    except Exception as e:
        logger.exception("Unable to Complete the data transformation: %s", e)
        raise

if __name__ == "__main__":
    main()
        
        
        
        

