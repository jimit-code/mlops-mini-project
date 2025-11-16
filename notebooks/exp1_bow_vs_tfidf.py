import mlflow, mlflow.sklearn, pandas as pd, numpy as np, re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functools import lru_cache
import dagshub
import re, os, joblib
from pathlib import Path

try: 

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except LookupError:
    import nltk
    nltk.download("stopwords")
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
mlflow.set_tracking_uri('https://dagshub.com/jimit-code/mlops-mini-project.mlflow')

dagshub.init(repo_owner='jimit-code', repo_name='mlops-mini-project', mlflow=True)

    
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
print("data added")

try: 
    STOPWORDS = set(stopwords('english'))
    
except:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))
    
LEMMNTIZE = WordNetLemmatizer()

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_WS = re.compile(r"\s+")

TRANS_TABLE = str.maketrans({c: " " for c in (string.punctuation + string.digits)})

@lru_cache(maxsize=100_000)
def lemmantize(token : str) -> str:
    return LEMMNTIZE.lemmatize(token)

def vector_cleaner(series: pd.Series) -> pd.Series:
    
    s = series.astype(str).str.lower()
    s = s.str.replace(RE_URL, "", regex=True)
    s = series.apply(lambda x: x.translate(x))
    s = series.replace(RE_WS, "", regex=True).str.strip()
    
    return s

def tokenise_stop_lemma(text: str) -> str:
    
    tok = []
    
    for w in text.split():
        
        if w in STOPWORDS:
            continue
        
        if not w.isalpha():
            continue
        
        tok.append(lemmantize(w))
        
    return " ".join(tok)

def normalise_text (df: pd.DataFrame, content_col: str = "content") -> pd.DataFrame:
    
    if content_col not in df.columns:
        raise KeyError(f"The {content_col} not found in df")
    
    out = df.copy()
    s = vector_cleaner(out[content_col])
    out[content_col] = out[content_col].apply(tokenise_stop_lemma)
    
    return out

filt = df['sentiment'].isin(['sadness', 'happiness'])
df = df[filt]
df = normalise_text(df)
df['sentiment'] = df['sentiment'].replace({'sadness': 0 , 'happiness': 1})

print("data cleaned and processing for mlflow and model training")

mlflow.set_experiment("bow vs TFIDF")

vectorizer = {
    "bow" : CountVectorizer(max_features=50_000, ngram_range=(1,1)),
    "tfidf" : TfidfVectorizer(max_features=50_000, ngram_range=(1,1)),
    "hasing_2^18" : HashingVectorizer(n_features=2*18, alternate_sign=False, norm="l2")
}

algorithms = {
    "LogisticRegression" : LogisticRegression(
        max_iter= 1000, n_jobs=-1, solver="saga", random_state=42
    ),
    "MultinomialNB" : MultinomialNB(),
    "RandomForest" : RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate = 0.1,
        max_depth = 8,
        sub_sample = 0.8,
        colsample_bytree = 0.8,
        reg_lambda = 1.0,
        n_jobs = -1, 
        tree_method = "hist",
        eval_metrics = "logloss",
        random_state = 42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
)

print("data split and now going for experimenting and registering it under mlflow")

with mlflow.start_run(run_name="All Experiments") as parent:
    base_params = {
        "data.test_size": 0.2,
        "data.stratify" : True,
        "data.random_State": 42
    }
    
    mlflow.log_params(base_params)
    
    for algo_name, clf in algorithms.items():
        for vec_name, ven in vectorizer.items():
            
            run_name = f"{algo_name}__{vec_name}"
            
            with mlflow.start_run(run_name=run_name, nested=True) as child:
                
                pipe = make_pipeline(ven, clf)
                pipe.fit(X_train, y_train)
                
                y_pred = pipe.predict(X_test)
                
                metrics_dict = {
                    "accuracy" : float(accuracy_score(y_test, y_pred)),
                    "precision" : float(precision_score(y_test, y_pred)),
                    "recall" : float(recall_score(y_test, y_pred)),
                    "f1_score" : float(f1_score(y_test, y_pred)),
                }
                
                mlflow.log_metrics(metrics_dict)
                
                mlflow.log_params({
                    "algorithm_name" : algo_name,
                    "vectorize" : vec_name,
                })
                
                if algo_name == "LogisticRegression":
                    mlflow.log_params({
                        "LOGR_C" : pipe.get_params().get("logisticregression__C", "NA"),
                        "LOGR_penalty" : pipe.get_params().get("logisticregression__penalty", "l2"), 
                    })
                    
                elif algo_name == "MultinomialNB":
                    mlflow.log_param("MulNB_alpha", pipe.get_params().get("Multinomialnb__alpha", "NA"))
                
                elif algo_name == "XGBoost":
                    p = pipe.get_params()
                    mlflow.log_params({
                        "XGB_n_estimators" : p.get("xgbclassifer__n_estimator", "NA"),
                        "XGB_max_depth" : p.get("xgbclassifer__max_depth", "NA"),
                        "XGB_learning_rate": p.get("xgbclassifer__learning_rate", "NA")
                    })
                    
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
                out_dir = Path("artifacts") / safe_name
                out_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipe, out_dir/"pipeline.pkl")
                mlflow.log_artifact(str(out_dir), artifact_path=f"model/{safe_name}")
                
                print(f"{run_name} -> acc={metrics_dict['accuracy']: .4} ")
    