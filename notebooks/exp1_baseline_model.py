import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import string
from functools import lru_cache
from sklearn.model_selection import train_test_split
import dagshub

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
print(df.head())

STOPWORDS = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_WS = re.compile(r"\s+")
TRANS_TABLE = str.maketrans({c : " " for c in (string.punctuation + string.digits)})

@lru_cache
def lemmentize(tocken: str) -> str:
    return(LEM.lemmatize(tocken, pos="v"))

def vector_clean(series: pd.Series) -> pd.Series:
    
    #start with lowering the string
    s = series.astype(str).str.lower()
    s = s.replace(RE_URL, " ", regex=True)
    s = s.apply(lambda x: x.translate(TRANS_TABLE))
    s = s.replace(RE_WS, " ", regex=True).str.strip()
    
    return s

def token_stop_lemm(text: str) -> str:
    
    toks = []
    
    for w in text.split():
        
        if w in STOPWORDS:
            continue
        
        if not w.isalpha():
            continue
        
        toks.append(lemmentize(w))
        
    return " ".join(toks)

def normalise_test(df: pd.DataFrame, content_col:str = "content") -> pd.DataFrame:
    """
    extremely fast normalised with NLTK only.
    1) Vectorised lower + removing url + removing punctuation + white space
    2) removing stop words and removing alpha 
    """
    
    if content_col not in df.columns:
        raise KeyError(f"column '{content_col}' not found")
    
    out = df.copy()
    s = vector_clean(out[content_col])
    out[content_col] = s.apply(token_stop_lemm)
    
    return out

df = normalise_test(df)
filt= df['sentiment'].isin(['sadness', 'happiness'])
df= df[filt]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, "happiness": 1})

vectorised = CountVectorizer(max_features=1000)

X = vectorised.fit_transform(df['content'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri('https://dagshub.com/jimit-code/mlops-mini-project.mlflow')
dagshub.init(repo_owner='jimit-code', repo_name='mlops-mini-project', mlflow=True)

mlflow.set_experiment('Logistic Regression Baseline')

with mlflow.start_run(run_name = 'LogReg BOW Base'):
    
    config = {
        "vectorizer" : "CountVectorizer",
        "vectorizer.max_Feature" : 1000,
        "data.test_size" : 0.2
    }
    
    mlflow.log_dict(config, "config.json")
    
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train, y_train)
    
    mlflow.log_param("model", "logistic_regression")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score (y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metrics({
        "accuracy" : float(accuracy),
        "precision" : float(precision),
        "recall" : float(recall),
        "f1_score" : float(f1)
    })
    
    import joblib, os
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    mlflow.log_artifact("artifacts/model.pkl", artifact_path="model")
    
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall:{recall}")
    print(f"f1_score{f1}")
