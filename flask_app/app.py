from flask import Flask, render_template, request

import os
import re
import string
import pickle
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None

import pandas as pd
import numpy as np
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ----------------------------
# Text cleaning helpers
# ----------------------------
def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    words = [word for word in str(text).split() if word not in stop_words]
    return " ".join(words)


def removing_numbers(text: str) -> str:
    return "".join([char for char in text if not char.isdigit()])


def lower_case(text: str) -> str:
    words = text.split()
    words = [word.lower() for word in words]
    return " ".join(words)


def removing_punctuations(text: str) -> str:
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def removing_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def normalize_text(text: str) -> str:
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


# ----------------------------
# Paths + env
# ----------------------------
# In Docker your file path is usually /app/flask_app/app.py
# parents[1] -> /app. parents[2] -> / (wrong for .env)
ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

# DagsHub creds
user = os.getenv("DAGSHUB_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
token = (
    os.getenv("DAGSHUB_PAT")
    or os.getenv("DAGSHUB_USER_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
)

if not user:
    raise EnvironmentError("DAGSHUB_USERNAME (or MLFLOW_TRACKING_USERNAME) is not set")

if not token:
    raise EnvironmentError("DAGSHUB_PAT (or DAGSHUB_USER_TOKEN or MLFLOW_TRACKING_PASSWORD) is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = user
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Tracking URI. Prefer env value from .env, else fallback to your repo URL
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "https://dagshub.com/jimit-code/mlops-mini-project.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

# ----------------------------
# NLTK resources (avoid runtime crash)
# ----------------------------
# If your Docker image does not contain NLTK data, these can crash.
# This download is safe for local testing. For production you should bake these into the image.
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    nltk.download("wordnet")


# ----------------------------
# Load model and vectorizer
# ----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Sentiment_XGB_BOW")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")

# Preferred. Load by alias that your promotion script sets.
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

try:
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    # Fallback. Load latest version if alias is missing.
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = list(client.search_model_versions(f"name = '{MODEL_NAME}'"))
    if not versions:
        raise RuntimeError(f"No versions found in registry for model '{MODEL_NAME}'. Original error: {e}")

    latest = max(versions, key=lambda v: int(v.version))
    fallback_uri = f"models:/{MODEL_NAME}/{latest.version}"
    model = mlflow.pyfunc.load_model(fallback_uri)

# Vectorizer path. Match your Docker COPY destination.
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", str(ROOT / "model" / "vectorizer.pkl")))
if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")

with VECTORIZER_PATH.open("rb") as f:
    vectorizer = pickle.load(f)


# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    # Clean
    text = normalize_text(text)

    # Vectorize
    features = vectorizer.transform([text])

    # Convert to DataFrame. Keep stable column naming.
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    result = model.predict(features_df)

    return render_template("index.html", result=result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
