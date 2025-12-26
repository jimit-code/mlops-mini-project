FROM python:3.11
WORKDIR /app

COPY flask_app/ /app/flask_app/
COPY model/vectorizer.pkl /app/model/vectorizer.pkl

RUN pip install --no-cache-dir -r /app/flask_app/requirements.txt

RUN python -m nltk.downloader punkt stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app.app:app"]