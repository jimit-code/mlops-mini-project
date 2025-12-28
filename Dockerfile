# Builder stage
FROM python:3.11-slim AS builder
WORKDIR /build

# Create a virtual environment so runtime stage is clean
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR=1

# Install only serving requirements (small)
COPY flask_app/requirements-serve.txt /build/requirements-serve.txt
RUN pip install --upgrade pip && pip install -r /build/requirements-serve.txt

# Download NLTK data once in builder
RUN python -m nltk.downloader -d /opt/nltk_data punkt stopwords wordnet


# Runtime stage
FROM python:3.11-slim
WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
ENV NLTK_DATA="/opt/nltk_data"

# Copy only runtime deps and nltk data
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data

# Copy only the app and model artifacts needed to run
COPY flask_app/ /app/flask_app/
COPY model/vectorizer.pkl /app/model/vectorizer.pkl

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app.app:app"]