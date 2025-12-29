#!/bin/bash
set -euo pipefail

APP_NAME="ecr_practice"
REGION="eu-north-1"
REGISTRY="774305594525.dkr.ecr.eu-north-1.amazonaws.com"
IMAGE="${REGISTRY}/ecr_practice:latest"

HOST_PORT="80"
CONTAINER_PORT="80"

# Required app env vars (set your real values)
DAGSHUB_USERNAME="jimit-code"
DAGSHUB_TOKEN="7b456baf38cb3c1b67806df8a072d4a070d8ff55"

echo "[start_docker] Ensuring Docker is running"
systemctl enable --now docker

echo "[start_docker] Checking AWS CLI"
aws --version

echo "[start_docker] Logging in to ECR (non-interactive)"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

echo "[start_docker] Pulling image: $IMAGE"
docker pull "$IMAGE"

echo "[start_docker] Stopping old container if exists"
docker rm -f "$APP_NAME" >/dev/null 2>&1 || true

echo "[start_docker] Starting new container on :${HOST_PORT} -> :${CONTAINER_PORT}"
docker run -d \
  --name "$APP_NAME" \
  --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e DAGSHUB_USERNAME="$DAGSHUB_USERNAME" \
  -e DAGSHUB_TOKEN="$DAGSHUB_TOKEN" \
  "$IMAGE"

echo "[start_docker] Running containers:"
docker ps --filter "name=$APP_NAME"