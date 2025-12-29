#!/bin/bash
set -euo pipefail

APP_NAME="ecr_practice"
REGION="eu-north-1"
REGISTRY="774305594525.dkr.ecr.eu-north-1.amazonaws.com"
IMAGE="${REGISTRY}/ecr_practice:v4.0"   # change to v4 if your tag is exactly v4

# ALB hits port 80 on the instance. Your app listens on 5000 inside the container.
HOST_PORT="80"
CONTAINER_PORT="5000"

# Required by your app
DAGSHUB_USERNAME="jimit-code"
DAGSHUB_PAT="7b456baf38cb3c1b67806df8a072d4a070d8ff55"   # this is the one your logs complain about

echo "[start_docker] Ensuring Docker is running"
systemctl enable --now docker

echo "[start_docker] Logging in to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

echo "[start_docker] Pulling image: $IMAGE"
docker pull "$IMAGE"

echo "[start_docker] Removing old container if exists"
docker rm -f "$APP_NAME" >/dev/null 2>&1 || true

echo "[start_docker] Starting container"
docker run -d \
  --name "$APP_NAME" \
  --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e DAGSHUB_USERNAME="$DAGSHUB_USERNAME" \
  -e DAGSHUB_PAT="$DAGSHUB_PAT" \
  "$IMAGE"

echo "[start_docker] Containers:"
docker ps --filter "name=$APP_NAME"