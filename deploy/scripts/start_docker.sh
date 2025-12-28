#!/bin/bash
set -euo pipefail

APP_NAME="ecr_practice"
REGION="eu-north-1"
REGISTRY="774305594525.dkr.ecr.eu-north-1.amazonaws.com"
IMAGE="${REGISTRY}/ecr_practice:latest"

HOST_PORT="80"
CONTAINER_PORT="80"

systemctl enable --now docker

aws ecr get-login-password --region "$REGION" \
| docker login --username AWS --password-stdin "$REGISTRY"

docker pull "$IMAGE"

docker rm -f "$APP_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$APP_NAME" \
  --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "$IMAGE"

docker ps --filter "name=$APP_NAME"