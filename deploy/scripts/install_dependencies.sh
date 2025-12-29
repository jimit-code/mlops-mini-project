#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[install_dependencies] Updating apt"
apt-get update -y

echo "[install_dependencies] Installing utilities + AWS CLI"
apt-get install -y unzip curl ca-certificates awscli

echo "[install_dependencies] Installing Docker"
apt-get install -y docker.io

echo "[install_dependencies] Enabling Docker"
systemctl enable --now docker

echo "[install_dependencies] Done"