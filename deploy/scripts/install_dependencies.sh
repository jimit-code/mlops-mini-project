#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[install_dependencies] Updating apt"
apt-get update -y

echo "[install_dependencies] Installing utilities"
apt-get install -y unzip curl ca-certificates docker.io

echo "[install_dependencies] Enabling Docker"
systemctl enable --now docker

echo "[install_dependencies] Installing AWS CLI v2"
cd /tmp
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q -o awscliv2.zip
./aws/install --update

echo "[install_dependencies] AWS CLI version:"
/usr/local/bin/aws --version || aws --version

echo "[install_dependencies] Done"