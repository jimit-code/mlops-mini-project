#!/bin/bash
set -euo pipefail

curl -fsS http://localhost:80 >/dev/null
echo "Service is responding on port 80"