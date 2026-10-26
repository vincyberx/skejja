#!/usr/bin/env bash
# Skejja phase-1 benchmark: installa le dipendenze in un venv e lancia il benchmark.
set -euo pipefail
cd "$(dirname "$0")"
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "arm64" ]; then
  echo "Serve un sistema operativo a 64 bit (aarch64). Trovato: $ARCH"; exit 1
fi
echo "Dispositivo: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || uname -a)"
free -m || true
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python bench_pi.py "$@"
