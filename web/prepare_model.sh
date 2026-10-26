#!/usr/bin/env bash
# Copia in web/model/ i file che la demo carica, da un bundle estratto (predefinito: quello in releases/).
# Uso: ./web/prepare_model.sh [cartella del bundle]
set -euo pipefail
cd "$(dirname "$0")"
B="${1:-../releases/skejja-edge32k-ft6_onnx}"
mkdir -p model
# solo l'int8 dinamico (predefinito della demo, il più veloce in WASM); per provare l'altro: EXTRA=int8
for f in tokenizer.json tokenizer_config.json calibration.json edge-v32k-ft6.int8dyn.onnx edge-v32k-ft6.int8dyn.onnx.data \
         ${EXTRA:+edge-v32k-ft6.$EXTRA.onnx edge-v32k-ft6.$EXTRA.onnx.data}; do
  cp "$B/$f" model/ && chmod 644 "model/$f"
done
ls -la model
