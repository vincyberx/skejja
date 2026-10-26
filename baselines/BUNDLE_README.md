# Skejja phase-1 baseline ONNX bundle: {MODEL_NAME} ({VARIANT})

Source checkpoint: `{REPO}` @ `{REV}` (Apache-2.0, original authors: {AUTHORS}).
This bundle is a format conversion (ONNX export{QUANT_NOTE}) made for Skejja phase-1 measurement; it is
not an official release of the original authors. Zero-shot model, no fine-tuning.

## Contents
- `{ONNX_FILE}` - ONNX graph (opset 18, exported with torch {TORCH} dynamo exporter; onnxruntime {ORT})
- `tokenizer.json` (+ config files) - original tokenizer from the source repo, unchanged
- `skejja_ortlib.py` - torch-free pre/post-processing (needs only `numpy`, `tokenizers`, `onnxruntime`)
- `model_info.json` - I/O spec, prompt format, calibration parameters fitted on the Skejja calib split

## Run (no torch needed)
```bash
pip install onnxruntime tokenizers numpy
```
```python
{RUN_SNIPPET}
```
Threads: `threads=` sets ORT `intra_op_num_threads` (inter_op=1, sequential). Use 4 on a Pi 4/5, 4 on Pi 3.

## Calibration (fitted on Skejja calib_*.jsonl, MASSIVE validation split, 5 langs pooled)
- Choice: `p = softmax(logits / T)` with `T = {TEMP}`
- Noul:   `p_yes = sigmoid(a * score + b)` with `a = {PLATT_A}`, `b = {PLATT_B}`

## Measured on the Skejja benchmark (Mac CPU proxy, see metrics_baseline.csv)
{METRICS_NOTE}

## Known memory caveat
Loading the 256k-entry `tokenizer.json` with HF `tokenizers` alone raised RSS by about 410-430 MB in our
macOS measurement (53 -> ~461-479 MB). On a 1 GB Pi 3 this tokenizer is the largest single memory item for the int8 bundles.
