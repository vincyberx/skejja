---
license: apache-2.0
language: [en, it, es, de, fr]
library_name: onnxruntime
pipeline_tag: zero-shot-classification
base_model: knowledgator/gliclass-multilang-edge
tags: [system-one, typed-decisions, choice, yes-no, calibration, onnx, int8, raspberry-pi, edge, multilingual]
datasets: [mteb/amazon_massive_intent]
---

# Skejja edge32k-ft6

Skejja is a small, open, multilingual **System One** decision model: it does not generate text. It reads a state
(text or JSON) and typed questions, and returns calibrated probabilities in a single forward pass:

- **Choice**: a probability distribution over closed options;
- **Noul**: the probability of a yes/no answer;
- **Score**: a position on an ordinal scale (a Choice over the levels).

It runs on modest CPUs, including a Raspberry Pi, with about 200 MB of RAM, and exposes an API compatible with
`POST /v1/systemone`, so existing System One clients work by changing only the URL.

Code, benchmark and every result file: [github.com/vincyberx/skejja](https://github.com/vincyberx/skejja). The file
paths in this card refer to that repository. **[Try it in your browser](https://huggingface.co/spaces/vincyber/skejja)**:
the demo runs the dynamic int8 file with onnxruntime-web, nothing is sent to a server.

| | |
|---|---|
| Parameters | 57.3M (44.7M without the embedding lookup) |
| Base model | [knowledgator/gliclass-multilang-edge](https://huggingface.co/knowledgator/gliclass-multilang-edge) (Apache-2.0), vocabulary trimmed to 32,768 tokens for en/it/es/de/fr |
| Version | v0.1.0 (Skejja package 0.1.0) |
| Downloads | `skejja-edge32k-ft6_onnx-int8.tar.gz` (default, 208 MB) and `skejja-edge32k-ft6_onnx-fp32.tar.gz` (exact weights, 212 MB); hashes in `SHA256SUMS` |
| Files in the archives | `edge-v32k-ft6.fp32.onnx`, `edge-v32k-ft6.int8.onnx` (weight-only int8), `edge-v32k-ft6.int8dyn.onnx` (dynamic int8, for ARM CPUs without dot-product), each with its `.onnx.data`; `tokenizer.json`; `calibration.json` |
| Languages | English, Italian, Spanish, German, French (trained and measured) |
| License | Apache-2.0 (attributions in `NOTICE`) |

## Use

```sh
pip install "skejja[server] @ git+https://github.com/vincyberx/skejja"   # until it is on PyPI
skejja serve --port 8080              # downloads this model on first use and checks its SHA-256
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "puoi spegnere le luci della cucina?",
  "questions": {"action": {"type": "choice", "criteria": {
      "lights_on": "turn on the lights", "lights_off": "turn off the lights", "music": "play music"}}}}'
```

`skejja serve <folder>` uses an archive extracted by hand instead. Also available: a Python API
(`skejja.SystemOne`), an MCP server for agents (`skejja mcp`) and a Dockerfile. Option descriptions work best in
English; the state can be in any of the five languages. Yes/no questions work best phrased as in training:
"Does the user want to <concrete action>?". The archive also contains `skejja_infer.py`, which runs the model with
only `onnxruntime`, `tokenizers` and `numpy`.

## Results

All quality numbers are on the Skejja benchmark built from MASSIVE (CC BY 4.0): 2,000 Choice questions with
8 options and 2,000 yes/no questions in five languages; "unseen" = 4 intent scenarios held out from training.
Measured in PyTorch fp32. In the export check the quantized files agree with fp32 on the top Choice option
99.3% (int8) and 96.7% (dynamic int8) of the time, and on the yes/no decision 98.3% and 97.3%
(`export_check.json` in the bundle).

| Metric (MASSIVE, 5 languages) | edge32k-ft6 | Source |
|---|---|---|
| Choice accuracy, all / seen / unseen | 0.873 / 0.895 / 0.741 | `results/fase2/metrics_edge32k-ft6.csv` |
| Choice accuracy en / it / es / de / fr | 0.893 / 0.875 / 0.878 / 0.850 / 0.868 | same |
| Choice ECE (temperature-calibrated) | 0.048 | same |
| Yes/no AUROC / ECE, test | 0.961 / 0.038 | same |
| Yes/no on unseen intents, no examples: AUROC / ECE | 0.840 / 0.150 | same |
| Same, after per-question calibration with 64 examples | 0.902 / 0.041 | `results/fase2/metrics_fewshot.csv` |
| Choice accuracy with options translated into the sentence's language | 0.849 | `results/fase2/metrics_edge32k-ft6.csv` |

**Other open models on the same benchmark** (run by us, zero-shot as released, calibration fitted only on the
calibration files): Choice accuracy / yes/no AUROC are 0.600 / 0.744 for Laya multilingual (322M), 0.547 / 0.697
for GLiClass multilang edge (143M, the base of this model) and 0.514 / 0.503 for Julia 1 (144M)
(`results/fase1/metrics_baseline.csv`, `results/fase3/metrics_julia-1.csv`). The benchmark is close to this
model's training data, so the gap shows what Skejja was built for, not general ability.

**Real requests** (167 requests to the author's voice assistant, not in the training data; only aggregate
metrics are published): routing to 5 destinations 0.589 on the 124 with an unambiguous label (English
descriptions), yes/no on concrete actions (mail and calendar, web search; images, only 3 positives) AUROC
0.86–0.87, on software-session requests 0.62
(`results/fase3/real_requests_routing_ft6.csv`, `docs/real_requests.md`). On a long-context decision (whether a voice assistant should pass a turn
to a larger model, inputs over 512 tokens) AUROC 0.79: not good enough (`results/fase3/real_requests_handoff_ft6.csv`).

**Raspberry Pi 4** (4 GB, dynamic int8, 8-option Choice, 14 October 2026; the board had no heatsink and
throttled thermally, so latencies are upper bounds): Choice 380 ms on 1 thread and 182 ms on 4, yes/no 126 and
81 ms, peak RAM 216–227 MB (`results/pi4/pi4_20261014-185806.json`). Longer inputs on one thread: 2.2 s and
203 MB at 255 tokens, 5.1 s and 246 MB at 511 tokens, 13.3 s and 398 MB at 1,023 tokens
(`results/fase2/costo_lunghezza_opzioni_pi4.csv`). The server shortens inputs above 512 tokens by default.

**Beyond intents** (zero-shot, datasets of the independent Jev evaluation, arXiv 2609.37647; own prompts, so
not a like-for-like comparison): AG News 0.782, SST-2 0.807, Emotion 0.358, SMS Spam F1 0.749
(`results/fase2/metrics_jev_suite_ft6.csv`). Large models and cloud System One services are 10–30 points higher on
these tasks (`docs/confronto_concorrenti.md`).

## Intended use

Fast, local, private decisions on short texts: routing commands and user intents, yes/no checks about what the
user wants, checks on agent actions, in the five languages above, on devices where a cloud call or a large model
is too slow, too costly or not allowed. Use per-question calibration for the questions that matter.

## Limitations

- **Not a general System One:** reasoning, entailment between two texts and Choice with many options are near
  chance (S1MB english-v1, `docs/fase2_ampiezza.md`).
- **Short inputs:** trained on inputs up to 192 tokens; quality drops on long contexts (see Real requests above).
- **Real requests are harder than the benchmark:** catch-all options ("anything else") and user jargon are the
  main failure modes. Prefer one yes/no per action and calibrate the questions that matter.
- **Uncalibrated on new yes/no questions without examples** (ECE 0.150 above): calibrate them with labelled
  examples (`POST /v1/skejja/calibrations`).
- **Options in the language of the sentence** score 2.4 points below English option descriptions (0.849 vs
  0.873), German the weakest.
- **Raspberry Pi:** measured on a Pi 3 (earlier candidates) and a Pi 4; no Pi 5 numbers yet.
- About 31% of the MASSIVE labels of the unseen sentences are questionable (`bench/data/annotations/`), so the
  unseen numbers are noisy.

## Training data

MASSIVE train utterances (CC BY 4.0, 4 scenarios held out), yes/no data from CLINC150 (CC BY 3.0), BANKING77 and
MINDS-14 (CC BY 4.0) with OPUS-MT translations, agent-action data from tldr-pages (CC BY 4.0), and breadth data:
texts from FineWeb/FineWeb-2 (ODC-By 1.0), the Aya Dataset (Apache-2.0) and MFAQ (CC0), plus synthetic texts and
soft labels generated by Qwen3.5-9B (Apache-2.0). No data from the evaluation suites above or from S1MB was used
for training. Full list and changes: `NOTICE`.
