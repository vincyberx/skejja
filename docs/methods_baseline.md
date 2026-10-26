# Skejja phase 1, Baseline track: methods and notes

Track `baseline`. Two existing open models measured **zero-shot**, as a user gets them, on the Skejja benchmark
(MASSIVE-derived, en/it/es/de/fr, 400 utterance ids x 5 languages per task). **No fine-tuning of any kind.** The only
fitted parameters are post-hoc calibrators (1 temperature for Choice, 2 Platt coefficients for Noul) fitted on the
`calib_*` files (MASSIVE validation split) and applied to `test_*`. All numbers below are copied by script from
`metrics_baseline.csv`, which was produced with the shared `skejja_measure.py` helpers.

## Summary table (Mac CPU proxy; quality is identical across thread counts)

| model | variant | weights MB | choice acc (macro) | worst-lang gap vs en | choice ECE raw/cal (pooled) | noul AUROC (macro) | noul ECE raw/cal (pooled) | choice p50/p95 ms @4t | choice p50 ms @1t | noul p50 ms @4t | peak RSS MB (choice) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-multilingual | torch-fp32 | 643.8 | 0.600 | 0.130 | 0.139 / 0.050 | 0.745 | 0.176 / 0.026 | 109.0 / 186.0 | 72.4 | 89.0 | 3160 |
| laya-multilingual | onnx-fp32 | 1290.5 | 0.600 | 0.130 | 0.139 / 0.050 | 0.745 | 0.176 / 0.026 | 71.6 / 91.7 | 219.0 | 50.4 | 3248 |
| laya-multilingual | onnx-int8 | 335.4 | 0.549 | 0.098 | 0.125 / 0.052 | 0.718 | 0.168 / 0.018 | 34.5 / 47.6 | 83.2 | 25.3 | 1447 |
| laya-multilingual | onnx-w8e4 | 265.9 | 0.599 | 0.125 | 0.135 / 0.052 | 0.743 | 0.174 / 0.021 | 149.0 / 373.6 | 355.9 | 97.5 | 1209 |
| laya-multilingual | onnx-w8e4a4 | 265.9 | 0.593 | 0.115 | 0.140 / 0.048 | 0.741 | 0.179 / 0.020 | 34.6 / 42.7 | 99.8 | 25.0 | 1317 |
| gliclass-multilang-edge | torch-fp32 | 286.0 | 0.547 | 0.152 | 0.106 / 0.029 | 0.698 | 0.265 / 0.035 | 51.0 / 63.2 | 32.2 | 42.9 | 1678 |
| gliclass-multilang-edge | onnx-fp32 | 574.8 | 0.547 | 0.152 | 0.106 / 0.029 | 0.698 | 0.265 / 0.035 | 24.9 / 33.0 | 68.1 | 11.0 | 2072 |
| gliclass-multilang-edge | onnx-int8 | 145.7 | 0.446 | 0.132 | 0.154 / 0.030 | 0.641 | 0.256 / 0.012 | 15.2 / 20.3 | 28.2 | 6.5 | 908 |
| gliclass-multilang-edge | onnx-w8e4 | 110.1 | 0.554 | 0.140 | 0.096 / 0.033 | 0.694 | 0.265 / 0.032 | 50.2 / 76.2 | 133.7 | 34.3 | 821 |
| gliclass-multilang-edge | onnx-w8e4a4 | 110.1 | 0.552 | 0.167 | 0.097 / 0.037 | 0.692 | 0.259 / 0.045 | 13.3 / 16.9 | 31.6 | 5.2 | 833 |

ECE "pooled" = computed on all 2000 test rows. The CSV also has per-language and `macro` (mean of the 5 per-language
values). Per-language ECE uses only 400 rows with 15 bins, so it is noisier and biased upward relative to pooled ECE.
Peak RSS for the ORT variants includes about 410-430 MB for the 256k-entry tokenizer (see Memory).

## Facts relevant to the phase-1 success criteria (Mac proxy, read from the CSV)
- Reference bar: Laya multilingual torch-fp32 macro Choice accuracy = 0.5995. "Within 3 points" therefore means >= 0.5695
  on this benchmark (8 options, English option text).
- GLiClass edge (the existing small model) torch-fp32: macro Choice accuracy 0.5475
  (-5.2 pts vs Laya), so it misses the 3-point bar. Its weight-only quantized
  variants reach 0.5540 (w8e4) and 0.5525 (w8e4a4).
- Worst-language Choice gap vs en (target <= 0.08): Laya 0.130, GLiClass 0.152. Neither baseline meets the target.
- ECE after calibration (target 0.05), pooled over 2000 rows: Laya Choice 0.050 / Noul 0.026;
  GLiClass Choice 0.029 / Noul 0.035. The per-language macro values are higher
  (Laya Choice 0.081, GLiClass Choice 0.060).
- Noul AUROC (target >= 0.9): Laya 0.745, GLiClass 0.698 (macro).
  Neither comes close on these hard near-miss negatives.
- Peak RSS (target <= 300 MB): the lowest measured value is gliclass-multilang-edge onnx-w8e4a4 Noul at 753 MB. The 256k tokenizer alone
  costs about 410-430 MB, so no configuration of these two models meets the target without a smaller tokenizer or vocabulary.
- Latency at 4 threads, ORT, Choice p50 (Mac proxy): Laya int8 34.5 ms / w8e4a4 34.6 ms;
  GLiClass int8 15.2 ms / w8e4a4 13.3 ms. Pi numbers must be measured on the device.

## Checkpoints and software
- Laya multilingual: `convaiinnovations/laya-multilingual` @ `1720e3e3357cfe1e281542e223f8273b0890ca34` (Hub `main` when downloaded,
  2026-10-04). Note: `laya` 0.3.26's opt-in `PINNED_REVISIONS` "reviewed" SHA for this repo is a different commit
  (`e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`). We used the default `main`, which is what `laya.load()` downloads.
  322M parameters (321.9M counted from the safetensors header), stored as **fp16** (643.8 MB); upcast to fp32 at load.
- GLiClass multilang edge: `knowledgator/gliclass-multilang-edge` @ `d16c08ef70547514081952104e6fc3d190d9ee39`;
  143.0M parameters stored as **bf16** (286.0 MB); loaded with `dtype=torch.float32`.
- laya 0.3.26, gliclass 0.1.20, torch 2.14.1 (CPU, BLAS = Apple Accelerate), transformers 5.18.0, onnx 1.23.1,
  onnxruntime 1.30.0, onnxscript 0.7.2 (added to `edgedecide` for the dynamo ONNX exporter), tokenizers 0.23.2.
- Machine: darwin arm64, 10 cores, 64 GB, CPU only (no MPS; Laya was given `device="cpu"` explicitly, because it would pick MPS by itself).

## How each model was fed
**Laya (torch)**: `laya.load(<local snapshot>, device="cpu")`; state = the raw utterance string.
- Choice: `{"intent": {"type": "choice", "instructions": "What does the user want to do?", "criteria": [8 option_desc]}}`.
  The list form makes each option's text exactly the English description. Probability = softmax of the raw marker logits
  divided by the shipped temperature. This checkpoint ships `temperature=[1,1,1]` and no per-option buckets, so the API probabilities are the raw softmax.
- Noul: `{"q": {"type": "noul", "instructions": <question>}}` with laya's default option texts
  ("false: no, the statement does not hold" / "true: yes, the statement holds"); score = z_true - z_false, p_yes = sigmoid(score),
  which is equal to the API's `noul` value.
- Raw logits come from the same encode -> `collate_items` -> `_forward` path that `Agent.predict_batch` uses, so that temperature/Platt
  can be fitted on logits rather than on the API's 4-decimal probabilities. Check against `agent.predict` on 50 items:
  choice max|dp| 5.0e-05 (the API rounds to 4 decimals), argmax 50/50; noul max|dp| 5.0e-05.

**GLiClass (torch)**: `ZeroShotClassificationPipeline(model, tokenizer, device="cpu")`, no prompt, no examples.
- Choice: `classification_type="single-label"`, labels = the 8 option_desc strings, i.e. input
  `<<LABEL>>d1...<<LABEL>>d8<<SEP>>text`, probabilities = softmax over the 8 logits (this is what single-label mode does).
- Noul: `classification_type="multi-label"`, **one** label = the English intent description from `intent_descriptions.json`
  (e.g. "turn on the lights", the same phrase inside "Does the user want to turn on the lights?"), p_yes = sigmoid(logit), threshold 0.5.
  We used the description rather than the full question string because GLiClass is trained with label names (a mapping choice, documented as a deviation).
- Raw logits come from the pipeline's own `prepare_inputs` + model forward. Check against the pipeline on 50 items: top label 50/50,
  max |dscore| 2.9e-06; noul max |dp| 2.1e-06.

Quality runs used batched inference for torch (batch 32, length-sorted, padded). We compared batched against single-item torch logits on 200 test items:
max |dlogit| 3.7e-05 (Laya) and 7.4e-05 (GLiClass). ONNX quality runs used **one decision per `session.run`** (batch 1),
because dynamic int8 picks its activation scale per tensor, so batch composition would change int8 results.

## Calibration and metrics
- Choice: `skejja_measure.fit_temperature` on calib_choice (1500 rows, all 5 languages pooled). Noul: `skejja_measure.fit_platt`
  on calib_noul (1500 rows). Calibrators are refitted **for every variant** (torch/ONNX/int8...).
  torch-fp32 fits: Laya T = 1.589, Platt (a, b) = (0.349, 0.408); GLiClass T = 1.324, Platt (a, b) = (0.491, 1.119).
- Choice metrics: `accuracy`, `nll`, `brier`, `ece_raw` (raw softmax) and `nll_cal`, `brier_cal`, `ece_cal` (after temperature;
  temperature does not change accuracy). Noul metrics: `auroc`, `accuracy`, `brier`, `nll`, `ece_raw` (raw) and `accuracy_cal`, `brier_cal`, `nll_cal`,
  `ece_cal` (after Platt; AUROC is unchanged because a > 0).
- Rows exist per language, `macro` (mean over the 5 languages) and `all` (pooled), and for split all/seen/unseen (`skejja_measure.is_unseen`).
  For these zero-shot baselines "unseen" is only a subset of intents: neither model was trained by us.
- Extra rows: `temperature`, `platt_a`, `platt_b` (split=all, lang=all, n = calib size); `agree_*` (ONNX vs torch, see below);
  latency `mean_ms`; memory `params_m`.
- The `threads` column of quality rows records the thread count used while computing them (torch 10; ORT 2 or 4). Quality does not depend on it.

## Latency protocol
For each (model, variant, task, threads in {4, 1}) we start a fresh child process (`skejja/lat_child.py`). It sets
`torch.set_num_threads(t)` + `set_num_interop_threads(1)` + OMP/MKL/VECLIB env = t, or ORT `intra_op_num_threads=t`,
`inter_op_num_threads=1`, sequential execution and `ORT_ENABLE_ALL`. It uses a fixed random sample (seed 0) of 310 test items with mixed languages,
and `skejja_measure.latency` (10 warm-up + 300 timed calls). Each call is one decision measured end to end
(tokenization + forward + softmax/sigmoid): one text with its 8 options, or one Noul question. Torch uses the public API
(`agent.predict`, pipeline `__call__`). ORT uses the torch-free `skejja_ortlib.py` (tokenizers + onnxruntime). We confirmed that
torch is not imported in the ORT children.
**Caveat**: torch on this Mac uses Apple Accelerate (which uses the AMX matrix units), so torch at 1 thread is *faster* than at 4 threads.
These torch timings are not a usable Raspberry Pi proxy. ORT uses its own MLAS NEON kernels and is the closer proxy, but the Mac
cores are still several times faster than a Pi's Cortex-A53/A72. Int8 speed-ups also depend on the dot-product instructions
(ARMv8.2), which the Pi 3/4 cores lack. Treat every latency here as relative only, until the numbers are measured on a Pi.

## Memory protocol
`peak_rss_mb` = `skejja_measure.peak_rss_mb` on a fresh child process that loads the model and runs 20 decisions (4 threads).
`skejja_measure` (which imports scipy and sklearn) is imported only **after** the model is freed, so its import cost does not inflate the peak.
`weights_mb` = file size on disk. For torch that is the safetensors file in its stored precision (fp16/bf16), so the fp32 model in memory is about 2x larger.
Findings from separate profiling runs (not in the CSV):
- Loading the 256k-entry `tokenizer.json` with HF `tokenizers` raises RSS from 53.0 to 460.7 MB for GLiClass (peak 488.3 MB),
  and from 53.6 to 478.9 MB for Laya (peak 506.5 MB). Both models share the same Gemma-derived 256k vocabulary. **The tokenizer by itself
  exceeds the 300 MB RAM target**, whichever model weights are used.
- ORT session-only RSS (GLiClass, from about 53 MB baseline, ENABLE_ALL): fp32 1582.2 MB (1446.7 MB with `session.disable_prepacking`),
  int8 458.1 MB (412.7 MB), w8e4 273.7 MB (291.9 MB), w8e4a4 291.9 MB. Laya session-only, prepacking on: fp32 2705.2 MB, int8 1044.7 MB,
  w8e4a4 545.8 MB. An fp32 ORT session holds about 2-2.5x the file size. With dynamic int8, session memory is far above the file size
  (Laya: about 993 MB above baseline for a 335 MB file). This fits ORT materialising the uint8-quantized embedding back to fp32
  (256k x 768 x 4 B = 786 MB), but we inferred that from the sizes and did not confirm it in the graph.

## ONNX export
- `torch.onnx.export(..., dynamo=True)` (opset 18, onnxscript optimizer), dynamic batch/sequence/option count, `external_data=False`.
- Laya: wrapper around laya's `DecisionModel.forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)`,
  with I/O names identical to `laya.onnx_agent.ONNXAgent` (laya ships no export script; ONNXAgent's error message mentions an `export_onnx.py` that the package does not include).
  The exported file loads in `ONNXAgent` (20 items vs torch `Agent`: max |dp| = 0.0 at the API's 4-decimal rounding).
  The fp32 file is 1290.5 MB because the weights are fp32 and the embedding table is 256k x 768.
- GLiClass: wrapper around `GLiClassUniEncoder`, with an extra input `label_slots` (an int64 vector whose *length* sets `max_num_classes`; its values
  are unused), so one graph serves both Choice (K=8) and Noul (K=1).
- Torch-free preprocessing (`skejja_ortlib.py`) reproduces laya's `build_head`/`build_sequence` and GLiClass's prompt format:
  0 token-id mismatches out of 7000 calib+test inputs for each model.
- fp32 agreement with torch on 200 test items (single item each): laya-multilingual choice: max|dlogit| 8.8e-05, max|dp| 6.0e-06, argmax agreement 1.000; laya-multilingual noul: max|dlogit| 3.2e-05, max|dp| 3.0e-06, argmax agreement 1.000; gliclass-multilang-edge choice: max|dlogit| 1.3e-04, max|dp| 6.7e-06, argmax agreement 1.000; gliclass-multilang-edge noul: max|dlogit| 5.1e-05, max|dp| 5.8e-06, argmax agreement 1.000. On the full test set, fp32 ONNX decisions agree with torch 100%.
- Before quantization we removed all `value_info` entries from the graphs (annotations only; the graph is unchanged). In Laya's export the
  dynamo optimizer left a stale annotation (`m.act_head.0.weight` recorded with dim 772, while shape inference gives 256), and this made
  `onnx.shape_inference` fail inside `quantize_dynamic`.

### Quantized variants
- `onnx-int8` (the requested recipe): `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)`, defaults (per-tensor weights,
  MatMul -> MatMulInteger with dynamic per-tensor uint8 activations, embedding Gather -> uint8).
- `onnx-w8e4` and `onnx-w8e4a4` (**not requested; added after int8 degraded both models**): weight-only quantization with
  `MatMulNBitsQuantizer`. MatMul weights -> 8-bit, block 128, symmetric (`MatMulNBits`); the embedding -> 4-bit, block 32 (`GatherBlockQuantized`; ORT's
  quantizer supports only 4-bit for Gather). `w8e4` keeps fp32 compute (accuracy_level unset). `w8e4a4` sets `accuracy_level=4`
  (int8 compute with block-wise quantized activations).
- Recipe screen on GLiClass, 200 test items (torch reference: choice accuracy 0.675, noul AUROC 0.753). Each entry gives
  choice argmax agreement with torch / choice accuracy / noul agreement / noul AUROC:
  dynamic int8 default 0.65 / 0.53 / 0.835 / 0.676; per_channel 0.655 / 0.58 / 0.855 / 0.699; MatMul-only (embedding fp32, 440.6 MB) 0.68 / 0.555 / 0.845 / 0.684;
  per_channel + MatMul-only (441.1 MB) 0.67 / 0.59 / 0.885 / 0.691; weight-only 8-bit MatMul (441.9 MB) 0.97 / 0.68 / 0.985 / 0.751;
  plus 4-bit embedding (= w8e4, 110.1 MB) 0.95 / 0.67 / 0.975 / 0.753; plus accuracy_level=4 (= w8e4a4) 0.925 / 0.68 / 0.975 / 0.750.
  Every dynamic-int8 recipe degrades, and the weight-only ones do not. So the loss comes from **dynamic per-tensor activation quantization**, not from
  weight rounding or from the embedding. That outlier activations in ModernBERT/mmBERT cause it is a plausible hypothesis that we did not test.

### Effect of each export (full test set, CSV values)
- laya-multilingual onnx-int8 vs torch-fp32: choice accuracy -5.1 pts (macro), noul AUROC -0.027 (macro), noul accuracy@0.5 -2.5 pts; decision agreement with torch on the full test set: choice 0.729, noul 0.836.
- laya-multilingual onnx-w8e4 vs torch-fp32: choice accuracy -0.1 pts (macro), noul AUROC -0.001 (macro), noul accuracy@0.5 +0.1 pts; decision agreement with torch on the full test set: choice 0.938, noul 0.961.
- laya-multilingual onnx-w8e4a4 vs torch-fp32: choice accuracy -0.7 pts (macro), noul AUROC -0.004 (macro), noul accuracy@0.5 -0.3 pts; decision agreement with torch on the full test set: choice 0.928, noul 0.958.
- gliclass-multilang-edge onnx-int8 vs torch-fp32: choice accuracy -10.1 pts (macro), noul AUROC -0.057 (macro), noul accuracy@0.5 -3.5 pts; decision agreement with torch on the full test set: choice 0.619, noul 0.879.
- gliclass-multilang-edge onnx-w8e4 vs torch-fp32: choice accuracy +0.7 pts (macro), noul AUROC -0.004 (macro), noul accuracy@0.5 -0.8 pts; decision agreement with torch on the full test set: choice 0.933, noul 0.982.
- gliclass-multilang-edge onnx-w8e4a4 vs torch-fp32: choice accuracy +0.5 pts (macro), noul AUROC -0.006 (macro), noul accuracy@0.5 -0.5 pts; decision agreement with torch on the full test set: choice 0.905, noul 0.978.

**laya-pi's report (int8 hurts Laya's Noul head): confirmed.** With the default dynamic int8, Laya's Noul AUROC and accuracy drop, and
16% of Noul yes/no decisions flip relative to torch. Choice is hurt even more. Weight-only 8-bit (w8e4/w8e4a4) mostly removes the problem.

## Limitations
- Options and Noul questions are **English for every language** (cross-lingual setting, as specified). Native-language options could change the results.
- Contamination: the Laya multilingual card reports an evaluation on MASSIVE (51 languages), and the GLiClass card lists MASSIVE as a zero-shot evaluation set.
  Neither card lists MASSIVE as training data. We cannot verify this.
- Noul negatives are hard near-misses: 72.4% of the 1050 test negatives ask about an intent from the same scenario as the gold intent.
- Mac latency and RSS are proxies. macOS allocator behaviour (memory not returned to the OS after the tokenizer JSON parse) can differ on Linux/aarch64.

## What failed / quirks
- `laya.load(..., backend=...)` imports `laya.backends`, which is missing from the 0.3.26 wheel. We did not use `backend=`.
- `quantize_dynamic` failed on the raw Laya export (stale value_info, see above). It worked after stripping the annotations.
- Laya fp32 ONNX (1290.5 MB) is **not saved as an artifact**, because it exceeds the 700 MB rule. Its int8 and w8e4a4 versions are saved.
  The fp32 file can be regenerated with the export code in this track.
- `onnx-w8e4` (fp32 compute) is accurate but 1.6-3.1x **slower** than fp32 ORT on this CPU (MatMulNBits 8-bit without accuracy_level),
  so it was not bundled. `w8e4a4` (accuracy_level=4) is the bundled weight-only variant.
