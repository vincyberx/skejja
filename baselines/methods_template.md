# Skejja phase 1, Baseline track: methods and notes

Track `baseline`. Two existing open models measured **zero-shot**, as a user gets them, on the Skejja benchmark
(MASSIVE-derived, en/it/es/de/fr, 400 utterance ids x 5 languages per task). **No fine-tuning of any kind.** The only
fitted parameters are post-hoc calibrators (1 temperature for Choice, 2 Platt coefficients for Noul) fitted on the
`calib_*` files (MASSIVE validation split) and applied to `test_*`. All numbers below are copied by script from
`metrics_baseline.csv`, which was produced with the shared `skejja_measure.py` helpers.

## Summary table (Mac CPU proxy; quality is identical across thread counts)

{TABLE}

ECE "pooled" = computed on all 2000 test rows. The CSV also has per-language and `macro` (mean of the 5 per-language
values). Per-language ECE uses only 400 rows with 15 bins, so it is noisier and biased upward relative to pooled ECE.
Peak RSS for the ORT variants includes about 410-430 MB for the 256k-entry tokenizer (see Memory).

{CRITERIA}
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
  torch-fp32 fits: Laya T = {LAYA_T}, Platt (a, b) = ({LAYA_PA}, {LAYA_PB}); GLiClass T = {GLI_T}, Platt (a, b) = ({GLI_PA}, {GLI_PB}).
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
- fp32 agreement with torch on 200 test items (single item each): {AGREE_FP32}. On the full test set, fp32 ONNX decisions agree with torch 100%.
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
{DELTAS}

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
