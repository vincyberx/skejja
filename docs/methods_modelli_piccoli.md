# Skejja phase 1 — track "Modelli piccoli" (small candidates): methods & notes

All numbers below are copied programmatically from `metrics_modelli_piccoli.csv` / `vocab_trim_modelli_piccoli.csv` (produced by the code in this session, metrics via `skejja_measure.py`). Machine: Apple-silicon Mac, darwin, 10 CPU cores, CPU only (no GPU/MPS). **Latency/RSS on this Mac are a proxy only, not Raspberry Pi numbers.**

## 1. What is zero-shot vs fine-tuned

| model id in CSV | what it is | training on MASSIVE? |
|---|---|---|
| `gliclass-edge-orig` | knowledgator/gliclass-multilang-edge as shipped (reference row) | no — **zero-shot** |
| `gliclass-edge-trim64k-zs` | same weights, vocabulary trimmed to 65,536 | no — **zero-shot** |
| `gliclass-edge-trim32k-zs` | same weights, vocabulary trimmed to 32,768 | no — **zero-shot** |
| `gliclass-edge-trim32k-ft` | trim32k, fine-tuned (label-conditioned) on MASSIVE train, 49 seen intents only, light KD from the zero-shot edge model | **fine-tuned** |
| `gliclass-edge-trim32k-L8-ft` | trim32k with 8 of 22 encoder layers kept, distilled from `trim32k-ft` + gold labels | **fine-tuned** |

**Fairness note.** The fine-tuned models have seen the 49 *seen* intents (and their English descriptions) during training, while the baselines are zero-shot. Seen and unseen are reported separately everywhere; **only the `unseen` split (gold intent in HELD_OUT_SCENARIOS = transport, takeaway, social, cooking — 10 intents never used as training utterances nor as options) is a fair comparison against zero-shot baselines.** The unseen split is small: 58 Choice items per language (290 total) and 60 Noul items per language (300 total); per-language unseen numbers carry roughly ±6 points sampling noise.

## 2. Checkpoints, data, software
- Base: `knowledgator/gliclass-multilang-edge` revision `d16c08ef70547514081952104e6fc3d190d9ee39` (Apache-2.0), uni-encoder, backbone `jhu-clsp/mmBERT-small` (ModernBERT, 22 layers, hidden 384, vocab 256,003), cross-attention scorer, class-token average pooling. Weights shipped in bf16; all our models are run/saved in fp32 (exact up-cast).
- Vocabulary-building corpus (no test data): MASSIVE train utterances (`train_utterances.jsonl`, 54,795 rows, all 5 languages, all intents — only token statistics are used), the 59 English intent descriptions plus their "Does the user want to …?" question forms and the formatted label prefix, and a Wikipedia sample: `wikimedia/wikipedia` revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configs `20231101.{en,it,es,de,fr}`, first parquet shard, row groups at 0, 1/3 and 2/3 of the shard (3,000 articles per language), ≤12 sentences per article, 20–300 chars, shuffled (seed 0): 30,000 sentences each for en/es/de/fr except es 29,958 and **it 22,256** (fewer usable sentences in those row groups). Wikipedia is CC BY-SA; only token frequencies are derived, no text is redistributed.
- Software: torch 2.14.1, transformers 5.18.0, onnxruntime 1.30.0, onnx 1.23.1, tokenizers 0.23.2, gliclass 0.1.20.

## 3. How options/questions are fed (identical for every model here)
GLiClass uni-encoder format with `prompt_first=True`, exactly as `gliclass.UniEncoderZeroShotClassificationPipeline.prepare_input`: `<<LABEL>>opt_1<<LABEL>>opt_2…<<LABEL>>opt_K<<SEP>>text` (+ `<bos>`/`<eos>` from the tokenizer).
- **Choice**: the 8 `option_desc` strings (English, for every language) in file order → 8 logits; probabilities = softmax(logits / T).
- **Noul**: the question "Does the user want to X?" is parsed back to the description X (checked: equals `intent_descriptions[intent]` for all 2,000 test rows) and fed as the single label → 1 logit; p_yes = sigmoid(a·logit + b). On the calibration split the alternative of feeding the whole question as the label gave AUROC 0.7171 vs 0.7138 (zero-shot trim32k), a negligible difference; the description form was kept so that Choice and Noul share label strings.
- Cross-lingual setting: options are English for it/es/de/fr inputs (limitation of the benchmark, not changed).
- One forward pass per decision (one text + 8 options, or one text + 1 label). A clean re-implementation of the GLiClass inference path (`skejja_glic.Scorer`, number of labels taken from the shape of a dummy `label_slots` input) is used for ONNX export; it matches the gliclass forward to ≤7.2e-05 on 200 Choice + 200 Noul items.

## 4. Vocabulary trimming
Method (`skejja_vocab.py`): the original tokenizer is a Gemma-style BPE (Metaspace pre-tokenizer, byte fallback, 580,604 merges). Seed set = every token produced on MASSIVE-train + description texts, plus the top-N Wikipedia tokens ranked by language-balanced relative frequency. For every corpus word whose final tokens are all in the seed, the exact BPE derivation was simulated (validated: 0 mismatches vs the HF tokenizer on 5,000 random words) and all its intermediate merge products and base characters were added; merges are kept iff both parts and the result are kept. This guarantees identical tokenization for those words (greedy BPE argmax never changes when only never-chosen merges are removed). All 256 byte-fallback tokens, `<pad>/<eos>/<bos>/<unk>/<mask>/<start_of_turn>/<end_of_turn>` and the three GLiClass tokens are kept (GLiClass tokens moved to the last 3 ids; config indices updated). N was binary-searched to hit exactly 32,768 and 65,536. Embedding rows are gathered from the original matrix (no re-training).

| | original | trim64k | trim32k |
|---|---|---|---|
| vocab size | 256,003 | 65,536 | 32,768 |
| total params | 142,992,642 | 69,853,314 | 57,270,402 |
| embedding params | 98,305,152 | 25,165,824 | 12,582,912 |
| fp32 weights MB | 571.97 | 279.41 | 229.08 |
| safetensors MB on disk (orig = bf16 as shipped) | 286 | 279.43 | 229.1 |
| tokenizer.json MB | 34.364 | 4.112 | 1.891 |
| BPE merges | 580,604 | 162,841 | 73,932 |
| Wikipedia tokens added on top of MASSIVE+descriptions seed | – | 40,228 | 7,676 |
| TEST-text token coverage (orig tokens surviving) | – | 0.992775 | 0.983262 |
| TEST items with unchanged tokenization | – | 0.947 | 0.9015 |
| TEST tokens per text, trimmed/orig | – | 1.00753 | 1.01878 |
| MASSIVE-train utterances identical | – | 1 | 1 |
| Wikipedia sample sentences identical (10k, used in build) | – | 0.6477 | 0.1075 |
| max abs Δlogit vs original, unchanged items (Choice) | – | 1.90735e-06 | 2.20239e-05 |
| max abs Δlogit vs original, unchanged items (Noul) | – | 0 | 9.05991e-06 |
| Choice argmax agreement on changed items | – | 0.858491 | 0.847716 |
| Choice argmax agreement, all items | – | 0.9925 | 0.985 |

Per-language test coverage:

| vocab | metric | en | it | es | de | fr |
|---|---|---|---|---|---|---|
| trim32k | token coverage | 0.9764 | 0.9882 | 0.9797 | 0.9824 | 0.9877 |
| trim32k | items unchanged | 0.8800 | 0.9300 | 0.8800 | 0.9000 | 0.9175 |
| trim64k | token coverage | 0.9873 | 0.9964 | 0.9911 | 0.9927 | 0.9950 |
| trim64k | items unchanged | 0.9200 | 0.9700 | 0.9450 | 0.9450 | 0.9550 |

Verification: (i) the trimmed `tokenizer.json` loaded with the plain `tokenizers` library gives the same ids as `transformers` on all 4,000 test inputs (needed for the torch-free Pi kit); (ii) on test items whose full model input tokenization is unchanged (90.15% / 94.7%), trimmed and original logits agree to ≤2.2e-05 (float noise from batching), i.e. identical outputs; (iii) changed items are ~1–2% more tokens per text and flip the Choice argmax in ~15% of the changed items. Embedding parameters drop from 98.3M to 12.6M (32k) — **the 32k model has 40% of the original parameter count (57.3M vs 143.0M)**.

## 5. Fine-tuning (`train_skejja.py`)
- Data: MASSIVE train rows whose intent is NOT in HELD_OUT_SCENARIOS: 47,645 of 54,795 rows, 49 intents. Held-out intents are never used as options or Noul labels (asserted in code). Sampled (seed 0, without replacement) **20,000 Choice examples and 20,000 Noul examples** = 0.42 epoch-equivalents of the seen utterances each (CPU budget); **1 pass, 1250 optimizer steps, batch 32**, alternating Choice/Noul batches.
- Choice example: gold + up to 3 other same-scenario seen intents + random other seen intents to 8, shuffled. Noul: 50% gold description (positive; realised 0.5009), else 50% hard negative from the same scenario (if any) / random seen intent.
- Model stays label-conditioned (GLiClass cross-attention scorer over arbitrary option texts; no fixed-class head).
- Loss: Choice = CE over 8 logits + 0.5·BCE(one-hot) + 0.2·KL(teacher‖student); Noul = BCE + 0.2·BCE(student, sigmoid(teacher)). **Distillation teacher = the untrimmed zero-shot GLiClass edge** (implemented with the trimmed-32k copy, whose tokenization is identical on 100% of MASSIVE-train utterances and descriptions, hence identical outputs); used as a weak anti-forgetting regulariser. Laya multilingual was NOT used as teacher (see deviations).
- Optimiser: AdamW, lr 3e-5 (encoder) / 1e-4 (projectors+scorer), wd 0.01, 6% warmup + linear decay, grad-clip 1.0, gliclass dropout as configured (0.3 on text features, 0.1 scorer attention). torch 2.14.1 CPU, 10 threads: 33.6 min.
- **Smaller encoder** `trim32k-L8-ft`: encoder layers [0,3,6,9,12,15,18,21] kept (exactly the global-attention layers of mmBERT-small; 30.4M params), head copied, then trained on the same 20,000+20,000 examples (same seed/sampling) with gold losses (weight 1) + KD weight 1.0 from `trim32k-ft`; lr 8e-5 / 2e-4, 8 threads, 13.5 min. Without training, every layer-dropped variant (11, 8, 6 layers; alternate or bottom) was at chance on the calibration split (Choice acc 0.076–0.133), so the layer subset was chosen by structure, not by score.
- Calibration: one temperature (Choice) and one Platt (a, b) (Noul) per model×variant, fitted ONLY on `calib_choice.jsonl` / `calib_noul.jsonl` (pooled over languages; the calib split contains all intents incl. held-out ones — only 1–2 scalar parameters are fitted) and applied to test. Values are in the CSV (`split=calibration`) and in each bundle's `calibration.json`.

## 6. ONNX export and quantization
- fp32: legacy TorchScript exporter (`torch.onnx.export(dynamo=False)`, opset 17), dynamic batch/sequence/number-of-labels axes (one graph serves Choice K=8 and Noul K=1). Saved with **external data** (`model.onnx` + `model.onnx.data`): onnxruntime memory-maps external initializers, which cut the session peak RSS roughly in half vs a single-file model (e.g. trim32k-ft int8-dyn 329 → 170 MB in a probe).
- `onnx-int8` = **weight-only 8-bit** `MatMulNBits` (onnxruntime `MatMulNBitsQuantizer`, bits=8, block 32, symmetric, accuracy_level=4, rotary MatMul excluded). Chosen because standard dynamic int8 destroyed the zero-shot model (calib Choice argmax agreement with fp32 61%); a per-module ablation showed the MLP down-projection (`mlp.Wo`, GeGLU output with activation outliers) is the culprit (59.5% agreement when only it is quantized). An accuracy_level=0 variant of the same weights was equally accurate but 2–4× slower than fp32 on this Mac and was dropped.
- `onnx-int8dyn` = `quantize_dynamic` QInt8 per-channel, MatMul with constant B (standard ORT dynamic quantization), reported for comparison. Its per-tensor activation scale depends on the whole batch, so its test quality was computed with **batch size 1** (deployment mode); fp32 / int8 were evaluated in length-bucketed batches of 64 (batch-invariant up to float noise, checked).
- Numerical check vs torch-fp32 on all 2,000+2,000 test items:

| model | variant | Choice max/mean abs Δlogit | Choice argmax agreement | Noul max/mean abs Δlogit | Noul decision agreement |
|---|---|---|---|---|---|
| gliclass-edge-trim32k-zs | onnx-fp32 | 1.61e-04 / 4.94e-06 | 1.0000 | 8.75e-05 / 5.20e-06 | 1.0000 |
| gliclass-edge-trim32k-zs | onnx-int8 | 1.06e+00 / 1.09e-01 | 0.9535 | 7.93e-01 / 1.18e-01 | 0.9870 |
| gliclass-edge-trim32k-zs | onnx-int8dyn | 7.09e+00 / 1.07e+00 | 0.6205 | 5.75e+00 / 1.10e+00 | 0.8765 |
| gliclass-edge-trim32k-ft | onnx-fp32 | 2.91e-05 / 1.61e-06 | 1.0000 | 5.13e-05 / 1.98e-06 | 1.0000 |
| gliclass-edge-trim32k-ft | onnx-int8 | 4.43e-01 / 3.22e-02 | 0.9930 | 7.32e-01 / 3.99e-02 | 0.9950 |
| gliclass-edge-trim32k-ft | onnx-int8dyn | 3.11e+00 / 4.95e-01 | 0.9275 | 4.94e+00 / 5.45e-01 | 0.9370 |
| gliclass-edge-trim32k-L8-ft | onnx-fp32 | 2.78e-05 / 1.67e-06 | 1.0000 | 1.76e-05 / 2.03e-06 | 1.0000 |
| gliclass-edge-trim32k-L8-ft | onnx-int8 | 4.97e-01 / 3.59e-02 | 0.9910 | 5.59e-01 / 4.83e-02 | 0.9930 |
| gliclass-edge-trim32k-L8-ft | onnx-int8dyn | 1.70e+00 / 1.18e-01 | 0.9625 | 2.11e+00 / 1.41e-01 | 0.9780 |

int8 degradation (macro, all split), torch-fp32 → onnx-int8 → onnx-int8dyn: trim32k-ft Choice acc 0.8690 → 0.8725 → 0.8525; Noul AUROC 0.9589 → 0.9593 → 0.9399. Noul suffers more than Choice under dynamic int8 (decision agreement 93.7%); weight-only int8 stays within 0.4 points of torch-fp32 on the macro `all` split; on the small `unseen` split the macro difference reaches 2.1 points (single language up to 3.4 points), i.e. within the sampling noise of 58–60 items per language.

## 7. Quality (test files; macro = mean over en/it/es/de/fr)

| model | variant | Choice acc all | seen | unseen | Choice ECE raw→cal (all) | Noul AUROC all | seen | unseen | Noul ECE raw→cal (all) | worst-lang Choice gap vs en |
|---|---|---|---|---|---|---|---|---|---|---|
| gliclass-edge-orig | torch-fp32 | 0.547 | 0.539 | 0.597 | 0.114→0.060 | 0.698 | 0.693 | 0.737 | 0.270→0.050 | 15.2 pts |
| gliclass-edge-trim64k-zs | torch-fp32 | 0.543 | 0.535 | 0.586 | 0.115→0.058 | 0.694 | 0.691 | 0.730 | 0.270→0.050 | 14.7 pts |
| gliclass-edge-trim32k-zs | torch-fp32 | 0.541 | 0.533 | 0.590 | 0.115→0.057 | 0.695 | 0.691 | 0.729 | 0.268→0.056 | 14.7 pts |
| gliclass-edge-trim32k-zs | onnx-fp32 | 0.541 | 0.533 | 0.590 | 0.115→0.057 | 0.695 | 0.691 | 0.729 | 0.268→0.056 | 14.7 pts |
| gliclass-edge-trim32k-zs | onnx-int8 | 0.538 | 0.532 | 0.579 | 0.120→0.062 | 0.691 | 0.687 | 0.721 | 0.268→0.059 | 15.8 pts |
| gliclass-edge-trim32k-zs | onnx-int8dyn | 0.458 | 0.447 | 0.524 | 0.117→0.066 | 0.643 | 0.643 | 0.669 | 0.261→0.039 | 12.8 pts |
| gliclass-edge-trim32k-ft | torch-fp32 | 0.869 | 0.889 | 0.748 | 0.063→0.045 | 0.959 | 0.981 | 0.714 | 0.043→0.034 | 5.0 pts |
| gliclass-edge-trim32k-ft | onnx-fp32 | 0.869 | 0.889 | 0.748 | 0.063→0.045 | 0.959 | 0.981 | 0.714 | 0.043→0.034 | 5.0 pts |
| gliclass-edge-trim32k-ft | onnx-int8 | 0.873 | 0.890 | 0.769 | 0.061→0.038 | 0.959 | 0.981 | 0.717 | 0.045→0.035 | 4.0 pts |
| gliclass-edge-trim32k-ft | onnx-int8dyn | 0.853 | 0.873 | 0.734 | 0.075→0.045 | 0.940 | 0.967 | 0.697 | 0.046→0.034 | 4.5 pts |
| gliclass-edge-trim32k-L8-ft | torch-fp32 | 0.762 | 0.815 | 0.452 | 0.060→0.051 | 0.909 | 0.940 | 0.705 | 0.053→0.051 | 5.2 pts |
| gliclass-edge-trim32k-L8-ft | onnx-fp32 | 0.762 | 0.815 | 0.452 | 0.060→0.051 | 0.909 | 0.940 | 0.705 | 0.053→0.051 | 5.2 pts |
| gliclass-edge-trim32k-L8-ft | onnx-int8 | 0.764 | 0.815 | 0.466 | 0.057→0.053 | 0.909 | 0.940 | 0.709 | 0.053→0.047 | 5.2 pts |
| gliclass-edge-trim32k-L8-ft | onnx-int8dyn | 0.755 | 0.804 | 0.469 | 0.056→0.050 | 0.907 | 0.937 | 0.711 | 0.056→0.051 | 4.0 pts |

Unseen split (fair comparison with zero-shot baselines):

| model (variant) | Choice acc unseen | Choice ECE_cal unseen | Noul acc unseen | Noul AUROC unseen | Noul ECE_cal unseen | n unseen items (choice / noul) |
|---|---|---|---|---|---|---|
| gliclass-edge-orig (torch-fp32) | 0.597 | 0.154 | 0.540 | 0.737 | 0.075 | 290 / 300 |
| gliclass-edge-trim32k-zs (torch-fp32) | 0.590 | 0.149 | 0.537 | 0.729 | 0.109 | 290 / 300 |
| gliclass-edge-trim32k-ft (torch-fp32) | 0.748 | 0.170 | 0.630 | 0.714 | 0.242 | 290 / 300 |
| gliclass-edge-trim32k-ft (onnx-int8) | 0.769 | 0.144 | 0.633 | 0.717 | 0.232 | 290 / 300 |
| gliclass-edge-trim32k-L8-ft (onnx-int8) | 0.466 | 0.208 | 0.603 | 0.709 | 0.190 | 290 / 300 |

Per language (all split):

| model (variant) | metric | en | it | es | de | fr | macro |
|---|---|---|---|---|---|---|---|
| gliclass-edge-trim32k-zs (torch-fp32) | choice accuracy | 0.642 | 0.495 | 0.550 | 0.502 | 0.517 | 0.541 |
| gliclass-edge-trim32k-zs (torch-fp32) | noul auroc | 0.751 | 0.666 | 0.702 | 0.651 | 0.704 | 0.695 |
| gliclass-edge-trim32k-ft (onnx-int8) | choice accuracy | 0.892 | 0.875 | 0.870 | 0.853 | 0.873 | 0.873 |
| gliclass-edge-trim32k-ft (onnx-int8) | noul auroc | 0.970 | 0.953 | 0.962 | 0.951 | 0.960 | 0.959 |
| gliclass-edge-trim32k-L8-ft (onnx-int8) | choice accuracy | 0.787 | 0.755 | 0.757 | 0.735 | 0.785 | 0.764 |
| gliclass-edge-trim32k-L8-ft (onnx-int8) | noul auroc | 0.922 | 0.899 | 0.909 | 0.887 | 0.930 | 0.909 |

Observations: fine-tuning on the 49 seen intents raised unseen Choice accuracy (0.590 → 0.748 torch-fp32, trim32k) but did not help unseen Noul (AUROC 0.729 → 0.714) and left unseen Noul badly calibrated (ECE_cal 0.24 with a single global Platt fit dominated by seen items). The 8-layer student keeps seen performance reasonably (Choice 0.815) but loses unseen generalisation (Choice 0.452 torch-fp32, below the zero-shot model's 0.590).

## 8. Latency and memory protocol
- Each measurement runs in a fresh child process (`skejja_bench_child.py`), sequentially: `skejja_measure.latency` with 10 warm-up + **300 timed calls**, one decision per call (tokenization with the `tokenizers` library + one forward; Choice = text + its 8 options, Noul = one question), batch 1, no padding. Inputs: 330 test items sampled with seed 123 (mixed languages). torch: `torch.set_num_threads(t)`, `torch.set_num_interop_threads(1)`, `inference_mode`. ORT: `intra_op_num_threads=t`, `inter_op_num_threads=1`, sequential execution, ORT_ENABLE_ALL, CPUExecutionProvider. t ∈ {1, 4}.
- `peak_rss_mb` = `skejja_measure.peak_rss_mb` on a fresh child that loads tokenizer + model and runs 20 decisions (10 Choice + 10 Noul) at 4 threads, child importing `skejja_measure` (as the shared protocol requires — this pulls scipy + sklearn, ~+150 MB). `peak_rss_mb_lean` = same child and same ru_maxrss formula but without importing skejja_measure/scipy/sklearn (what a deployed runtime would load: numpy + tokenizers + onnxruntime only). macOS `ru_maxrss` (bytes) is converted as in skejja_measure. Loading the 1.9 MB trimmed tokenizer.json alone costs ~50 MB RSS in the `tokenizers` library.
- `weights_mb` = on-disk size (safetensors fp32 for torch; .onnx + .onnx.data for ORT).
- Caveat: another workload was running on the same Mac during the final benchmark window (2026-10-04T11:00:26Z – 2026-10-04T11:13:41Z): 1.7–5.3 busy cores sampled per minute (10-core machine). An earlier full benchmark pass (single-file ONNX, under heavier contention) was superseded and is not reported. torch at 4 threads shows no speed-up over 1 thread for these batch-1, ~50-token inputs.

| model | variant | weights MB | peak RSS MB (protocol child) | peak RSS MB (lean child) | Choice p50 ms t1 | t4 | Noul p50 ms t1 | t4 | Choice p95 ms t4 |
|---|---|---|---|---|---|---|---|---|---|
| gliclass-edge-trim32k-ft | torch-fp32 | 229.1 | 759 | 764 | 25.8 | 26.8 | 24.3 | 27.4 | 33.3 |
| gliclass-edge-trim32k-ft | onnx-fp32 | 229.6 | 445 | 306 | 57.7 | 20.8 | 21.3 | 9.4 | 26.8 |
| gliclass-edge-trim32k-ft | onnx-int8 | 101.2 | 350 | 190 | 32.6 | 12.8 | 10.8 | 5.5 | 16.0 |
| gliclass-edge-trim32k-ft | onnx-int8dyn | 96.5 | 331 | 176 | 24.1 | 11.1 | 8.6 | 5.0 | 13.2 |
| gliclass-edge-trim32k-L8-ft | torch-fp32 | 121.7 | 634 | 626 | 10.4 | 11.5 | 9.7 | 11.0 | 13.9 |
| gliclass-edge-trim32k-L8-ft | onnx-fp32 | 121.9 | 338 | 193 | 21.9 | 8.0 | 8.1 | 3.6 | 9.8 |
| gliclass-edge-trim32k-L8-ft | onnx-int8 | 70.7 | 302 | 138 | 12.4 | 4.9 | 4.2 | 2.1 | 7.4 |
| gliclass-edge-trim32k-L8-ft | onnx-int8dyn | 68.8 | 283 | 140 | 9.2 | 4.2 | 3.3 | 1.9 | 4.9 |
| gliclass-edge-trim32k-zs | torch-fp32 | 229.1 | 727 | 733 | 25.5 | 28.1 | 23.4 | 27.5 | 33.6 |
| gliclass-edge-trim32k-zs | onnx-fp32 | 229.6 | 462 | 306 | 58.7 | 20.8 | 21.3 | 9.2 | 27.3 |
| gliclass-edge-trim32k-zs | onnx-int8 | 101.2 | 345 | 190 | 32.5 | 12.8 | 10.8 | 5.1 | 15.6 |
| gliclass-edge-trim32k-zs | onnx-int8dyn | 96.5 | 322 | 170 | 24.3 | 11.0 | 8.6 | 4.7 | 13.5 |
| gliclass-edge-trim64k-zs | torch-fp32 | 279.4 | 812 | 796 | 26.1 | 27.6 | 24.8 | 27.2 | 32.3 |
| gliclass-edge-trim64k-zs | onnx-fp32 | 280.0 | 506 | 358 | 57.5 | 20.9 | 21.1 | 9.3 | 24.3 |
| gliclass-edge-trim64k-zs | onnx-int8 | 151.6 | 409 | 250 | 32.7 | 12.8 | 10.7 | 5.5 | 15.2 |
| gliclass-edge-trim64k-zs | onnx-int8dyn | 146.8 | 390 | 242 | 24.1 | 11.0 | 8.6 | 4.9 | 13.3 |
| gliclass-edge-orig | torch-fp32 | 286.0 | 1685 | 1679 | 24.4 | 25.0 | 15.5 | 23.6 | 30.0 |
| gliclass-edge-orig | onnx-fp32 | 572.5 | 796 | 649 | 58.4 | 20.8 | 21.0 | 9.1 | 24.8 |
| gliclass-edge-orig | onnx-int8 | 444.1 | 652 | 565 | 32.6 | 12.8 | 10.9 | 5.2 | 15.2 |
| gliclass-edge-orig | onnx-int8dyn | 439.4 | 640 | 501 | 24.2 | 11.1 | 8.5 | 4.9 | 13.7 |

## 9. Raspberry kit bundles (tar.gz)
Each bundle: ONNX model(s) with external `.data`, `tokenizer.json` (+config/special tokens), `calibration.json` (per ONNX file, fitted on calib only), `skejja_infer.py` (onnxruntime + tokenizers + numpy only; verified to run without torch/transformers and to reproduce the torch-fp32 logits of the L8 model with the int8 file (max abs logit difference 0.121 on a test item, same argmax)). Keep `x.onnx` and `x.onnx.data` in the same folder.

## 10. Deviations, failures, limitations
- Teacher for distillation = the zero-shot GLiClass edge (untrimmed-equivalent), not Laya multilingual (the plan said "best baseline"): Laya would have required an unfamiliar API and a 322M-parameter model on CPU within the budget. Distillation weight 0.2 for the 22-layer model; the 8-layer model is distilled from our fine-tuned model.
- Training budget: 20,000 Choice + 20,000 Noul examples (0.42 epoch-equivalents of the 47,645 seen utterances), one run per model, no hyper-parameter search, no seeds repetition.
- `datasets` streaming failed in the sandbox (`torch_shm_manager: Operation not permitted`); Wikipedia was read directly as parquet row groups through `HfFileSystem`. Italian contributed 22,256 sentences instead of 30,000.
- Standard ORT dynamic int8 is reported but not recommended (see §6); the recommended int8 is weight-only MatMulNBits (accuracy_level 4). Its speed on Cortex-A53 (Pi 3) / A72 (Pi 4), which lack the ARMv8.2 dot-product instructions of Apple silicon, is unknown and must be measured on device; the int8dyn file is included in the int8 bundles as a fallback.
- Embedding tables stay fp32 in all int8 files (ORT did not quantize the Gather); ~50 MB of the 101 MB trim32k int8 file is the embedding table.
- The v64k vocabulary was evaluated zero-shot only (no fine-tune); its ONNX int8/int8dyn quality was not evaluated (only latency/RSS/size).
- Raspberry Pi 3 running a 32-bit OS (armv7l) has no official onnxruntime wheel; a 64-bit OS is assumed.
