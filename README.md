# Skejja

**Small, local, multilingual typed decisions for AI agents, fast enough for a Raspberry Pi.**

Skejja is a *System One* decision model: it does not generate text. It reads a state (text or JSON) and one or
more typed questions, and returns probabilities in a single forward pass:

- **Choice**: a probability distribution over closed options;
- **Noul**: the probability of a yes/no answer;
- **Score**: a position on an ordinal scale (a Choice over the levels, with an expected value).

It has 57M parameters, runs on onnxruntime without PyTorch, works in English, Italian, Spanish, German and French,
and speaks the same `POST /v1/systemone` API as cloud System One services, so their clients work by changing only
the URL. The name comes from Italian *scheggia*, a splinter: small and fast.

> **Status: v0.1.0.** Model on [Hugging Face](https://huggingface.co/vincyber/skejja-edge32k-ft6),
> **[try it in your browser](https://huggingface.co/spaces/vincyber/skejja)**. Not yet on PyPI: install from this
> repository (see [Install](#install)).

## At a glance

| | edge32k-ft6 (int8) | Where it was measured |
|---|---|---|
| Choice accuracy, 8 options, 5 languages | **0.873** (intents never seen in training: 0.741) | MASSIVE benchmark, PyTorch fp32 |
| Yes/no AUROC | **0.961** | same |
| Choice latency, 1 / 4 threads | **380 / 182 ms** | Raspberry Pi 4, dynamic int8 |
| Yes/no latency, 1 / 4 threads | **126 / 81 ms** | same |
| Peak RAM | **216–227 MB** | same |
| Choice latency, 1 / 4 threads | 34.5 / 14.8 ms | Mac (Apple Silicon), an approximation only |

Sources: [metrics_edge32k-ft6.csv](results/fase2/metrics_edge32k-ft6.csv),
[pi4_20261014-185806.json](results/pi4/pi4_20261014-185806.json),
[latenza_julia-1_ft6_mac.json](results/fase3/latenza_julia-1_ft6_mac.json). The Pi 4 had no heatsink and
throttled, so its latencies are upper bounds. No Raspberry Pi 5 numbers yet.

## Install

```sh
pip install "skejja[server] @ git+https://github.com/vincyberx/skejja"   # until it is on PyPI
skejja --version
```

The model is an archive of about 208 MB. Without a path, `skejja` downloads the default model
(`edge32k-ft6-int8`) on first use, checks its SHA-256 and caches it (`$SKEJJA_HOME`, else `~/.cache/skejja`).
To use an archive you downloaded yourself (release page or Hugging Face), pass its folder:

```sh
tar xzf skejja-edge32k-ft6_onnx-int8.tar.gz
skejja serve skejja-edge32k-ft6_onnx-int8 --port 8080
```

## Use

HTTP server (listens on 127.0.0.1 by default; set `SKEJJA_API_KEY` before exposing it):

```sh
skejja serve --port 8080
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "puoi spegnere le luci della cucina?",
  "questions": {
    "action": {"type": "choice", "criteria": {
        "lights_on": "turn on the lights", "lights_off": "turn off the lights", "music": "play music"}},
    "off": {"type": "noul", "instructions": "Does the user want to turn off the lights?"}}}'
```

Python, without a server:

```python
from skejja import Skejja, SystemOne
from skejja.models import resolve_bundle

engine = SystemOne(Skejja(str(resolve_bundle())), name="skejja-edge32k-ft6")
engine.run({"state": "turn the music down", "questions": {"q": {"type": "choice",
            "criteria": {"vol_down": "lower the volume", "vol_up": "raise the volume"}}}})
```

Also available: an MCP server for agents (`skejja mcp`, stdio) with a [skill file](skills/skejja/SKILL.md) that
tells an agent when and how to use it, a one-shot command (`skejja run request.json`), a [Dockerfile](Dockerfile) and
a [demo that runs in the browser](web/) (onnxruntime-web, nothing sent to a server). The full contract, the extras (per-question calibration, input truncation) and
the differences from cloud services are in [docs/api.md](docs/api.md).

**Tips.** Write options as concrete actions in English; the state can be in any of the five languages. Phrase
yes/no questions as in training, "Does the user want to <concrete action>?": other phrasings can give answers near
0.5. Prefer one yes/no per action to a catch-all "something else" option. For the questions that matter, give a few dozen labelled
examples, in their real proportions, to `POST /v1/skejja/calibrations`: on unseen yes/no questions this takes the
calibration error from 0.150 to 0.041 with 64 examples ([metrics_fewshot.csv](results/fase2/metrics_fewshot.csv)).

## Compared with other open models

Same benchmark and protocol, run by us; the other models are used as released (zero-shot), with only a temperature
or Platt calibration fitted on the calibration files. "Unseen" means the four MASSIVE scenarios held out from
Skejja's training.

| | Parameters | Choice, all / unseen | Yes/no AUROC |
|---|---|---|---|
| **Skejja edge32k-ft6** | **57M** | **0.873 / 0.741** | **0.961** |
| Laya multilingual | 322M | 0.600 / 0.610 | 0.744 |
| GLiClass multilang edge (Skejja's base) | 143M | 0.547 / 0.597 | 0.697 |
| Julia 1 | 144M | 0.514 / 0.514 | 0.503 |

Sources: [metrics_baseline.csv](results/fase1/metrics_baseline.csv) (PyTorch fp32),
[metrics_julia-1.csv](results/fase3/metrics_julia-1.csv), method in [docs/fase3_julia.md](docs/fase3_julia.md).
MASSIVE intents are close to Skejja's training data, so this table shows what Skejja was built for, not general
ability. Julia 1 may have seen MASSIVE during training; its published results are on easier, broader tasks.

## Limitations

- **Short inputs.** Trained on inputs up to 192 tokens. On a real task with long contexts (deciding whether a
  voice assistant should hand a turn to a larger model, inputs over 512 tokens) it reached AUROC 0.79 and was not
  good enough ([real_requests.md](docs/real_requests.md)).
- **Real requests are harder than benchmarks.** On real requests to the author's voice assistant, routing to 5
  destinations reached 0.589 (124 requests with an unambiguous label); yes/no questions on concrete actions (mail,
  web search) reached AUROC about 0.86 (167 requests). Catch-all options and user jargon are the main failure modes (same document).
- **Not a general System One.** Reasoning, entailment between two texts and Choice with many options are near
  chance on S1MB english-v1 ([fase2_ampiezza.md](docs/fase2_ampiezza.md)).
- **New yes/no questions without examples are not calibrated**: AUROC 0.840, ECE 0.150 on unseen intents.
- **Options in the language of the sentence** score 2.4 points below English options (0.849 vs 0.873), German the
  weakest.
- **Noisy unseen split**: about 31% of the MASSIVE labels of the unseen sentences are questionable
  ([bench/data/annotations/](bench/data/annotations/)).

## How it was built

A [GLiClass multilang edge](https://huggingface.co/knowledgator/gliclass-multilang-edge) encoder with its vocabulary
trimmed to 32,768 tokens for five languages, fine-tuned on MASSIVE (four scenarios held out), multilingual yes/no
data, agent-action checks and breadth data labelled by a teacher model, then exported to ONNX and quantized to int8.
Every candidate was accepted or rejected by rules fixed before training (paired bootstrap, Holm correction); two
later rounds were rejected that way ([fase2_giro_G.md](docs/fase2_giro_G.md)). Model card:
[docs/model_card.md](docs/model_card.md).

| Folder | Contents |
|---|---|
| `runtime/` | `skejja_infer.py` (inference without PyTorch) and the `skejja` package: API, server, MCP, CLI |
| `bench/` | benchmark builder (reproduces the data byte for byte), shared metrics, evaluation scripts, data |
| `training/` | vocabulary trimming, fine-tuning, ONNX export |
| `baselines/` | measurements of other models |
| `kit/` | Raspberry Pi benchmark kit |
| `skills/`, `web/` | skill file for agents, browser demo |
| `results/` | every number in the documents comes from a CSV or JSON here |
| `docs/` | phase reports and methods (mostly in Italian) |

## License

Code and model: Apache License 2.0 ([LICENSE](LICENSE)); attributions for the base model and the data in
[NOTICE](NOTICE). Benchmark data derives from [MASSIVE](https://github.com/alexa/massive) (CC BY 4.0).
"Jev", "System One" and "TypeSafe" belong to their owners; Skejja is not affiliated with them and uses none of their
code.
