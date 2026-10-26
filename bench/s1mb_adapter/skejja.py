"""Skejja: small open System One model (GLiClass uni-encoder, ONNX bundle, CPU only).

Skejja's native input is a single sequence `<<LABEL>>option…<<SEP>>text`: it has no slot for instructions,
documents or a system prompt. The mapping below is fixed in advance and declared in `settings`, so the
measurement reports what the shipped model actually does:

- Choice: the authored criterion descriptions become the labels, in the stored order; option IDs never reach the
  model. Probabilities are the bundle's calibrated softmax over those labels.
- Noul: the authored "true" definition becomes the single label; the probability is the bundle's calibrated
  Platt output, which is what `skejja_infer.Skejja.noul` returns.
- Score: the authored level descriptions become the labels, in the STORED order (never re-sorted), and the
  probability mass over them is combined with the levels' own `value` fields into the expected value that the
  benchmark scores. This is `Skejja.score`, added in giro G: a Score is a Choice over an ordinal scale, so the
  ONNX graph is unchanged.
- Fields with no slot in Skejja (system prompt, instructions) are placed in the text, before the state, because
  that is the only place they can go. Nothing is dropped.

Training length (`training_max_len` in `settings`) is lower than the 8192 positions the encoder accepts, so long
states are shortened head/tail with an explicit marker and the counts are written to `--skejja-stats`.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction, TrainingExposure

from .base import questions_for_api

ELLIPSIS = "\n[…]\n"
# Sorgenti di addestramento di Skejja (fase 1 e 2) che coincidono con una famiglia di sorgenti di S1MB.
EXPOSURE = [
    ("massive-choice-test-v1", "AmazonScience/massive", "https://huggingface.co/datasets/AmazonScience/massive",
     "Skejja is trained on the MASSIVE train split (4 scenarios held out). The training manifest is not public yet."),
    ("clinc-choice-test-v1", "clinc/clinc_oos", "https://huggingface.co/datasets/clinc/clinc_oos",
     "Skejja phase 2 uses CLINC150 train utterances as Noul/Choice sources, minus the excluded domains."),
    ("banking77-choice-test-v1", "PolyAI/banking77", "https://huggingface.co/datasets/PolyAI/banking77",
     "Skejja phase 2 uses the Banking77 train split; its results on Banking77 are in-domain, not zero-shot."),
]


class SkejjaAdapter:
    def __init__(self, model: str, revision: str = "main", runtime: str | None = None,
                 model_file: str | None = None, max_len: int = 4096, threads: int = 4,
                 stats_path: str | None = None, model_id: str | None = None,
                 backend: str = "onnx", ckpt: str | None = None, training_dir: str | None = None,
                 device: str = "cpu", training_max_len: int = 192):
        if max_len < 128:
            raise ValueError("max_len must leave room for labels and text")
        if backend not in ("onnx", "torch"):
            raise ValueError("backend must be onnx or torch")
        runtime = runtime or os.environ.get("SKEJJA_RUNTIME", "")
        if not runtime or not Path(runtime, "skejja_infer.py").is_file():
            raise ValueError("Pass --skejja-runtime <dir with skejja_infer.py> (or set SKEJJA_RUNTIME)")
        sys.path.insert(0, str(Path(runtime).resolve()))
        import skejja_infer

        bundle = Path(model).expanduser()
        if not bundle.is_dir():
            raise ValueError(f"Skejja bundle directory not found: {bundle}")
        self.backend, self.device = backend, device
        if backend == "onnx":
            self.sk = skejja_infer.Skejja(str(bundle), threads=threads, model_file=model_file, int8dyn="no")
            digest = hashlib.sha256(Path(bundle, self.sk.model_file).read_bytes()).hexdigest()
            self.revision = f"{self.sk.model_file}+sha256:{digest[:16]}"
        else:
            self.sk = self._torch_backend(skejja_infer, bundle, ckpt, training_dir, model_file, device)
            digest = hashlib.sha256(Path(ckpt, "model.safetensors").read_bytes()).hexdigest()
            self.revision = f"model.safetensors+sha256:{digest[:16]}"
        self.max_len = max_len
        self.training_max_len = training_max_len
        self.threads = threads
        self.stats_path = stats_path
        self.declared_revision = revision
        self.model_id = model_id or f"skejja/{bundle.name}"
        self.stats = dict(questions=0, shortened=0, label_overflow=0, tokens_max=0, tokens_sum=0,
                          options_max=0, over_training_length=0)

    @staticmethod
    def _torch_backend(skejja_infer, bundle, ckpt, training_dir, model_file, device):
        """Same weights, tokenizer and calibration as the ONNX fp32 bundle, run by PyTorch fp32.

        The ONNX fp32 file was exported from this checkpoint through `skejja_glic.Scorer`; the runtime's own
        `choice`/`noul` code (temperature, Platt) is reused unchanged, only `logits` is replaced. TF32 is off so
        CUDA matmuls stay fp32.
        """
        if not ckpt or not Path(ckpt, "model.safetensors").is_file():
            raise ValueError("backend=torch needs --skejja-ckpt <checkpoint dir with model.safetensors>")
        if not training_dir or not Path(training_dir, "skejja_glic.py").is_file():
            raise ValueError("backend=torch needs --skejja-training-dir <dir with skejja_glic.py>")
        import numpy as np
        import tokenizers
        import torch

        sys.path.insert(0, str(Path(training_dir).resolve()))
        import skejja_glic
        from gliclass import GLiClassModel

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        scorer = skejja_glic.Scorer(GLiClassModel.from_pretrained(ckpt, dtype=torch.float32)).to(device).eval()
        calibration = json.loads(Path(bundle, "calibration.json").read_text())
        model_file = model_file or sorted(k for k in calibration if k.endswith(".fp32.onnx"))[0]
        sk = object.__new__(skejja_infer.Skejja)
        sk.model_file = model_file
        sk.tok = tokenizers.Tokenizer.from_file(str(Path(bundle, "tokenizer.json")))
        sk.cal = calibration[model_file]
        zeros = {}

        def logits(text, labels):
            s = "".join(f"<<LABEL>>{label}" for label in labels) + "<<SEP>>" + text
            ids = torch.tensor([sk.tok.encode(s).ids], dtype=torch.long, device=device)
            k = len(labels)
            if k not in zeros:
                zeros[k] = torch.zeros(k, device=device)
            with torch.inference_mode():
                z = scorer(ids, torch.ones_like(ids), zeros[k])
            return z[0].float().cpu().numpy().astype(np.float32)

        sk.logits = logits
        return sk

    # ---------- rendering, fixed in advance ----------
    def state_text(self, case: InferenceCase) -> str:
        if isinstance(case.state, str):
            return case.state
        return json.dumps(case.state, ensure_ascii=False, sort_keys=False)

    def tokens(self, text: str) -> int:
        return len(self.sk.tok.encode(text).ids)

    def shorten(self, text: str, budget: int) -> str:
        """Head 60% / tail 40% of the allowed tokens, with the harness's own ellipsis marker."""
        ids = self.sk.tok.encode(text).ids
        if len(ids) <= budget or budget < 8:
            return text
        head, tail = (budget * 3) // 5, budget - (budget * 3) // 5
        return self.sk.tok.decode(ids[:head]) + ELLIPSIS + self.sk.tok.decode(ids[len(ids) - tail:])

    def render(self, case: InferenceCase, question) -> tuple[str, list[str]]:
        native = questions_for_api([question])[question.id]
        if question.task == "noul":
            true_option = next(o for o in question.options if o.id == "true")
            labels = [true_option.description]
        elif question.task in ("choice", "score"):
            # L'ordine memorizzato non si tocca: per le Score i valori dei livelli viaggiano con le opzioni.
            labels = [o.description for o in question.options]
        else:
            raise NotImplementedError(f"Skejja has no {question.task} primitive")
        head = f"{native['instructions']}\n\n" if native["instructions"] else ""
        label_tokens = self.tokens("".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>")
        if label_tokens > self.max_len - 32:
            self.stats["label_overflow"] += 1
            raise ValueError("Skejja: criteria alone exceed the declared token budget")
        body = head + self.state_text(case)
        budget = self.max_len - label_tokens
        shortened = self.shorten(body, budget)
        if shortened is not body:
            self.stats["shortened"] += 1
        total = label_tokens + self.tokens(shortened)
        self.stats["questions"] += 1
        self.stats["tokens_sum"] += total
        self.stats["tokens_max"] = max(self.stats["tokens_max"], total)
        self.stats["options_max"] = max(self.stats["options_max"], len(labels))
        self.stats["over_training_length"] += int(total > self.training_max_len)
        return shortened, labels

    # ---------- inference ----------
    def predict(self, case: InferenceCase) -> list[Prediction]:
        out = []
        for q in case.questions:
            text, labels = self.render(case, q)
            if q.task == "noul":
                p = float(self.sk.noul(text, labels[0]))
                probabilities = {"false": 1.0 - p, "true": p}
            elif q.task == "score":
                scale = [float(o.value) for o in q.options]
                _, mass = self.sk.score(text, labels, scale)
                total = float(sum(mass))
                probabilities = {o.id: float(v) / total for o, v in zip(q.options, mass, strict=True)}
            else:
                values = [float(v) for v in self.sk.choice(text, labels)]
                total = sum(values)
                probabilities = {o.id: v / total for o, v in zip(q.options, values, strict=True)}
            out.append(Prediction(case_id=case.case_id, question_id=q.id, probabilities=probabilities))
        return out

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            adapter="skejja",
            revision=self.revision,
            total_params=57_270_402,
            active_params=44_687_490,
            parameter_count_method="non_lookup_parameters_v1",
            training_exposure=[
                TrainingExposure(benchmark_id=b, relationship="same_source_family", dataset=d,
                                 split="train", evidence_url=u, note=n)
                for b, d, u, n in EXPOSURE
            ],
            settings={
                "runtime": ("onnxruntime CPU, no torch" if self.backend == "onnx" else
                            "PyTorch fp32 (TF32 off) on " + self.device + ": the checkpoint the ONNX fp32 bundle "
                            "was exported from, with the bundle's tokenizer and fp32 calibration"),
                "backend": self.backend,
                "onnx_file": self.sk.model_file,
                "declared_bundle_revision": self.declared_revision,
                "threads": self.threads,
                "max_len": self.max_len,
                "training_max_len": self.training_max_len,
                "encoder_max_positions": 8192,
                "bundle_tokenizer_truncation": 4096,
                "max_len_rationale": "the shipped bundle's tokenizer truncates at 4096 tokens; the budget is "
                                     "that limit, not a constant chosen for this benchmark",
                "renderer": "instructions-then-state-in-text, criteria-as-labels-v1",
                "state_renderer": "raw-string-or-compact-json-v1",
                "input_length_policy": "shorten-state-head60-tail40-with-marker; reject if criteria alone "
                                       "exceed max_len - 32",
                "choice_probabilities": "bundle temperature softmax over criterion descriptions (fitted on "
                                        "calib_choice.jsonl only)",
                "noul_probabilities": "bundle global Platt on the single-label logit (fitted on calib_noul*.jsonl "
                                      "only); no per-question few-shot calibration",
                "score_support": "supported from giro G: Choice over the authored level descriptions in stored "
                                 "order, expected value = sum(p * value) from the levels' own `value` fields",
                "score_probabilities": "bundle temperature softmax over level descriptions ("
                                       "`temperature_score` when the bundle declares one, otherwise the Choice "
                                       "temperature; fitted on calib_* only)",
                "calibration": self.sk.cal,
                "batching": "none, one decision per forward pass",
            },
        )

    def close(self) -> None:
        if self.stats_path:
            stats = dict(self.stats)
            if stats["questions"]:
                stats["tokens_mean"] = round(stats["tokens_sum"] / stats["questions"], 1)
            Path(self.stats_path).parent.mkdir(parents=True, exist_ok=True)
            previous = {}
            if Path(self.stats_path).is_file():
                previous = json.loads(Path(self.stats_path).read_text())
            for key, value in stats.items():
                if key.endswith("_max"):
                    previous[key] = max(previous.get(key, 0), value)
                elif key != "tokens_mean":
                    previous[key] = previous.get(key, 0) + value
            if previous.get("questions"):
                previous["tokens_mean"] = round(previous["tokens_sum"] / previous["questions"], 1)
            Path(self.stats_path).write_text(json.dumps(previous, indent=1, ensure_ascii=False))
        del self.sk
