#!/usr/bin/env python3
"""Skejja phase-1 benchmark for Raspberry Pi (and any CPU box).

Dependencies: onnxruntime, tokenizers, numpy. No torch, no scipy, no sklearn.
Each (model, threads) runs in a FRESH child process so peak RSS is measured per model.

    python3 bench_pi.py                      # default: all models that fit in RAM, threads 1 and 4
    python3 bench_pi.py --models ft32k-int8  # one model only
    python3 bench_pi.py --quality-n 2000     # full test set (slow on a Pi 3)
    python3 bench_pi.py --list               # show models in the kit
    python3 bench_pi.py --cost --models ft6-int8dyn --threads 1,4   # cost grid: tokens x options (giro G)

Writes results/<host>_<timestamp>.json and .csv. Send the .json file back for the report.
On a Raspberry Pi, `vcgencmd get_throttled` is logged for the whole run (start/end of each model and every
--throttle-every seconds): the run is marked "throttled": true if under-voltage or throttling was active at any sample.
"""
import argparse, json, os, platform, re, resource, socket, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
Q_RE = re.compile(r"^Does the user want to (.*)\?$")
METRIC_COLUMNS = ["track", "model", "variant", "runtime", "threads", "task", "split", "lang", "metric", "value", "n"]
THROTTLED_NOW = 0x1 | 0x4  # get_throttled bits: under-voltage now, throttled now


# ---------------------------------------------------------------- system info
def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return default


def meminfo_mb():
    out = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts and parts[0].isdigit():
            out[k] = int(parts[0]) / 1024
    return out


def sysinfo():
    import numpy, onnxruntime, tokenizers
    cpu = _read("/proc/cpuinfo")
    feats = sorted(set(" ".join(l.split(":", 1)[1] for l in cpu.splitlines() if l.startswith("Features")).split()))
    mem = meminfo_mb()
    thr = ""
    try:
        thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    temp = _read("/sys/class/thermal/thermal_zone0/temp").strip()
    osrel = dict(l.split("=", 1) for l in _read("/etc/os-release").splitlines() if "=" in l)
    return {
        "device_model": _read("/proc/device-tree/model").strip("\x00\n ") or platform.platform(),
        "machine": platform.machine(), "system": platform.system(), "os": osrel.get("PRETTY_NAME", "").strip('"'),
        "python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
        "tokenizers": tokenizers.__version__, "numpy": numpy.__version__, "cpu_count": os.cpu_count(),
        "cpu_features": feats, "has_dotprod": "asimddp" in feats,
        "mem_total_mb": round(mem.get("MemTotal", 0)), "mem_available_mb": round(mem.get("MemAvailable", 0)),
        "swap_total_mb": round(mem.get("SwapTotal", 0)),
        "cpu_temp_c": round(int(temp) / 1000, 1) if temp.isdigit() else None, "throttled": thr,
    }


def throttled_hex():
    """`vcgencmd get_throttled` as '0x...', or None where vcgencmd is missing (not a Raspberry Pi)."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    m = re.search(r"0x[0-9a-fA-F]+", out)
    return m.group(0) if m else None


class ThrottleLog:
    """get_throttled samples: explicit marks (start/end of each model) plus one every `every` s from a thread."""
    def __init__(self, every):
        self.t0, self.samples, self.lock, self.stop_ev = time.time(), [], threading.Lock(), threading.Event()
        self.enabled = throttled_hex() is not None
        if self.enabled:
            self.thread = threading.Thread(target=self._loop, args=(every,), daemon=True)
            self.thread.start()

    def _loop(self, every):
        while not self.stop_ev.wait(every):
            self.mark("sample")

    def mark(self, event, **kw):
        v = throttled_hex() if self.enabled else None
        if v is not None:
            with self.lock:
                self.samples.append({"t_s": round(time.time() - self.t0, 1), "event": event, **kw, "value": v})

    def stop(self):
        if self.enabled:
            self.stop_ev.set(); self.thread.join(timeout=10)
        self.mark("end")
        bad = [s for s in self.samples if int(s["value"], 16) & THROTTLED_NOW]
        return {"throttled": bool(bad) if self.samples else None, "throttled_bad_n": len(bad),
                "throttled_samples": self.samples}


# ---------------------------------------------------------------- metrics (numpy only)
def softmax(z):
    import numpy as np
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / e.sum()


def auroc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)  # average ties
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); ranks = (sums / cnt)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def ece(conf, correct, bins=15):
    import numpy as np
    conf = np.asarray(conf, float); correct = np.asarray(correct, float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def pct(xs, q):
    import numpy as np
    return float(np.percentile(np.asarray(xs, float), q))


# ---------------------------------------------------------------- child: one model, one thread count
def run_child(a):
    import numpy as np, onnxruntime as ort, tokenizers
    if hasattr(ort, "disable_telemetry_events"):
        ort.disable_telemetry_events()  # keep everything local
    m = json.load(open(os.path.join(HERE, "models.json")))["models"][a.child]
    mdir = os.path.join(HERE, m["dir"])
    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if a.low_mem:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    sess = ort.InferenceSession(os.path.join(mdir, m["onnx"]), so, providers=["CPUExecutionProvider"])
    tok = tokenizers.Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
    load_s = time.perf_counter() - t0
    ins = {i.name: i for i in sess.get_inputs()}
    slot_dtype = np.int64 if "int64" in ins["label_slots"].type else np.float32
    cal = m.get("calibration", {})
    T = float(cal.get("temperature", 1.0)); pa = float(cal.get("platt_a", 1.0)); pb = float(cal.get("platt_b", 0.0))

    def encode(text, labels):
        s = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + text
        return np.array([tok.encode(s).ids], dtype=np.int64)

    def run(ids, k):
        return sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids),
                                     "label_slots": np.zeros(k, slot_dtype)})[0][0]

    def noul_label(q):
        mm = Q_RE.match(q.strip()); return mm.group(1) if mm else q

    ch = [json.loads(l) for l in open(os.path.join(HERE, "data", "test_choice.jsonl"), encoding="utf-8")]
    no = [json.loads(l) for l in open(os.path.join(HERE, "data", "test_noul.jsonl"), encoding="utf-8")]
    out = {"model": a.child, "threads": a.threads, "load_s": load_s, "low_mem": a.low_mem}

    # ---- cost grid: one (tokens, options) point per child, so peak RSS belongs to that point ----
    if a.cost_point:
        want_tok, want_opt = (int(x) for x in a.cost_point.lower().split("x"))
        labels = []
        for r in ch:                                   # descrizioni vere delle opzioni, senza ripetizioni
            for d in r["option_desc"]:
                if d not in labels:
                    labels.append(d)
            if len(labels) >= want_opt:
                break
        if len(labels) < want_opt:
            labels = (labels * (want_opt // max(len(labels), 1) + 1))[:want_opt]
        labels = labels[:want_opt]
        head = len(tok.encode("".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>").ids)
        out.update(cost_tokens_target=want_tok, cost_options=want_opt, cost_label_tokens=head)
        if head >= want_tok:                            # i criteri da soli non ci stanno: punto non misurabile
            out["status_point"] = "label_overflow"
            print("RESULT=" + json.dumps(out), flush=True)
            return
        body_ids = tok.encode(" ".join(r["text"] for r in ch)).ids   # testo vero del test, concatenato
        text = tok.decode(body_ids[:max(want_tok - head, 1)])
        ids = encode(text, labels)
        for _ in range(2):
            run(ids, want_opt)                          # warmup del punto
        tt, tf = [], []
        t_start = time.perf_counter()
        while len(tt) < a.cost_n and (len(tt) < 5 or time.perf_counter() - t_start < a.cost_budget):
            t1 = time.perf_counter(); ids = encode(text, labels); t2 = time.perf_counter()
            run(ids, want_opt); t3 = time.perf_counter()
            tt.append((t3 - t1) * 1000); tf.append((t3 - t2) * 1000)
        r_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out.update(cost_tokens=int(ids.shape[1]), cost_n=len(tt), status_point="ok",
                   cost_e2e_p50_ms=pct(tt, 50), cost_e2e_p95_ms=pct(tt, 95),
                   cost_fwd_p50_ms=pct(tf, 50), cost_fwd_p95_ms=pct(tf, 95),
                   peak_rss_mb=r_ / 1e6 if sys.platform == "darwin" else r_ / 1024)
        print("RESULT=" + json.dumps(out), flush=True)
        return

    for r in ch[:a.warmup]:  # warmup
        run(encode(r["text"], r["option_desc"]), len(r["option_desc"]))

    # latency: end-to-end (tokenize + forward) and forward-only, one decision per call
    for task, rows in (("choice", ch), ("noul", no)):
        tt, tf, ntok = [], [], []
        for r in rows[:a.latency_n]:
            labels = r["option_desc"] if task == "choice" else [noul_label(r["question"])]
            t1 = time.perf_counter(); ids = encode(r["text"], labels); t2 = time.perf_counter()
            run(ids, len(labels)); t3 = time.perf_counter()
            tt.append((t3 - t1) * 1000); tf.append((t3 - t2) * 1000); ntok.append(ids.shape[1])
        out[f"{task}_e2e_p50_ms"] = pct(tt, 50); out[f"{task}_e2e_p95_ms"] = pct(tt, 95)
        out[f"{task}_fwd_p50_ms"] = pct(tf, 50); out[f"{task}_fwd_p95_ms"] = pct(tf, 95)
        out[f"{task}_tok_median"] = pct(ntok, 50); out[f"{task}_latency_n"] = len(tt)

    # quality on the first quality_n rows (rows come in groups of 5 languages per utterance id)
    if a.quality_n > 0:
        P, Y, L = [], [], []
        for r in ch[:a.quality_n]:
            z = run(encode(r["text"], r["option_desc"]), len(r["option_desc"]))
            P.append(softmax(np.asarray(z, float) / T)); Y.append(r["gold_idx"]); L.append(r["lang"])
        P = np.array(P); Y = np.array(Y); L = np.array(L)
        correct = P.argmax(1) == Y
        out["choice_accuracy"] = float(correct.mean()); out["choice_ece_cal"] = ece(P.max(1), correct)
        out["choice_quality_n"] = int(len(Y))
        out["choice_accuracy_by_lang"] = {l: float(correct[L == l].mean()) for l in sorted(set(L.tolist()))}
        S, Yn = [], []
        for r in no[:a.quality_n]:
            z = float(run(encode(r["text"], [noul_label(r["question"])]), 1)[0])
            S.append(1 / (1 + np.exp(-(pa * z + pb)))); Yn.append(r["label"])
        S = np.array(S); Yn = np.array(Yn); pr = (S >= 0.5).astype(int)
        out["noul_auroc"] = auroc(S, Yn); out["noul_accuracy"] = float((pr == Yn).mean())
        out["noul_ece_cal"] = ece(np.where(pr == 1, S, 1 - S), pr == Yn); out["noul_quality_n"] = int(len(Yn))

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["peak_rss_mb"] = r / 1e6 if sys.platform == "darwin" else r / 1024
    print("RESULT=" + json.dumps(out), flush=True)


# ---------------------------------------------------------------- parent
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="auto", help="comma list of model keys, 'all' or 'auto' (auto = skip if RAM short)")
    p.add_argument("--threads", default="1,4")
    p.add_argument("--latency-n", type=int, default=200)
    p.add_argument("--quality-n", type=int, default=500, help="test rows per task (max 2000)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=int, default=3600, help="seconds per child")
    p.add_argument("--low-mem", action="store_true", help="disable ORT memory arena (less RAM, maybe slower)")
    p.add_argument("--force", action="store_true", help="run models even if RAM looks insufficient")
    p.add_argument("--throttle-every", type=int, default=30, help="seconds between get_throttled samples")
    p.add_argument("--cost", action="store_true",
                   help="banco di costo: latenza e RAM per lunghezza dell'ingresso e numero di opzioni")
    p.add_argument("--cost-grid", default="64x8,256x8,512x8,1024x8,256x20,512x20,1024x20,256x50,512x50,1024x50",
                   help="punti <token>x<opzioni> del banco di costo")
    p.add_argument("--cost-n", type=int, default=15, help="decisioni cronometrate per punto")
    p.add_argument("--cost-budget", type=int, default=180, help="secondi massimi per punto (minimo 5 decisioni)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--cost-point", help=argparse.SUPPRESS)
    a = p.parse_args()
    a.threads_list = [int(x) for x in a.threads.split(",")]
    if a.child:
        a.threads = a.threads_list[0]
        return run_child(a)

    man = json.load(open(os.path.join(HERE, "models.json")))["models"]
    if a.list:
        for k, m in man.items():
            print(f"{k:22s} ~{m['est_rss_mb']:>4} MB  {m['description']}")
        return
    info = sysinfo(); print(json.dumps(info, indent=1))
    if info["machine"] in ("armv7l", "armv6l"):
        print("ATTENZIONE: sistema a 32 bit. Serve Raspberry Pi OS a 64 bit (aarch64)."); return
    keys = list(man) if a.models in ("all", "auto") else a.models.split(",")
    results, csv_rows = [], []
    tl = ThrottleLog(a.throttle_every); tl.mark("start")
    for k in keys:
        m = man[k]
        for th in a.threads_list:
            avail = meminfo_mb().get("MemAvailable", 1e9)
            if not a.force and avail < m["est_rss_mb"] * 1.15:
                print(f"SKIP {k} t={th}: MemAvailable {avail:.0f} MB < ~{m['est_rss_mb'] * 1.15:.0f} MB stimati (usa --force)")
                results.append({"model": k, "threads": th, "status": "skipped_ram", "mem_available_mb": avail}); continue
            if a.cost:
                base = dict(track="pi", model=m["model"], variant=m["variant"], runtime="onnxruntime", threads=th)
                for point in a.cost_grid.split(","):
                    cmd = [sys.executable, os.path.abspath(__file__), "--child", k, "--threads", str(th),
                           "--cost-point", point, "--cost-n", str(a.cost_n), "--cost-budget", str(a.cost_budget)]
                    cmd += ["--low-mem"] if a.low_mem else []
                    print(f"COST {k} threads={th} {point} ...", flush=True)
                    tl.mark("cost_start", model=k, threads=th, point=point)
                    try:
                        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        pr = None
                    tl.mark("cost_end", model=k, threads=th, point=point)
                    lines = [l for l in pr.stdout.splitlines() if l.startswith("RESULT=")] if pr else []
                    if not lines:
                        st = "timeout" if pr is None else (f"killed_signal_{-pr.returncode} (probabile memoria "
                                                          "esaurita)" if pr.returncode < 0 else "error")
                        results.append({"model": k, "threads": th, "cost_point": point, "status": st,
                                        "stderr_tail": (pr.stderr[-1500:] if pr else "")})
                        print(f"     {st}"); continue
                    r = json.loads(lines[-1][7:]); r["status"] = "ok"; r["cost_point"] = point
                    results.append(r)
                    if r.get("status_point") != "ok":
                        print(f"     {r.get('status_point')}: i criteri da soli superano il budget di token"); continue
                    print(f"     {r['cost_tokens']} token, {r['cost_options']} opzioni: p50 {r['cost_e2e_p50_ms']:.0f} ms "
                          f"(p95 {r['cost_e2e_p95_ms']:.0f}) | RSS {r['peak_rss_mb']:.0f} MB | n={r['cost_n']}", flush=True)
                    split = f"{r['cost_tokens']}tok_{r['cost_options']}opt"
                    for mt in ("e2e_p50_ms", "e2e_p95_ms", "fwd_p50_ms", "fwd_p95_ms"):
                        csv_rows.append({**base, "task": "cost", "split": split, "lang": "all",
                                         "metric": mt, "value": r[f"cost_{mt}"], "n": r["cost_n"]})
                    for mt in ("peak_rss_mb", "load_s"):
                        csv_rows.append({**base, "task": "cost", "split": split, "lang": "all", "metric": mt,
                                         "value": r[mt], "n": 1})
                continue
            qn = a.quality_n if th == a.threads_list[-1] else 0   # quality once per model (last thread setting)
            cmd = [sys.executable, os.path.abspath(__file__), "--child", k, "--threads", str(th),
                   "--latency-n", str(a.latency_n), "--quality-n", str(qn), "--warmup", str(a.warmup)]
            cmd += ["--low-mem"] if a.low_mem else []
            print(f"RUN  {k} threads={th} ...", flush=True)
            t0 = time.time()
            tl.mark("model_start", model=k, threads=th)
            try:
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                pr = None
            tl.mark("model_end", model=k, threads=th)
            if pr is None:
                results.append({"model": k, "threads": th, "status": "timeout"}); print("     timeout"); continue
            lines = [l for l in pr.stdout.splitlines() if l.startswith("RESULT=")]
            if pr.returncode != 0 or not lines:
                st = f"killed_signal_{-pr.returncode} (probabile memoria esaurita)" if pr.returncode < 0 else "error"
                results.append({"model": k, "threads": th, "status": st, "stderr_tail": pr.stderr[-1500:]})
                print(f"     {st}\n{pr.stderr[-600:]}"); continue
            r = json.loads(lines[-1][7:]); r["status"] = "ok"; r["wall_s"] = time.time() - t0; results.append(r)
            print(f"     choice p50 {r['choice_e2e_p50_ms']:.1f} ms (p95 {r['choice_e2e_p95_ms']:.1f}) | "
                  f"noul p50 {r['noul_e2e_p50_ms']:.1f} ms | RSS {r['peak_rss_mb']:.0f} MB | load {r['load_s']:.1f} s"
                  + (f" | acc {r['choice_accuracy']:.3f} AUROC {r['noul_auroc']:.3f}" if "choice_accuracy" in r else ""),
                  flush=True)
            base = dict(track="pi", model=m["model"], variant=m["variant"], runtime="onnxruntime", threads=th)
            for t in ("choice", "noul"):
                for mt in ("e2e_p50_ms", "e2e_p95_ms", "fwd_p50_ms", "fwd_p95_ms"):
                    csv_rows.append({**base, "task": t, "split": "latency", "lang": "all", "metric": mt,
                                     "value": r[f"{t}_{mt}"], "n": r[f"{t}_latency_n"]})
            for mt in ("peak_rss_mb", "load_s"):
                csv_rows.append({**base, "task": "all", "split": "memory", "lang": "all", "metric": mt, "value": r[mt], "n": 1})
            if "choice_accuracy" in r:
                for mt in ("accuracy", "ece_cal"):
                    csv_rows.append({**base, "task": "choice", "split": "subset", "lang": "all", "metric": mt,
                                     "value": r[f"choice_{mt}"], "n": r["choice_quality_n"]})
                for mt in ("auroc", "accuracy", "ece_cal"):
                    csv_rows.append({**base, "task": "noul", "split": "subset", "lang": "all", "metric": mt,
                                     "value": r[f"noul_{mt}"], "n": r["noul_quality_n"]})
    thr = tl.stop()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    host = re.sub(r"[^A-Za-z0-9]+", "-", socket.gethostname())
    stem = os.path.join(HERE, "results", f"{host}_{time.strftime('%Y%m%d-%H%M%S')}")
    json.dump({"system": info, "args": {k: v for k, v in vars(a).items()}, **thr, "results": results},
              open(stem + ".json", "w"), indent=1)
    with open(stem + ".csv", "w") as f:
        f.write(",".join(METRIC_COLUMNS) + "\n")
        for row in csv_rows:
            f.write(",".join(str(row[c]) for c in METRIC_COLUMNS) + "\n")
    print(f"\nRisultati salvati in:\n  {stem}.json\n  {stem}.csv\nInvia il file .json per il report.")
    n = len(thr["throttled_samples"])
    if thr["throttled"]:
        print(f"\nATTENZIONE: throttled = true. In {thr['throttled_bad_n']} campioni su {n} get_throttled segnava "
              "sottotensione o CPU rallentata in corso (bit 0x1 o 0x4): le latenze di questo giro non sono affidabili.")
    elif thr["throttled"] is False:
        print(f"\nthrottled = false: nessuna sottotensione né rallentamento in {n} campioni di get_throttled.")


if __name__ == "__main__":
    main()
