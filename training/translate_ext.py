"""Traduzione automatica delle frasi esterne inglesi in it/es/de/fr con OPUS-MT (Helsinki-NLP, revisioni fissate).

Traduce: (1) le frasi inglesi di addestramento di bench/data/ext/ext_utterances.jsonl (CLINC, Banking77; MINDS-14 è
già nativo) e (2) le frasi dei domini tenuti fuori scelte per calibrazione e sviluppo (`build_external.py mt_list`).
Decodifica greedy, max 64 token. Risultato in bench/data/ext/mt_cache.jsonl ({src, lang, text, model, revision}):
in git c'è la versione compressa mt_cache.jsonl.gz (rigenerarla con `gzip -9 -k -n`); lo script riprende da dove si era fermato, quindi la cache versionata rende tutto riproducibile senza ritradurre.

uso: python training/translate_ext.py [--threads 10] [--bs 64]
"""
import argparse, json, os, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "bench", "data", "ext")
MODELS = {"it": ("Helsinki-NLP/opus-mt-en-it", "be5a254d936f1a8c8da080406ca582a6615e6658"),
          "es": ("Helsinki-NLP/opus-mt-en-es", "5bc4493d463cf000c1f0b50f8d56886a392ed4ab"),
          "de": ("Helsinki-NLP/opus-mt-en-de", "6183067f769a302e3861815543b9f312c71b0ca4"),
          "fr": ("Helsinki-NLP/opus-mt-en-fr", "dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18")}


def translate_descriptions(a):
    """Descrizioni e parafrasi (tutte le intenzioni di descriptions.json: train e heldout; le 10 MASSIVE mai viste non
    ci sono) -> bench/data/ext/descriptions_mt.json {testo inglese: {lingua: traduzione}}."""
    torch.set_num_threads(a.threads)
    from transformers import MarianMTModel, MarianTokenizer
    D = json.load(open(os.path.join(EXT, "descriptions.json")))["intents"]
    src = sorted({d for v in D.values() for d in v["desc"]})
    out = {s: {} for s in src}
    for lang, (name, rev) in MODELS.items():
        tok = MarianTokenizer.from_pretrained(name, revision=rev)
        m = MarianMTModel.from_pretrained(name, revision=rev).eval()
        for i in range(0, len(src), a.bs):
            b = src[i:i + a.bs]
            with torch.inference_mode():
                enc = tok(b, return_tensors="pt", padding=True)
                o = m.generate(**enc, num_beams=4, max_new_tokens=48)
            for s, t in zip(b, tok.batch_decode(o, skip_special_tokens=True)):
                out[s][lang] = t
    meta = {l: {"model": n, "revision": r} for l, (n, r) in MODELS.items()}
    json.dump({"_models": meta, "translations": out}, open(os.path.join(EXT, "descriptions_mt.json"), "w"),
              ensure_ascii=False, indent=0)
    print("DONE", len(src), "descrizioni")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--descriptions", action="store_true",
                    help="traduce invece le descrizioni di descriptions.json -> descriptions_mt.json")
    a = ap.parse_args()
    if a.descriptions:
        return translate_descriptions(a)
    torch.set_num_threads(a.threads)
    from transformers import MarianMTModel, MarianTokenizer
    src = [json.loads(l)["text"] for l in open(os.path.join(EXT, "ext_utterances.jsonl"))
           if json.loads(l)["lang"] == "en" and json.loads(l)["source"] != "minds14"]
    ho = subprocess.run([sys.executable, os.path.join(HERE, "..", "bench", "build_external.py"), "mt_list"],
                        check=True, capture_output=True, text=True).stdout.splitlines()
    src = sorted(set(src) | {json.loads(l)["text"] for l in ho})
    cache_path = os.path.join(EXT, "mt_cache.jsonl")
    done = set()
    if not os.path.exists(cache_path) and os.path.exists(cache_path + ".gz"):
        import gzip, shutil
        with gzip.open(cache_path + ".gz", "rb") as fi, open(cache_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); done.add((r["src"], r["lang"]))
    with open(cache_path, "a") as f:
        for lang, (name, rev) in MODELS.items():
            todo = [s for s in src if (s, lang) not in done]
            if not todo:
                continue
            tok = MarianTokenizer.from_pretrained(name, revision=rev)
            m = MarianMTModel.from_pretrained(name, revision=rev).eval()
            todo.sort(key=len); t0 = time.time()
            for i in range(0, len(todo), a.bs):
                b = todo[i:i + a.bs]
                with torch.inference_mode():
                    enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    out = m.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=64)
                for s, t in zip(b, tok.batch_decode(out, skip_special_tokens=True)):
                    f.write(json.dumps(dict(src=s, lang=lang, text=t, model=name, revision=rev), ensure_ascii=False) + "\n")
                f.flush()
                if (i // a.bs) % 50 == 0:
                    print(lang, i + len(b), "/", len(todo), f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", len(src), "frasi")


if __name__ == "__main__":
    main()
