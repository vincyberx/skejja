"""Descrizioni del benchmark tradotte nelle 4 lingue (OPUS-MT, beam 4, revisioni fissate in training/translate_ext.py)
per misurare MASSIVE con opzioni e domande nella lingua della frase (variante `native` di training/eval_noul.py).
Solo per valutazione: queste traduzioni (comprese le 10 intenzioni mai viste) non entrano mai in addestramento.

uso: python bench/build_native_options.py   -> bench/data/intent_descriptions_mt.json
"""
import json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
from translate_ext import MODELS  # noqa: E402


def main():
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(2)
    desc = json.load(open(os.path.join(HERE, "data", "intent_descriptions.json")))
    out = {k: {"en": v} for k, v in desc.items()}
    keys = sorted(desc)
    for lang, (name, rev) in MODELS.items():
        tok = MarianTokenizer.from_pretrained(name, revision=rev); m = MarianMTModel.from_pretrained(name, revision=rev).eval()
        with torch.inference_mode():
            o = m.generate(**tok([desc[k] for k in keys], return_tensors="pt", padding=True), num_beams=4, max_new_tokens=48)
        for k, t in zip(keys, tok.batch_decode(o, skip_special_tokens=True)):
            out[k][lang] = t
    json.dump(out, open(os.path.join(HERE, "data", "intent_descriptions_mt.json"), "w"), ensure_ascii=False, indent=1)
    print("DONE", len(out))


if __name__ == "__main__":
    main()
