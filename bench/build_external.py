"""Dati sì/no esterni a MASSIVE (fase 2), solo licenze CC BY.

Fonti (revisioni fissate):
- CLINC150 `clinc/clinc_oos` config `plus` (CC BY 3.0), inglese, 150 intenzioni + out-of-scope;
- Banking77 `PolyAI-LDN/task-specific-datasets` (CC BY 4.0), inglese, 77 intenzioni;
- MINDS-14 `PolyAI/minds14` (CC BY 4.0), trascrizioni native it-IT/es-ES/de-DE/fr-FR/en-US/en-GB, 14 intenzioni
  (si legge solo il testo, mai l'audio).
Le descrizioni e i ruoli delle intenzioni sono in bench/data/ext/descriptions.json. Le intenzioni CLINC affini agli
ambiti MASSIVE esclusi (trasporti, cibo a domicilio e ristoranti, social, cucina; più voli/hotel/viaggi) sono
scartate (lista EXCLUDED_CLINC) e non compaiono da nessuna parte.

Passi:
  python bench/build_external.py prepare --download --raw data/ext_raw
      -> bench/data/ext/ext_utterances.jsonl (frasi di addestramento) e ext_heldout_en.jsonl (domini tenuti fuori)
  python training/translate_ext.py ...           (traduzioni OPUS-MT, cache in bench/data/ext/mt_cache.jsonl)
  python bench/build_external.py items
      -> bench/data/calib_noul_ext.jsonl, bench/data/dev_noul_ext.jsonl (domande su intenzioni tenute fuori)
"""
import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from skejja_measure import LANGS, is_unseen, open_mt_cache  # noqa: E402

CLINC = ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec")
B77 = ("PolyAI-LDN/task-specific-datasets", "57ec275d8078af65b7731c2a98be812d844a6d6b")
MINDS = ("PolyAI/minds14", "40ce77cb32a384e4d50a568e1ec39ac804019d33")
MINDS_LOCALES = {"it-IT": "it", "es-ES": "es", "de-DE": "de", "fr-FR": "fr", "en-US": "en", "en-GB": "en"}
EXCLUDED_CLINC = ("uber traffic book_flight flight_status directions distance car_rental lost_luggage carry_on order "
                  "order_status restaurant_reservation restaurant_suggestion restaurant_reviews accept_reservations "
                  "confirm_reservation cancel_reservation how_busy recipe cook_time ingredients_list "
                  "ingredient_substitution meal_suggestion food_last nutrition_info calories book_hotel "
                  "travel_suggestion international_visa travel_alert").split()
EXT = os.path.join(HERE, "data", "ext")
SEED = 31
N_CALIB_PER_INTENT, N_DEV_PER_INTENT = 40, 30
P_SAME = 0.7


def load_desc():
    return json.load(open(os.path.join(EXT, "descriptions.json")))


def download(raw):
    from huggingface_hub import hf_hub_download
    import urllib.request
    os.makedirs(raw, exist_ok=True)
    for sp in ("train", "validation", "test"):
        hf_hub_download(CLINC[0], f"plus/{sp}-00000-of-00001.parquet", repo_type="dataset", revision=CLINC[1], local_dir=raw)
    for sp in ("train", "test"):
        url = f"https://raw.githubusercontent.com/{B77[0]}/{B77[1]}/banking_data/{sp}.csv"
        urllib.request.urlretrieve(url, os.path.join(raw, f"banking77_{sp}.csv"))
    for loc in MINDS_LOCALES:
        hf_hub_download(MINDS[0], f"{loc}/train-00000-of-00001.parquet", repo_type="dataset", revision=MINDS[1], local_dir=raw)


def read_raw(raw):
    import pandas as pd
    import pyarrow.parquet as pq
    rows = []
    for sp in ("train", "validation", "test"):
        p = os.path.join(raw, "plus", f"{sp}-00000-of-00001.parquet")
        names = json.loads(pq.read_schema(p).metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            rows.append(dict(source="clinc", split=sp, uid=f"clinc-{sp}-{i}", lang="en", text=r["text"],
                             intent="clinc:" + names[r["intent"]]))
    for sp in ("train", "test"):
        d = pd.read_csv(os.path.join(raw, f"banking77_{sp}.csv"))
        for i, r in enumerate(d.itertuples()):
            rows.append(dict(source="banking77", split=sp, uid=f"b77-{sp}-{i}", lang="en", text=r.text,
                             intent="banking77:" + r.category))
    for loc, l in MINDS_LOCALES.items():
        pf = pq.ParquetFile(os.path.join(raw, loc, "train-00000-of-00001.parquet"))
        names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["intent_class"]["names"]
        for i, r in enumerate(pf.read(columns=["transcription", "intent_class"]).to_pylist()):
            rows.append(dict(source="minds14", split="train", uid=f"minds-{loc}-{i}", lang=l, text=r["transcription"],
                             intent="minds14:" + names[r["intent_class"]]))
    return rows


def prepare(raw):
    D = load_desc()["intents"]
    rows = read_raw(raw)
    out_tr, out_ho, skipped = [], [], collections.Counter()
    for r in rows:
        name = r["intent"].split(":", 1)[1]
        if r["source"] == "clinc" and name in EXCLUDED_CLINC:
            skipped["excluded"] += 1; continue
        if not str(r["text"]).strip():
            skipped["empty"] += 1; continue
        if r["intent"] == "clinc:oos":
            r["role"] = "train"; r["neg_only"] = True
        else:
            info = D[r["intent"]]; r["role"] = info["role"]; r["neg_only"] = False
        if r["role"] == "heldout":
            out_ho.append(r)
        elif r["split"] == "train":   # per l'addestramento solo gli split di training delle fonti
            out_tr.append(r)
    os.makedirs(EXT, exist_ok=True)
    for name, rr in (("ext_utterances.jsonl", out_tr), ("ext_heldout_en.jsonl", out_ho)):
        with open(os.path.join(EXT, name), "w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train", len(out_tr), collections.Counter(r["source"] + "/" + r["lang"] for r in out_tr))
    print("heldout", len(out_ho), dict(skipped))


def heldout_selection():
    """Frasi dei domini tenuti fuori: calibrazione dallo split train di CLINC, sviluppo da validation+test."""
    ho = [json.loads(l) for l in open(os.path.join(EXT, "ext_heldout_en.jsonl"))]
    rng = np.random.default_rng(SEED); sel = {"calib": [], "dev": []}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ho:
        by["calib" if r["split"] == "train" else "dev"][r["intent"]].append(r)
    for part, n in (("calib", N_CALIB_PER_INTENT), ("dev", N_DEV_PER_INTENT)):
        for it in sorted(by[part]):
            rr = by[part][it]
            for j in sorted(rng.choice(len(rr), size=min(n, len(rr)), replace=False)):
                sel[part].append(rr[j])
    return sel


def items(massive_train):
    """Item Noul (stesso formato di test_noul_unseen_ext.jsonl) con domande sulle intenzioni tenute fuori.
    pos / neg_same (sorella dello stesso dominio CLINC) per ogni frase; neg_other = frasi MASSIVE di addestramento
    (intenzioni viste, native nelle 5 lingue) con una domanda su un'intenzione tenuta fuori."""
    J = load_desc(); D = J["intents"]
    mt = {}
    for l in open_mt_cache(EXT):
        r = json.loads(l); mt[(r["src"], r["lang"])] = r["text"]
    groups = collections.defaultdict(list)
    for k, v in D.items():
        if v["role"] == "heldout":
            groups[v["group"]].append(k)
    ho_int = sorted(k for k, v in D.items() if v["role"] == "heldout")
    mtr = [json.loads(l) for l in open(massive_train)]
    mtr_by = collections.defaultdict(dict)
    for r in mtr:
        if not is_unseen(r["intent"]):
            mtr_by[r["id"]][r["lang"]] = r["text"]
    mids = sorted((i for i, v in mtr_by.items() if all(l in v for l in LANGS)), key=int)
    sel = heldout_selection()
    for part, seed in (("calib", SEED + 1), ("dev", SEED + 2)):
        rng = np.random.default_rng(seed); its = []
        for r in sel[part]:
            g = D[r["intent"]]["group"]
            texts = {l: (r["text"] if l == "en" else mt[(r["text"], l)]) for l in LANGS}
            its.append((r["uid"], texts, r["intent"], r["intent"], 1, "pos"))
            sib = [i for i in groups[g] if i != r["intent"]]
            its.append((r["uid"], texts, r["intent"], str(rng.choice(sib)), 0, "neg_same"))
        n_other = int(round(len(sel[part]) * (1 - P_SAME) / P_SAME))
        for mid in rng.choice(mids, size=n_other, replace=False):
            its.append((f"massive-train-{mid}", mtr_by[mid], "massive:other", str(rng.choice(ho_int)), 0, "neg_other"))
        path = os.path.join(HERE, "data", f"{part}_noul_ext.jsonl")
        with open(path, "w") as f:
            for uid, texts, gold, q, lab, nt in its:
                for l in LANGS:
                    f.write(json.dumps(dict(id=uid, lang=l, text=texts[l], intent=q, gold=gold, neg_type=nt,
                                            question=f"Does the user want to {D[q]['desc'][0]}?", label=lab),
                                       ensure_ascii=False) + "\n")
        print(path, len(its), "item per lingua", collections.Counter(x[5] for x in its))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "items", "mt_list"])
    ap.add_argument("--raw", default="data/ext_raw")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--massive_train", default=os.path.join(HERE, "data", "train_utterances.jsonl"))
    a = ap.parse_args()
    if a.cmd == "prepare":
        if a.download:
            download(a.raw)
        prepare(a.raw)
    elif a.cmd == "mt_list":   # frasi inglesi dei domini tenuti fuori da tradurre in tutte e 4 le lingue
        sel = heldout_selection()
        for part in ("calib", "dev"):
            for r in sel[part]:
                print(json.dumps({"text": r["text"]}, ensure_ascii=False))
    else:
        items(a.massive_train)


if __name__ == "__main__":
    main()
