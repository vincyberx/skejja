"""Pool di testi multilingue per i dati di ampiezza della fase 2, e richieste per l'insegnante.

Fonti (licenze verificate il 13 ottobre 2026, vedi docs/fase2_ampiezza.md):
- fineweb-2 (ODC-BY), una scheggia del train per it/es/de/fr: testo web vario, incluso quello promozionale;
- fineweb campione 10BT (ODC-BY) per l'inglese, che fineweb-2 non copre;
- aya_dataset (Apache-2.0): richieste e messaggi scritti da persone nelle nostre 5 lingue;
- mfaq (CC0-1.0), split di validazione: domande vere di assistenza in 5 lingue.
I parquet grandi non si scaricano interi: si leggono per gruppi di righe dal filesystem dell'Hub.

L'insegnante fornisce le ETICHETTE; qui si prepara solo il testo e la domanda. Le famiglie di domande, con
opzioni e affermazioni nelle 5 lingue, sono in bench/data/broad/families.json. La famiglia `lingua` è l'unica
con etichetta già nota (la lingua dichiarata dal corpus).

Regola di separazione (docs/fase2_ampiezza.md §1): ogni testo del pool passa un controllo di sovrapposizione
contro tutti i file di test, sviluppo e calibrazione del progetto; i dataset dell'articolo su Jev non entrano
nel pool in nessuna forma.

uso:
  python bench/build_broad.py pool --n_per_lang 500 --out bench/data/broad --name trial
  python bench/build_broad.py requests --pool bench/data/broad/pool_trial.jsonl --out bench/data/broad --name trial
"""
import argparse, collections, hashlib, json, os, random, re, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "it", "es", "de", "fr"]
FW2 = {"it": "ita_Latn", "es": "spa_Latn", "de": "deu_Latn", "fr": "fra_Latn"}
AYA = {"eng": "en", "ita": "it", "spa": "es", "deu": "de", "fra": "fr"}
MIN_CHAR, MAX_CHAR, MIN_PAROLE = 40, 1500, 6
# File del progetto che non devono mai condividere un testo con il pool.
FILE_DA_NON_TOCCARE = [
    "test_choice.jsonl", "test_noul.jsonl", "test_noul_unseen_ext.jsonl", "dev_noul_ext.jsonl",
    "calib_choice.jsonl", "calib_noul.jsonl", "calib_noul_ext.jsonl", "calib_noul_fewshot.jsonl",
    "train_utterances.jsonl", "ood/ood_v0.jsonl", "agent/tldr_test.jsonl", "agent/tldr_dev.jsonl",
    "ext/ext_utterances.jsonl", "ext/ext_heldout_en.jsonl",
]


def norm(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def chiave(t):
    return hashlib.sha1(norm(t).encode()).hexdigest()[:16]


def testi_del_progetto(data_dir):
    """Tutti i testi già usati dal progetto (test, sviluppo, calibrazione, addestramento)."""
    visti = set()
    for rel in FILE_DA_NON_TOCCARE:
        path = os.path.join(data_dir, rel)
        if not os.path.isfile(path):
            print(f"  attenzione: manca {rel}", file=sys.stderr)
            continue
        for line in open(path):
            r = json.loads(line)
            for campo in ("text", "cmd"):
                if r.get(campo):
                    visti.add(chiave(r[campo]))
            for v in (r.get("desc") or {}).values():
                visti.add(chiave(v))
    # testi del metro di qualità dell'insegnante (split di training dei dataset dell'articolo su Jev): mai nel pool
    import glob
    for path in glob.glob(os.path.join(data_dir, "broad", "teacher_*_in.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("uso") == "solo_metro":
                visti.add(chiave(r["text"]))
    return visti


def accettabile(t):
    t = (t or "").strip()
    if not (MIN_CHAR <= len(t) <= MAX_CHAR) or len(t.split()) < MIN_PAROLE:
        return None
    lettere = sum(c.isalpha() or c.isspace() for c in t)
    if lettere / len(t) < 0.75:
        return None
    return re.sub(r"\s+", " ", t)


def da_parquet_hub(path, colonne, righe_volute, rng):
    """Legge gruppi di righe da un parquet dell'Hub senza scaricare il file intero."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    f = pq.ParquetFile(fs.open(path))
    gruppi = list(range(f.num_row_groups))
    rng.shuffle(gruppi)
    out = []
    for g in gruppi:
        for r in f.read_row_group(g, columns=colonne).to_pylist():
            out.append(r)
        if len(out) >= righe_volute * 4:
            break
    return out


def raccogli(n_per_lang, seed, out_dir, nome):
    rng = random.Random(seed)
    import urllib.request
    visti = testi_del_progetto(os.path.join(HERE, "data"))
    print(f"testi già usati dal progetto: {len(visti)}")
    pool, scartati = [], collections.Counter()
    chiavi = set()

    def aggiungi(source, lang, testo, extra=None):
        t = accettabile(testo)
        if t is None:
            scartati["forma"] += 1
            return
        k = chiave(t)
        if k in visti:
            scartati["sovrapposizione_col_progetto"] += 1
            return
        if k in chiavi:
            scartati["duplicato"] += 1
            return
        chiavi.add(k)
        pool.append(dict(id=f"{source}-{lang}-{len(pool)}", source=source, lang=lang, text=t,
                         n_char=len(t), **(extra or {})))

    quota = max(1, n_per_lang // 4)   # quattro fonti per lingua, dove esistono
    for lang in LANGS:
        if lang in FW2:
            import urllib.request as u
            alberi = json.load(u.urlopen(
                f"https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/tree/main/data/{FW2[lang]}/train"))
            primo = sorted(e["path"] for e in alberi if e["type"] == "file")[0]
            righe = da_parquet_hub(f"datasets/HuggingFaceFW/fineweb-2/{primo}",
                                   ["text", "language_score"], quota, rng)
            rng.shuffle(righe)
            for r in righe[: quota * 3]:
                if float(r.get("language_score") or 1) >= 0.9:
                    aggiungi("fineweb2", lang, r["text"])
        else:
            righe = da_parquet_hub(
                "datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet", ["text"], quota, rng)
            rng.shuffle(righe)
            for r in righe[: quota * 3]:
                aggiungi("fineweb", lang, r["text"])
        n_prima = len(pool)
        print(f"  {lang}: web {n_prima}", flush=True)

    from huggingface_hub import hf_hub_download
    aya = hf_hub_download("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset")
    import pyarrow.parquet as pq
    tb = pq.read_table(aya, columns=["inputs", "language_code"]).to_pylist()
    per_lingua = collections.defaultdict(list)
    for r in tb:
        if r["language_code"] in AYA:
            per_lingua[AYA[r["language_code"]]].append(r["inputs"])
    for lang, testi in per_lingua.items():
        rng.shuffle(testi)
        for t in testi[: quota * 3]:
            aggiungi("aya", lang, t)

    for lang in LANGS:
        try:
            p = hf_hub_download("clips/mfaq", f"data/{lang}/valid.jsonl", repo_type="dataset")
        except Exception as e:
            print(f"  mfaq {lang}: {type(e).__name__}", file=sys.stderr)
            continue
        righe = [json.loads(l) for l in open(p)]
        rng.shuffle(righe)
        for r in righe[: quota * 3]:
            for coppia in (r.get("qa_pairs") or [])[:1]:
                aggiungi("mfaq", lang, coppia.get("question", ""))

    prima_del_livellamento = collections.Counter((p["source"], p["lang"]) for p in pool)
    # pareggia le lingue: al massimo n_per_lang per lingua, distribuiti fra le fonti disponibili
    finale, conta = [], collections.Counter()
    rng.shuffle(pool)
    for p in sorted(pool, key=lambda p: (p["lang"], p["source"])):
        if conta[p["lang"]] < n_per_lang:
            finale.append(p)
            conta[p["lang"]] += 1
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pool_{nome}.jsonl")
    with open(path, "w") as f:
        for p in finale:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # Tutti i conteggi qui sotto descrivono il file scritto (`finale`), tranne quello segnato «prima del
    # livellamento», che descrive il pool raccolto prima del taglio a n_per_lang per lingua.
    finale_per_fonte_lingua = collections.Counter((p["source"], p["lang"]) for p in finale)
    riepilogo = dict(file=os.path.relpath(path, os.path.dirname(HERE)), n=len(finale), per_lingua=dict(conta),
                     per_fonte=dict(collections.Counter(p["source"] for p in finale)),
                     per_fonte_e_lingua={f"{s}/{l}": n for (s, l), n in sorted(finale_per_fonte_lingua.items())},
                     raccolti_prima_del_livellamento={f"{s}/{l}": n for (s, l), n in
                                                      sorted(prima_del_livellamento.items())},
                     scartati=dict(scartati), caratteri_mediani=sorted(p["n_char"] for p in finale)[len(finale) // 2],
                     seed=seed, min_char=MIN_CHAR, max_char=MAX_CHAR,
                     testi_del_progetto_esclusi=len(visti))
    json.dump(riepilogo, open(os.path.join(out_dir, f"pool_{nome}_riepilogo.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


def richieste(pool_path, out_dir, nome, seed, p_domanda_nativa, n_choice, n_noul):
    """Per ogni testo: n_choice famiglie con opzioni + n_noul affermazioni sì/no, in lingua nativa o inglese."""
    rng = random.Random(seed + 1)
    F = json.load(open(os.path.join(HERE, "data", "broad", "families.json")))["famiglie"]
    con_opzioni = [k for k, v in F.items() if v["opzioni"]]
    con_affermazioni = [k for k, v in F.items() if v["affermazioni"]]
    pool = [json.loads(l) for l in open(pool_path)]
    out, conta = [], collections.Counter()
    for p in pool:
        ql = lambda: p["lang"] if (p["lang"] != "en" and rng.random() < p_domanda_nativa) else "en"
        for fam in rng.sample(con_opzioni, min(n_choice, len(con_opzioni))):
            lq = ql()
            opzioni = sorted(F[fam]["opzioni"])
            rng.shuffle(opzioni)
            oro = p["lang"] if fam == "lingua" else None
            out.append(dict(rid=f"{p['id']}-{fam}-choice", text_id=p["id"], lang=p["lang"], text=p["text"],
                            famiglia=fam, tipo="choice", lang_domanda=lq, domanda=F[fam]["domanda"][lq],
                            opzioni=opzioni, opzioni_desc=[F[fam]["opzioni"][o][lq] for o in opzioni],
                            gold=oro, gold_da="corpus" if oro else "insegnante"))
            conta[f"choice/{fam}"] += 1
        for fam in rng.sample(con_affermazioni, min(n_noul, len(con_affermazioni))):
            lq = ql()
            aff = rng.choice(sorted(F[fam]["affermazioni"]))
            oro = None
            if fam == "lingua":
                oro = int((aff == "e_italiano" and p["lang"] == "it") or (aff == "e_inglese" and p["lang"] == "en"))
            out.append(dict(rid=f"{p['id']}-{fam}-{aff}", text_id=p["id"], lang=p["lang"], text=p["text"],
                            famiglia=fam, tipo="noul", lang_domanda=lq, domanda=F[fam]["domanda"][lq],
                            affermazione=aff, affermazione_desc=F[fam]["affermazioni"][aff][lq],
                            gold=oro, gold_da="corpus" if oro is not None else "insegnante"))
            conta[f"noul/{fam}"] += 1
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"requests_{nome}.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    riepilogo = dict(file=os.path.relpath(path, os.path.dirname(HERE)), n_richieste=len(out), n_testi=len(pool), per_famiglia=dict(conta),
                     domande_in_lingua_nativa=sum(1 for r in out if r["lang_domanda"] != "en"),
                     etichette_dal_corpus=sum(1 for r in out if r["gold_da"] == "corpus"),
                     p_domanda_nativa=p_domanda_nativa, seed=seed)
    json.dump(riepilogo, open(os.path.join(out_dir, f"requests_{nome}_riepilogo.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "requests"])
    ap.add_argument("--n_per_lang", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "broad"))
    ap.add_argument("--name", default="trial")
    ap.add_argument("--pool", default="")
    ap.add_argument("--p_domanda_nativa", type=float, default=0.5)
    ap.add_argument("--n_choice", type=int, default=2)
    ap.add_argument("--n_noul", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "pool":
        raccogli(a.n_per_lang, a.seed, a.out, a.name)
    else:
        richieste(a.pool or os.path.join(a.out, f"pool_{a.name}.jsonl"), a.out, a.name, a.seed,
                  a.p_domanda_nativa, a.n_choice, a.n_noul)


if __name__ == "__main__":
    main()
