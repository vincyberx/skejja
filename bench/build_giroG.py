"""Pool di testi del giro G: stati lunghi, relazioni fra due testi, pertinenza graduata e giudizi ordinali.

Il giro F copriva un testo solo per volta (argomento, sentiment, emozione...). Il giro G aggiunge quello che
manca per il formato di S1MB (docs/fase2_giro_G.md): ingressi fino a 1.024 token, relazioni fra DUE testi,
Choice fino a 50 opzioni e la primitiva Score su scala ordinale.

Fonti, tutte approvate dal controllo delle licenze di `bench/fonti_relazioni_audit.py`
(results/fase2/fonti_relazioni_licenze.csv) e nessuna coincidente con una delle 110 famiglie di sorgenti delle
prove di S1MB (bench/data/broad/s1mb_sottoinsiemi.json):

| Gruppo | Fonte | Licenza | Cosa dà |
|---|---|---|---|
| `lungo` | fineweb, fineweb-2 | ODC-BY | passaggi web lunghi (300-1.000 token) nelle 5 lingue |
| `pertinenza` | mfaq | CC0-1.0 | domanda + sua risposta, negativi difficili dalla stessa pagina |
| `pertinenza` | deepset/germanquad | CC BY 4.0 | domanda + paragrafo (de) |
| `pertinenza` | AgentPublic/piaf | MIT | domanda + paragrafo (fr) |
| `pertinenza` | miracl/miracl | Apache-2.0 | domanda + passaggi positivi e negativi (de, es, fr, en) |
| `relazione` | google-research-datasets/circa | CC BY 4.0 | risposte indirette con la classe vera (en) |
| `score` | nvidia/HelpSteer3 | CC BY 4.0 | giudizi su scala 0-4, anche it/de/es/fr |
| `score` | openbmb/UltraFeedback | MIT | giudizi su scala 1-5 (en) |

Ogni testo passa il controllo di sovrapposizione del giro F (tutti i file di test, sviluppo, calibrazione e i
testi del metro dell'insegnante) **più** gli stati delle prove di S1MB, confrontati per impronta con
`bench/data/broad/s1mb_testi_hash.json`.

uso:
  python bench/build_giroG.py impronte_s1mb --s1mb <cartella del dataset> --out bench/data/broad
  python bench/build_giroG.py pool --out bench/data/broad --name giroG [--n_lungo 600] [--n_pertinenza 1500]
"""

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_broad import LANGS, FW2, accettabile, chiave, da_parquet_hub, testi_del_progetto  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Gli stati lunghi hanno un minimo e un massimo diversi dal giro F (40-1.500 caratteri): qui servono testi che
# arrivino al budget di 1.024 token, cioè circa 4.000 caratteri.
LUNGO_MIN_CHAR, LUNGO_MAX_CHAR = 1200, 4000
MIRACL_LANG = {"de": "de", "es": "es", "fr": "fr", "en": "en"}


def parquet_convertito(repo, path):
    """Parquet del ramo `refs/convert/parquet` dell'Hub: serve per i dataset che avevano uno script di
    caricamento (datasets 4.x non li esegue più). Stesso repository, stessa licenza."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision="refs/convert/parquet"))


def nomi_classi(tabella, colonna):
    """Nomi delle classi di una colonna ClassLabel, letti dai metadati del parquet."""
    meta = json.loads(tabella.schema.metadata[b"huggingface"].decode())
    return meta["info"]["features"][colonna]["names"]


def accettabile_coppia(t, min_char=10, min_parole=3):
    """Primo testo di una coppia (domanda, richiesta): può essere molto più corto di un documento."""
    t = re.sub(r"\s+", " ", (t or "").strip())
    if len(t) < min_char or len(t.split()) < min_parole or len(t) > 2000:
        return None
    return t


def accettabile_lungo(t):
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not (LUNGO_MIN_CHAR <= len(t) <= LUNGO_MAX_CHAR):
        return None
    if sum(c.isalpha() or c.isspace() for c in t) / len(t) < 0.75:
        return None
    return t


def impronte_s1mb(cartella, out_dir):
    """Impronte (solo hash) degli stati delle prove di S1MB, per il controllo di sovrapposizione.

    Nel repository non finisce il testo del banco: solo le impronte normalizzate, come per i testi del metro.
    """
    import glob
    import pyarrow.parquet as pq
    impronte, n = set(), 0
    for f in sorted(glob.glob(os.path.join(cartella, "data", "*", "*", "*.parquet"))):
        for row in pq.read_table(f, columns=["input"]).to_pylist():
            stato = (row["input"] or {}).get("state_json") or ""
            if not stato:
                continue
            n += 1
            impronte.add(chiave(stato))
            for pezzo in re.split(r'","|\\n\\n', stato):     # anche i campi interni dello stato
                pezzo = re.sub(r'^[^:]{0,40}":"', "", pezzo)
                if len(pezzo) >= 80:
                    impronte.add(chiave(pezzo))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "s1mb_testi_hash.json")
    json.dump(dict(_about="Impronte normalizzate (sha1 a 16 cifre) degli stati delle prove di S1MB english-v1 e "
                          "dei loro campi piu lunghi di 80 caratteri. Servono solo a verificare che nessun testo "
                          "di addestramento del giro G coincida con il banco: il testo del banco non entra nel "
                          "repository.",
                   dataset="hotchpotch/S1MB-english-v1", stati_letti=n, impronte=len(impronte),
                   hash=sorted(impronte)), open(path, "w"), indent=1)
    print(json.dumps(dict(file=path, stati_letti=n, impronte=len(impronte))))


def carica_impronte(data_dir):
    visti = testi_del_progetto(data_dir)
    path = os.path.join(data_dir, "broad", "s1mb_testi_hash.json")
    if os.path.isfile(path):
        d = json.load(open(path))
        visti |= set(d["hash"])
        print(f"impronte: progetto + S1MB = {len(visti)} ({d['impronte']} da S1MB)")
    else:
        raise SystemExit("manca bench/data/broad/s1mb_testi_hash.json: esegui prima `impronte_s1mb`")
    return visti


class Pool:
    """Raccolta con controllo di sovrapposizione e di duplicati, uguale nello spirito a quella del giro F."""

    def __init__(self, visti):
        self.visti, self.chiavi, self.righe = visti, set(), []
        self.scartati = collections.Counter()

    def aggiungi(self, gruppo, source, lang, testo, **extra):
        # `lungo` vuole documenti interi; negli altri gruppi il primo testo è una domanda o una richiesta,
        # che può essere corta: conta la coppia, non la lunghezza del primo pezzo.
        t = accettabile_lungo(testo) if gruppo == "lungo" else accettabile_coppia(testo)
        if t is None:
            self.scartati[f"{gruppo}/forma"] += 1
            return None
        k = chiave(t)
        if k in self.visti:
            self.scartati[f"{gruppo}/sovrapposizione"] += 1
            return None
        # Il duplicato si giudica sulla COPPIA: in Circa la stessa domanda torna con risposte diverse, e
        # ognuna è un esempio a sé. Il controllo di sovrapposizione col banco resta sul primo testo.
        secondo = extra.get("secondo_testo") or extra.get("positivo") or extra.get("risposta_a") or ""
        k_dup = chiave(t + " || " + secondo[:400]) if secondo else k
        if k_dup in self.chiavi:
            self.scartati[f"{gruppo}/duplicato"] += 1
            return None
        k = k_dup
        self.chiavi.add(k)
        r = dict(id=f"{source}-{lang}-{len(self.righe)}", gruppo=gruppo, source=source, lang=lang, text=t,
                 n_char=len(t), **extra)
        self.righe.append(r)
        return r

    def controlla_coppia(self, testo):
        """Per i secondi testi (risposte, passaggi): non entrano nel pool come righe proprie."""
        k = chiave(testo or "")
        if k in self.visti:
            self.scartati["coppia/sovrapposizione"] += 1
            return False
        return True


def raccogli_lungo(pool, n_per_lang, rng):
    for lang in LANGS:
        if lang in FW2:
            import urllib.request as u
            alberi = json.load(u.urlopen(
                f"https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/tree/main/data/{FW2[lang]}/train"))
            primo = sorted(e["path"] for e in alberi if e["type"] == "file")[0]
            righe = da_parquet_hub(f"datasets/HuggingFaceFW/fineweb-2/{primo}",
                                   ["text", "language_score"], n_per_lang, rng)
            righe = [r for r in righe if float(r.get("language_score") or 1) >= 0.9]
            source = "fineweb2"
        else:
            righe = da_parquet_hub("datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet",
                                   ["text"], n_per_lang, rng)
            source = "fineweb"
        rng.shuffle(righe)
        presi = 0
        for r in righe:
            if presi >= n_per_lang:
                break
            if pool.aggiungi("lungo", source, lang, r["text"]):
                presi += 1
        print(f"  lungo/{lang}: {presi}", flush=True)


def raccogli_pertinenza(pool, n_per_fonte, rng):
    """Domanda + passaggio giusto + passaggi sbagliati difficili (stessa pagina o stesso documento)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    # mfaq: le coppie domanda/risposta della stessa pagina sono negativi difficili per definizione
    for lang in LANGS:
        try:
            p = hf_hub_download("clips/mfaq", f"data/{lang}/valid.jsonl", repo_type="dataset")
        except Exception as e:
            print(f"  mfaq {lang}: {type(e).__name__}", file=sys.stderr)
            continue
        righe = [json.loads(l) for l in open(p)]
        rng.shuffle(righe)
        presi = 0
        for r in righe:
            coppie = [c for c in (r.get("qa_pairs") or []) if c.get("question") and c.get("answer")]
            if len(coppie) < 3 or presi >= n_per_fonte // len(LANGS):
                continue
            rng.shuffle(coppie)
            giusta, sbagliate = coppie[0], coppie[1:4]
            if not pool.controlla_coppia(giusta["answer"]):
                continue
            if pool.aggiungi("pertinenza", "mfaq", lang, giusta["question"],
                             positivo=giusta["answer"][:2000],
                             negativi=[c["answer"][:2000] for c in sbagliate], negativi_da="stessa pagina"):
                presi += 1
        print(f"  pertinenza/mfaq/{lang}: {presi}", flush=True)

    # GermanQuAD: lo script di caricamento non esiste più, si legge il parquet convertito dello stesso repo
    try:
        t = parquet_convertito("deepset/germanquad", "plain_text/train/0000.parquet")
        righe = t.to_pylist()
        rng.shuffle(righe)
        contesti = [(r.get("context") or "")[:2000] for r in righe[:4000] if r.get("context")]
        presi = 0
        for r in righe:
            if presi >= n_per_fonte // 2:
                break
            dom, ctx = r.get("question"), (r.get("context") or "")[:2000]
            if not dom or not ctx or not pool.controlla_coppia(ctx):
                continue
            negativi = [cc for cc in rng.sample(contesti, 4) if cc != ctx][:3]
            if pool.aggiungi("pertinenza", "germanquad", "de", dom, positivo=ctx, negativi=negativi,
                             negativi_da="altri paragrafi dello stesso corpus"):
                presi += 1
        print(f"  pertinenza/germanquad: {presi}", flush=True)
    except Exception as e:
        print(f"  germanquad: {type(e).__name__}: {e}", file=sys.stderr)

    # PIAF: parquet normale sul ramo principale
    try:
        from datasets import load_dataset
        ds = load_dataset("AgentPublic/piaf", split="train")
        righe = list(ds)
        rng.shuffle(righe)
        contesti = [(r.get("context") or "")[:2000] for r in righe[:4000] if r.get("context")]
        presi = 0
        for r in righe:
            if presi >= n_per_fonte // 2:
                break
            dom, ctx = r.get("question"), (r.get("context") or "")[:2000]
            if not dom or not ctx or not pool.controlla_coppia(ctx):
                continue
            negativi = [cc for cc in rng.sample(contesti, 4) if cc != ctx][:3]
            if pool.aggiungi("pertinenza", "piaf", "fr", dom, positivo=ctx, negativi=negativi,
                             negativi_da="altri paragrafi dello stesso corpus"):
                presi += 1
        print(f"  pertinenza/piaf: {presi}", flush=True)
    except Exception as e:
        print(f"  piaf: {type(e).__name__}: {e}", file=sys.stderr)

    # MIRACL: ha già i negativi scelti dal recupero (negativi difficili veri), dal ramo parquet convertito
    try:
        from huggingface_hub import HfApi
        file_miracl = HfApi().list_repo_files("miracl/miracl", repo_type="dataset",
                                              revision="refs/convert/parquet")
    except Exception as e:
        print(f"  miracl: {type(e).__name__}: {e}", file=sys.stderr)
        file_miracl = []
    for lang in MIRACL_LANG.values():
        attesi = [f for f in file_miracl if f.startswith(f"{lang}/dev/") and f.endswith(".parquet")]
        if not attesi:
            print(f"  miracl {lang}: nessun parquet dev nel ramo convertito", file=sys.stderr)
            continue
        try:
            righe = []
            for f in attesi[:2]:
                righe += parquet_convertito("miracl/miracl", f).to_pylist()
        except Exception as e:
            print(f"  miracl {lang}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        rng.shuffle(righe)
        presi = 0
        for r in righe:
            if presi >= n_per_fonte // len(MIRACL_LANG):
                break
            pos = [p_["text"][:2000] for p_ in (r.get("positive_passages") or [])]
            neg = [p_["text"][:2000] for p_ in (r.get("negative_passages") or [])]
            if not pos or len(neg) < 2 or not pool.controlla_coppia(pos[0]):
                continue
            if pool.aggiungi("pertinenza", "miracl", lang, r["query"], positivo=pos[0], negativi=neg[:3],
                             negativi_da="negativi del recupero (stessa domanda)"):
                presi += 1
        print(f"  pertinenza/miracl/{lang}: {presi}", flush=True)


def raccogli_relazione(pool, n, rng):
    """Circa: domanda, risposta indiretta e classe vera scritta da persone (CC BY 4.0, inglese).

    `goldstandard2` è una ClassLabel: i nomi delle classi si leggono dai metadati del parquet, non a memoria.
    Il valore -1 vuol dire «senza etichetta concorde» e si scarta.
    """
    try:
        t = parquet_convertito("google-research-datasets/circa", "default/train/0000.parquet")
    except Exception as e:
        print(f"  circa: {type(e).__name__}: {e}", file=sys.stderr)
        return
    nomi = nomi_classi(t, "goldstandard2")
    righe = t.to_pylist()
    rng.shuffle(righe)
    presi = 0
    for r in righe:
        if presi >= n:
            break
        gold = r.get("goldstandard2")
        if not isinstance(gold, int) or not 0 <= gold < len(nomi):
            continue
        domanda, risposta = r.get("question-X"), r.get("answer-Y")
        if not domanda or not risposta or not pool.controlla_coppia(risposta):
            continue
        if pool.aggiungi("relazione", "circa", "en", domanda, secondo_testo=risposta, gold=nomi[gold],
                         gold_da="corpus", contesto=r.get("context", "")):
            presi += 1
    print(f"  relazione/circa: {presi} (classi: {nomi})", flush=True)


def raccogli_score(pool, n_per_fonte, rng):
    """Giudizi su scala ordinale già nei dati: HelpSteer3 (preferenza -3..3) e UltraFeedback (1-10).

    In HelpSteer3 la configurazione `preference` confronta due risposte alla stessa richiesta: il voto va da -3
    («la prima è molto meglio») a +3 («la seconda è molto meglio»). Il campo `language` contiene sia lingue
    naturali sia linguaggi di programmazione: teniamo solo le nostre cinque lingue naturali.
    """
    lingue = {"english": "en", "italian": "it", "spanish": "es", "german": "de", "french": "fr"}
    try:
        righe = parquet_convertito("nvidia/HelpSteer3", "preference/train/0000.parquet").to_pylist()
    except Exception as e:
        print(f"  helpsteer3: {type(e).__name__}: {e}", file=sys.stderr)
        righe = []
    rng.shuffle(righe)
    presi = collections.Counter()
    for r in righe:
        lang = lingue.get((r.get("language") or "").lower())
        if not lang or presi[lang] >= n_per_fonte // 5:
            continue
        ctx = r.get("context")
        if isinstance(ctx, str):
            try:
                import ast
                ctx = ast.literal_eval(ctx)
            except Exception:
                ctx = [{"role": "user", "content": ctx}]
        richiesta = " ".join(m.get("content", "") for m in (ctx or []) if isinstance(m, dict)
                             and m.get("role") == "user")
        a, b = r.get("response1") or "", r.get("response2") or ""
        voto = r.get("overall_preference")
        if not richiesta or not a or not b or voto is None:
            continue
        if not (pool.controlla_coppia(a) and pool.controlla_coppia(b)):
            continue
        if pool.aggiungi("score", "helpsteer3", lang, richiesta[:1500], risposta_a=a[:1500], risposta_b=b[:1500],
                         gold=float(voto), gold_da="corpus", scala="preferenza -3..3",
                         dominio=r.get("domain")):
            presi[lang] += 1
    print(f"  score/helpsteer3: {dict(presi)}", flush=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("openbmb/UltraFeedback", split="train")
    except Exception as e:
        print(f"  ultrafeedback: {type(e).__name__}: {e}", file=sys.stderr)
        return
    righe = list(ds.select(range(min(len(ds), 20000))))
    rng.shuffle(righe)
    presi = 0
    for r in righe:
        if presi >= n_per_fonte:
            break
        compl = [cc for cc in (r.get("completions") or []) if cc.get("overall_score") is not None]
        if not compl:
            continue
        cc = rng.choice(compl)
        risposta = (cc.get("response") or "")[:1500]
        if not risposta or not pool.controlla_coppia(risposta):
            continue
        if pool.aggiungi("score", "ultrafeedback", "en", (r.get("instruction") or "")[:1500],
                         secondo_testo=risposta, gold=float(cc["overall_score"]), gold_da="corpus",
                         scala="1-10 del corpus"):
            presi += 1
    print(f"  score/ultrafeedback: {presi}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("impronte_s1mb", help="impronte degli stati di S1MB per il controllo di sovrapposizione")
    a1.add_argument("--s1mb", required=True, help="cartella dello snapshot del dataset S1MB-english-v1")
    a1.add_argument("--out", default=os.path.join(HERE, "data", "broad"))
    a2 = sub.add_parser("pool", help="raccoglie il pool del giro G")
    a2.add_argument("--out", default=os.path.join(HERE, "data", "broad"))
    a2.add_argument("--name", default="giroG")
    a2.add_argument("--data", default=os.path.join(HERE, "data"))
    a2.add_argument("--n_lungo", type=int, default=600, help="passaggi lunghi per lingua")
    a2.add_argument("--n_pertinenza", type=int, default=1500, help="per famiglia di fonti")
    a2.add_argument("--n_relazione", type=int, default=3000)
    a2.add_argument("--n_score", type=int, default=2500, help="per fonte")
    a2.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "impronte_s1mb":
        return impronte_s1mb(a.s1mb, a.out)

    rng = random.Random(a.seed)
    pool = Pool(carica_impronte(a.data))
    raccogli_lungo(pool, a.n_lungo, rng)
    raccogli_pertinenza(pool, a.n_pertinenza, rng)
    raccogli_relazione(pool, a.n_relazione, rng)
    raccogli_score(pool, a.n_score, rng)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"pool_{a.name}.jsonl")
    with open(path, "w") as f:
        for r in pool.righe:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_gruppo = collections.Counter(r["gruppo"] for r in pool.righe)
    car = sorted(r["n_char"] for r in pool.righe if r["gruppo"] == "lungo")
    riepilogo = dict(
        file=os.path.relpath(path, os.path.dirname(HERE)), n=len(pool.righe),
        per_gruppo=dict(per_gruppo),
        per_fonte=dict(collections.Counter(r["source"] for r in pool.righe)),
        per_fonte_e_lingua={f"{s}/{l}": n for (s, l), n in
                            sorted(collections.Counter((r["source"], r["lang"]) for r in pool.righe).items())},
        caratteri_lungo=dict(mediana=car[len(car) // 2] if car else None, massimo=car[-1] if car else None),
        scartati=dict(pool.scartati), seed=a.seed,
        nota="Il controllo di sovrapposizione copre i file del progetto, i testi del metro dell'insegnante e le "
             "impronte degli stati di S1MB (bench/data/broad/s1mb_testi_hash.json).")
    json.dump(riepilogo, open(os.path.join(a.out, f"pool_{a.name}_riepilogo.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
