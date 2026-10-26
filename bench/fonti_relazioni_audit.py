"""Licenze delle fonti candidate per le relazioni fra due testi e per i giudizi ordinali (giro G).

Il giro G ha bisogno di due cose che le 8 famiglie del giro F non danno: **relazioni fra due testi**
(implicazione, pertinenza fra domanda e documento, regola/caso) e **giudizi su scala ordinale** per la primitiva
Score. La regola del §4 di `docs/fase2_giro_G.md` dice da dove possono venire:

1. niente fonti che coincidono con una delle famiglie di sorgenti delle prove di S1MB
   (`bench/data/broad/s1mb_sottoinsiemi.json`): userebbe come addestramento le stesse sorgenti su cui ci
   misuriamo, e la misura non sarebbe più zero-shot;
2. solo licenze di sola attribuzione (CC BY, CC0, Apache-2.0, MIT, BSD, ODC-BY): niente ShareAlike, non
   commerciale, GPL, OpenRAIL o licenza assente;
3. dove la coincidenza è inevitabile e la fonte è già in casa, la prova va dichiarata come esposizione.

Lo script legge per ogni candidato l'API di Hugging Face (licenza dichiarata, revisione, ultima modifica, se il
repository è gated) e scrive la tabella con la decisione. Le licenze vanno comunque lette anche nella scheda: la
classificazione automatica parte dalla stringa dichiarata, come in `bench/bekko_license_audit.py`.

    python bench/fonti_relazioni_audit.py --out results/fase2/fonti_relazioni_licenze.csv
"""

import argparse
import csv
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from bekko_license_audit import classe

HERE = Path(__file__).resolve().parent
SOTTOINSIEMI = HERE / "data" / "broad" / "s1mb_sottoinsiemi.json"
AMMESSE = {"CC BY", "CC0 o dominio pubblico", "Apache, MIT o BSD", "ODC-BY"}

# (repository, a cosa serve, tipo di compito, lingue, famiglia di sorgenti di S1MB con cui coincide o "")
CANDIDATI = [
    ("google-research-datasets/circa", "risposte sì/no indirette a una domanda: relazione domanda/risposta",
     "relazione", "en", ""),
    ("deepset/germanquad", "pertinenza fra domanda e paragrafo, scritta da persone", "pertinenza", "de", ""),
    ("deepset/germandpr", "pertinenza con negativi difficili dallo stesso dominio", "pertinenza", "de", ""),
    ("AgentPublic/piaf", "pertinenza fra domanda e paragrafo in francese", "pertinenza", "fr", ""),
    ("miracl/miracl", "pertinenza multilingue, anche it/es/de/fr", "pertinenza", "it,es,de,fr,en", ""),
    ("clips/mfaq", "domande vere di assistenza con la loro risposta (già approvata al §2 del giro F)",
     "pertinenza", "it,es,de,fr,en", ""),
    ("nyu-mll/multi_nli", "implicazione fra premessa e ipotesi", "relazione", "en", ""),
    ("nvidia/HelpSteer", "giudizi 0-4 su utilità, correttezza, coerenza, complessità, prolissità",
     "score", "en", ""),
    ("nvidia/HelpSteer3", "come sopra, versione nuova e multilingue", "score", "en,de,es,fr,it", ""),
    ("ucberkeley-dlab/measuring-hate-speech", "gravità su scala continua e voci ordinali", "score", "en", ""),
    ("tasksource/tasksource-instruct-v0", "compiti di ragionamento convertiti (da filtrare compito per compito)",
     "relazione", "en", ""),
    ("PhilipMay/stsb_multi_mt", "somiglianza fra due frasi su scala 0-5: relazione e scala ordinale insieme",
     "score", "it,es,de,fr,en", ""),
    ("nvidia/HelpSteer2", "giudizi 0-4 sulle risposte, versione intermedia", "score", "en", ""),
    ("openbmb/UltraFeedback", "giudizi 1-5 su istruzione e risposta, con motivazione", "score", "en", ""),
    # Fonti vicine che la regola 1 esclude: restano in tabella con il motivo, per non ridiscuterle ogni volta.
    ("stanfordnlp/snli", "implicazione: è la prova `snli` di S1MB", "relazione", "en", "snli"),
    ("allenai/scitail", "implicazione: è la prova `scitail` di S1MB", "relazione", "en", "scitail"),
    ("google-research-datasets/paws", "parafrasi: è la prova `paws` di S1MB", "relazione", "en", "paws"),
    ("Anthropic/hh-rlhf", "giudizi di preferenza: è la prova `hh_rlhf` di S1MB", "score", "en", "hh_rlhf"),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "sicurezza: è la prova `aegis2` di S1MB", "score", "en",
     "aegis2"),
    ("google/civil_comments", "tossicità: è la prova `civil_comments` di S1MB", "score", "en", "civil_comments"),
    ("BeIR/msmarco", "pertinenza: famiglia delle prove `synthetic_relevance__nanobeir__NanoMSMARCO`",
     "pertinenza", "en", "synthetic_relevance__nanobeir__NanoMSMARCO"),
]


def api(repo):
    """Metadati del dataset dall'API di Hugging Face: licenza dichiarata, revisione, data, gated."""
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=60))
    except urllib.error.HTTPError as e:
        return dict(errore=f"HTTP {e.code}")
    except Exception as e:  # rete assente o repository sparito: va scritto nella tabella, non nascosto
        return dict(errore=type(e).__name__)
    tag = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")]
    carta = (d.get("cardData") or {}).get("license")
    if isinstance(carta, list):
        carta = ", ".join(carta)
    lingue = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("language:")]
    return dict(licenza_tag=", ".join(tag), licenza_scheda=carta or "", revisione=d.get("sha", ""),
                ultima_modifica=(d.get("lastModified") or "")[:10], gated=str(d.get("gated", False)),
                lingue_dichiarate=",".join(sorted(lingue)[:25]), errore="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/fase2/fonti_relazioni_licenze.csv")
    a = ap.parse_args()
    s1mb = set(json.loads(SOTTOINSIEMI.read_text())["sottoinsiemi"])
    righe = []
    for repo, serve, tipo, lingue, coincide in CANDIDATI:
        meta = api(repo)
        lic = (meta.get("licenza_tag") or "") + " " + (meta.get("licenza_scheda") or "")
        cl = classe(lic) if not meta.get("errore") else "sconosciuta"
        if coincide and coincide not in s1mb:
            raise SystemExit(f"{repo}: la coincidenza dichiarata «{coincide}» non è fra i sottoinsiemi di S1MB")
        if coincide:
            decisione, motivo = "escluso", f"coincide con la prova S1MB «{coincide}» (regola 1)"
        elif cl not in AMMESSE:
            decisione, motivo = "escluso", f"licenza «{lic.strip() or 'assente'}» in classe {cl} (regola 2)"
        else:
            decisione, motivo = "da usare", f"licenza {cl}, nessuna coincidenza con le prove di S1MB"
        righe.append(dict(repository=repo, tipo=tipo, a_cosa_serve=serve, lingue_utili=lingue,
                          classe_licenza=cl, decisione=decisione, motivo=motivo,
                          coincide_con_prova_s1mb=coincide, **meta))
    cols = ["repository", "tipo", "a_cosa_serve", "lingue_utili", "licenza_tag", "licenza_scheda",
            "classe_licenza", "decisione", "motivo", "coincide_con_prova_s1mb", "lingue_dichiarate", "gated",
            "revisione", "ultima_modifica", "errore"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in righe:
            w.writerow({k: r.get(k, "") for k in cols})
    riepilogo = dict(
        letto_il=__import__("time").strftime("%Y-%m-%d"),
        candidati=len(righe),
        da_usare=[r["repository"] for r in righe if r["decisione"] == "da usare"],
        esclusi_per_coincidenza=[r["repository"] for r in righe if r["coincide_con_prova_s1mb"]],
        esclusi_per_licenza=[r["repository"] for r in righe
                             if r["decisione"] == "escluso" and not r["coincide_con_prova_s1mb"]],
        per_tipo={t: sum(1 for r in righe if r["tipo"] == t and r["decisione"] == "da usare")
                  for t in ("relazione", "pertinenza", "score")},
        errori={r["repository"]: r["errore"] for r in righe if r["errore"]},
        nota="Classificazione automatica dalla stringa di licenza dichiarata sull'Hub: non è una consulenza "
             "legale, e le schede vanno lette anche a mano prima di usare una fonte.")
    json.dump(riepilogo, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
