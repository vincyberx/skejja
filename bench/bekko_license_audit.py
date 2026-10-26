"""Licenze del dataset di addestramento di Bekko System One: cosa è utilizzabile sotto la nostra regola.

Il dataset `hotchpotch/bekko-system-one-dataset-v0` (6,59 milioni di casi, 153 sottoinsiemi attivi) non dichiara
una licenza a livello di repository: la scheda dice «No blanket relicensing is applied» e rimanda alla licenza di
ogni dataset a monte. Questo script scarica la scheda a una revisione fissata, legge la tabella delle fonti
generata e `license-audit.json`, e classifica i casi di addestramento per licenza dichiarata, segnalando:

- `tocca_banco`: sottoinsiemi che si sovrappongono al nostro banco di prova (MASSIVE e i dataset dell'articolo
  su Jev), che non possono entrare nel nostro addestramento senza rendere «in dominio» le nostre misure;
- `derivato_jev`: righe acquisite da Open-Jev, dove le descrizioni delle domande vengono dalla documentazione di
  TypeSafe senza licenza verificata (regola del progetto: mai Jev);
- `generato_da_modello`: sottoinsiemi sintetici il cui autore è un modello (es. le prove di generalizzazione di
  S1MB, scritte con GPT-6-Astra, «license: not separately declared»).

uso: python bench/bekko_license_audit.py --out results/fase2/bekko_dataset_licenze.csv
"""
import argparse, csv, json, os, re, urllib.request

REPO = "hotchpotch/bekko-system-one-dataset-v0"
# Sottoinsiemi che toccano il nostro banco di prova: MASSIVE (il nostro test) e i dataset dell'articolo su Jev.
BANCO = re.compile(r"^(massive|minds14|clinc|banking77|agnews|laya__ag_news|synthetic_relevance__agnews|sms_spam|"
                   r"laya__enron_spam|laya__phishing|go_emotions|hwu64|mtop|snips|nlupp|sib)", re.I)
JEV = re.compile(r"open_jev|typesafe", re.I)
GENERATO = re.compile(r"^(s1mb-generalization|synthetic_relevance|laya__typed_decisions)|followir", re.I)
PERM_CCBY = re.compile(r"cc\s?by[\s-]?[1-4]|cc-by-[1-4]|attribution 4\.0", re.I)
PERM_CC0 = re.compile(r"cc0|public domain|unlicense", re.I)
PERM_SW = re.compile(r"apache|\bmit\b|bsd|oanc", re.I)
SA = re.compile(r"cc[\s-]?by[\s-]?sa", re.I)
RISTRETTA = re.compile(r"non-?commercial|research only|cc[\s-]?by[\s-]?nc|\bgpl\b|lgpl|openrail|afl-3|cdla|"
                       r"terms of use|not verified|unresolved", re.I)
SCONOSCIUTA = re.compile(r"unknown|not separately declared|no aggregate license", re.I)


def classe(lic):
    """Classe della licenza dichiarata a monte. L'ordine conta: prima i vincoli, poi i permessi."""
    if not lic.strip() or SCONOSCIUTA.search(lic):
        return "sconosciuta"
    if RISTRETTA.search(lic):
        return "ristretta"
    if SA.search(lic):
        return "CC BY-SA"
    if PERM_CC0.search(lic):
        return "CC0 o dominio pubblico"
    if PERM_CCBY.search(lic):
        return "CC BY"
    if PERM_SW.search(lic):
        return "Apache, MIT o BSD"
    return "sconosciuta"


def fetch(path, revision="main"):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}"
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")


def revisione():
    d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60))
    tag = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")]
    return d["sha"], (tag[0] if tag else None), d.get("lastModified", "")[:10]


def parse_tabella(testo):
    """Tabella «Config | fonte a monte | train | validation | test | licenza» generata nella scheda."""
    corpo = testo.split("<!-- BEGIN GENERATED SOURCE TABLE -->", 1)[1]
    righe = []
    for line in corpo.splitlines():
        if not line.startswith("|"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 6 or c[0].startswith("---") or c[0] == "Config":
            continue
        num = lambda s: int(re.sub(r"[^0-9]", "", s) or 0)
        nolink = lambda s: re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s).replace("<br>", " · ")
        righe.append(dict(config=c[0].strip("`"), fonte=nolink(c[1]), train=num(c[2]), validation=num(c[3]),
                          test=num(c[4]), licenza=nolink(c[5]),
                          url_fonte=" ".join(re.findall(r"\((https?://[^)]+)\)", c[1]))))
    return righe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="main", help="revisione del dataset (default: main, poi fissata nel CSV)")
    ap.add_argument("--out", default="results/fase2/bekko_dataset_licenze.csv")
    a = ap.parse_args()
    sha, tag_repo, modificato = revisione()
    rev = sha if a.revision == "main" else a.revision
    righe = parse_tabella(fetch("README.md", rev))
    audit = json.loads(fetch("license-audit.json", rev))
    stato = {}
    for s in audit["sources"]:
        u = (s.get("url") or "").rstrip("/")
        stato[u] = s.get("status", "")
    for r in righe:
        r["classe"] = classe(r["licenza"])
        r["tocca_banco"] = int(bool(BANCO.search(r["config"])))
        r["derivato_jev"] = int(bool(JEV.search(r["config"] + " " + r["licenza"])))
        r["generato_da_modello"] = int(bool(GENERATO.search(r["config"])))
        r["stato_audit"] = next((stato[u.rstrip("/")] for u in r["url_fonte"].split()
                                 if u.rstrip("/") in stato), "")
        r["usabile_per_noi"] = int(r["classe"] in ("CC BY", "CC0 o dominio pubblico", "Apache, MIT o BSD")
                                   and not r["tocca_banco"] and not r["derivato_jev"]
                                   and not r["generato_da_modello"] and r["train"] > 0)
    cols = ["config", "fonte", "train", "validation", "test", "licenza", "classe", "stato_audit",
            "tocca_banco", "derivato_jev", "generato_da_modello", "usabile_per_noi", "url_fonte"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(righe, key=lambda r: -r["train"]):
            w.writerow({k: r[k] for k in cols})
    tot = sum(r["train"] for r in righe)
    per_classe = {}
    for r in righe:
        v = per_classe.setdefault(r["classe"], dict(sottoinsiemi=0, casi_train=0))
        v["sottoinsiemi"] += 1
        v["casi_train"] += r["train"]
    riepilogo = dict(
        dataset=REPO, revisione=rev, licenza_dichiarata_nel_repository=tag_repo, ultima_modifica=modificato,
        fonti_nell_audit=audit.get("source_count"), audit_del=audit.get("checked_at"),
        stati_audit={s: sum(1 for x in audit["sources"] if x.get("status") == s)
                     for s in sorted({x.get("status", "") for x in audit["sources"]})},
        sottoinsiemi_in_tabella=len(righe), casi_train_totali=tot, per_classe=per_classe,
        casi_che_toccano_il_banco=sum(r["train"] for r in righe if r["tocca_banco"]),
        casi_derivati_jev=sum(r["train"] for r in righe if r["derivato_jev"]),
        casi_generati_da_modello=sum(r["train"] for r in righe if r["generato_da_modello"]),
        casi_usabili_per_noi=sum(r["train"] for r in righe if r["usabile_per_noi"]),
        sottoinsiemi_usabili_per_noi=sum(1 for r in righe if r["usabile_per_noi"]))
    out_json = os.path.splitext(a.out)[0] + "_riepilogo.json"
    json.dump(riepilogo, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
