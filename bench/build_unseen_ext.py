"""Test Noul esteso sulle intenzioni mai viste (fase 2).

Il test Noul della fase 1 ha solo 300 domande su intenzioni mai viste, cioè 60 frasi in 5 lingue: troppo poche per
misurare un AUROC con precisione. Qui si usano TUTTE le frasi del test MASSIVE dei 4 ambiti esclusi
(HELD_OUT_SCENARIOS), meno le 60 già presenti in test_noul.jsonl, così i due file sono disgiunti.

Ogni domanda riguarda un'intenzione mai vista (campo `intent` = intenzione della domanda, come in test_noul.jsonl);
il campo `gold` è l'intenzione vera della frase e `neg_type` dice il tipo di esempio:
- `pos`:       domanda sull'intenzione vera (una per frase degli ambiti esclusi);
- `neg_same`:  domanda su un'intenzione sorella dello stesso ambito (una per frase degli ambiti esclusi);
- `neg_other`: frase di un ambito visto con una domanda su un'intenzione mai vista; tante quante servono per avere
               la stessa proporzione del benchmark originale (negativi dello stesso ambito con p = 0,7).
Le stesse domande sono usate in tutte e 5 le lingue. Seed fisso (21).

uso: python bench/build_unseen_ext.py --raw data/massive [--download] --out bench/data
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_benchmark import DESC, LANGS, download, load  # noqa: E402
from skejja_measure import HELD_OUT_SCENARIOS, scenario  # noqa: E402

SEED = 21
P_SAME = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/massive")
    ap.add_argument("--out", default="bench/data")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download:
        download(a.raw)
    M = load(a.raw)
    T = M[M.split == "test"]
    en = T[T.lang == "en"].set_index("id").label_text
    old_ids = {json.loads(l)["id"] for l in open(os.path.join(a.out, "test_noul.jsonl"))}
    unseen = sorted(i for i in DESC if scenario(i) in HELD_OUT_SCENARIOS)
    by = collections.defaultdict(list)
    for i in unseen:
        by[scenario(i)].append(i)
    rng = np.random.default_rng(SEED)
    ho_ids = sorted((i for i, g in en.items() if g in unseen and i not in old_ids), key=int)
    seen_ids = sorted((i for i, g in en.items() if g not in unseen and i not in old_ids), key=int)
    items = []  # (id, question intent, label, neg_type)
    for uid in ho_ids:
        g = en[uid]
        items.append((uid, g, 1, "pos"))
        sib = [i for i in by[scenario(g)] if i != g]
        items.append((uid, str(rng.choice(sib)), 0, "neg_same"))
    n_other = int(round(len(ho_ids) * (1 - P_SAME) / P_SAME))
    for uid in rng.choice(seen_ids, size=n_other, replace=False):
        items.append((str(uid), str(rng.choice(unseen)), 0, "neg_other"))
    items.sort(key=lambda x: (int(x[0]), x[1]))
    d = T.set_index(["id", "lang"])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "test_noul_unseen_ext.jsonl")
    with open(path, "w") as f:
        for uid, q, lab, nt in items:
            for l in LANGS:
                f.write(json.dumps(dict(id=uid, lang=l, text=d.loc[(uid, l), "text"], intent=q, gold=en[uid],
                                        neg_type=nt, question=f"Does the user want to {DESC[q]}?", label=lab),
                                   ensure_ascii=False) + "\n")
    c = collections.Counter(nt for _, _, _, nt in items)
    print(path, "frasi mai viste:", len(ho_ids), "item per lingua:", len(items), dict(c))


if __name__ == "__main__":
    main()
