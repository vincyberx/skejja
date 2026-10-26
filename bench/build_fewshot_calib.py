"""Pool per la calibrazione con pochi esempi per domanda (fase 2), dallo split VALIDATION di MASSIVE (mai dal test).

Simula l'uso reale: chi crea una domanda nuova fornisce qualche esempio etichettato (sì/no) e il runtime adatta la
calibrazione di quella domanda. Per ogni frase di validation dei 4 ambiti esclusi: un positivo (domanda sulla sua
intenzione) e un negativo fratello (stesso ambito); più negativi di altri ambiti (frasi di intenzioni viste) nella
stessa proporzione del test esteso. Stesso formato di test_noul_unseen_ext.jsonl. Seed fisso (41).

uso: python bench/build_fewshot_calib.py --raw data/massive --out bench/data
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_benchmark import DESC, LANGS, load  # noqa: E402
from skejja_measure import HELD_OUT_SCENARIOS, scenario  # noqa: E402

SEED, P_SAME = 41, 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/massive")
    ap.add_argument("--out", default="bench/data")
    a = ap.parse_args()
    M = load(a.raw); V = M[M.split == "validation"]
    en = V[V.lang == "en"].set_index("id").label_text
    unseen = sorted(i for i in DESC if scenario(i) in HELD_OUT_SCENARIOS)
    by = collections.defaultdict(list)
    for i in unseen:
        by[scenario(i)].append(i)
    rng = np.random.default_rng(SEED)
    ho = sorted((i for i, g in en.items() if g in unseen), key=int)
    seen_ids = sorted((i for i, g in en.items() if g not in unseen), key=int)
    items = []
    for uid in ho:
        g = en[uid]; items.append((uid, g, 1, "pos"))
        items.append((uid, str(rng.choice([i for i in by[scenario(g)] if i != g])), 0, "neg_same"))
    for uid in rng.choice(seen_ids, size=int(round(len(ho) * (1 - P_SAME) / P_SAME)), replace=False):
        items.append((str(uid), str(rng.choice(unseen)), 0, "neg_other"))
    items.sort(key=lambda x: (int(x[0]), x[1]))
    d = V.set_index(["id", "lang"])
    path = os.path.join(a.out, "calib_noul_fewshot.jsonl")
    with open(path, "w") as f:
        for uid, q, lab, nt in items:
            for l in LANGS:
                f.write(json.dumps(dict(id=uid, lang=l, text=d.loc[(uid, l), "text"], intent=q, gold=en[uid], neg_type=nt,
                                        question=f"Does the user want to {DESC[q]}?", label=lab), ensure_ascii=False) + "\n")
    print(path, "frasi:", len(ho), "item per lingua:", len(items), collections.Counter(x[3] for x in items))


if __name__ == "__main__":
    main()
