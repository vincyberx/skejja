"""Banco di costo del giro G: lunghezza dell'ingresso x numero di opzioni.

Legge i file JSON prodotti da `kit/bench_pi.py --cost` (su Raspberry Pi o su qualsiasi CPU) e scrive una tabella
sola con tutte le macchine, più la figura della curva di costo.

    python bench/costo_formato.py --in <file.json> [<altri.json> ...] \
        --macchina pi4 --out results/fase2/costo_lunghezza_opzioni_pi4.csv --fig results/fase2/costo_formato_pi4.png

`--macchina` è l'etichetta della macchina nella tabella (`pi4`, `pi3`, `mac`): i numeri del Mac sono
approssimazioni, valgono solo quelli misurati sulla scheda (regola di CLAUDE.md). I nomi host dei file del kit
non entrano nella tabella.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

COLONNE = ["macchina", "dispositivo", "modello", "variante", "thread", "arena", "token", "opzioni",
           "e2e_p50_ms", "e2e_p95_ms", "fwd_p50_ms", "rss_mb", "n", "esito"]


def leggi(percorsi, macchina):
    righe = []
    for p in percorsi:
        dati = json.loads(Path(p).read_text())
        sistema = dati.get("system", {})
        dispositivo = sistema.get("device_model", "")
        for r in dati.get("results", []):
            if "cost_point" not in r:
                continue
            esito = r.get("status_point") or r.get("status", "")
            righe.append({
                "macchina": macchina,
                "dispositivo": dispositivo,
                "modello": r.get("model", ""),
                "variante": "int8dyn" if "int8dyn" in r.get("model", "") else "int8",
                "thread": r.get("threads"),
                "arena": "off" if r.get("low_mem") else "on",
                "token": r.get("cost_tokens", r.get("cost_tokens_target")),
                "opzioni": r.get("cost_options"),
                "e2e_p50_ms": r.get("cost_e2e_p50_ms"),
                "e2e_p95_ms": r.get("cost_e2e_p95_ms"),
                "fwd_p50_ms": r.get("cost_fwd_p50_ms"),
                "rss_mb": r.get("peak_rss_mb"),
                "n": r.get("cost_n"),
                "esito": esito,
            })
    return pd.DataFrame(righe, columns=COLONNE)


def figura(df, percorso, limite_rss=300, limite_ms=400):
    """Due pannelli: latenza e RAM per lunghezza dell'ingresso. Colore = numero di opzioni, tratto = macchina,
    thread e arena. La legenda sta sotto i pannelli, fuori dall'area dei dati."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[df.esito == "ok"].copy()
    colori = {8: "#1f77b4", 20: "#d1711c", 50: "#8c2d2d"}
    tratti = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
    chiavi = sorted({(r.macchina, r.thread, r.arena) for r in ok.itertuples()})
    stile = {k: tratti[i % len(tratti)] for i, k in enumerate(chiavi)}

    fig, assi = plt.subplots(1, 2, figsize=(11, 4.6))
    for macchina, thread, arena in chiavi:
        for opzioni in sorted(ok.opzioni.unique()):
            sel = ok[(ok.macchina == macchina) & (ok.thread == thread) & (ok.arena == arena)
                     & (ok.opzioni == opzioni)].sort_values("token")
            if len(sel) < 2:
                continue
            for asse, colonna in zip(assi, ("e2e_p50_ms", "rss_mb")):
                asse.plot(sel.token, sel[colonna], marker="o", ms=4, color=colori.get(opzioni, "#444444"),
                          linestyle=stile[(macchina, thread, arena)], lw=1.4)
    assi[0].axhline(limite_ms, color="grey", lw=1, ls="--")
    assi[0].annotate(f"{limite_ms} ms, carico normale", (0.02, limite_ms), xycoords=("axes fraction", "data"),
                     va="bottom", fontsize=8, color="grey")
    assi[0].set(xlabel="token dell'ingresso (criteri compresi)", ylabel="latenza p50 (ms)", yscale="log",
                title="Costo per decisione")
    assi[1].axhline(limite_rss, color="grey", lw=1, ls="--")
    assi[1].annotate(f"limite {limite_rss} MB", (0.02, limite_rss), xycoords=("axes fraction", "data"),
                     va="bottom", fontsize=8, color="grey")
    assi[1].set(xlabel="token dell'ingresso (criteri compresi)", ylabel="RAM di picco (MB)", title="Memoria")
    for asse in assi:
        asse.grid(alpha=0.3)
        asse.set_xticks(sorted(ok.token.unique()))
        asse.set_xticklabels([str(t) for t in sorted(ok.token.unique())], fontsize=8)

    voci = [plt.Line2D([], [], color=colori.get(o, "#444444"), lw=1.6, label=f"{o} opzioni")
            for o in sorted(ok.opzioni.unique())]
    voci += [plt.Line2D([], [], color="#444444", lw=1.6, linestyle=stile[k],
                        label=f"{k[0]}, {k[1]} thread" + (", arena off" if k[2] == "off" else ""))
             for k in chiavi]
    fig.legend(handles=voci, loc="lower center", ncol=min(len(voci), 4), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.3, wspace=0.22)
    fig.savefig(percorso, dpi=150)
    return percorso


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="ingresso", nargs="+", default=[], help="file JSON di kit/bench_pi.py --cost")
    ap.add_argument("--da-csv", default="", help="rifà solo la figura da una tabella già scritta")
    ap.add_argument("--macchina", default="", help="etichetta della macchina: pi4, pi3, mac")
    ap.add_argument("--out", default="", help="CSV da scrivere")
    ap.add_argument("--fig", default="", help="figura da scrivere (facoltativa)")
    ap.add_argument("--unisci", default="", help="CSV di un'altra macchina da unire nella tabella e nella figura")
    a = ap.parse_args()
    if a.da_csv:
        print("rifatta", figura(pd.read_csv(a.da_csv), a.fig or a.out))
        return
    if not a.ingresso or not a.out or not a.macchina:
        ap.error("servono --in <file.json>, --macchina e --out, oppure --da-csv <tabella.csv> con --fig")
    df = leggi(a.ingresso, a.macchina)
    if a.unisci:
        df = pd.concat([pd.read_csv(a.unisci), df], ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["macchina", "thread", "arena", "opzioni", "token"]).to_csv(a.out, index=False)
    print(f"scritto {a.out}: {len(df)} punti, {df.esito.value_counts().to_dict()}")
    ok = df[df.esito == "ok"]
    if len(ok):
        print(ok.pivot_table(index=["macchina", "thread", "arena", "opzioni"], columns="token",
                             values="e2e_p50_ms").round(0).to_string())
        print(ok.pivot_table(index=["macchina", "thread", "arena", "opzioni"], columns="token",
                             values="rss_mb").round(0).to_string())
    if a.fig:
        print("scritta", figura(df, a.fig))


if __name__ == "__main__":
    main()
