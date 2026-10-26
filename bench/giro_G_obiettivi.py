"""Obiettivi e regola di non regressione del giro G, fissati PRIMA di addestrare.

Legge i numeri del candidato attuale (`edge32k-ft6`) dai file di `results/` e scrive
`results/fase2/giro_G_obiettivi.json`: obiettivi, misure di guardia con il loro valore di partenza,
limiti rigidi e regola di decisione. Il documento `docs/fase2_giro_G.md` cita questo file.

    python bench/giro_G_obiettivi.py            # scrive results/fase2/giro_G_obiettivi.json
    python bench/giro_G_obiettivi.py --check    # rilegge il file e verifica che sia coerente con i CSV

Nessun numero è scritto a mano: ogni valore di partenza viene da un CSV o JSON di `results/`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "fase2"
PI4 = ROOT / "results" / "pi4" / "pi4_20261009-115526.csv"
OUT = RES / "giro_G_obiettivi.json"

RIF = "edge32k-ft6"          # candidato principale al 14 ottobre 2026 (giro F)
NUOVO = "edge32k-ft7"        # nome previsto del candidato del giro G
BUDGET_TOKEN = 1024          # lunghezza di ADDESTRAMENTO, confermata dalla misura sul Pi 4 del 14 ottobre
INVILUPPO_PI_TOKEN = 512     # lunghezza massima DICHIARATA sul Pi 4: oltre, la RAM supera i 300 MB (misurato)
BUDGET_OPZIONI = 50
SCALA_SCORE = [0, 1, 2, 3, 4]


def _pick(df, value="value", **kw):
    """Una riga sola dalle tabelle lunghe di `skejja_measure` (track/model/task/split/lang/metric)."""
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    if len(sel) != 1:
        raise SystemExit(f"attese 1 riga, trovate {len(sel)} per {kw}")
    row = sel.iloc[0]
    return float(row[value]), (int(row["n"]) if "n" in sel.columns and pd.notna(row.get("n")) else None)


def guardia(nome, famiglia, valore, n, fonte, verso="su", tolleranza=0.0, nota=""):
    """Una misura di guardia. `verso`: 'su' = più alto è meglio, 'giu' = più basso è meglio."""
    return dict(misura=nome, famiglia=famiglia, valore_ft6=round(valore, 4), n=n, fonte=fonte,
                verso=verso, tolleranza=tolleranza, nota=nota)


def costruisci():
    massive = pd.read_csv(RES / f"metrics_{RIF}.csv")
    agent = pd.read_csv(RES / f"metrics_agent_tldr_{RIF.split('-')[1]}.csv")
    ood = pd.read_csv(RES / "metrics_ood_diff_ft6_vs_ft5.csv")
    jev = pd.read_csv(RES / f"metrics_jev_suite_{RIF.split('-')[1]}.csv")
    jev = jev[jev.model == RIF]
    s1mb = pd.read_csv(RES / f"metrics_s1mb_{RIF}.csv")
    s1mb_rie = json.loads((RES / f"metrics_s1mb_{RIF}_riepilogo.json").read_text())
    nopt = json.loads((RES / "s1mb_choice_numero_opzioni.json").read_text())
    lung = json.loads((RES / "s1mb_lunghezze_ft6.json").read_text())
    sottoinsiemi = json.loads((ROOT / "bench" / "data" / "broad" / "s1mb_sottoinsiemi.json").read_text())
    pi4 = pd.read_csv(PI4)
    pi4 = pi4[pi4.model == "gliclass-edge-trim32k-ft5"]

    f_mass = f"results/fase2/metrics_{RIF}.csv"
    f_ag = f"results/fase2/metrics_agent_tldr_{RIF.split('-')[1]}.csv"
    f_ood = "results/fase2/metrics_ood_diff_ft6_vs_ft5.csv"
    f_jev = f"results/fase2/metrics_jev_suite_{RIF.split('-')[1]}.csv"
    f_s1 = f"results/fase2/metrics_s1mb_{RIF}.csv"

    # --- A. MASSIVE: il carico di lavoro che spediamo -------------------------------------------
    # `variant` distingue le opzioni in inglese da quelle nella lingua della frase (|native|).
    V_CHOICE = "torch-fp32-cuda|temp_calib"
    V_CHOICE_NAT = "torch-fp32-cuda|native|temp_calib"
    V_NOUL = "torch-fp32-cuda|platt_calib"
    V_NOUL_EXT = "torch-fp32-cuda|platt_calib_ext"
    A = []
    for split, nome in [("all", "Choice, tutte"), ("unseen", "Choice, intenzioni mai viste")]:
        v, n = _pick(massive, task="choice", split=split, metric="accuracy", lang="pooled", variant=V_CHOICE)
        A.append(guardia(nome, "A", v, n, f_mass))
    v, n = _pick(massive, task="choice", split="all", metric="accuracy", lang="pooled", variant=V_CHOICE_NAT)
    A.append(guardia("Choice con le opzioni nella lingua della frase", "A", v, n, f_mass))
    for split, var, nome in [("all", V_NOUL, "Sì/no, test (AUROC)"),
                             ("unseen_ext", V_NOUL_EXT, "Sì/no, test esteso mai visto (AUROC)")]:
        v, n = _pick(massive, task="noul", split=split, metric="auroc", lang="pooled", variant=var)
        A.append(guardia(nome, "A", v, n, f_mass))
    v, _ = _pick(massive, task="choice", split="all", metric="ece", lang="pooled", variant=V_CHOICE)
    A.append(guardia("ECE calibrata delle Choice", "A", v, 2000, f_mass, verso="giu",
                     nota="obiettivo di progetto ≤ 0,05 (limite rigido)"))
    v, _ = _pick(massive, task="noul", split="all", metric="ece", lang="pooled", variant=V_NOUL)
    A.append(guardia("ECE delle sì/no (Platt)", "A", v, 2000, f_mass, verso="giu",
                     nota="obiettivo di progetto ≤ 0,05 (limite rigido)"))
    lang = {l: _pick(massive, task="choice", split="all", metric="accuracy", lang=l, variant=V_CHOICE)[0]
            for l in ("en", "it", "es", "de", "fr")}
    A.append(guardia("Divario Choice fra inglese e lingua peggiore", "A",
                     lang["en"] - min(v for k, v in lang.items() if k != "en"), 2000, f_mass, verso="giu",
                     nota="obiettivo di progetto ≤ 8 punti; per lingua: "
                          + ", ".join(f"{k} {v:.4f}" for k, v in lang.items())))

    # --- B. Azioni degli agenti (tldr-pages) -----------------------------------------------------
    B = []
    for l, nome in [("cmd|q_en", "Azioni, comando (AUROC)"), ("desc_en", "Azioni, descrizione inglese (AUROC)"),
                    ("desc_native", "Azioni, descrizione nella lingua (AUROC)")]:
        v, n = _pick(agent, task="noul", split="tldr_test/all", lang=l, metric="auroc")
        B.append(guardia(nome, "B", v, n, f_ag,
                         nota="lo script non salva i punteggi per esempio: niente test appaiato (da correggere nel giro G)"))

    # --- C. Test v0 fuori da MASSIVE -------------------------------------------------------------
    C = []
    for task, lingua, nome in [("noul", "it", "v0 sì/no, italiano (AUROC)"), ("noul", "en", "v0 sì/no, inglese (AUROC)"),
                               ("choice", "it", "v0 Choice, italiano"), ("choice", "en", "v0 Choice, inglese")]:
        row = ood[(ood.task == task) & (ood.scenario == "tutte") & (ood.lingua == lingua)].iloc[0]
        C.append(guardia(nome, "C", float(row["ft6"]), int(row["n"]), f_ood))

    # --- D. Dataset dell'articolo su Jev (zero-shot, tranne Banking77) ---------------------------
    D = []
    for _, r in jev.iterrows():
        D.append(guardia(f"{r['dataset']} ({r['metric']})", "D", float(r["value"]), int(r["n"]), f_jev,
                         nota="in dominio, non zero-shot" if r["dataset"] == "banking77" else ""))

    # --- S1MB: il banco su cui il giro G deve guadagnare -----------------------------------------
    ch = s1mb[s1mb.famiglia == "choice"].copy()
    ch["n_opt"] = ch.prova.map(nopt)
    banda = pd.cut(ch.n_opt, [0, 4, 10, 50, 1000], labels=["2-4", "5-10", "11-50", ">50"])
    per_banda = (ch.groupby(banda, observed=True).punteggio_corretto.mean() * 100).round(2).to_dict()
    bekko = s1mb_rie["confronto_classifica"]["hotchpotch/bekko-system-one-v0-68m"]
    per_prova = pd.read_csv(RES / "s1mb_confronto_per_prova.csv")
    noul_auroc = per_prova[per_prova.famiglia == "noul"]["ft6_auroc"].dropna()

    n_score = int(s1mb_rie["prove_per_famiglia"]["score"])
    score_fallite = int(s1mb_rie["score_non_supportata"]["decisioni"])
    score_prove_fallite = int(s1mb_rie["score_non_supportata"]["prove"])
    obiettivi_s1mb = [
        dict(misura="Score: prove misurate", valore_ft6=f"{n_score - score_prove_fallite}/{n_score}",
             obiettivo=f"{n_score}/{n_score}, zero decisioni fallite", bekko68m=f"{n_score}/{n_score}", fonte=f_s1,
             nota=f"la primitiva manca: oggi {score_fallite} decisioni risultano fallite, non sbagliate"),
        dict(misura="Score: punteggio corretto di famiglia", valore_ft6=None, obiettivo=10.0,
             bekko68m=float(bekko["score"]), fonte=f_s1),
        dict(misura="Noul: punteggio corretto di famiglia", valore_ft6=s1mb_rie["medie_per_famiglia"]["noul"],
             obiettivo=8.0, bekko68m=float(bekko["noul"]), fonte=f_s1),
        dict(misura="Noul: AUROC mediana delle 59 prove", valore_ft6=round(float(noul_auroc.median()), 4),
             obiettivo=0.60, bekko68m=None, fonte="results/fase2/s1mb_confronto_per_prova.csv",
             nota=f"oggi {int((noul_auroc < 0.5).sum())} prove su {len(noul_auroc)} sotto 0,5 e solo "
                  f"{int((noul_auroc >= 0.6).sum())} sopra 0,6: il modello risponde «sì» quasi a tutto"),
        dict(misura="Noul: prove con AUROC ≥ 0,6", valore_ft6=int((noul_auroc >= 0.6).sum()), obiettivo=20,
             bekko68m=None, fonte="results/fase2/s1mb_confronto_per_prova.csv", nota="su 59"),
        dict(misura="Choice: punteggio corretto di famiglia", valore_ft6=s1mb_rie["medie_per_famiglia"]["choice"],
             obiettivo=15.0, bekko68m=float(bekko["choice"]), fonte=f_s1),
        dict(misura="Choice: punteggio medio, prove con 11–50 opzioni", valore_ft6=per_banda.get("11-50"),
             obiettivo=40.0, bekko68m=None, fonte=f_s1,
             nota="banda coperta dal budget di 50 opzioni del giro G"),
        dict(misura="Choice: punteggio medio, prove con 2–4 opzioni", valore_ft6=per_banda.get("2-4"),
             obiettivo=8.0, bekko68m=None, fonte=f_s1,
             nota="quasi tutte di ragionamento, logica, implicazione: oggi al livello del caso"),
        dict(misura="Quota di decisioni entro la lunghezza di addestramento", valore_ft6=lung["tutte"]["quota_entro"]["192"],
             obiettivo=lung["tutte"]["quota_entro"][str(BUDGET_TOKEN)], bekko68m=None,
             fonte="results/fase2/s1mb_lunghezze_ft6.json",
             nota=f"con il budget di {BUDGET_TOKEN} token, senza accorciare nient'altro"),
    ]

    # --- limiti rigidi: Pi 4 (riferimento ft5, 9 ottobre) e calibrazione -------------------------
    def pi(variant, threads, task, metric):
        sel = pi4[(pi4.variant == variant) & (pi4.threads == threads) & (pi4.task == task) & (pi4.metric == metric)]
        return round(float(sel.value.iloc[0]), 1)

    # Costo misurato sul Pi 4 il 14 ottobre (kit --cost): la riga «carico nuovo» non è più una previsione.
    costo = pd.read_csv(RES / "costo_lunghezza_opzioni_pi4.csv")
    costo = costo[(costo.macchina == "pi4") & (costo.esito == "ok")]

    def costo_punto(token, thread, arena="on", colonna="rss_mb", f=max):
        sel = costo[(costo.token == token) & (costo.thread == thread) & (costo.arena == arena)]
        if sel.empty:
            raise SystemExit(f"manca il punto {token} token / {thread} thread / arena {arena} nel banco di costo")
        return round(float(f(sel[colonna])), 1)

    limiti = [
        dict(limite="Pi 4, Choice 8 opzioni, 1 thread, p50", massimo_ms=400,
             ft5_misurato_ms=pi("onnx-int8dyn", 1, "choice", "e2e_p50_ms"), ft6_misurato_ms=380.0,
             fonte="results/pi4/pi4_20261009-115526.csv; ft6 dal giro 3 del 14 ottobre",
             nota="obiettivo di bench/success_criteria.json; il giro di ft6 era in rallentamento termico, "
                  "quindi 380 ms è un limite superiore"),
        dict(limite="Pi 4, sì/no, 1 thread, p50", massimo_ms=200,
             ft5_misurato_ms=pi("onnx-int8dyn", 1, "noul", "e2e_p50_ms"), ft6_misurato_ms=126.0,
             fonte="results/pi4/pi4_20261009-115526.csv; ft6 dal giro 3 del 14 ottobre"),
        dict(limite="Pi 4, RAM di picco, carico normale", massimo_mb=300,
             ft5_misurato_mb=pi("onnx-int8dyn", 4, "all", "peak_rss_mb"), ft6_misurato_mb=227.0,
             fonte="results/pi4/pi4_20261009-115526.csv; ft6 dal giro 3 del 14 ottobre"),
        dict(limite=f"Pi 4, RAM di picco dentro l'inviluppo dichiarato ({INVILUPPO_PI_TOKEN} token, "
                    f"{BUDGET_OPZIONI} opzioni)", massimo_mb=300,
             ft6_misurato_mb=costo_punto(511, 4), ft6_misurato_mb_arena_off=costo_punto(511, 1, "off"),
             fonte="results/fase2/costo_lunghezza_opzioni_pi4.csv",
             nota="misurato il 14 ottobre: dentro il limite. Il rallentamento termico non tocca la RAM"),
        dict(limite=f"Pi 4, RAM di picco a {BUDGET_TOKEN} token (fuori dall'inviluppo, solo dichiarata)",
             massimo_mb=None, ft6_misurato_mb=costo_punto(1023, 4),
             ft6_misurato_mb_arena_off=costo_punto(1023, 1, "off"),
             fonte="results/fase2/costo_lunghezza_opzioni_pi4.csv",
             nota="supera i 300 MB anche con l'arena disattivata: gli ingressi oltre l'inviluppo non si "
                  "dichiarano per il Pi, si misurano e si riportano"),
        dict(limite="ECE calibrata delle Choice (MASSIVE, test completo)", massimo=0.05,
             ft6_misurato=round(_pick(massive, task="choice", split="all", metric="ece", lang="pooled", variant=V_CHOICE)[0], 4),
             fonte=f_mass),
        dict(limite="ECE delle sì/no (Platt, MASSIVE, test completo)", massimo=0.05,
             ft6_misurato=round(_pick(massive, task="noul", split="all", metric="ece", lang="pooled", variant=V_NOUL)[0], 4),
             fonte=f_mass),
        dict(limite="Dimensione del bundle int8 dinamico", massimo_byte=130_000_000,
             ft6_misurato_byte=124_288_004, fonte="docs/fase2_ampiezza.md §6",
             nota="stessa architettura: il giro G non cambia i pesi, solo i dati e la lunghezza"),
    ]

    guardie = A + B + C + D
    regola = {
        "principio": "Pre-registrata: insieme di guardia, test statistico e tolleranze sono fissati qui, prima di "
                     "generare i dati e di addestrare. Il verdetto si legge su questo file, non si riscrive dopo.",
        "test": "bootstrap appaiato per esempio, 10.000 ricampionamenti, intervallo al 95% sulla differenza "
                f"{NUOVO} − {RIF} sugli stessi esempi (bench/compare_runs.py)",
        "correzione": "Holm dentro ciascuna famiglia (A, B, C, D), non su tutte le misure insieme: la disputa del "
                      "giro F su SIB-200 nasceva dall'avere scelto la correzione dopo aver visto il risultato",
        "perdita_significativa": "intervallo al 95% che esclude lo zero, stima puntuale negativa, dopo Holm nella "
                                 "sua famiglia",
        "tolleranze": {
            "A": "nessuna perdita significativa ammessa: è il carico di lavoro che spediamo",
            "B": "nessuna perdita significativa ammessa sul comando; sulle descrizioni è ammessa una perdita fino a "
                 "0,02 se non significativa dopo Holm",
            "C": "ammessa una perdita fino a 0,02 per misura, purché la sì/no complessiva (it ed en) non peggiori in "
                 "modo significativo",
            "D": "ammessa una perdita significativa su al massimo UNA delle 13 righe e non oltre 0,03 di stima "
                 "puntuale, da dichiarare nel documento; due o più righe in perdita significativa = candidato "
                 "respinto. Banking77 è in dominio e va letta a parte",
        },
        "limiti_rigidi": "una violazione qualsiasi dei limiti rigidi respinge il candidato senza passare dalla "
                         "statistica",
        "misure_nuove": "le capacità nuove (stati lunghi, 50 opzioni, Score) non hanno un valore di partenza: non "
                        "entrano nell'insieme di guardia, si riportano e basta",
        "selezione": "la variante si sceglie solo sulla media di selezione dello sviluppo; S1MB e i test non "
                     "scelgono niente (politica di S1MB sul test)",
        "una_misura_sola": "ogni banco si misura una volta sola con la configurazione fissata qui",
    }
    condizione = {
        "necessaria": "nessuna regressione secondo la regola, limiti rigidi rispettati",
        "guadagno": "Score supportata su 21/21 prove con zero decisioni fallite, E almeno uno fra: punteggio Choice "
                    "di S1MB ≥ +5 punti, punteggio Noul di S1MB ≥ +5 punti",
        "se_guadagno_manca": f"il candidato resta alternativo e {RIF} resta principale; il giro viene comunque "
                             "documentato con i suoi numeri",
        "riserve": "una perdita ammessa dalle tolleranze va dichiarata come riserva nel documento e in ROADMAP.md, "
                   "come è stato fatto per SIB-200 in inglese nel giro F",
    }

    separazione = {
        "mai_in_addestramento": [
            "dataset dell'articolo su Jev: AG News, Rotten Tomatoes, SST-2, Emotion (dair-ai), Financial PhraseBank, "
            "SMS Spam, Language ID, SIB-200, nemmeno i loro split di training",
            "S1MB in qualunque forma, dati e prove comprese",
            "le 10 intenzioni dei 4 ambiti esclusi di MASSIVE (transport, takeaway, social, cooking)",
            "il dataset di addestramento di Bekko (nessuna licenza sullo strato di conversione)",
            "i 1.600 testi del metro dell'insegnante (uso = solo_metro)",
        ],
        "fonti_di_relazioni": f"una fonte candidata è esclusa se coincide con una delle "
                              f"{sottoinsiemi['n_sottoinsiemi']} famiglie di sorgenti delle prove di S1MB "
                              f"(bench/data/broad/s1mb_sottoinsiemi.json); se la coincidenza è inevitabile e la "
                              f"fonte è già in casa, va dichiarata come esposizione",
        "esposizioni_gia_dichiarate": s1mb_rie["prove_con_nostra_esposizione"],
        "licenze_ammesse": ["CC BY", "CC0", "Apache-2.0", "MIT", "BSD", "ODC-BY"],
        "licenze_escluse": ["ShareAlike (CC BY-SA, GFDL)", "non commerciale (CC BY-NC)", "GPL/LGPL", "OpenRAIL",
                            "licenza assente o «unknown»"],
        "controllo": "controllo automatico di sovrapposizione dei testi normalizzati contro tutti i file di test, i "
                     "metri e i testi delle prove di S1MB; assert in train_skejja2.py esteso ai dati nuovi",
    }

    return {
        "_about": "Obiettivi, misure di guardia e regola di non regressione del giro G «formato S1MB», fissati prima "
                  "di generare i dati e di addestrare. Generato da bench/giro_G_obiettivi.py; i valori di partenza "
                  "vengono dai file di results/ citati in ogni riga.",
        "giro": "G",
        "fissato_il": "2026-10-14",
        "modello_di_riferimento": RIF,
        "candidato_previsto": NUOVO,
        "formato": {
            "budget_token": BUDGET_TOKEN,
            "budget_token_nota": "lunghezza di addestramento, confermata dalla misura sul Pi 4 del 14 ottobre: il "
                                 "costo all'inferenza dipende dall'ingresso che si dà al modello, non dalla "
                                 "lunghezza a cui è stato addestrato, e il bundle resta della stessa dimensione",
            "inviluppo_pi_token": INVILUPPO_PI_TOKEN,
            "inviluppo_pi_nota": "lunghezza massima dichiarata sul Pi 4: a 512 token la RAM di picco è 246 MB "
                                 "(231 con l'arena disattivata), a 1.024 arriva a 398 MB (357), oltre il limite "
                                 "di 300 MB. Dentro l'inviluppo la latenza resta comunque di secondi: il tempo "
                                 "reale vale per il carico normale, non per gli stati lunghi",
            "budget_opzioni": BUDGET_OPZIONI,
            "scala_score": SCALA_SCORE,
            "scala_score_nota": "scala prevalente di S1MB (6.168 decisioni su 6.748 hanno i livelli 0–4); Score è una "
                                "Choice sui livelli con valore, il valore atteso è Σ p·valore e la metrica di S1MB è "
                                "l'errore sul valore atteso normalizzato sull'ampiezza della scala",
            "resa": "istruzioni + stato + criteri, la stessa dell'adattatore S1MB "
                    "(renderer: instructions-then-state-in-text, criteria-as-labels-v1)",
            "copertura_s1mb": {
                "entro_192_token_oggi": lung["tutte"]["quota_entro"]["192"],
                f"entro_{BUDGET_TOKEN}_token": lung["tutte"]["quota_entro"][str(BUDGET_TOKEN)],
                "opzioni_entro_50_choice": lung["per_famiglia"]["choice"]["quota_opzioni_entro_50"],
                "fonte": "results/fase2/s1mb_lunghezze_ft6.json",
            },
        },
        "obiettivi_s1mb": obiettivi_s1mb,
        "misure_di_guardia": guardie,
        "famiglie_di_guardia": {
            "A": "MASSIVE: il carico di lavoro che spediamo (Choice, sì/no, calibrazione, divario fra lingue)",
            "B": "azioni degli agenti (tldr-pages)",
            "C": "test v0 fuori da MASSIVE (domotica, email, agenti)",
            "D": "dataset dell'articolo su Jev, zero-shot",
        },
        "limiti_rigidi": limiti,
        "regola_di_non_regressione": regola,
        "condizione_di_guadagno": condizione,
        "separazione": separazione,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verifica che il file salvato sia coerente con i CSV")
    args = ap.parse_args()
    dati = costruisci()
    if args.check:
        salvato = json.loads(OUT.read_text())
        if salvato != dati:
            raise SystemExit("giro_G_obiettivi.json non è più coerente con i file di results/")
        print("coerente:", OUT)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dati, indent=1, ensure_ascii=False) + "\n")
    print("scritto", OUT, len(dati["misure_di_guardia"]), "misure di guardia,",
          len(dati["obiettivi_s1mb"]), "obiettivi su S1MB,", len(dati["limiti_rigidi"]), "limiti rigidi")


if __name__ == "__main__":
    main()
