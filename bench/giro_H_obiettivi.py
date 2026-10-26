"""Obiettivi, ricetta e regola del giro H, fissati PRIMA di addestrare (16 ottobre 2026).

Il giro H ripete il giro G con una sola correzione di sostanza, quella diagnosticata nel §6 di
docs/fase2_giro_G.md: il tetto unico sui dati di ampiezza aveva ridotto le famiglie del giro F a circa un terzo.
Qui le famiglie del giro F entrano tutte, come nel giro F, e le fonti nuove del giro G hanno un tetto per fonte.

L'insieme di guardia, il test, le tolleranze e i limiti rigidi sono QUELLI del giro G, letti da
results/fase2/giro_G_obiettivi.json e non riscritti. Cambia solo la condizione di guadagno (motivata sotto).

    python bench/giro_H_obiettivi.py            # scrive results/fase2/giro_H_obiettivi.json
    python bench/giro_H_obiettivi.py --check    # verifica che guardia e regola coincidano con il giro G
"""

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "fase2"
G = RES / "giro_G_obiettivi.json"
OUT = RES / "giro_H_obiettivi.json"

NUOVO = "edge32k-ft8"

# tetti per fonte sulle richieste del giro G (chiave = tipo/origine, come in broad_stats del trainer);
# le fonti non elencate entrano tutte. Valori del giro G fra parentesi nel commento.
TETTI = {
    "noul/circa": 4000,        # 16.000 nel giro G
    "choice/circa": 3000,      # 8.000
    "noul/mfaq": 1500,         # 6.483
    "noul/germanquad": 1500,   # 4.500
    "noul/piaf": 1500,         # 4.500
    "noul/miracl": 1500,       # 1.719
    "choice/casa": 6000,       # 12.000 (Choice con 11-49 opzioni)
}

RICETTA = {
    "init": "run_F/best (edge32k-ft6), come nel giro G",
    "teacher_skejja": "run_C/best, distillazione solo su MASSIVE (kd_w 0,5, kd_agent 0, kd_broad 0)",
    "iperparametri": "lr 2e-5, head_lr 6e-5, p_desc_native 0,5, n_agent 12.000, n_choice 24.000, n_noul_anchor "
                     "24.000, max_len 1.024, bs 32, token_budget 16.384, lambda_score 0,5, seed 0: identici al giro G",
    "dati_giro_F": "teacher_trial_in.jsonl + requests_synth.jsonl con le etichette Qwen3.5-9B, TUTTI, come nel giro F "
                   "(30.114 Choice e 37.135 sì/no di ampiezza). Il giro G usava requests_trial.jsonl, che non ha le "
                   "2.261 varianti di ordine e di lingua del giro F: qui tornano",
    "dati_giro_G": "richieste_giroG_pulite.jsonl con i tetti per fonte qui sotto; stati lunghi (lungo/fineweb, "
                   "lungo/fineweb2) e Score tutti",
    "tetti_per_fonte": TETTI,
    "tetti_per_tipo": "nessuno (n_broad_choice, n_broad_noul, n_broad_score = -1)",
    "modifiche_tecniche": [
        "le sì/no di ampiezza vanno in batch propri con tetto di 16.384 token (--bnoul_budget), invece che nei batch "
        "sì/no da 32 con padding sul più lungo: nel giro G un solo stato lungo portava a 1.024 token un batch di 32",
        "il tetto di token conta i token dopo il padding, e gli esempi sono ordinati per lunghezza dentro blocchi "
        "casuali di 100 batch",
        "il teacher Skejja gira solo dove la sua uscita entra nella perdita (Choice e sì/no di MASSIVE): stessa "
        "perdita, meno calcolo",
        "il giro va portato a termine: il tetto di tempo del lavoro si fissa dalla prova di ritmo con almeno il 50% "
        "di margine, e se il ritmo misurato non lo permette si accorcia la ricetta PRIMA di partire, non a metà",
    ],
    "selezione": "come nel giro G: checkpoint migliore sulla media di selezione dello sviluppo",
}

CONDIZIONE = {
    "necessaria": "nessuna perdita oltre le tolleranze del giro G (stesse 28 misure, stesso test, Holm dentro ogni "
                  "famiglia), limiti rigidi rispettati",
    "guadagno": "Score supportata su 21/21 prove di S1MB con zero decisioni fallite, E almeno uno fra: "
                "(i) Choice di MASSIVE con le opzioni nella lingua della frase, differenza appaiata ft8 − ft6 con "
                "intervallo al 95% sopra lo zero; (ii) punteggio della famiglia Score di S1MB ≥ quello di ft6 + 1 "
                "punto, ft6 misurato con lo stesso adattatore e la stessa configurazione prima del verdetto; "
                "(iii) Choice o Noul di S1MB ≥ +5 punti (la condizione del giro G, che resta valida)",
    "perche_cambia": "il giro G ha mostrato che con questi dati S1MB sale di 1-2 punti, non di 5: con la condizione "
                     "del giro G il giro H sarebbe respinto per costruzione anche senza regressioni. Lo scopo del giro "
                     "H è tenere le capacità nuove (Score, stati lunghi, molte opzioni) senza perdere niente; la "
                     "condizione lo dice, ed è fissata qui prima di addestrare",
    "se_guadagno_manca": "ft8 resta alternativo e edge32k-ft6 resta principale; la fase 2 si chiude comunque",
    "riserve": "una perdita ammessa dalle tolleranze va dichiarata come riserva, come nel giro G",
}

AVVERTENZE = [
    "terzo candidato misurato sugli stessi test di guardia (ft6 nel giro F, ft7 nel giro G, ft8 qui): la "
    "probabilità di un falso guadagno cresce con i candidati; per questo la condizione di guadagno chiede un "
    "intervallo appaiato sopra lo zero o soglie fissate prima, e ogni banco si misura una volta sola",
    "la Choice con le opzioni nella lingua della frase è anche una misura di guardia della famiglia A: lo stesso "
    "numero può respingere (perdita significativa) o qualificare (guadagno significativo) il candidato",
    "il bootstrap è a 10.000 ricampionamenti come scritto nella regola del giro G; il confronto appaiato del giro G "
    "è stato eseguito con 1.000 (valore fisso in bench/compare_runs.py, senza opzione), contro i 10.000 dichiarati: "
    "va rifatto a 10.000 prima del verdetto del giro H, e il verdetto del giro G va riletto su quel rifacimento",
]


def costruisci():
    g = json.loads(G.read_text())
    h = copy.deepcopy(g)
    h["_about"] = ("Obiettivi, ricetta e regola del giro H, fissati il 16 ottobre 2026 prima di addestrare. Guardia, "
                   "test, tolleranze e limiti rigidi sono copiati senza modifiche da giro_G_obiettivi.json; cambiano "
                   "la ricetta dei dati e la condizione di guadagno. Generato da bench/giro_H_obiettivi.py.")
    h["giro"] = "H"
    h["fissato_il"] = "2026-10-16"
    h["candidato_previsto"] = NUOVO
    h["giro_precedente"] = {"giro": "G", "candidato": "edge32k-ft7", "esito": "respinto",
                            "documento": "docs/fase2_giro_G.md, §6"}
    h["ricetta"] = RICETTA
    h["condizione_di_guadagno"] = CONDIZIONE
    h["avvertenze"] = AVVERTENZE
    h["regola_di_non_regressione"] = dict(g["regola_di_non_regressione"])
    h["regola_di_non_regressione"]["test"] = g["regola_di_non_regressione"]["test"].replace("edge32k-ft7", NUOVO)
    return g, h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    g, h = costruisci()
    if a.check:
        s = json.loads(OUT.read_text())
        for k in ("misure_di_guardia", "famiglie_di_guardia", "limiti_rigidi", "separazione"):
            assert s[k] == g[k], f"{k} diverso dal giro G"
        assert s["regola_di_non_regressione"]["tolleranze"] == g["regola_di_non_regressione"]["tolleranze"]
        assert s["ricetta"]["tetti_per_fonte"] == TETTI
        print("ok: guardia, tolleranze, limiti e separazione coincidono con il giro G")
        return
    OUT.write_text(json.dumps(h, indent=1, ensure_ascii=False) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
