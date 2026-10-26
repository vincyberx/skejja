# Julia 1 contro Skejja ft6 sul benchmark MASSIVE (22 ottobre 2026)

Julia 1 (SupersonicLabs) è uscito il 24–26 settembre 2026 e nel confronto del 9 ottobre
([confronto_concorrenti.md](confronto_concorrenti.md)) mancava. È il concorrente più vicino a Skejja per
posizionamento: piccolo (144,3M parametri, su mmBERT-small), multilingue, gira su CPU, interfaccia Choice / Score / Noul.
Qui lo misuriamo sul nostro benchmark con lo stesso protocollo di ft6.

## Modello e licenza

- `SupersonicLabs/Julia-1` @ `a85b127321d580d65176c89ced8273f305745d85`; hash dei pesi verificato
  (`df853bf7…5b72`, uguale a quello di `provenance.json` e `inference-policy.json`).
- **Licenza Apache-2.0** per i pesi e il codice di inferenza (scheda del modello e `pyproject.toml`); la pipeline di
  addestramento non è pubblica. Non ridistribuiamo nulla di Julia, quindi `NOTICE` non cambia.
- Runtime: il pacchetto `julia` del repository, torch fp32 su CPU, transformers 5.0.x (richiesto dal modello).
- **Esposizione a MASSIVE:** il `provenance.json` del modello riporta una validazione su MASSIVE (scenari, 200 frasi per
  52 lingue). I dati di addestramento non sono pubblici: non si può escludere che il modello abbia visto frasi del nostro
  test. Se è successo, il risultato qui sotto è un limite superiore per Julia.

## Come è stato interrogato

Script: [baselines/eval_julia.py](../baselines/eval_julia.py) (qualità) e [baselines/lat_julia.py](../baselines/lat_julia.py)
(latenza e RAM). Le metriche usano le stesse funzioni di `training/eval_noul.py`, quindi le righe di
[metrics_julia-1.csv](../results/fase3/metrics_julia-1.csv) si confrontano una per una con
[metrics_edge32k-ft6.csv](../results/fase2/metrics_edge32k-ft6.csv).

- Choice: domanda "What does the user want to do?" (la stessa data a Laya in fase 1), opzioni = le 8 descrizioni
  inglesi; variante `native` con le descrizioni tradotte nella lingua della frase, come per ft6.
- Sì/no: domanda = il campo `question` del benchmark, opzioni `false`/`true`, cioè quello che fa l'API di Julia
  quando la domanda non ha criteri. Punteggio = logit di `true` meno logit di `false`.
- Calibrazione solo sui file `calib_*` (temperatura per le Choice, Platt per le sì/no); nessun addestramento.
- Controlli: le probabilità del nostro adattatore coincidono con quelle dell'API `predict` di Julia (Choice e sì/no);
  i logit in batch coincidono con quelli a una riga per chiamata (differenza massima 1,2e-6).
- Il runtime di Julia rimette i thread di torch a `JULIA_CPU_THREADS` (predefinito 4) quando carica il modello
  (`julia/cuda.py`): gli script impostano quella variabile. Una prima misura di latenza l'aveva ignorata ed è stata
  rifatta.

## Qualità (test MASSIVE, 400 frasi × 5 lingue)

| | Julia 1 (zero-shot) | Skejja ft6 |
|---|---|---|
| Choice, tutte le intenzioni | 0,514 | **0,873** |
| Choice, intenzioni mai viste da ft6 | 0,514 | **0,741** |
| Choice, opzioni nella lingua della frase | 0,468 | **0,849** |
| Choice, ECE dopo calibrazione | 0,059 | **0,048** |
| Sì/no, AUROC test | 0,503 (IC 95% 0,457–0,547) | **0,961** |
| Sì/no, AUROC test esteso mai visto | 0,522 (0,500–0,542) | **0,840** |

Per lingua (Choice, opzioni inglesi): Julia en 0,543, it 0,510, es 0,508, de 0,493, fr 0,515; ft6 0,893 / 0,875 /
0,878 / 0,850 / 0,868. Con le opzioni tradotte Julia perde 4,6 punti (ft6 2,4).

Il confronto giusto per un modello zero-shot è la colonna "mai viste" e le baseline della fase 1, anch'esse zero-shot
([methods_baseline.md](methods_baseline.md)): Laya multilingue 0,600 nelle Choice e AUROC 0,745 nelle sì/no,
GLiClass edge 0,547 e 0,698. **Julia 1 è sotto entrambe.**

### Sì/no al livello del caso

Le sì/no del benchmark hanno negativi difficili (intenzioni sorelle dello stesso ambito). Con la formulazione
documentata Julia non le distingue (AUROC 0,50). Diagnosi sul test (solo AUROC, che non dipende dalla calibrazione, e
nessuna scelta presa sul test), righe `ablazione_*` del CSV:

| opzioni della sì/no | AUROC |
|---|---|
| `false` / `true` (predefinito dell'API) | 0,503 |
| `no` / `yes` | 0,545 |
| "No, the user does not want this." / "Yes, the user wants this." | 0,553 |
| le opzioni predefinite di Laya | 0,556 |
| come Choice a due opzioni: "something else" contro la descrizione dell'intenzione | 0,699 |

Le descrizioni aiutano poco; anche il ripiego migliore (la Choice a due opzioni) resta sotto Laya e GLiClass e
lontano da ft6. Dopo Platt le probabilità di Julia stanno tutte vicino a 0,5, perciò l'ECE è basso (0,012) senza
che le sì/no siano utili.

## Latenza e RAM (Mac arm64, 10 core: approssimazione, non vale per il Pi)

[latenza_julia-1_ft6_mac.json](../results/fase3/latenza_julia-1_ft6_mac.json): 300 decisioni misurate dopo 10 di
riscaldamento, una per chiamata, processo nuovo per ogni misura.

| | Choice p50, 1 / 4 thread | sì/no p50, 1 / 4 thread | picco RAM |
|---|---|---|---|
| Julia 1, torch fp32 | 25,3 / 27,2 ms | 17,4 / 22,6 ms | circa 1.200 MB |
| ft6, onnx fp32 | 62,6 / 23,7 ms | 22,7 / 10,4 ms | circa 430 MB |
| ft6, onnx int8 | 34,5 / 14,8 ms | 11,7 / 6,5 ms | circa 320 MB |

A 1 thread sul Mac Julia è più veloce di ft6 fp32 nelle Choice (runtime diversi: torch contro onnxruntime). Il
peso conta di più per il Pi: 1,2 GB di RAM non entrano in un Pi 3 da 1 GB e sono 4–5 volte l'inviluppo di ft6
(216–246 MB sul Pi 4). Julia non è stato misurato su un Pi.

## Cosa se ne ricava

- Sul nostro benchmark (intenzioni di casa e servizi, opzioni sorelle, 5 lingue europee) Julia 1 non è un concorrente:
  Choice 0,51 contro 0,87, sì/no al livello del caso. I suoi numeri pubblicati riguardano compiti più facili
  (MASSIVE a 18 scenari, AG News, Emotion su campioni di 100) e non si trasferiscono a intenzioni fini.
- Il vantaggio dichiarato di Skejja ("piccolo, multilingue, su CPU modeste") regge anche contro Julia, che pesa
  550 MB in fp32 e usa circa 1,2 GB di RAM.
- Julia ha ciò che Skejja non ha: contesto fino a 8.192 token, Score, fino a 20 opzioni native. Su compiti generici
  (AG News, Emotion) i suoi numeri pubblicati sono sopra quelli di ft6 sui dataset dell'articolo su Jev, ma le misure
  non sono confrontabili (campioni e protocolli diversi): non le abbiamo rifatte.
