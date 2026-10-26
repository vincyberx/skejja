# Skejja — Fase 2: controlli sulle azioni di un agente

_7 ottobre 2026. Numeri da [metrics_agent_tldr.csv](../results/fase2/metrics_agent_tldr.csv), [metrics_agent_tldr_base.csv](../results/fase2/metrics_agent_tldr_base.csv) e [metrics_ood_v0.csv](../results/fase2/metrics_ood_v0.csv); Mac, PyTorch fp32, calibrazione del protocollo._

## Il problema

Nel test v0 ([fase2_ood_v0.md](fase2_ood_v0.md)) le domande sì/no sulle proprietà di un'azione ("cancella file?", "si connette a internet?", "usa privilegi di amministratore?", "installa software?", "è di sola lettura?", "pubblica codice?") erano quasi a caso: AUROC 0,64 per ft3. Il modello non aveva mai visto comandi shell né domande di questo tipo.

## Dati: tldr-pages con etichette per regole

- **Fonte:** [tldr-pages](https://github.com/tldr-pages/tldr) (CC BY 4.0, revisione fissata in `bench/build_agent_actions.py`), cioè comandi reali con descrizione in inglese e, per una parte, in italiano, spagnolo, tedesco e francese. Le pagine sono quelle `common`, `linux` e `osx`; i rimandi `tldr <comando>` sono scartati, così come 3 comandi identici a quelli del test v0.
- **Etichette:** regole deterministiche sul comando (funzione `labels`), senza testo generato. Una proprietà riceve 1 o 0 solo quando la regola è sicura, altrimenti resta senza etichetta. Esempi: per gli strumenti remoti o cloud (`ssh`, `aws`, `gh`, …) "cancella", "installa" e "pubblica" non sono decidibili; `docker run` non dice se installa qualcosa.
- **Divisione per programma** (hash della pagina): 23.789 esempi di addestramento, 2.964 di sviluppo, 2.928 di test. Test e addestramento non condividono mai lo stesso programma.
- **Controllo delle regole:**
  - sulle 158 coppie azione-proprietà etichettate del test v0 coincidono al 100% con le mie etichette a mano. Le une e le altre vengono dalla stessa persona, quindi questo non è un controllo indipendente;
  - lettura di circa 85 esempi del test tldr: corretti due errori sistematici (`tar t`, `git config`), il resto è coerente.
- **Domande:** `bench/data/agent/props.json`, 6 proprietà × 5 lingue × 3 formulazioni, scritte dall'agente. La prima formulazione inglese e italiana è identica a quella del test v0: **nel test v0 le domande sono già viste in addestramento, le azioni no.**
- **Testi in addestramento:** solo comando, solo descrizione (nella lingua della traduzione), oppure "descrizione: comando"; domanda nella lingua della descrizione con probabilità 0,6. Per "usa privilegi di amministratore" il comando c'è sempre.

## Giro D (`edge32k-ft4`)

Parte da ft3, con gli stessi dati del giro C più 12.000 esempi sulle azioni bilanciati per proprietà ed etichetta (`train_skejja2.py --agent_dir bench/data/agent`). Distillazione dallo zero-shot, come nei giri precedenti. 2.983 passi; checkpoint migliore al passo 1500, scelto con la media dello sviluppo CLINC (domande inglesi e native) e dello sviluppo tldr.

| | ft3 | ft4 |
|---|---|---|
| **Test v0 agente** (36 azioni scritte a mano, anche descritte in italiano), AUROC, domande it / en | 0,641 / 0,586 | **0,908 / 0,915** |
| … "connettersi a internet" / "usare privilegi di amministratore" / "pubblicare" | 0,39 / 0,75 / 0,70 | 0,88 / 1,00 / 1,00 |
| … "solo lettura" / "cancellare o sovrascrivere" / "installare" | 0,83 / 0,68 / 0,83 | 0,96 / 0,75 / 0,88 |
| Test tldr (programmi mai visti): solo comando, AUROC | 0,528 [0,478–0,576] | 0,996 [0,991–0,999] |
| Test tldr: solo descrizione inglese / nella lingua della traduzione | 0,400 / 0,413 | 0,880 / 0,805 |
| MASSIVE Choice, tutte / mai viste | 0,878 / 0,772 | 0,869 / 0,734 |
| MASSIVE sì/no mai viste (test esteso), AUROC | 0,839 | 0,829 |
| Test v0 domotica: Choice con opzioni in italiano / sì/no in italiano | 0,788 / 0,919 | 0,738 / 0,919 |

**Lettura.**
1. Sul test tldr con solo il comando, 0,996 misura soprattutto quanto il modello impara le regole: le etichette sono generate dalle regole stesse. Il numero credibile è quello del test v0 (0,91), che contiene anche azioni scritte a parole in italiano ("Svuota il cestino", "Carica la cartella delle foto su Google Drive"), con etichette scritte a mano.
2. Il costo è sulla Choice: −3,8 punti sulle intenzioni mai viste di MASSIVE e −5 punti nella domotica v0 in italiano. Per questo è stato lanciato il giro E (sotto).
3. "Cancellare o sovrascrivere file" resta la proprietà più debole: nel test v0 0,75 (ft5: 0,79), e sulle descrizioni tradotte del test tldr circa 0,31.

## Giro E (`edge32k-ft5`, nuovo candidato principale)

Come il giro D, ma con due differenze pensate per non perdere la Choice:
- il teacher è **ft3** e non lo zero-shot, con peso 0,5;
- la distillazione **non si applica agli esempi sulle azioni**, dove ft3 sbaglia (`--kd_agent 0`).

Checkpoint migliore al passo 2000, scelto con la stessa regola del giro D. Calibrazione per domanda con il runtime corretto (vedi [fase2_noul.md](fase2_noul.md)).

| | ft3 | ft4 (giro D) | **ft5 (giro E)** |
|---|---|---|---|
| Test v0 agente, AUROC, domande it / en | 0,641 / 0,586 | 0,908 / 0,915 | **0,932 / 0,943** |
| … "cancellare" / "internet" / "sudo" / "installare" / "solo lettura" / "pubblicare" (it) | 0,68 / 0,39 / 0,75 / 0,83 / 0,83 / 0,70 | 0,75 / 0,88 / 1,00 / 0,88 / 0,96 / 1,00 | 0,79 / 0,93 / 1,00 / 0,98 / 0,97 / 1,00 |
| Test v0 agente, ECE (it) | 0,081 | 0,058 | 0,058 |
| Test tldr: solo comando / descrizione inglese / descrizione nativa | 0,528 / 0,400 / 0,413 | 0,996 / 0,880 / 0,805 | 0,996 / 0,881 / 0,797 |
| MASSIVE Choice, tutte / mai viste | 0,878 / **0,772** | 0,869 / 0,734 | 0,873 / 0,731 |
| MASSIVE Choice con opzioni native | 0,848 | 0,845 | **0,855** |
| MASSIVE sì/no, tutte / viste | 0,964 / 0,983 | 0,957 / 0,982 | 0,964 / 0,984 |
| Sì/no mai viste, test esteso: AUROC / ECE senza esempi | **0,839** / 0,143 | 0,829 / 0,146 | 0,832 / 0,156 |
| … con 32 / 64 esempi per domanda: AUROC / ECE | 0,890 / 0,048 · **0,904 / 0,032** | 0,875 / 0,039 · 0,894 / 0,035 | 0,874 / 0,043 · 0,895 / 0,054 |
| Test v0 domotica: Choice it / sì/no it | 0,788 / 0,919 | 0,738 / 0,919 | 0,788 / 0,904 |
| Test v0 email: Choice it / sì/no it | 0,650 / 0,768 | 0,575 / 0,766 | 0,650 / 0,784 |

Differenze appaiate della Choice rispetto a ft3 ([metrics_diff_choice_vs_ft3.csv](../results/fase2/metrics_diff_choice_vs_ft3.csv)): ft5 sul totale −0,006 [−0,018; +0,007], sulle mai viste −0,041 [−0,090; +0,003]; ft4 sulle mai viste −0,038 [−0,093; +0,014].

**Decisione: ft5 è il nuovo candidato principale.** Regola: non deve perdere in modo significativo su nessuna misura, e deve guadagnare nettamente su almeno una. ft5 aggiunge i controlli sulle azioni (da 0,64 a 0,93 nel test v0) e recupera la Choice in italiano che ft4 perdeva. Restano due perdite non significative ma da tenere d'occhio:
- **Choice sulle intenzioni mai viste** (−4 punti, intervallo che tocca lo zero);
- **calibrazione per domanda:** con 64 esempi AUROC 0,895, contro 0,904 di ft3, cioè appena sotto l'obiettivo 0,90; ECE 0,054.

**ft3** resta disponibile per chi usa solo Choice e sì/no su intenzioni nuove, senza controlli sulle azioni.

## Quantizzazione per il Raspberry (ft5)

Sul Pi 3 e sul Pi 4, che non hanno le istruzioni dot-product, il runtime sceglie l'int8 dinamico (fase 1, [results/pi3/ANALISI.md](../results/pi3/ANALISI.md)). Su ft5 l'int8 dinamico completo **perde 9 punti di Choice sulle intenzioni mai viste**. Lasciando in fp32 la proiezione d'uscita dell'MLP (`mlp.Wo`, 22 MatMul), il modulo già individuato nella fase 1 come il più sensibile, la perdita scende a meno di 1 punto. Test completo, batch 1:

| | Choice tutte | Choice mai viste | sì/no tutte | sì/no mai viste (test esteso) |
|---|---|---|---|---|
| ft5, PyTorch fp32 | 0,873 | 0,731 | 0,964 | 0,832 |
| ft5, int8 solo pesi | 0,872 | 0,738 | 0,963 | 0,832 |
| ft5, int8 dinamico completo | 0,849 | **0,638** | 0,952 | 0,819 |
| **ft5, int8 dinamico con `mlp.Wo` in fp32 (nel bundle)** | 0,871 | 0,724 | 0,963 | 0,829 |
| ft3, int8 dinamico completo (confronto) | 0,864 | 0,721 | 0,951 | 0,824 |

Fonti: [metrics_edge32k-ft5_int8.csv](../results/fase2/metrics_edge32k-ft5_int8.csv), [metrics_edge32k-ft5_int8dyn.csv](../results/fase2/metrics_edge32k-ft5_int8dyn.csv), [metrics_edge32k-ft5_int8dyn-noWo.csv](../results/fase2/metrics_edge32k-ft5_int8dyn-noWo.csv), [metrics_edge32k-ft3_int8dyn.csv](../results/fase2/metrics_edge32k-ft3_int8dyn.csv). Verifica del bundle contro PyTorch, su 300+300 item ([export_check_edge32k-ft5.json](../results/fase2/export_check_edge32k-ft5.json)): accordo Choice 0,987 e sì/no 0,993, contro 0,907 e 0,940 dell'int8 dinamico completo.

- `training/export_bundle.py` ora lascia `mlp.Wo` in fp32 nell'int8 dinamico per default (`--dyn_exclude`). Il file si chiama sempre `*.int8dyn.onnx`, quindi il runtime lo sceglie da solo sulle CPU senza dot-product.
- **Costo misurato sul Pi 3** (giro 3, 8 ottobre, 1 thread, [results/pi3/ANALISI.md](../results/pi3/ANALISI.md#giro-3-ft5-contro-ft32k-8-ottobre)): Choice 991 ms e sì/no 322 ms, contro 908 e 288 ms di ft32k con l'int8 dinamico completo; RAM 215 MB (+29 MB). La stima fatta prima, partendo dal 23% delle moltiplicazioni di matrici che passa per `mlp.Wo`, era circa 990 ms. Sul Pi le risposte coincidono con quelle del Mac.
- Il bundle ft3 ha ancora l'int8 dinamico completo (Choice sulle mai viste 0,721).
- **Problema dell'export corretto il 7 ottobre:** `onnx.save` con dati esterni *accoda* a un file `.onnx.data` già esistente. Rieseguendo l'export nella stessa cartella i file raddoppiavano (modelli corretti, ma due volte più grandi). Ora ogni file si scrive in una cartella nuova e poi sostituisce quello vecchio.

## Limiti

- Le etichette di addestramento e di test tldr vengono dalle stesse regole; il test v0 è scritto dalla stessa persona che ha scritto le regole. Per un numero citabile servono azioni reali di un agente, etichettate da un'altra persona.
- Le 6 domande sono fisse: il modello impara 6 controlli specifici, non a rispondere a qualunque domanda su un'azione.
- "Pubblicare" ha solo 36 esempi positivi in tutto il dataset, nessuno nel test tldr.

## Riproduzione

```sh
python bench/build_agent_actions.py --tldr data/tldr --download --out bench/data/agent
python training/train_skejja2.py --init releases/checkpoints/run_C/best --out releases/checkpoints/run_D \
    --teacher releases/checkpoints/edge32k-zs --kd_w 0.2 --threads 10 --lr 2e-5 --head_lr 6e-5 \
    --p_desc_native 0.5 --agent_dir bench/data/agent --n_agent 12000
python training/train_skejja2.py --init releases/checkpoints/run_C/best --out releases/checkpoints/run_E \
    --teacher releases/checkpoints/run_C/best --kd_w 0.5 --kd_agent 0 --threads 10 --lr 2e-5 --head_lr 6e-5 \
    --p_desc_native 0.5 --agent_dir bench/data/agent --n_agent 12000
python training/eval_agent.py --model releases/checkpoints/edge32k-ft --name edge32k-ft --model releases/checkpoints/run_C/best --name edge32k-ft3 \
    --model releases/checkpoints/run_D/best --name edge32k-ft4 --model releases/checkpoints/run_E/best --name edge32k-ft5 \
    --out results/fase2/metrics_agent_tldr.csv
python training/export_bundle.py --ckpt releases/checkpoints/run_E/best --name edge32k-ft5 --out_dir releases \
    --calib_ext bench/data/calib_noul_ext.jsonl
```
