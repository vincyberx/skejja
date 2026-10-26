# Skejja e gli altri System One: confronto da fonti pubbliche

_9 ottobre 2026. Fonti: documentazione ufficiale di TypeSafe, valutazioni indipendenti, schede dei modelli. Nessuna chiamata all'API di Jev (vedi "Perché nessuna misura diretta su Jev")._

**Sintesi.** Fra i System One di cui ci sono dati pubblici, Skejja è di gran lunga il più piccolo: 57M parametri contro 322M–27B. È anche l'unico con misure pubblicate su un Raspberry Pi. Jev è più forte in quasi tutto il resto: ampiezza dei compiti, contesto lungo, `Score`, molte opzioni. Sulle lingue molto diffuse, compreso l'italiano, è buono anche lui. Il nostro spazio è quindi **decisioni locali su dispositivi piccoli, in più lingue, con probabilità calibrate**, non l'accuratezza generale.

## Perché nessuna misura diretta su Jev

Il Master Customer Agreement di TypeSafe (sezione 2.3(b)) vieta di usare il servizio o le sue risposte per distillare o addestrare modelli che lo imitano, e per sviluppare, o facilitare lo sviluppo di, un prodotto simile o concorrente. Skejja è un prodotto concorrente: per questo non abbiamo chiamato l'API. I numeri di Jev qui sotto sono di TypeSafe o di terzi, misurati sui loro dati e non sul nostro benchmark: **nessun numero di questa pagina è un confronto alla pari con Skejja.** Lettura del contratto fatta da noi, non da un legale.

## Jev (TypeSafe AI)

**Cosa dichiara TypeSafe** (docs.typesafe.ai, annuncio del 15 settembre 2026):
- API solo in cloud (servizio negli Stati Uniti), pesi chiusi, architettura e dimensioni non pubblicate.
- Prezzo: 0,042 $ per milione di token in ingresso; l'uscita è gratis.
- Latenza: 70–500 ms end to end, «la maggior parte delle domande in circa 100 ms», misurati dalla costa ovest degli Stati Uniti.
- Primitive: `Choice` (fino a 255 opzioni), `Score` (livelli ordinati), `Noul`. Più domande nella stessa chiamata quasi senza costo aggiuntivo.
- Contesto: 64k token per richiesta, 32k per lo stato più la domanda più lunga. Ingresso solo testo: stringa, JSON o lista di testi.
- Lingue: l'inglese è la lingua principale; le altre sono accettate ma funzionano «non altrettanto bene».
- Debolezze ammesse ("jaggedness"): lettura letterale delle istruzioni, numeri e date, domande indirette, stati lunghi con dettagli irrilevanti, contenuti avversari, istruzioni in contraddizione con i criteri, tendenza a scegliere la prima opzione.
- `confidence` della Choice = (p_max − 1/n)/(1 − 1/n); la Noul restituisce solo la probabilità.
- Valutazioni: solo i "workflow eval", con risposte di LLM di frontiera come riferimento al posto di etichette vere. TypeSafe non pubblica ECE, diagrammi di affidabilità o Brier.

**Cosa trovano le valutazioni indipendenti:**

| Fonte | Risultato principale |
|---|---|
| Deußer, Sparrenberg, Sifa (arXiv 2609.37647; 37 dataset, 346.009 richieste, risposte pubblicate su Zenodo 10.5281/zenodo.23039006) | CLINC150 89,5% con 151 opzioni; Banking77 79,7%; Belebele: inglese 97,2%, mediana su 122 lingue 91,1%, in calo forte solo sulle lingue poco diffuse. ECE delle Choice 0,028 (aggregata); probabilità sì/no ben ordinate ma mal posizionate rispetto alla soglia 0,5. Latenza media 0,36 s per richiesta. Ruotare le opzioni non cambia l'accuratezza. |
| PriorBench (pre-registrata, dall'Europa) | Latenza minima di circa 430 ms, fissa; 800 giudizi in una chiamata in 985 ms. |
| Rassegna su dev.to (otto giorni di test) | Accuratezza alla pari con LLM di fascia media, 6,5–11,5 punti sotto la frontiera; Banking77 0,753–0,840 in otto prove indipendenti; calibrazione buona su compiti inglesi familiari, sbagliata fuori da essi. |
| Altri test pubblici | Phishing (2.000 email): 62,6% con una sola domanda, 95% scomponendo la domanda in cinque; Score: confidenza 0,9 con 48% di risposte giuste; rischio delle chiamate a tool di un agente: 91,7%, ECE 0,05–0,07 (solo 60 casi). |

## Gli altri System One aperti

| Modello | Parametri | Pesi | Dove gira | Latenza pubblicata | Lingue | Note |
|---|---|---|---|---|---|---|
| **Skejja ft5** | **57M** (101–124 MB int8) | Apache 2.0 | **CPU, anche Raspberry Pi 3** | Mac 12 ms (Choice, 4 thread); **Pi 3, 1 core: sì/no 322 ms, Choice 991 ms; RAM 215 MB** | en, it, es, de, fr (addestrato e misurato) | ingressi fino a 192 token in addestramento; niente `Score` |
| Laya multilingue (Convai) | 322M (mmBERT-base) | Apache 2.0 | CPU/GPU | 33–40 ms su GPU T4 (checkpoint non specificato) | multilingue | da noi su MASSIVE: Choice 0,593, mai viste 0,586; RAM circa 1,3 GB (harness); contesto 1.024 token |
| Laya inglese / typed-decisions | 421M (ModernBERT-large) | Apache 2.0 | CPU/GPU | — | inglese | contesto 512 / 1.024 token |
| Julia 1 (SupersonicLabs, aggiunto il 22 ottobre) | 144M (mmBERT-small) | Apache 2.0 | CPU/GPU | da noi sul Mac: Choice 25 ms a 1 thread, RAM circa 1,2 GB | multilingue | da noi su MASSIVE: Choice 0,514, sì/no AUROC 0,503 (caso); contesto 8.192 token, Score, 2–20 opzioni; [fase3_julia.md](fase3_julia.md) |
| GLiNER2.5-Decide (Fastino) | 340M (DeBERTa-v3-large) | Apache 2.0 | CPU/GPU | 167 ms su Xeon a 48 vCPU; 38–47 ms su GPU | inglese (variante per il tedesco) | 60,2% sul benchmark del produttore |
| Kev (Jared Palmer) | 0,8B–27B (Qwen) | Apache 2.0 | GPU, Mac | 0,8B: circa 110 ms su CPU desktop, circa 1 GB di RAM; 2 s su M4 | — | stessa API di Jev; contesto fino a 65k |
| pplx-decider (Perplexity), Solar Decide (Upstage), CLM-8B | — | vari | API o GPU | — | — | non analizzati qui |

Le cifre degli altri modelli vengono dai rispettivi autori o da test di terzi: non sono confrontabili fra loro né con le nostre.

## Dove Skejja è più forte (oggi, con prove)

1. **Dimensioni e hardware.** Ha 6 volte meno parametri di GLiNER2.5-Decide e Laya e 14 volte meno di Kev-0.8B. È l'unico con misure pubblicate su Raspberry Pi: RAM 215 MB sul Pi 3, sotto l'obiettivo di 300 MB. Gli altri dichiarano CPU da server, GPU o Mac.
2. **Locale e aperto.** Niente rete, niente costo per chiamata, niente dati che escono dalla macchina. La latenza non include i 100–430 ms di rete di Jev. Su un Mac una Choice richiede 12 ms, meno dei soli tempi di rete di Jev.
3. **Lingue europee su un modello piccolo.** Misurato su 5 lingue, con un divario fra inglese e lingua peggiore di 3,3 punti (ft5). Rispetto agli altri modelli aperti piccoli è un vantaggio reale: GLiNER2.5-Decide è solo inglese, e su MASSIVE Laya multilingue ha 11,5 punti di divario. **Rispetto a Jev non lo è necessariamente:** sulle lingue molto diffuse Jev resta sopra il 90%.
4. **Calibrazione per domanda.** Con 64 esempi etichettati: ECE 0,032 e AUROC 0,904 sulle intenzioni mai viste. Jev non si può adattare: le sue soglie vanno tarate a mano, come mostra l'arXiv sul sì/no.

## Dove Skejja è più debole

1. **Contesto corto:** addestrato su ingressi fino a 192 token, contro 1.024 di Laya e 64k di Jev. Per gli agenti, che passano log, JSON e conversazioni, è il limite più serio.
2. **Niente `Score`**, che invece hanno Jev, Laya, Kev e GLiNER2.5-Decide.
3. **Poche opzioni provate:** misurato con 8 opzioni; Jev ne gestisce 151 su CLINC150 con 89,5%.
4. **Ampiezza:** misurata dal 9 ottobre sui dataset dell'articolo su Jev (sezione sotto) e dal 13 ottobre su S1MB ([fase2_ampiezza.md](fase2_ampiezza.md)). Con ft6 la classificazione di testi generici migliora molto, ma ragionamento, implicazione fra due testi e Choice con molte opzioni restano al livello del caso.
5. **Choice sulle intenzioni mai viste** 0,73 e ECE delle Choice di ft5 0,047–0,050.

## Confronto sui dataset dell'articolo arXiv (9 ottobre)

Skejja valutato zero-shot sugli stessi dataset e split dell'articolo arXiv 2609.37647, confrontato con i suoi numeri **aggregati** (Tabella 3). Non abbiamo chiamato l'API di Jev né usato le risposte di Jev pubblicate su Zenodo, che non hanno una licenza dichiarata. Script: [bench/eval_jev_suite.py](../bench/eval_jev_suite.py); risultati: [metrics_jev_suite.csv](../results/fase2/metrics_jev_suite.csv) (con intervalli bootstrap al 95% ed ECE); numeri dell'articolo: [metrics_jev_suite_paper.json](../results/fase2/metrics_jev_suite_paper.json). Modelli Skejja in ONNX fp32 sul Mac; GLiClass di partenza = GLiClass edge originale (vocabolario 256k).

| Dataset | Jev | Qwen3.8-27B | Gemma-4-E4B | GLiClass di partenza | Skejja ft (fase 1) | ft3 | ft5 | **ft6** (14 ott.) |
|---|---|---|---|---|---|---|---|---|
| AG News (4 argomenti) | 0,885 | 0,867 | 0,862 | 0,776 | 0,721 | 0,729 | 0,716 | **0,782** |
| Rotten Tomatoes (sentiment) | 0,933 | 0,932 | 0,901 | 0,758 | 0,673 | 0,643 | 0,682 | **0,731** |
| SST-2 (sentiment) | 0,964 | 0,954 | 0,952 | 0,799 | 0,737 | 0,697 | 0,711 | **0,807** |
| Emotion (6 emozioni) | 0,585 | 0,571 | 0,569 | 0,474 | 0,375 | 0,307 | 0,276 | **0,358** |
| Financial PhraseBank (3) | 0,730 | 0,677 | 0,652 | 0,497 | 0,333 | 0,385 | 0,442 | **0,472** |
| Language ID (20 lingue) | 0,996 | 0,998 | 0,987 | 0,056 | 0,034 | 0,032 | 0,040 | **0,208** |
| SMS Spam (sì/no, F1) | 0,938 | 0,940 | 0,902 | 0,375 | 0,000 | 0,134 | 0,092 | **0,749** |
| Banking77 (77 intenzioni) ⚠ | 0,797 | 0,774 | 0,670 | 0,324 | 0,383 | 0,570 | 0,581 | **0,575** |
| SIB-200, media en/it/es/de/fr (7 argomenti) | 0,815 su 205 lingue; 0,91–0,92 sulle lingue migliori | 0,800 (205 lingue) | 0,761 (205 lingue) | 0,719 | 0,702 | 0,729 | 0,729 | **0,740** |

**Aggiornamento del 14 ottobre: ft6 (giro F, dati di ampiezza etichettati da Qwen3.5-9B).** Misurato sulla GPU in PyTorch fp32, nello stesso giro di ft5; i numeri di ft5 coincidono con quelli del Mac. Differenze appaiate ft6 − ft5: AG News +6,6 punti, Rotten Tomatoes +4,9, SST-2 +9,6, Emotion +8,2, Language ID +16,8, SMS Spam F1 +0,66 (AUROC da 0,451 a 0,943), tutte con intervallo al 95% lontano da zero. Financial PhraseBank +3,0 e Banking77 −0,6 non sono significative. **Una perdita:** SIB-200 in inglese da 0,730 a 0,676 (−5,4 [−9,3; −1,5]), compensata nella media dalle altre lingue. Dettagli, test v0, MASSIVE e S1MB: [fase2_ampiezza.md, §6](fase2_ampiezza.md#6-giro-f-il-candidato-edge32k-ft6-1314-ottobre-2026); risultati: [metrics_jev_suite_ft6.csv](../results/fase2/metrics_jev_suite_ft6.csv), [metrics_jev_suite_diff_ft6_vs_ft5.csv](../results/fase2/metrics_jev_suite_diff_ft6_vs_ft5.csv).

**Differenze di metodo da tenere presenti.** Le descrizioni delle opzioni e la domanda sì/no le abbiamo scritte noi, una sola volta e prima dei risultati; l'articolo usa template propri, quindi il confronto non è sullo stesso testo di richiesta. Per Skejja abbiamo valutato SIB-200 solo in 5 lingue, mentre il numero di Jev è sulla media di 205 lingue. ⚠ Banking77 non è zero-shot per ft3 e ft5, che hanno visto in addestramento le frasi di training di Banking77 (dati sì/no della fase 2): il loro 0,57–0,58 va letto come «in dominio».

**Cosa dicono i numeri**
1. **Fuori dalle intenzioni Skejja è molto sotto Jev:** da 17 a 31 punti per ft5 su argomento, sentiment, emozioni e finanza. Su questi compiti ampi la dimensione conta: Qwen3.8-27B, circa 500 volte più grande di Skejja, è vicino a Jev.
2. **Il nostro fine-tuning ha fatto perdere ampiezza.** ft, ft3 e ft5 sono sotto il modello di partenza su AG News (−5–6 punti), sentiment (−6–12), emozioni (−10–20) e finanza (−5–16). Unica eccezione SIB-200: ft3 e ft5 salgono da 0,719 a 0,729 in media e migliorano in italiano e francese (0,676 → 0,755 e 0,716 → 0,755).
3. **La sì/no di Skejja è una sì/no sulle intenzioni dell'utente, non su un'affermazione qualsiasi.** Su SMS Spam la domanda va espressa come intenzione («receive spam: …»). Il modello di partenza ordina bene i messaggi (AUROC 0,768); ft5 scende a 0,451, sotto il caso, e con la soglia 0,5 trova pochi spam (F1 0,09). Jev invece valuta affermazioni generali («questo messaggio è spam?»). Per un System One generale è il limite più importante.
4. **Language ID non è un compito per Skejja** (0,03–0,06, al livello del caso, anche per il modello di partenza a 256k token). Un modello che confronta il significato fra testo e opzioni non riconosce la lingua da descrizioni come «Italian».
5. **Latenza:** sul Mac circa 50 ms per esempio con 2 thread e 4 processi in parallelo. Per Jev l'articolo misura 0,36 s per richiesta, rete compresa.

**Conclusione (9 ottobre).** Skejja era un modello **specializzato** in intenzioni e controlli, piccolo e veloce, non un System One generale. Per avvicinarsi a Jev servivano addestramento su molti tipi di compito, per non perdere ampiezza, e una sì/no su affermazioni generali, non solo su intenzioni.

**Dopo il giro F (14 ottobre).** Con ft6 i punti 2 e 3 sono in parte superati:
- su AG News e SST-2 ft6 torna sopra il modello di partenza, e la sì/no sullo spam funziona (AUROC 0,943);
- MASSIVE e i controlli sulle azioni restano come in ft5.

La distanza da Jev resta di 10–26 punti, e su S1MB, cioè ragionamento, implicazione e molte opzioni, ft6 resta al livello del caso (Noul 0,69, Choice 9,75 contro 42,91 e 51,62 di Bekko 68M).

## Cosa cambia per la ROADMAP

1. **Dati di addestramento di molti tipi** (argomento, sentiment, emozioni, spam, intenzioni, controlli, affermazioni sì/no generali, in 5 lingue), etichettati da un modello insegnante aperto: è la leva principale per l'ampiezza, e serve una GPU.
2. **Sì/no su affermazioni generali** («il testo è X?») accanto alla sì/no sulle intenzioni.
3. **Stati lunghi e JSON.** Obiettivo minimo: 512 token, con misure su Pi.
4. **`Score`** e **Choice con molte opzioni** (almeno 77), codificando le opzioni una volta sola.
5. **Test sull'ordine delle opzioni.** Costa poco, e Jev lo dichiara come proprio difetto.
6. **Posizionamento:** «il System One più piccolo, che gira su un Raspberry Pi, nelle lingue europee». Non «più accurato di Jev».

## Fonti

- TypeSafe: docs.typesafe.ai (models, primitives, confidence, model-jaggedness/jev-1.13, concepts/state); typesafe.ai/blog/introducing-system-one-models-and-jev; typesafe.ai/legal/mca (sezione 2.3). Consultati il 9 ottobre 2026.
- Deußer T., Sparrenberg L., Sifa R., *Evaluating and Benchmarking the System One Model Jev*, arXiv:2609.37647 (2026).
- PriorBench, github.com/priorbench/jev; github.com/ejs-5/jev-benchmark; github.com/themsquared/jev-benchmark; dev.to/gde (rassegna "Jev after eight days"); beri.net (phishing).
- Laya: huggingface.co/convaiinnovations; misure nostre in [fase1_report.md](fase1_report.md).
- GLiNER2.5-Decide: huggingface.co/fastino/GLiNER2.5-Decide e schede di terzi.
- Kev: github.com/jaredpalmer/kev; github.com/Zerolxgic/kev-decision-benchmark.
- Skejja: [fase2_agente.md](fase2_agente.md), [results/pi3/ANALISI.md](../results/pi3/ANALISI.md).
