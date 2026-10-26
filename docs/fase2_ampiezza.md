# Skejja — Fase 2: ampiezza, fonti dei dati e misura su S1MB

_13 ottobre 2026. Numeri da [bekko_dataset_licenze.csv](../results/fase2/bekko_dataset_licenze.csv),
[bekko_dataset_licenze_riepilogo.json](../results/fase2/bekko_dataset_licenze_riepilogo.json),
[s1mb_riferimenti.csv](../results/fase2/s1mb_riferimenti.csv) e
[s1mb_riferimenti_riepilogo.json](../results/fase2/s1mb_riferimenti_riepilogo.json); licenze lette sull'Hub
e nelle schede dei dataset il 13 ottobre 2026. Questo documento raccoglie le decisioni prese **prima** di
etichettare e addestrare (§1–§5); i risultati del giro F, cioè del candidato `edge32k-ft6`, sono al §6
(14 ottobre 2026)._

Il punto di partenza è la conclusione di [confronto_concorrenti.md](confronto_concorrenti.md): fuori dalle
intenzioni ft5 è 17–31 punti sotto Jev e **sotto il proprio modello di partenza**, e la sua sì/no funziona solo
su intenzioni dell'utente, non su affermazioni qualsiasi. La leva è addestrare su molti tipi di compito. Questo
documento fissa: da dove vengono i testi, cosa non deve mai entrare nell'addestramento, cosa è utilizzabile del
dataset di Bekko, e come ci misuriamo sulla classifica pubblica S1MB.

## 1. Regola di separazione dal banco di prova

Il confronto con gli altri System One vale solo se resta **zero-shot**. Quindi:

1. **I dataset dell'articolo su Jev non entrano nell'addestramento, nemmeno con i loro split di training:**
   AG News, Rotten Tomatoes, SST-2, Emotion (dair-ai), Financial PhraseBank, SMS Spam, Language ID, SIB-200.
   Non basta escludere lo split di test: addestrare sul training dello stesso dataset renderebbe la riga «in
   dominio», come è già successo a **Banking77** (unica riga già dichiarata tale, vedi l'avviso ⚠ nel confronto).
   I dati nuovi per argomento, sentiment, emozioni e spam vengono quindi da corpora **diversi** da questi.
2. **MASSIVE**: resta la regola della fase 1. Le 10 intenzioni dei 4 ambiti esclusi (`transport`, `takeaway`,
   `social`, `cooking`) non compaiono mai, né come frasi né come domande o opzioni; l'assert in
   `train_skejja2.py` va esteso ai dati nuovi.
3. **S1MB** è un banco di sola prova: non entra nell'addestramento in nessuna forma. Le sue sei prove di
   generalizzazione sono scritte da un modello commerciale (GPT-6-Astra) e la loro licenza è «not separately
   declared»: non sono riusabili nemmeno volendo.
4. **Eccezione per misurare l'insegnante.** Gli split di training dei dataset al punto 1 si possono usare
   *soltanto* come metro per stimare la qualità delle etichette dell'insegnante (accordo con l'etichetta vera),
   mai come dati di Skejja. Il pool di addestramento passa un controllo automatico di sovrapposizione dei testi
   (normalizzati) contro tutti i file di test e contro questi metri.

## 2. Fonti dei testi (licenze verificate il 13 ottobre 2026)

L'insegnante fornisce le **etichette**; i corpora forniscono solo il **testo**. Così la licenza che conta è
quella del testo, e i tipi di compito li scegliamo noi invece di ereditarli da un dataset.

| Fonte | Licenza dichiarata | Lingue | A cosa serve |
|---|---|---|---|
| [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (una scheggia per lingua) | ODC-BY | it, es, de, fr | testo web vario: notizie, recensioni, messaggi promozionali (serve per la sì/no generale e per lo spam) |
| [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (campione) | ODC-BY | en | come sopra, per l'inglese (fineweb-2 non ha l'inglese) |
| [aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Apache-2.0 | 70, fra cui le nostre 5 | richieste e messaggi scritti da persone |
| [mfaq](https://huggingface.co/datasets/clips/mfaq) | CC0-1.0 | 21, fra cui le nostre 5 | domande vere di assistenza, molti ambiti |
| MASSIVE (già in casa) | CC BY 4.0 | le nostre 5, native | comandi vocali: continuità con la fase 1 |
| tldr-pages (già in casa) | CC BY 4.0 | le nostre 5 | azioni di un agente: continuità con il giro E |
| CLINC150, Banking77, MINDS-14 (già in casa) | CC BY | en + MINDS-14 native | intenzioni: continuità con il giro C |

**Scartate, con il motivo:** XNLI e wiki_lingua/xlsum (CC BY-NC: non commerciale);
`textdetox/multilingual_toxicity_dataset` (OpenRAIL++: pone vincoli d'uso che si propagherebbero a Skejja, che è
Apache-2.0); le copie multilingui di SMS Spam su HF (GPL); `Helsinki-NLP/opus-100`, `statmt/cc100` e
`unimelb-nlp/wikiann` (licenza «unknown»); `MoritzLaurer/multilingual-NLI-26lang` (nessuna licenza dichiarata);
`coastalcph/multi_eurlex` (CC BY-SA 4.0, usabile ma rinviata: testo legale lontano dal nostro carico di lavoro e
serve il lavoro sugli stati lunghi); gli aggregati di sentiment multilingue `tyqiangz` e `clapAI` (etichetta
Apache-2.0 sull'aggregato ma composizione a monte non verificata: da ricontrollare prima di un eventuale uso).

Nota sulle licenze con **ShareAlike** (CC BY-SA, GFDL): la regola del progetto fin qui è stata «solo CC BY»
(vedi `bench/build_external.py`). Le fonti della tabella sopra sono tutte CC BY, CC0, Apache-2.0 o ODC-BY, cioè
solo attribuzione: nessuna obbligherebbe a condividere alle stesse condizioni. Le attribuzioni vanno aggiunte in
`NOTICE` prima di pubblicare i dati o i pesi.

## 3. Il dataset di addestramento di Bekko: cosa è utilizzabile

[`hotchpotch/bekko-system-one-dataset-v0`](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)
contiene 6.589.188 casi di addestramento in 182 sottoinsiemi (153 attivi), ed è il dato che ha prodotto Bekko,
il modello della nostra fascia. Analisi riproducibile: `python bench/bekko_license_audit.py` (revisione fissata
`9c673a09`, scheda del 8 ottobre 2026).

**Il repository non dichiara nessuna licenza.** La scheda dice esplicitamente «No blanket relicensing is
applied» e rimanda alla licenza di ogni dataset a monte; l'autore ha fatto una revisione delle licenze il 30
settembre 2026 su 131 fonti, con esito 41 «verified», 84 «qualified», 6 «unresolved». Casi di addestramento per
licenza dichiarata a monte:

| Classe della licenza a monte | Sottoinsiemi | Casi | Quota |
|---|---:|---:|---:|
| Apache, MIT o BSD | 53 | 2.387.701 | 36,2% |
| CC BY-SA | 33 | 2.369.876 | 36,0% |
| CC BY | 38 | 695.331 | 10,6% |
| ristretta (non commerciale, GPL/LGPL, AFL, CDLA) | 11 | 387.210 | 5,9% |
| sconosciuta | 17 | 375.426 | 5,7% |
| CC0 o dominio pubblico | 30 | 373.644 | 5,7% |

Per licenza escludiamo tre classi intere: «ristretta», «sconosciuta» e anche **CC BY-SA**, cioè 0 dei suoi 33
sottoinsiemi. ShareAlike è fuori dalla regola del progetto «solo CC BY o equivalenti» (vedi la nota del §2).
Togliendo poi i sottoinsiemi che non possiamo usare per altri motivi restano **3.081.249 casi in 65 sottoinsiemi**
(47% del totale): 37 Apache/MIT/BSD, 24 CC BY, 4 CC0. I 182 sottoinsiemi comprendono 29 di sola valutazione,
senza casi di addestramento; i 153 «attivi» sono quelli del `training-manifest.json` e contengono tutti i
6.589.188 casi. Gli altri motivi di esclusione, indipendenti dalla licenza:

- **283.181 casi si sovrappongono al nostro banco di prova** (`laya__ag_news` 120.000, `agnews` 19.992,
  `sms_spam` 5.171, `massive` 10.463, `clinc` 15.189, `banking77` 10.003, più `mtop`, `snips`, `go_emotions`,
  `hwu64`, `laya__enron_spam`, `laya__phishing`): usarli violerebbe la regola del paragrafo 1;
- **266.793 casi in 24 sottoinsiemi `open_jev__*`** sono acquisiti da Open-Jev. La revisione delle licenze
  (`license-audit.json`, voce `ZefanCai/Open-Jev`) dice che la dichiarazione CC0 copre solo il materiale generato,
  e che una parte dei contenuti, cioè le descrizioni delle domande di alcuni compiti, viene dalla documentazione di
  TypeSafe con licenza «not verified». Per la nostra regola «mai Jev» sono esclusi;
- **93.133 casi** sono scritti o riscritti da modelli: 90.208 nei 19 sottoinsiemi `synthetic_relevance__*` con casi
  di addestramento, 1.725 di FollowIR (documenti sintetici generati con GPT-3.5-Turbo) e 1.200 di
  `laya__typed_decisions`. Licenza non dichiarata o termini del fornitore non verificabili. Le prove di
  generalizzazione di S1MB scritte con GPT-6-Astra sono fra i sottoinsiemi di sola valutazione (0 casi di
  addestramento).

**Resta però un ostacolo che riguarda tutti i 6,59 milioni di casi, licenza a monte compresa.** Il dataset non è
solo una raccolta di testi: la conversione in casi tipizzati (definizioni «vera/falsa» delle Noul, istruzioni,
ordine dei candidati, probabilità morbide) è lavoro originale di chi l'ha pubblicato, e **su quello strato non
c'è alcuna concessione**, perché il repository non dichiara una licenza. In più il file `AGENTS.md` del dataset
dice «Keep this repository private unless explicitly authorized to change visibility», mentre la scheda parla di
«Public Hugging Face dataset distribution»: le due frasi si contraddicono. Con una licenza assente e una
contraddizione sulla visibilità, la nostra regola non ci permette di usare il pacchetto.

**Verdetto.** Non usiamo il dataset di Bekko. Usiamo invece, come punto di partenza del nostro lavoro,
l'**elenco delle fonti a monte** con la revisione delle licenze fatta dall'autore: è pubblico, dettagliato e
indica dove andare a prendere i dati originali, cosa che per CLINC, Banking77 e MINDS-14 facciamo già. Fra i 65
sottoinsiemi compatibili i più vicini al nostro buco di ampiezza sono
[Circa](https://huggingface.co/datasets/google-research-datasets/circa) (CC BY, risposte sì/no indirette,
34.015 casi), [Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) (CC BY,
sicurezza dei contenuti, 27.060), [civil_comments](https://huggingface.co/datasets/google/civil_comments)
(CC0, 42.857), [measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
(CC BY, 30.000), [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) e HelpSteer3 (CC BY, giudizi di
qualità, utili più avanti per `Score`), e i compiti di ragionamento `tasksource/*` (Apache-2.0). Sono tutti in
**inglese**: andrebbero estesi alle nostre 5 lingue, che è il nostro vantaggio, con le traduzioni OPUS-MT già in
uso nel progetto oppure rietichettando testo nativo con l'insegnante. Due cose da decidere con l'utente: se
chiedere all'autore di dichiarare una licenza (è un'interazione pubblica) e se vale la pena convertire noi
stessi questi dataset invece di generare etichette su testo libero, che è la strada della fase 2.

## 4. S1MB: la classifica pubblica e il confronto con Bekko 68M

[S1MB](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark) (System One Mosaic Benchmark) è il
banco pubblico con classifica per i System One: `english-v1` ha **137 prove su 106 sottoinsiemi, 14.009 casi e
26.269 giudizi** (59 Noul, 57 Choice, 21 Score); esiste anche un `japanese-v1` con dati privati. Codice di
valutazione MIT ([github.com/hotchpotch/S1MB](https://github.com/hotchpotch/S1MB)), dati di prova e risultati
su Hugging Face, 170 modelli misurati. Metrica primaria: `binary_brier` per le Noul,
`target_mass_at_prediction` per le Choice, `normalized_expected_score_mae` per le Score; i punteggi pubblicati
sono corretti per la linea di base (`baseline_adjusted_score`). **La «Task Avg» della classifica è la media non
pesata delle tre medie di famiglia**, non la media sulle 137 prove: verificato su Bekko 68M,
(42,91 + 51,62 + 26,85) / 3 = 40,46.

Riferimenti dalla classifica ufficiale (`python bench/s1mb_reference.py`): riepilogo scaricato dal repository
dei risultati alla revisione `ec0a2b73`, che aggrega le misure alla revisione `35853a9d` ed è stato generato il
13 ottobre 2026 alle 08:04 UTC. L'ultima colonna è il numero di prove per cui il modello **dichiara** di avere
la stessa famiglia di sorgenti nei dati di addestramento:

| Modello | Parametri totali / non-lookup | Task Avg | Noul | Choice | Score | Esposizione dichiarata |
|---|---|---:|---:|---:|---:|---:|
| Jev 1.13 | non pubblicati | 59,59 | 64,63 | 67,22 | 46,92 | 0/137 |
| Bekko 400M | 394,8M / 343,2M | 50,60 | 51,24 | 61,32 | 39,25 | 98/137 |
| Kev 4B | 4,21B / 3,57B | 42,33 | 48,48 | 52,03 | 26,47 | 0/137 |
| **Bekko 68M** | **68,1M / 42,4M** | **40,46** | **42,91** | **51,62** | **26,85** | **98/137** |
| Bekko 17M | 16,8M / 3,9M | 27,57 | 31,43 | 35,57 | 15,70 | 98/137 |
| Kev 0.8B | 752,9M / 498,6M | 18,68 | 23,61 | 27,36 | 5,07 | 0/137 |
| Laya typed-decisions | 421,3M / 369,7M | 15,00 | 20,06 | 18,93 | 5,99 | 0/137 |
| Laya (inglese) | 421,3M / 369,7M | 13,36 | 20,19 | 16,31 | 3,58 | 0/137 |
| ModernBERT-large zeroshot v2.0 | 395,8M / 344,3M | 9,56 | 7,33 | 17,67 | 3,68 | 0/137 |
| Laya multilingue | 321,9M / 125,3M | 9,28 | 14,01 | 13,03 | 0,79 | 0/137 |
| GLiNER2.5-Decide | 486,4M / 354,8M † | 9,15 | 5,40 | 18,19 | 3,87 | 0/137 |

_Parametri «non-lookup» = senza le tabelle di embedding, con il metodo `non_lookup_parameters_v1` della
classifica. † GLiNER2.5-Decide è dichiarato con un metodo diverso (`embedding_excluded_parameters_v1`)._

**Bekko 68M è davvero la nostra fascia, e con il metodo di conteggio della classifica è Skejja a essere un po'
più grande dove conta:** Skejja ft5 ha 57.270.402 parametri in tutto, di cui 12.582.912 nella tabella di
embedding (32.768 × 384), quindi **44.687.490 parametri non-lookup** contro i 42.357.763 di Bekko 68M (conteggio
fatto sul checkpoint `releases/checkpoints/run_E/best`). Bekko ha più parametri totali perché ha un vocabolario
più grande; a parità di calcolo siamo pari.

**Cosa possiamo e non possiamo misurare.** Tre limiti del nostro formato, da dichiarare insieme ai numeri:

1. **Niente `Score`**: 21 prove su 137. La Task Avg, che è la media delle tre famiglie, per noi non è
   confrontabile; i numeri leggibili sono Noul e Choice, da confrontare con 42,91 e 51,62 di Bekko 68M.
2. **Niente spazio per le istruzioni.** Gli ingressi di S1MB hanno `state_json`, `instructions_json`,
   `system_prompt`, criteri e documenti; Skejja prende solo `<<LABEL>>opzioni<<SEP>>testo`. L'adattatore mette lo
   stato nel testo e i criteri nelle opzioni, e **perde le istruzioni**: è il limite che già conoscevamo
   («contesto corto» nel confronto) e la misura serve proprio a quantificarlo.
3. **Lunghezza.** L'architettura regge 8.192 posizioni, ma l'addestramento è arrivato a 192 token: gli stati
   lunghi vanno troncati e la quota di troncati va riportata accanto ai punteggi.

**L'esposizione cambia la lettura del confronto.** Bekko dichiara esposizione su 98 prove su 137 (e la scheda di
S1MB avverte che il banco «does not establish unseen-task generalization or training-data non-overlap»). Delle
137 prove, quelle che usano sorgenti presenti nel nostro addestramento sono **3**: `massive-choice-test-v1`,
`clinc-choice-test-v1` e `banking77-choice-test-v1`. Riporteremo quindi i numeri su tutte le prove e, a parte,
sulle 134 dove **noi** non abbiamo esposizione, dichiarando le nostre 3 come fa Bekko con le sue 98.

**Regole che accettiamo misurandoci**: la politica sul test di S1MB vieta di tarare qualunque parametro, soglia,
calibrazione o scelta di checkpoint guardando i punteggi del test, anche un solo parametro condiviso. Fissiamo
configurazione e calibrazione sui nostri file `calib_*` e di sviluppo, poi misuriamo una volta. Nessun invio alla
classifica pubblica senza chiederlo prima all'utente (vedi `CLAUDE.md`, «Cosa chiedere all'utente prima di
farlo»): l'invio è una pull request su un dataset pubblico.

### 4.1 Misura di Skejja su S1MB english-v1 (13 ottobre 2026)

Configurazione fissata prima di misurare (adattatore `bench/s1mb_adapter/`, dati alla revisione `6aa5f397`, harness
al commit `16a23df9`): criteri come opzioni della Choice, definizione «vera» come etichetta della Noul, istruzioni
e stato nel testo, calibrazione globale del bundle (fitted solo sui `calib_*`), limite 4.096 token del tokenizer del
bundle. Eseguito sulla workstation RTX 3090 con PyTorch fp32 sugli stessi pesi del bundle ONNX fp32: sulle 2.991
decisioni misurate anche sul Mac con il bundle ONNX la differenza massima di probabilità è 5,7·10⁻⁶ e le metriche
coincidono, quindi i numeri valgono per il bundle fp32 (non per l'int8 dinamico del Raspberry, non misurato qui).
Tempo: 493 s per ft5 e 501 s per ft3 sulla GPU. Fonti: [metrics_s1mb_edge32k-ft5.csv](../results/fase2/metrics_s1mb_edge32k-ft5.csv),
[metrics_s1mb_edge32k-ft3.csv](../results/fase2/metrics_s1mb_edge32k-ft3.csv) con i `_riepilogo.json`,
[s1mb_confronto_per_prova.csv](../results/fase2/s1mb_confronto_per_prova.csv) e il suo riepilogo.

| Modello | Noul | Choice | Score | Esposizione dichiarata |
|---|---:|---:|---:|---:|
| Bekko 68M | 42,91 | 51,62 | 26,85 | 98/137 |
| Laya multilingue | 14,01 | 13,03 | 0,79 | 0/137 |
| GLiNER2.5-Decide | 5,40 | 18,19 | 3,87 | 0/137 |
| **Skejja ft3** | **0,92** | **10,94** | non supportata | 3/137 |
| **Skejja ft5** | **0,78** | **8,66** | non supportata | 3/137 |

Senza le 3 prove su cui abbiamo esposizione (MASSIVE, CLINC, Banking77): ft5 Choice 7,32 e ft3 9,82; le Noul non
cambiano. Le 21 prove Score (5.148 decisioni) risultano fallite, non sbagliate. Per ft6 (giro F), con la stessa
configurazione: Noul 0,69 e Choice 9,75 (§6, tabella E).

**Cosa dicono i numeri.** Sono a configurazione fissata: nessuna di queste diagnosi ha cambiato l'adattatore.
1. **Le sì/no generali non funzionano.** AUROC mediana delle 59 Noul 0,48 per ft5 (0,47 per ft3), 37 prove su 59
   sotto 0,5, probabilità media di «vero» 0,86 in mediana: Skejja risponde «sì» quasi a tutto. Il formato non è il
   problema (le definizioni «vere» sono frasi complete, es. «The premise supports the hypothesis.»): il modello,
   addestrato a riconoscere intenzioni, premia le affermazioni che hanno parole in comune con il testo. Solo 3
   Noul hanno AUROC ≥ 0,6 (drone-control 0,73, scitail 0,67, ruletaker 0,66).
2. **Le Choice da 2–4 opzioni sono al livello del caso.** Sono 39 su 57, quasi tutte di ragionamento, logica,
   implicazione o conoscenza (ARC, AQuA-RAT, BBQ, COPA, contract-NLI…): accuratezza mediana 0,340 contro 0,333
   della scelta a caso.
3. **Sulle intenzioni Skejja capisce, ma con molte opzioni perde molto.** Sulle 4 prove con più di 50 opzioni
   accuratezza mediana 0,32 contro 0,014 del caso: CLINC (151 opzioni) 0,325, Banking77 (77) 0,51, HWU64 (64) 0,24,
   MTOP (62) 0,315. MASSIVE ha 39 opzioni (fascia 11–50, 5 prove, mediana 0,275) e ft5 fa 0,275, mentre sul nostro
   banco MASSIVE, con 8 opzioni, fa 0,873: il calo viene dal numero di opzioni e dalle descrizioni scritte da altri.
   Bekko 68M su queste cinque prove sta fra 79 e 92.
4. **Dove Skejja va meglio:** AG News (ft3 77,6, ft5 61,2; Bekko 92,5), DBpedia 50,6, la prova di generalizzazione
   «contextual choice» 53,6 (Bekko 64,0). Le prove in cui un nostro modello supera Bekko 68M sono una per ciascuno
   (ft5: followir-robust04; ft3: drone-control).
5. **ft3 è un po' meglio di ft5** anche qui (Choice 10,94 contro 8,66), coerente con la perdita di ampiezza del
   fine-tuning già vista sui dataset dell'articolo su Jev.

**Conseguenza per i dati della fase 2.** Le 8 famiglie di `families.json` riguardano un testo solo (argomento,
sentiment, emozione, spam…): servono per i dataset dell'articolo su Jev, ma non toccano la parte più grande di
S1MB, che chiede **relazioni fra due testi** (premessa/ipotesi, domanda/documento, regola/caso) e **Choice con molte
opzioni**. Per avvicinarsi a Bekko servono anche dati di quel tipo: implicazione, pertinenza fra domanda e
documento, e Choice con 30–150 opzioni. Molte delle fonti adatte sono fra i 65 sottoinsiemi compatibili del
paragrafo 3 (ruletaker, nlsat, FOL-NLI, temporal-NLI, PLANE, Circa, HANS, WIQA…), ma sono le stesse famiglie di
sorgenti di prove S1MB: usarle (split di training) è ammesso dalle regole di S1MB a patto di dichiararlo, come fa
Bekko con 98 prove, ma la nostra misura su quelle prove non sarebbe più zero-shot. Per il giro F (§6) abbiamo
tenuto la via zero-shot: relazioni fra due testi solo con la famiglia `relazione`, generata dall'insegnante e non
presa da fonti di S1MB. La scelta per i giri successivi resta aperta (va decisa con l'utente).

## 5. Prova di etichettatura e scelta dell'insegnante (13 ottobre 2026)

Workstation RTX 3090, vLLM 0.31 (ambiente `teacher`). Richieste: le 10.000 del pool di prova
(`bench/data/broad/requests_trial.jsonl`), 788 con le opzioni in ordine inverso, 1.473 con la domanda nell'altra
lingua e 1.600 del **metro** (400 testi dagli split di training di SST-2, AG News, Emotion e SMS Spam con la loro
etichetta vera, solo per misurare l'insegnante), in tutto 13.861 (`bench/data/broad/teacher_trial_in.jsonl`). Un solo
token generato, probabilità lette dai logprob delle risposte valide (`bench/teacher_label.py`). Numeri da
[teacher_trial.csv](../results/fase2/teacher_trial.csv) e [teacher_trial_riepilogo.json](../results/fase2/teacher_trial_riepilogo.json).

| Metro (etichetta vera) | Qwen3.5-9B (bf16) | Qwen3.8-27B (INT4, RedHatAI) |
|---|---:|---:|
| SST-2, accuratezza (3 opzioni, compreso «neutro») | 0,838 | 0,853 |
| AG News, accuratezza (8 argomenti, «World» → politica) | 0,820 | 0,810 |
| Emotion, accuratezza (7 opzioni) | 0,478 | 0,488 |
| SMS Spam, AUROC / F1 con soglia 0,5 | 0,993 / 0,953 | 0,997 / 0,895 |
| Lingua del testo (etichetta del corpus), Choice / sì/no | 0,995 / 0,994 | 0,996 / 0,995 |
| Probabilità sulle risposte valide, Choice / sì/no | 0,9996 / 0,993 | 0,999 / 0,995 |
| Richieste al secondo | 20,1 | 6,6 sul metro (testi brevi); più lento sui testi web lunghi del pool (giro interrotto, non misurato fino in fondo) |

- **Scelta: Qwen3.5-9B.** La qualità è la stessa entro uno o due punti, e sul pool i due insegnanti danno la
  stessa risposta nell'80–99% delle richieste a seconda della famiglia (registro 0,80, intento 0,83–0,86, le altre
  ≥ 0,87). Il 9B è tre volte più veloce: 100.000 richieste in circa 1,4 ore.
- **Stabilità del 9B:** invertendo l'ordine delle opzioni la risposta resta la stessa nel 90% dei casi (distanza
  media fra le distribuzioni 0,09); con la domanda nell'altra lingua nel 92% (Choice) e nel 97% (sì/no). Il 10% di
  dipendenza dall'ordine è il motivo per cui le opzioni dei dati di Skejja sono sempre rimescolate.
- **I numeri del metro non sono quelli dell'articolo su Jev**, anche se i dataset sono gli stessi: split di training
  invece di test, le nostre opzioni (SST-2 con «neutro», AG News con 8 argomenti), 400 testi per dataset. Servono
  solo a confrontare gli insegnanti fra loro.
- **Problemi della macchina risolti** (annotati nelle note della workstation): FlashInfer chiede `nvcc`, che non c'è
  → `attention_backend="FLASH_ATTN"` ed `enforce_eager`; il checkpoint RedHatAI chiede una cache KV in FP8, che
  FlashAttention 2 non supporta su Ampere → `kv_cache_dtype="bfloat16"`.

**Le etichette dei testi naturali sono molto sbilanciate.** Sui testi del pool (richieste di Aya, domande di mfaq,
pagine web) il 9B dà sentiment neutro nel 77,4% dei casi e negativo nel 4,1%; emozione «nessuna» nell'82,3%;
«indesiderato» nell'1,1% e testo offensivo, violento o con dati personali nel 3,3%. Addestrare solo su questi
insegnerebbe a Skejja a rispondere «neutro» e «no».

**Rimedio: testi sintetici scritti dall'insegnante** (`bench/teacher_generate.py`). Qwen3.5-9B (Apache-2.0: i suoi
testi si possono usare per addestrare) ha scritto 19.296 testi brevi nelle 5 lingue, circa 3.860 per lingua, con
una classe voluta: recensioni positive e negative, messaggi con un'emozione, SMS ed email di spam e di phishing
contro messaggi legittimi, risposte scortesi, notizie di aggressioni, messaggi con dati personali inventati, testi
per argomento e per registro, e 3.750 coppie A/B per la nuova famiglia `relazione` (implicazione, contraddizione,
nessuna delle due, risposta e non risposta a una domanda), che copre la parte di S1MB sulle relazioni fra due testi
senza usarne le fonti. Tipo di testo, argomento (58 soggetti quotidiani) e lunghezza sono scelti a caso;
temperatura 0,9; 4 testi scartati per formato, 1 duplicato esatto; 1.386 s di generazione
([synth_texts_tempi.json](../bench/data/broad/synth_texts_tempi.json)). Ne vengono 61.392 richieste, 29.292 con un
valore atteso: nell'addestramento si scartano quelle in cui il 9B non conferma la classe voluta.

Lettura di un campione da parte dell'agente (10 testi, non una revisione umana): testi per lo più realistici e
nella lingua giusta; qualche parola innaturale (in italiano «piatti… insalati» per «insipidi»); alcune coppie di
«implicazione» in cui B segue da A solo in modo debole; le coppie «non risponde» sono spesso troppo facili (B parla
d'altro). Una revisione a campione da parte di una persona resta da fare.

### 5.1 Qualità delle etichette sui testi sintetici (13 ottobre 2026)

Le 61.392 richieste sintetiche sono state etichettate da Qwen3.5-9B in 2.741 s (22,4 richieste al secondo), con un
lavoro riprendibile a blocchi di 1.000 ([labels_synth_qwen9b_tempi.json](../bench/data/broad/labels_synth_qwen9b_tempi.json);
etichette in `bench/data/broad/labels_synth_qwen9b.jsonl.gz`). Il rapporto applica **gli stessi filtri
dell'addestramento** (copertura ≥ 0,9; scartata la richiesta se l'insegnante non conferma la classe voluta):
`bench/synth_quality.py` → [qualita_ampiezza_sintetico.csv](../results/fase2/qualita_ampiezza_sintetico.csv) e
[qualita_ampiezza_sintetico_riepilogo.json](../results/fase2/qualita_ampiezza_sintetico_riepilogo.json).

- **Tenute 58.623 su 61.392 (95,5%).** Scartate 377 per copertura bassa (tutte sì/no) e 2.392 perché l'insegnante
  non conferma la classe voluta (1.482 Choice, 910 sì/no).
- **Accordo fra insegnante e classe voluta:** Choice 0,867 (11.150 richieste con una classe voluta), sì/no 0,949
  (17.970). Per famiglia, Choice: sentiment 0,988, registro 0,936, emozione 0,854, intento 0,847, **argomento 0,740**
  (il più basso: gli 8 argomenti si sovrappongono, per esempio «casa» con salute e cultura); sì/no fra 0,929
  (registro) e 0,976 (emozione), `relazione` 0,941. Nessuna lingua è sistematicamente peggiore; i valori più bassi
  sono argomento sì/no in italiano (0,867) e registro sì/no in italiano e spagnolo (0,89–0,90).
- **Etichette bilanciate, come voluto:** la classe più frequente dell'emozione («niente») è ora al 22,3%, contro
  l'82,3% dei testi naturali; il 38,9% delle sì/no tenute è «sì». Resta sbilanciato il registro (Choice «informale»
  al 68,2%) e la sì/no del sentiment («sì» al 64,5%).
- **Etichette incerte:** 15,6% delle Choice con confidenza sotto 0,8; 11,0% delle sì/no fra 0,2 e 0,8, di più per
  `relazione` (26,8%). Sono tenute come bersagli morbidi.

## 6. Giro F: il candidato `edge32k-ft6` (13–14 ottobre 2026)

**Configurazione** (fissata prima dei risultati; meta in [train_meta_run_F.json](../results/fase2/train_meta_run_F.json)).
Si parte da ft5 (`run_E/best`) con distillazione da ft3 (`run_C/best`, peso 0,5) solo sui dati MASSIVE, non su azioni
e ampiezza; lr 2·10⁻⁵, testa 6·10⁻⁵, descrizioni nella lingua della frase con probabilità 0,5, 12.000 esempi sulle
azioni, ingressi fino a 320 token; workstation RTX 3090. I dati di ampiezza sono il pool di prova naturale (Aya, mfaq,
FineWeb) e i testi sintetici, etichettati da Qwen3.5-9B e filtrati come al §5.1. Le 1.600 righe del metro sono
escluse, e un assert lo verifica. Il 5% dei testi (per `text_id`) è tenuto da parte come sviluppo di ampiezza
(3.574 esempi). In addestramento entrano 30.114 Choice e 37.135 sì/no di ampiezza, accanto alle 24.000 Choice
MASSIVE e alle sì/no esistenti (108.569 sì/no in tutto); 5.086 passi, 1.211 s.

**Scelta della variante, solo sullo sviluppo.** F (tutti i dati di ampiezza) contro Fh (12.000 Choice e 15.000 sì/no
di ampiezza; 3.829 passi): media di selezione 0,9488 contro 0,9431, quindi F. Alla fine di F, sviluppo MASSIVE (AUROC
sì/no) 0,931 in inglese, 0,944 con la domanda nella lingua della frase; azioni 0,999; sì/no di ampiezza 0,983 e
Choice di ampiezza 0,860. Attenzione: lo sviluppo di ampiezza ha la stessa distribuzione dei dati di addestramento,
quindi favorisce la variante con più dati di ampiezza. Le misure qui sotto sono fuori dallo sviluppo.

**Export.** Bundle fp32, int8 (solo pesi) e int8 dinamico con `mlp/Wo` in fp32, calibrazione sui soli `calib_*`
([calibration_edge32k-ft6.json](../results/fase2/calibration_edge32k-ft6.json)). Controllo contro PyTorch su 300+300
esempi ([export_check_edge32k-ft6.json](../results/fase2/export_check_edge32k-ft6.json)): fp32 identico (differenza
massima 2·10⁻⁵); int8 dinamico, accordo Choice 0,967 e sì/no 0,973, contro 0,987 e 0,993 di ft5. Sul test completo
però non perde nulla (tabella A). File in `releases/skejja-edge32k-ft6_onnx-{fp32,int8}.tar.gz`, checkpoint in
`releases/checkpoints/run_F/best`.

**A. MASSIVE** (test originale ed esteso; differenze appaiate ft6 − ft5 con bootstrap al 95%,
`bench/compare_runs.py` → [metrics_diff_ft6_vs_ft5.csv](../results/fase2/metrics_diff_ft6_vs_ft5.csv); fp32 sulla GPU,
[metrics_edge32k-ft6.csv](../results/fase2/metrics_edge32k-ft6.csv))

| Misura | ft5 | ft6 | ft6 − ft5 [IC 95%] | n |
|---|---:|---:|---:|---:|
| Choice, tutte | 0,8725 | 0,8725 | 0,000 [−0,012; +0,014] | 2.000 |
| Choice, intenzioni mai viste | 0,731 | 0,741 | +0,010 [−0,041; +0,062] | 290 |
| Sì/no, test (AUROC) | 0,9636 | 0,9612 | −0,002 [−0,007; +0,003] | 2.000 |
| Sì/no, test esteso mai visto (AUROC) | 0,8317 | 0,8399 | +0,008 [−0,000; +0,016] | 3.655 |
| int8 dinamico sul Mac: Choice / mai viste | — | 0,872 / 0,748 | +0,002 [−0,012; +0,015] / +0,024 [−0,028; +0,079] | |
| int8 dinamico sul Mac: sì/no test / esteso mai visto | — | 0,961 / 0,840 | −0,002 [−0,007; +0,003] / **+0,011 [+0,003; +0,019]** | |

ECE calibrata delle Choice 0,050 → 0,048 (int8 dinamico 0,047): ancora al limite dell'obiettivo 0,05. ECE sì/no
(Platt) 0,033 → 0,038. Divario Choice fra inglese e lingua peggiore 4,0 → 4,25 punti. Contro ft3
([metrics_diff_ft6_vs_ft3.csv](../results/fase2/metrics_diff_ft6_vs_ft3.csv)) nessuna differenza significativa
(Choice −0,006 [−0,018; +0,006]; mai viste −0,031 [−0,090; +0,021]). Int8: [metrics_edge32k-ft6_int8dyn-noWo.csv](../results/fase2/metrics_edge32k-ft6_int8dyn-noWo.csv),
[metrics_diff_ft6_vs_ft5_int8dyn.csv](../results/fase2/metrics_diff_ft6_vs_ft5_int8dyn.csv) (contro l'int8 dinamico di ft5).

**B. Azioni degli agenti (tldr-pages, test)** ([metrics_agent_tldr_ft6.csv](../results/fase2/metrics_agent_tldr_ft6.csv)):
AUROC con il comando 0,996 → 0,997 (3.819), con la descrizione inglese 0,882 → 0,911 (910), con la descrizione nella
lingua 0,797 → 0,836 (326; accuratezza 0,801 → 0,782). Resta debole «cancellare o sovrascrivere file» con descrizione
nella lingua (0,399 → 0,380, 93 esempi). Lo script non salva i punteggi per esempio, quindi qui non c'è un test appaiato.

**C. Test v0 fuori da MASSIVE** (bootstrap appaiato per frase: [metrics_ood_diff_ft6_vs_ft5.csv](../results/fase2/metrics_ood_diff_ft6_vs_ft5.csv),
[metrics_ood_diff_ft6_vs_ft3.csv](../results/fase2/metrics_ood_diff_ft6_vs_ft3.csv)): sì/no complessiva in italiano
0,851 → 0,883 (**+0,032 [+0,011; +0,055]**), in inglese 0,839 → 0,858 (**+0,020 [+0,001; +0,038]**); email in italiano
**+0,067 [+0,017; +0,121]**; agenti e domotica senza differenze significative. Choice (120 frasi) 0,742 → 0,725 in
italiano e 0,592 → 0,592 in inglese, differenze non significative. Contro ft3 la sì/no cresce di +0,12 in entrambe le
lingue, quasi tutto sulle azioni degli agenti.

**D. Dataset dell'articolo su Jev** (zero-shot; sulla GPU nello stesso giro per ft5 e ft6, PyTorch fp32. I numeri di
ft5 coincidono con quelli del Mac del 9 ottobre. Bootstrap appaiato:
[metrics_jev_suite_diff_ft6_vs_ft5.csv](../results/fase2/metrics_jev_suite_diff_ft6_vs_ft5.csv); valori, intervalli
ed ECE: [metrics_jev_suite_ft6.csv](../results/fase2/metrics_jev_suite_ft6.csv))

| Dataset | Jev | GLiClass di partenza | ft5 | **ft6** | ft6 − ft5 [IC 95%] |
|---|---:|---:|---:|---:|---:|
| AG News | 0,885 | 0,776 | 0,716 | **0,782** | +0,066 [+0,057; +0,075] |
| Rotten Tomatoes | 0,933 | 0,758 | 0,682 | **0,731** | +0,049 [+0,025; +0,073] |
| SST-2 | 0,964 | 0,799 | 0,711 | **0,807** | +0,096 [+0,069; +0,124] |
| Emotion | 0,585 | 0,474 | 0,276 | **0,358** | +0,082 [+0,061; +0,103] |
| Financial PhraseBank | 0,730 | 0,497 | 0,442 | **0,472** | +0,030 [−0,001; +0,062] |
| Language ID (20 lingue) | 0,996 | 0,056 | 0,040 | **0,208** | +0,168 [+0,161; +0,176] |
| SMS Spam, F1 (AUROC) | 0,938 | 0,375 | 0,092 (0,451) | **0,749 (0,943)** | +0,657 [+0,622; +0,691] |
| Banking77 ⚠ in dominio | 0,797 | 0,324 | 0,581 | **0,575** | −0,006 [−0,017; +0,005] |
| SIB-200, media en/it/es/de/fr | 0,815 (205 lingue) | 0,719 | 0,729 | **0,740** | +0,011 [−0,016; +0,037] |
| SIB-200, solo inglese | — | 0,775 | 0,730 | **0,676** | **−0,054 [−0,093; −0,015]** |

1. **L'ampiezza persa con il fine-tuning è in parte recuperata.** ft6 torna sopra il modello di partenza su AG News e
   SST-2, e su SMS Spam lo supera di molto. Resta sotto su Rotten Tomatoes (−2,7 punti), Emotion (−11,6) e Financial
   PhraseBank (−2,5).
2. **La sì/no su un'affermazione generale ora funziona** sul caso dello spam: AUROC 0,943 e F1 0,749 alla soglia 0,5,
   con ECE 0,024 (ft5: 0,451, 0,092 e 0,079).
3. **La distanza da Jev resta grande:** da 10 punti (AG News) a 26 (Financial PhraseBank); su Language ID
   Skejja è ancora lontano dall'essere utilizzabile (0,208, anche se ha imparato le nostre 5 lingue dalla famiglia
   `lingua`).
4. **Una perdita: SIB-200 in inglese**, −5,4 punti (15 frasi peggiorate, 4 migliorate; McNemar esatto p = 0,019).
   Dopo la correzione di Bonferroni sui 13 dataset della tabella non è più significativa, e sulle 5 lingue insieme
   ft6 guadagna (+1,1; spagnolo +4,4 [+0,5; +8,3]). Il modello sovrastima «geography» (52 predizioni per 17 frasi
   vere) e perde su sport ed entertainment. ECE su SIB-200 inglese 0,079 → 0,141.

**E. S1MB english-v1** (stessa configurazione del §4.1, GPU fp32, 21.121 decisioni riuscite e 5.148 Score non
supportate: [metrics_s1mb_edge32k-ft6.csv](../results/fase2/metrics_s1mb_edge32k-ft6.csv) con il `_riepilogo.json`,
[s1mb_confronto_per_prova.csv](../results/fase2/s1mb_confronto_per_prova.csv) per ft6, ft5 e ft3)

| Modello | Noul | Choice | Choice senza le 3 prove esposte |
|---|---:|---:|---:|
| Bekko 68M | 42,91 | 51,62 | — |
| Skejja ft3 | 0,92 | 10,94 | 9,82 |
| Skejja ft5 | 0,78 | 8,66 | 7,32 |
| **Skejja ft6** | **0,69** | **9,75** | **8,88** |

- **Su S1MB il giro F non cambia quasi nulla.** 20 prove migliorano, 14 peggiorano, 82 restano uguali (quasi tutte a
  0). I guadagni sono sulle prove di classificazione: email-selection-control 0 → 53,0 (come Bekko 68M), painting-
  geometry 42,1 → 79,0 (Bekko 68,4), AG News 61,2 → 71,6 (Bekko 92,5), ARCT 8,5 → 19,1, DBpedia 50,6 → 56,2. Le
  perdite più grandi sono followir-robust04 (18,1 → 0), la generalizzazione «diverse choice» (20,7 → 8,8), CLINC
  (29,7 → 20,0) e SGD (11,9 → 3,4).
- **Le sì/no di S1MB restano al livello del caso:** AUROC mediana 0,474 (ft5 0,480), 39 prove su 59 sotto 0,5,
  probabilità media di «vero» 0,79 in mediana (ft5 0,86). La famiglia `relazione` (5.601 sì/no generate) non basta
  per le prove di implicazione, logica e pertinenza di S1MB, che hanno testi lunghi (media 598 token, 12.175 domande
  oltre i 192 token dell'addestramento) e formati diversi dai nostri.
- Prove in cui ft6 supera Bekko 68M: drone-control e painting-geometry.

**Decisione.** Il criterio fissato nel piano era: adottare il candidato solo se non perde in modo significativo su
nessuna misura e guadagna chiaramente sull'ampiezza. Il guadagno c'è (tabella D, test v0). Una misura lo viola
alla lettera: SIB-200 in inglese, significativa senza correzione ma non dopo Bonferroni. Proponiamo quindi `edge32k-ft6`
come **nuovo candidato principale, con questa riserva dichiarata**. ft5 resta disponibile, uguale su MASSIVE e
sulle azioni. Su Raspberry Pi ft6 non è ancora misurato. Architettura e file int8 dinamico hanno la stessa
dimensione di ft5 (124.288.004 byte di pesi), quindi latenza e RAM dovrebbero essere le stesse, ma va verificato.

## 7. Limiti di questo documento

- Le licenze sono lette dalle schede sull'Hub e dai file di licenza collegati, con revisioni fissate dove
  possibile. Non è una consulenza legale, come già dichiarato per il contratto di TypeSafe nel confronto.
- La classificazione delle licenze di Bekko è automatica (`bench/bekko_license_audit.py`) e parte dalle stringhe
  dichiarate dall'autore: le 6 fonti «unresolved» e le 17 «sconosciute» andrebbero lette una per una prima di
  usare qualsiasi sottoinsieme.
- I numeri degli altri modelli su S1MB sono loro misure, pubblicate nel dataset dei risultati; i nostri (§4.1)
  sono misurati da noi con l'adattatore del paragrafo 4 e non sono stati inviati alla classifica. L'adattatore è
  nostro: un altro modo di mettere istruzioni e criteri nel formato di Skejja potrebbe dare numeri diversi, ma
  sceglierlo guardando il test violerebbe la politica di S1MB.
- **Giro F:**
  - un solo seme, senza un giro di controllo con la stessa ricetta partendo dal modello zero-shot: non sappiamo
    quanto del guadagno dipenda dal partire da ft5;
  - le etichette di ampiezza sono di un solo insegnante e nessuna persona le ha controllate;
  - l'accordo con la classe voluta misura la coerenza fra il generatore e l'etichettatore, che sono lo stesso
    modello, non la correttezza;
  - le Choice con molte opzioni (S1MB, CLINC con 151) non sono state addestrate;
  - le misure sono in fp32 sulla GPU, tranne l'int8 dinamico su MASSIVE (Mac); su Raspberry Pi ft6 non è ancora
    misurato;
  - per le azioni degli agenti manca il test appaiato.
- I testi del metro (split di training di SST-2, AG News, Emotion e SMS Spam) restano solo in locale, in
  `bench/data/broad/teacher_*_in.jsonl`, escluso da git. Si rigenerano con `bench/teacher_label.py prepara`.
