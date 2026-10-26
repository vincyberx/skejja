# Skejja — Fase 2, giro G: il formato di S1MB

_14 ottobre 2026. Questo documento fissa **prima di generare i dati e di addestrare** cosa cambia nel giro G,
quali obiettivi ci diamo e con quale regola decidiamo se il candidato sostituisce `edge32k-ft6`. La stessa cosa in
forma leggibile da uno script è in [giro_G_obiettivi.json](../results/fase2/giro_G_obiettivi.json), generato da
`python bench/giro_G_obiettivi.py` dai file di `results/`: nessun numero di partenza è scritto a mano, e
`--check` verifica che il file sia ancora coerente con i CSV. Il punto di partenza è il §6 di
[fase2_ampiezza.md](fase2_ampiezza.md): il giro F ha recuperato ampiezza sui testi singoli (SMS Spam F1 da 0,092
a 0,749, SST-2 da 0,711 a 0,807) ma su S1MB non ha quasi cambiato niente (Noul 0,69, Choice 9,75), perché quel
banco chiede altre tre cose: **ingressi lunghi**, **molte opzioni** e la **primitiva Score**._

## 1. Cosa cambia: il formato

Le misure del formato vengono da [s1mb_lunghezze_ft6.json](../results/fase2/s1mb_lunghezze_ft6.json): la
lunghezza in token della resa di Skejja (istruzioni e stato nel testo, criteri come opzioni) con il tokenizer del
bundle ft6, su tutte le 29.469 decisioni presenti nei file di `english-v1` (le 137 prove ammesse ne usano 26.269).

### 1.1 Ingressi fino a 1.024 token

L'addestramento si è fermato a 192 token (giro F: 320), mentre l'architettura regge 8.192 posizioni. Oggi solo il
**31,4%** delle decisioni di S1MB sta dentro la lunghezza di addestramento; a 1.024 token ci sta l'**84,4%**.
Più in là il guadagno costa molto: 1.536 token aggiungono 7 punti di copertura e 2.048 ne aggiungono altri 2,6.

| Budget | Quota di decisioni coperte | Mediana / p90 per famiglia |
|---|---:|---|
| 192 (oggi) | 31,4% | Choice 253 / 2.541, sì/no 424 / 1.030, Score 666 / 1.255 |
| 512 | 54,2% | |
| **1.024 (giro G)** | **84,4%** | Choice 78,1%, sì/no 89,9%, Score 81,1% |
| 2.048 | 94,1% | |

#### Costo misurato sul Pi 4 (14 ottobre 2026, ft6 int8 dinamico)

Misura fatta **prima** di generare i dati, con `kit/bench_pi.py --cost` su Raspberry Pi 4 Model B Rev 1.5
(tabella [costo_lunghezza_opzioni_pi4.csv](../results/fase2/costo_lunghezza_opzioni_pi4.csv), figura
[costo_formato_pi4.png](../results/fase2/costo_formato_pi4.png); 30 punti, tutti riusciti, nessuno ucciso per
memoria). Il massimo sulle 8/20/50 opzioni:

| Token dell'ingresso | RAM di picco (arena on / off) | p50 a 1 thread | p50 a 4 thread |
|---:|---:|---:|---:|
| 63 | 191 / 191 MB | 468 ms | 206 ms |
| 255 | 203 / 196 MB | 2,20 s | 0,96–1,23 s |
| 511 | **246 / 231 MB** | 5,12–5,20 s | 2,44–3,08 s |
| 1.023 | **398 / 357 MB** | 13,3–13,5 s | 7,15–8,31 s |

Tre cose cambiano il quadro rispetto alla stima sul Mac:

1. **A 1.024 token la RAM sfora i 300 MB, e disattivare l'arena non basta** (398 MB, 357 con `--low-mem`). A 512
   token si sta dentro con margine (246 MB, 231 con l'arena disattivata).
2. **Sul Pi gli ingressi lunghi non sono tempo reale, a nessun budget.** Già a 512 token servono 2,4–3,1 s a 4
   thread. Il tempo reale resta il carico normale: frasi corte e 8 opzioni, dove ft6 fa 380 ms a 1 thread e 182 ms
   a 4 (giro 3 del 14 ottobre, stesse cifre di ft5: 379 e 170 ms), con 216–227 MB di RAM, accuratezza Choice 0,854
   e AUROC sì/no 0,975 sulle 500 domande del kit. **ft6 è quindi confermato anche su Pi 4**, uguale a ft5.
3. **A 4 thread le opzioni non sono gratis**, al contrario di quanto sembrava sul Mac: a parità di lunghezza, 50
   opzioni costano dal +16% (1.023 token) al +29% (255 token) rispetto a 8. A 1 thread la differenza sparisce,
   nascosta dal costo dell'encoder.

Avvertenza sul rallentamento termico: il Pi 4 è senza dissipatore e ha toccato 81 °C; `get_throttled` finale
`0xe0008`, nessuna sottotensione. Il kit ha segnato come non affidabili le latenze dei giri 1 e 3 (4 campioni su
99 e 6 su 51 con rallentamento in corso); il giro con l'arena disattivata era pulito ed è anche il più lento
(15,8 s a 1.023 token), quindi le latenze qui sopra vanno lette come **limiti inferiori**. La RAM non ne risente,
ed è la RAM a decidere il budget.

#### Decisione: 1.024 token in addestramento, 512 come inviluppo dichiarato sul Pi

- **Addestriamo a 1.024 token.** Il costo all'inferenza dipende dall'ingresso che si dà al modello, non dalla
  lunghezza a cui è stato addestrato: a parità di ingresso corto, un modello addestrato a 1.024 costa quanto uno
  addestrato a 512, e il bundle ha la stessa dimensione. Addestrare a 512 ridurrebbe la copertura di S1MB dall'84%
  al 54% senza far risparmiare un byte sul Pi.
- **Sul Pi 4 dichiariamo 512 token e fino a 50 opzioni**, con 246 MB di RAM misurati. Gli ingressi più lunghi
  restano supportati dal modello e si misurano (398 MB, 7–8 s a 4 thread), ma non rientrano in quello che
  dichiariamo per quella scheda.
- **Emendamento al §3, fatto ora e non dopo aver visto i risultati:** la riga «RAM ≤ 300 MB sul carico nuovo
  (1.024 token, 50 opzioni, Score)» diventa «RAM ≤ 300 MB dentro l'inviluppo dichiarato (512 token, 50 opzioni)»,
  e il costo a 1.024 token passa da limite rigido a numero dichiarato. Il motivo è un fatto misurato
  sull'hardware, non un risultato del modello: nessuna misura di qualità esisteva ancora quando questa riga è
  stata cambiata. Gli altri limiti rigidi restano come erano.

### 1.2 Choice fino a 50 opzioni

Le Choice di S1MB arrivano a 151 opzioni; il **93,7%** delle decisioni Choice ne ha al massimo 50. Addestriamo
fino a 50: copre la banda 11–50 (dove ft6 prende 31,1 su 100, il suo risultato migliore) e la fascia dei 36
dell'estrazione di dati personali, lasciando fuori per ora CLINC (151), Banking77 (77), HWU64 (64) e MTOP (62).
Le opzioni restano sempre rimescolate (il 10% di dipendenza dall'ordine dell'insegnante, §5 di
fase2_ampiezza.md, è il motivo per cui lo facciamo).

### 1.3 Primitiva Score

Una Score di S1MB è una scala ordinale con un valore numerico per livello: 6.286 decisioni su 6.748 hanno 5
livelli e 6.168 usano la scala 0–4. Il punteggio è l'errore sul **valore atteso**, normalizzato sull'ampiezza
della scala (`abs(Σ valore·(p − t)) / (max − min)`, letto in `evaluator/src/s1mb/metrics.py`).

Questo entra nell'architettura senza cambiarla: **Score è una Choice sui livelli**, dove ogni livello è
un'opzione con la sua descrizione e il suo valore, e la risposta è la distribuzione sui livelli più il valore
atteso Σ p·valore. Serve quindi:

- in addestramento, bersagli morbidi dell'insegnante sui livelli, con una penalità che tiene conto della distanza
  fra livelli (sbagliare di un livello non è come sbagliare di quattro);
- nel runtime, `Skejja.score(testo, livelli, valori)` che restituisce distribuzione e valore atteso, con la sua
  calibrazione (temperatura fittata solo sui `calib_*`, come per le Choice);
- nell'adattatore S1MB, la Score al posto dell'eccezione, dichiarando che la scala viene presa dai `value` della
  prova e che l'ordine dei livelli non viene toccato.

Scala predefinita dei nostri dati: **0–4 a cinque livelli**, quella prevalente del banco.

### 1.4 Resa «istruzioni + stato + criteri»

I dati del giro G usano la stessa resa dell'adattatore (`instructions-then-state-in-text, criteria-as-labels-v1`):
istruzione in testa, poi lo stato (testo o JSON compatto), criteri come opzioni. Oggi le istruzioni sono il
campo che Skejja perde: su S1MB sono 143 token in media (mediana 33, massimo 337). Addestrare con quella resa
serve a far usare al modello l'istruzione invece di ignorarla, ed è anche il formato che un'API compatibile con
`POST /v1/systemone` dovrà accettare (fase 3).

## 2. Obiettivi del giro G

Primari, su S1MB english-v1, con la configurazione dell'adattatore fissata e una misura sola (§4.1 di
fase2_ampiezza.md). Fra parentesi il riferimento: Bekko 68M, il modello della nostra fascia.

| Misura | ft6 | Obiettivo | Bekko 68M |
|---|---:|---:|---:|
| Score: prove misurate | 0/21 | **21/21, zero decisioni fallite** | 21/21 |
| Score: punteggio di famiglia | — | **≥ 10** | 26,85 |
| Noul: punteggio di famiglia | 0,69 | **≥ 8** | 42,91 |
| Noul: AUROC mediana delle 59 prove | 0,474 | **≥ 0,60** | — |
| Noul: prove con AUROC ≥ 0,6 | 5 su 59 | **≥ 20 su 59** | — |
| Choice: punteggio di famiglia | 9,75 | **≥ 15** | 51,62 |
| Choice: prove con 11–50 opzioni | 31,1 | **≥ 40** | — |
| Choice: prove con 2–4 opzioni | 3,4 | **≥ 8** | — |
| Decisioni entro la lunghezza di addestramento | 31,4% | **84,4%** | — |

Gli obiettivi sono deliberatamente modesti rispetto a Bekko 68M, che dichiara esposizione su 98 prove su 137 e si
è addestrato proprio sulle famiglie di sorgenti del banco: qui si tratta di passare dal livello del caso a un
livello utile, non di raggiungerlo. Le sì/no generali sono il punto più difficile (39 prove su 59 sotto 0,5) e il
rischio maggiore del giro.

Secondari, in casa: un banco «formato» tenuto da parte per `text_id` (stati lunghi, molte opzioni, relazioni,
Score) misurato e riportato, mai usato per scegliere; e il costo misurato sul Pi 4 per il carico nuovo.

## 3. Regola di non regressione (pre-registrata)

Il giro F ha mostrato il problema: il criterio era «nessuna perdita significativa su nessuna misura», SIB-200 in
inglese ha perso 5,4 punti, e la correzione per confronti multipli è stata scelta **dopo** aver visto il
risultato. Qui l'insieme di guardia, il test e le tolleranze sono fissati prima.

**Insieme di guardia** (28 misure, valori di partenza di ft6 nel file JSON):

| Famiglia | Misure | Tolleranza |
|---|---|---|
| **A** — MASSIVE, il carico che spediamo | Choice tutte (0,8725) e mai viste (0,7414), Choice con opzioni nella lingua (0,8485), sì/no test (0,9612) ed esteso mai visto (0,8399), ECE Choice (0,048) e sì/no (0,038), divario fra lingue (0,0425) | nessuna perdita significativa |
| **B** — azioni degli agenti | AUROC con il comando (0,997), con la descrizione inglese (0,911), nella lingua (0,836) | niente sul comando; fino a 0,02 sulle descrizioni se non significativa |
| **C** — test v0 fuori da MASSIVE | sì/no it (0,883) ed en (0,858), Choice it (0,725) ed en (0,592) | fino a 0,02 per misura, purché la sì/no complessiva non perda in modo significativo |
| **D** — dataset dell'articolo su Jev | 13 righe, da AG News (0,782) a SIB-200 per lingua e SMS Spam F1 (0,749) | al massimo **una** riga in perdita significativa e non oltre 0,03, da dichiarare |

**Test:** bootstrap appaiato per esempio, 10.000 ricampionamenti, intervallo al 95% sulla differenza
`ft7 − ft6` sugli stessi esempi (`bench/compare_runs.py`). Una misura è «in perdita significativa» se
l'intervallo esclude lo zero con stima puntuale negativa **dopo la correzione di Holm dentro la sua famiglia**
(A, B, C, D separate, non tutte insieme). Per le azioni degli agenti va prima aggiunto il salvataggio dei
punteggi per esempio, che oggi manca.

**Limiti rigidi** (una violazione respinge il candidato senza passare dalla statistica):

| Limite | Valore | Riferimento |
|---|---:|---|
| Pi 4, Choice 8 opzioni, 1 thread, p50 | ≤ 400 ms | ft5 379 ms, ft6 380 ms |
| Pi 4, sì/no, 1 thread, p50 | ≤ 200 ms | ft5 126 ms, ft6 126 ms |
| Pi 4, RAM di picco, carico normale | ≤ 300 MB | ft5 227 MB, ft6 227 MB |
| Pi 4, RAM di picco dentro l'inviluppo dichiarato (512 token, fino a 50 opzioni) | ≤ 300 MB | ft6 246 MB, il massimo fra 8, 20 e 50 opzioni a 511 token (243 con 50; 231 con arena off) |
| Pi 4, RAM di picco a 1.024 token (fuori dall'inviluppo) | solo dichiarata | ft6 398 MB (357) |
| ECE calibrata delle Choice (MASSIVE) | ≤ 0,05 | ft6: 0,048 |
| ECE delle sì/no (Platt, MASSIVE) | ≤ 0,05 | ft6: 0,038 |
| Dimensione del bundle int8 dinamico | ≤ 130 MB | ft6: 124.288.004 byte |

I primi tre limiti valgono sul carico normale di `bench/success_criteria.json` (frasi corte, 8 opzioni): il giro G
non deve peggiorare il caso d'uso che il kit misura da tre giri, e la misura del 14 ottobre dice che ft6 lo
rispetta (380 ms, 126 ms, 227 MB, dentro i limiti anche con il rallentamento termico in corso). Per gli ingressi
lunghi il limite vale **dentro l'inviluppo dichiarato** di 512 token, misurato a 246 MB; il costo a 1.024 token si
riporta senza limite, perché su quella scheda non lo dichiariamo (§1.1). La latenza degli ingressi lunghi non ha
un limite: a 512 token sono 2,4–3,1 s a 4 thread, cioè fuori dal tempo reale per definizione.

**Capacità nuove:** stati lunghi, 50 opzioni e Score non hanno un valore di partenza. Non entrano nell'insieme di
guardia e non possono «compensare» una perdita: si riportano e basta.

**Condizione di adozione.** Il candidato diventa principale se, tutte insieme: (a) nessuna perdita significativa
oltre le tolleranze; (b) limiti rigidi rispettati; (c) Score supportata su 21/21 prove con zero decisioni
fallite; (d) almeno uno fra «Choice di S1MB ≥ +5 punti» e «Noul di S1MB ≥ +5 punti». Se mancano solo (c) o (d),
il candidato resta alternativo e ft6 resta principale, e il giro si documenta lo stesso con i suoi numeri. Ogni
perdita ammessa dalle tolleranze va dichiarata come riserva qui e in `ROADMAP.md`, come per SIB-200 nel giro F.

**Scelta della variante:** solo sulla media di selezione dello sviluppo. S1MB e i test non scelgono niente: la
politica del banco vieta di tarare qualunque parametro, soglia o checkpoint guardando i suoi punteggi, e ogni
banco si misura una volta sola con la configurazione fissata qui.

## 4. Separazione dal banco di prova e fonti dei dati

Vale il §1 di [fase2_ampiezza.md](fase2_ampiezza.md), con un'aggiunta per le relazioni fra due testi.

1. **Mai in addestramento:** i dataset dell'articolo su Jev (nemmeno i loro split di training), S1MB in
   qualunque forma, le 10 intenzioni dei 4 ambiti esclusi di MASSIVE, il dataset di addestramento di Bekko, i
   1.600 testi del metro dell'insegnante.
2. **Fonti delle relazioni.** Una fonte candidata è esclusa se coincide con una delle 110 famiglie di sorgenti
   delle prove di S1MB, elencate in
   [bench/data/broad/s1mb_sottoinsiemi.json](../bench/data/broad/s1mb_sottoinsiemi.json) (unione fra la chiave
   `evaluation` del `training-manifest.json` del dataset e le cartelle dei dati, revisione `6aa5f397`). Questo
   esclude in blocco le strade più comode: SNLI, SciTail, HANS, RuleTaker, FOL-NLI, temporal-NLI, PAWS, WIQA e
   tutta la pertinenza costruita su BEIR (`synthetic_relevance__nanobeir__*`, cioè MS MARCO, NQ, HotpotQA,
   FEVER, SciFact, Quora e le altre).
3. **Dove prendiamo le relazioni allora.** Testo con licenza permissiva già in casa o approvato al §2 di
   fase2_ampiezza.md (FineWeb e FineWeb-2 ODC-BY, Aya Apache-2.0, mfaq CC0) più i testi sintetici di
   Qwen3.5-9B (Apache-2.0), con le coppie A/B e i livelli scritti ed etichettati dall'insegnante. Alle fonti
   esterne si aggiunge quello che passa il controllo delle licenze di `python bench/fonti_relazioni_audit.py`
   (21 candidati letti sull'Hub il 14 ottobre 2026,
   [fonti_relazioni_licenze.csv](../results/fase2/fonti_relazioni_licenze.csv) e il suo riepilogo):

   | Fonte | Licenza | Lingue dichiarate | A cosa serve |
   |---|---|---|---|
   | google-research-datasets/circa | CC BY 4.0 | en | risposte sì/no indirette: relazione domanda/risposta |
   | deepset/germanquad, germandpr | CC BY 4.0 | de | pertinenza domanda/paragrafo, con negativi difficili |
   | AgentPublic/piaf | MIT | fr | pertinenza domanda/paragrafo |
   | miracl/miracl | Apache-2.0 | 18, fra cui de, es, fr, en (**non** it) | pertinenza multilingue |
   | clips/mfaq | CC0-1.0 | 21, comprese le nostre 5 | domande vere con la loro risposta |
   | nvidia/HelpSteer, HelpSteer2, HelpSteer3 | CC BY 4.0 | HelpSteer3 anche it, de, es, fr | giudizi 0–4: scala ordinale pronta per Score |
   | openbmb/UltraFeedback | MIT | en | giudizi 1–5 su istruzione e risposta |
   | ucberkeley-dlab/measuring-hate-speech | CC BY 4.0 | en | gravità su scala, voci ordinali |

   **Escluse per coincidenza con una prova di S1MB** (regola 2): SNLI, SciTail, PAWS, hh-rlhf, Aegis 2.0,
   civil_comments e tutta la pertinenza su BEIR. **Escluse per licenza:** MultiNLI (nel mucchio c'è
   CC BY-SA 3.0), `tasksource-instruct-v0` e `stsb_multi_mt` (entrambe «other», cioè licenza non dichiarata in
   chiaro). Resta il fatto che per le relazioni vere e proprie sopravvive quasi solo Circa, in inglese: la
   famiglia `relazione` del giro G resta per la maggior parte scritta dall'insegnante, e le fonti esterne
   servono soprattutto per la pertinenza e per Score.
4. **Esposizione dichiarata.** Dove una fonte coincide comunque con una famiglia di sorgenti del banco, la prova
   corrispondente va dichiarata nell'adattatore (`TrainingExposure`, `same_source_family`) e nei documenti, come
   facciamo già per `massive-choice-test-v1`, `clinc-choice-test-v1` e `banking77-choice-test-v1` e come fa
   Bekko con le sue 98 prove. Il conteggio delle nostre esposizioni va riportato accanto ai punteggi, insieme
   alla versione senza quelle prove.
5. **Controllo automatico:** sovrapposizione dei testi normalizzati contro tutti i file di test, i metri
   dell'insegnante e i testi delle prove di S1MB; l'assert di `train_skejja2.py` esteso ai dati nuovi.

## 5. I dati del giro G (15 ottobre 2026)

Costruiti con `bench/build_giroG.py` (pool), `bench/build_giroG_richieste.py` (richieste nella resa del §1.4) e
`bench/teacher_label.py` (etichette), controllati con `bench/qualita_giroG.py`. Numeri da
[pool_giroG_riepilogo.json](../bench/data/broad/pool_giroG_riepilogo.json),
[richieste_giroG_riepilogo.json](../bench/data/broad/richieste_giroG_riepilogo.json),
[qualita_giroG.csv](../results/fase2/qualita_giroG.csv) e il suo riepilogo.

**Pool: 21.862 testi.** 3.000 passaggi web lunghi (600 per lingua, mediana 2.136 caratteri), 5.734 coppie
domanda/documento (mfaq 2.161 nelle 5 lingue, GermanQuAD 1.500, PIAF 1.500, MIRACL 573), 8.000 coppie di Circa
e 5.128 giudizi ordinali (HelpSteer3 2.128 nelle 5 lingue, UltraFeedback 3.000). Di MIRACL è entrato solo lo
spagnolo: il ramo parquet convertito non ha il `dev` per tedesco, francese e inglese, e `datasets` 4.x non
esegue più gli script di caricamento.

**Richieste: 70.362** (39.202 sì/no, 24.028 Choice, 7.132 Score), con mediana 61 token, p90 546 e massimo 1.024;
12.000 Choice hanno più di 10 opzioni, fino a 49. Per lingua: 33.966 inglese, 11.828 tedesco, 11.284 francese,
8.145 spagnolo, 5.139 italiano — l'inglese pesa più che nel giro F perché Circa e UltraFeedback sono solo
inglesi.

**Etichette: 58.330 dai corpora, 12.032 dall'insegnante.** La parte grande non costa niente all'insegnante e non
è un'etichetta sintetica: pertinenza (il documento giusto contro negativi difficili della stessa pagina o dello
stesso recupero), classi di Circa scritte da persone, giudizi di HelpSteer3 e UltraFeedback, intenzioni delle
fonti di casa per le Choice con molte opzioni. L'insegnante (Qwen3.5-9B, 1.669 s, 7,2 richieste al secondo,
4.464 token di prompt al secondo) serve per la pertinenza graduata 0-4 e per le famiglie del giro F applicate ai
testi lunghi. Tenute 11.858 su 12.032: 17 scartate per copertura bassa e 157 perché l'insegnante non conferma la
classe voluta.

**Separazione verificata.** Il controllo confronta ogni pezzo di stato lungo almeno 80 caratteri con i testi di
test, sviluppo, calibrazione e metro del progetto **e** con le 31.605 impronte degli stati di S1MB
([s1mb_testi_hash.json](../bench/data/broad/s1mb_testi_hash.json), solo hash: il testo del banco non entra nel
repository). Risultato: **nessun testo in comune**. Un primo giro aveva segnalato 336 richieste: erano le frasi
di `train_utterances.jsonl`, cioè i nostri stessi dati di addestramento, e il controllo è stato corretto perché
l'insieme proibito contiene i file di prova, non quelli di addestramento.

**Livelli delle Score.** Pertinenza graduata: 1.003 al livello 0, 493 al 4, solo 13 al 2 — l'insegnante usa poco
i livelli intermedi, e la scala resta sbilanciata verso gli estremi. Preferenza di HelpSteer3: distribuzione
larga sui sette livelli (da 135 a 444 per livello). Giudizio di UltraFeedback: concentrato sul livello 3 (1.583
su 3.000).

**Prova del trainer (150 passi, RTX 3090).** `train_skejja2.py` con la primitiva Score, ingressi a 1.024 token e
batch a budget di token gira: la perdita Score scende da 1,92 a 1,85, l'errore sul valore atteso dello sviluppo
(la metrica di S1MB) da 0,262 a 0,232, le sì/no di ampiezza stanno a 0,781 di AUROC e le Choice di ampiezza a
0,792, mentre MASSIVE resta a 0,935 di AUROC in inglese. **Costo: 6,1 s per passo** contro 0,236 del giro F, per
via dei 1.024 token e delle molte opzioni: un giro intero come il giro F (5.086 passi) richiederebbe circa 8,5
ore, quindi la ricetta definitiva dovrà scegliere fra meno passi, un budget di token per batch più basso o una
quota minore di esempi lunghi.

## 6. Il candidato `edge32k-ft7` e il verdetto (16 ottobre 2026)

**Addestramento.** Partenza da ft6 con distillazione da ft3 sui soli dati MASSIVE (peso 0,5), lr 2·10⁻⁵ e testa
6·10⁻⁵, 12.000 esempi sulle azioni, ingressi fino a 1.024 token, batch da 32 con tetto di 16.384 token; dati di
ampiezza uniti (giro F e giro G) con tetto di 20.000 Choice, 24.000 sì/no e tutte le 6.616 Score tenute dai
filtri. Il giro è stato **fermato dal tetto delle 12 ore al 96% dei passi** (exit 124): il candidato è il
checkpoint scelto sullo sviluppo al passo 4.000 su 4.649 pianificati, salvato prima dello stop. Il tetto era una
mia stima sbagliata del ritmo (9,6 s per passo misurati contro 6 previsti); lo sviluppo saliva ancora, quindi la
ricetta non è stata portata a termine.

**Sviluppo** (stessa distribuzione dei dati di addestramento, quindi favorevole): errore sul valore atteso delle
Score 0,262 → 0,174, sì/no di ampiezza 0,825 → 0,964 di AUROC, Choice di ampiezza 0,799 → 0,862, media di
selezione 0,915 → 0,938, azioni 0,999 → 0,998.

**A. MASSIVE, differenze appaiate ft7 − ft6** (bootstrap al 95%, [metrics_diff_ft7_vs_ft6.csv](../results/fase2/metrics_diff_ft7_vs_ft6.csv),
valori in [metrics_edge32k-ft7.csv](../results/fase2/metrics_edge32k-ft7.csv))

| Misura | ft6 | ft7 | ft7 − ft6 [IC 95%] | n |
|---|---:|---:|---:|---:|
| Choice, tutte | 0,8725 | 0,8760 | +0,004 [−0,009; +0,016] | 2.000 |
| Choice, intenzioni mai viste | 0,7414 | 0,7207 | −0,021 [−0,055; +0,017] | 290 |
| Choice, opzioni nella lingua della frase | 0,8485 | 0,8520 | — ¹ | 2.000 |
| Divario Choice fra inglese e lingua peggiore | 0,0425 | 0,0425 | — | 2.000 |
| Sì/no, test (AUROC) | 0,9612 | 0,9628 | +0,002 [−0,002; +0,006] | 2.000 |
| **Sì/no, test esteso mai visto (AUROC)** | 0,8399 | 0,8299 | **−0,010 [−0,015; −0,005]** | 3.655 |
| ECE calibrata delle Choice | 0,0478 | 0,0453 | — | 2.000 |
| ECE delle sì/no (Platt) | 0,0377 | 0,0453 | — | 2.000 |

¹ Senza test appaiato: `eval_noul.py --native` non salvava i punteggi per esempio delle opzioni nella lingua
(corretto il 16 ottobre per il giro H). Valori in [metrics_edge32k-ft7_native.csv](../results/fase2/metrics_edge32k-ft7_native.csv).

**Le opzioni nella lingua della frase restano 2,4 punti sotto quelle inglesi** (0,852 contro 0,876; in ft6 0,8485
contro 0,8725): il giro G non chiude quel divario, e il tedesco resta la lingua peggiore (0,850 con le opzioni
inglesi, 0,7925 con quelle tedesche). *Correzione del 16 ottobre:* la prima versione di questo paragrafo dava
0,876 con le opzioni nella lingua e scriveva che il divario «si chiude (4,25 punti → 0)»; 0,876 era il valore con
le opzioni inglesi, letto dalla riga sbagliata, e 4,25 punti è un'altra misura (inglese contro lingua peggiore,
rimasta uguale). La sì/no sul test esteso delle intenzioni mai viste perde in modo significativo, e la famiglia A
non tollera perdite significative.

**B. Azioni degli agenti** ([metrics_agent_tldr_ft7.csv](../results/fase2/metrics_agent_tldr_ft7.csv)): comando
0,997 → 0,996, descrizione inglese 0,911 → 0,901, descrizione nella lingua 0,836 → 0,841. Niente test appaiato:
lo script non salva i punteggi per esempio, come già dichiarato nel giro F.

**C. Test v0 fuori da MASSIVE** ([metrics_ood_v0_ft7.csv](../results/fase2/metrics_ood_v0_ft7.csv)): agenti
0,938 → 0,951 in italiano e 0,941 → 0,954 in inglese, domotica 0,910 → 0,914 e 0,876 → 0,880, **email 0,851 →
0,787 in italiano e 0,776 → 0,717 in inglese**. Il confronto appaiato non è stato eseguito perché `eval_ood.py`
non ha l'opzione, quindi la famiglia C resta valutata solo sulle stime puntuali: è un buco dichiarato.

**D. Dataset dell'articolo su Jev** ([metrics_jev_suite_ft7.csv](../results/fase2/metrics_jev_suite_ft7.csv))

| Dataset | ft6 | ft7 | Differenza |
|---|---:|---:|---:|
| AG News | 0,782 | 0,774 | −0,008 |
| Rotten Tomatoes | 0,731 | 0,728 | −0,003 |
| SST-2 | 0,807 | 0,789 | −0,018 |
| Emotion | 0,358 | 0,326 | −0,033 |
| Financial PhraseBank | 0,472 | **0,519** | +0,046 |
| Language ID | 0,208 | 0,199 | −0,009 |
| **SMS Spam, F1** | 0,749 | **0,557** | **−0,192** |
| Banking77 ⚠ in dominio | 0,575 | 0,598 | +0,022 |
| SIB-200, solo inglese | 0,676 | **0,745** | +0,069 |
| SIB-200, media 5 lingue | 0,740 | 0,767 | +0,027 |

**E. S1MB english-v1** (configurazione del §4.1, una misura sola, 26.269 decisioni, **zero fallite**:
[metrics_s1mb_edge32k-ft7.csv](../results/fase2/metrics_s1mb_edge32k-ft7.csv))

| Misura | ft6 | ft7 | Obiettivo |
|---|---:|---:|---:|
| Score: prove misurate | 0/21 | **21/21, zero decisioni fallite** | 21/21 ✔ |
| Score: punteggio di famiglia | — | 1,75 | ≥ 10 ✘ |
| Noul: punteggio di famiglia | 0,69 | 2,49 | ≥ 8 ✘ |
| Noul: prove sopra la linea di base | 10/59 | 17/59 | — |
| Choice: punteggio di famiglia | 9,75 | 9,44 | ≥ 15 ✘ |
| Choice: prove con 11–50 opzioni | 31,1 | 22,0 | ≥ 40 ✘ |
| Choice: prove con 2–4 opzioni | 3,4 | 3,9 | ≥ 8 ✘ |
| Decisioni oltre la lunghezza di addestramento | 57,6% | **15,8%** | ≤ 15,6% ≈ ✔ |

La primitiva Score funziona davvero (21 prove su 21, nessuna decisione fallita) ma sta appena sopra la linea di
base: solo 2 prove su 21 guadagnano qualcosa, e le prove di pertinenza graduata di S1MB, che sono i tre quarti
della famiglia, non si muovono. Le Choice con molte opzioni **peggiorano** (31,1 → 22,0) nonostante 12.000
esempi addestrati fino a 49 opzioni.

### Verdetto: candidato respinto, `edge32k-ft6` resta principale

Contro la condizione di adozione del §3: (a) **violata** — perdita significativa sulla sì/no del test esteso
(famiglia A, che non ammette perdite) e più righe in perdita nella famiglia D, fra cui SMS Spam a −19 punti,
contro il massimo di una riga e 0,03 ammessi (quattro righe con il test appaiato, sotto); (b) limiti rigidi rispettati su ECE e dimensione del bundle (modello int8 dinamico:
124.288.004 byte di pesi più 1,3 MB di grafo, identico a ft6; il `-int8.tar.gz` da 210 MB contiene due varianti,
int8 solo pesi e int8 dinamico), non misurati sul Pi perché il candidato è respinto prima; (c) **soddisfatta** — Score su 21/21 con zero decisioni
fallite; (d) **non soddisfatta** — Choice −0,3 e Noul +1,8, contro i +5 richiesti.

**Rilettura con il test dichiarato (16 ottobre).** Il verdetto sopra era stato scritto su un confronto appaiato a
1.000 ricampionamenti (il valore fisso di `compare_runs.py`), con Holm applicato a mano e con le famiglie C e D
lette sulle stime puntuali. `bench/verdetto_giro.py` applica la regola com'è scritta: bootstrap appaiato a
10.000 ricampionamenti su tutte le famiglie con punteggi per esempio, Holm dentro ogni famiglia, tolleranze lette
alla lettera ([verdetto_giro_G.csv](../results/fase2/verdetto_giro_G.csv)). Il verdetto non cambia, anzi si
rafforza:

- **A violata:** sì/no del test esteso −0,010 [−0,016; −0,005], p di Holm 0,001. La Choice con le opzioni nella
  lingua resta senza test, perché mancano i punteggi per esempio.
- **B non misurata:** `eval_agent.py` non salvava i punteggi per esempio (corretto il 16 ottobre).
- **C violata:** v0 Choice in inglese −0,025, oltre lo 0,02 ammesso per misura. Non è significativa: 120 domande,
  intervallo [−0,10; +0,05]. Le sì/no del v0, invece, sono ferme (−0,002 in italiano e in inglese).
- **D violata:** **quattro** righe in perdita significativa dopo Holm, non due:
  - SMS Spam −0,192;
  - Emotion −0,032;
  - Language ID −0,009;
  - AG News −0,008.

  Tre righe sono in guadagno significativo:
  - SIB-200 inglese +0,069;
  - Financial PhraseBank +0,046;
  - Banking77 +0,022, in dominio.

**Perché, e cosa cambiare.** Il tetto unico sui dati di ampiezza (20.000 Choice e 24.000 sì/no sull'unione) ha
diluito le famiglie del giro F: con 17.202 sì/no di pertinenza e 16.000 di Circa nel giro G, la quota che teneva
in vita spam ed emozione è scesa a circa un terzo (stima dalla quota del giro F nel pool: circa 11.100 Choice
contro 30.114 del giro F, circa 11.900 sì/no contro 37.135), e SMS Spam lo mostra. In più il giro G leggeva
`requests_trial.jsonl`, che non ha le 2.261 varianti di ordine e di lingua del giro F. Per un giro H servono quote separate per fonte
(oggi il trainer non le ha), le famiglie del giro F a piena forza, meno sì/no lunghi di pertinenza, e la ricetta
portata a termine invece che fermata al 96%.

**Cosa resta buono del giro G**, indipendentemente dal candidato: la primitiva Score esiste e misura; il budget
di 1.024 token porta la copertura di S1MB dal 31% all'84% e la quota di ingressi troncati dal 58% al 16%;
SIB-200 in inglese recupera i 5,4 punti persi nel giro F; e i
dati del §5, con 58.330 etichette vere, restano riusabili.

## 7. Giro H: ricetta e regola (fissate il 16 ottobre, prima di addestrare)

File: [giro_H_obiettivi.json](../results/fase2/giro_H_obiettivi.json), generato da `bench/giro_H_obiettivi.py`;
`--check` verifica che guardia, tolleranze, limiti rigidi e separazione siano **copiati senza modifiche** dal
giro G.

**Ricetta.** La stessa del giro G (partenza da ft6, stessi iperparametri, 1.024 token), con una correzione di
sostanza e tre tecniche:

1. **Famiglie del giro F a piena forza**: `teacher_trial_in.jsonl` e `requests_synth.jsonl`, tutti, come nel giro F
   (30.114 Choice e 37.135 sì/no di ampiezza, contati nella prova di ritmo). Tornano anche le 2.261 varianti di
   ordine e di lingua che il giro G aveva perso leggendo `requests_trial.jsonl`.
2. **Tetti per fonte sulle fonti nuove** (`--broad_caps`): Circa 4.000 sì/no e 3.000 Choice (16.000 e 8.000 nel
   giro G), pertinenza 1.500 per fonte, cioè 6.000 in tutto (17.202), Choice con 11–49 opzioni 6.000 (12.000).
   Stati lunghi e Score entrano tutti. Niente tetti per tipo. Totale: 42.955 Choice, 52.832 sì/no e 6.616 Score
   di ampiezza.
3. Tecnica: le sì/no di ampiezza in batch propri con un tetto di 16.384 token (`--bnoul_budget`); un tetto che
   conta i token dopo il padding, con gli esempi ordinati per lunghezza dentro blocchi casuali; il teacher Skejja
   calcolato solo dove entra nella perdita. Le perdite sono le stesse; cambia la composizione dei batch.
4. Il giro va portato a termine. **Prova di ritmo** di 300 passi: 134 s, cioè 0,45 s per passo contro gli 8–10 del
   giro G. La causa era il padding: nel giro G un solo stato lungo portava a 1.024 token un batch sì/no da 32. La
   ricetta intera è di circa 6.400 passi, quindi meno di un'ora; il tetto del lavoro è di 4 ore.

**Regola.** Insieme di guardia, test, Holm per famiglia, tolleranze e limiti rigidi sono quelli del §3. Cambia
solo la condizione di guadagno: Score su 21/21 con zero decisioni fallite, **e** almeno uno fra (i) la Choice di
MASSIVE con le opzioni nella lingua della frase in guadagno appaiato significativo su ft6, (ii) la famiglia Score
di S1MB almeno 1 punto sopra ft6, misurato con lo stesso adattatore, (iii) Choice o Noul di S1MB a +5. Il motivo
è scritto nel file: il giro G ha mostrato che con questi dati S1MB sale di 1–2 punti, quindi con la condizione
(d) del giro G il giro H sarebbe respinto per costruzione anche senza regressioni. Se il guadagno manca, ft6 resta
principale e la fase 2 si chiude lo stesso.

**Avvertenze dichiarate prima.** ft8 è il terzo candidato misurato sugli stessi test di guardia. La Choice con le
opzioni nella lingua è sia una guardia sia una via di guadagno. E il confronto appaiato del giro G girava a 1.000
ricampionamenti contro i 10.000 dichiarati (valore fisso in `compare_runs.py`, ora opzione `--B` con 10.000 come
default): va rifatto a 10.000 prima del verdetto del giro H. Fatto il 16 ottobre, dopo l'avvio del giro H e prima
di qualunque sua misura: [verdetto_giro_G.csv](../results/fase2/verdetto_giro_G.csv), rilettura nel §6.

## 8. Il candidato `edge32k-ft8` e il verdetto del giro H (16 ottobre 2026)

**Addestramento** ([train_meta_run_H.json](../results/fase2/train_meta_run_H.json)): ricetta del §7 **portata a
termine**, 6.404 passi in 2.716 s, cioè 45 minuti sulla RTX 3090. Il checkpoint migliore sullo sviluppo è l'ultimo
passo. Dati di ampiezza: 42.955 Choice, 52.832 sì/no e 6.616 Score. Sviluppo, da ft6 a ft8:

- MASSIVE (AUROC): inglese 0,931 → 0,926, nella lingua della frase 0,944 → 0,939;
- sì/no di ampiezza: 0,827 → 0,968;
- Choice di ampiezza: 0,802 → 0,869;
- errore delle Score: 0,262 → 0,169.

**Misure di guardia** ([verdetto_giro_H.csv](../results/fase2/verdetto_giro_H.csv), `bench/verdetto_giro.py`).
Il test è quello dichiarato: bootstrap appaiato a 10.000 ricampionamenti e Holm dentro ogni famiglia. **Per la
prima volta tutte e 28 le misure hanno un test appaiato**: ft6 è stato rimisurato con le opzioni nella lingua e
sulle azioni, salvando i punteggi per esempio, e i valori di partenza coincidono con quelli del file degli
obiettivi.

| Famiglia | Misura | ft6 | ft8 | ft8 − ft6 [IC 95%] | Esito |
|---|---|---:|---:|---:|---|
| A | Choice, tutte | 0,8725 | 0,8750 | +0,003 [−0,010; +0,015] | |
| A | Choice, intenzioni mai viste | 0,7414 | 0,7241 | −0,017 [−0,059; +0,024] | |
| A | Choice, opzioni nella lingua della frase | 0,8485 | 0,8550 | +0,007 [−0,004; +0,017] | |
| A | Sì/no, test (AUROC) | 0,9612 | 0,9627 | +0,002 [−0,002; +0,005] | |
| A | **Sì/no, test esteso mai visto (AUROC)** | 0,8399 | 0,8292 | **−0,011 [−0,016; −0,006]** | **perdita, Holm p < 0,001** |
| A | ECE Choice / ECE sì/no | 0,048 / 0,038 | 0,040 / 0,033 | | meglio |
| A | Divario inglese − lingua peggiore | 0,0425 | 0,0500 | +0,008 [−0,018; +0,038] | |
| B | Azioni: comando, descrizione inglese, nella lingua | 0,997 / 0,911 / 0,836 | 0,996 / 0,915 / 0,853 | nessuna differenza significativa | rispettata |
| C | v0 sì/no it / en (AUROC) | 0,883 / 0,858 | 0,873 / 0,852 | −0,010 / −0,006, non significative | rispettata |
| C | v0 Choice it / en | 0,725 / 0,592 | 0,750 / 0,583 | +0,025 / −0,008 | rispettata |
| D | **SMS Spam, F1** | 0,749 | 0,693 | **−0,056 [−0,079; −0,032]** | **perdita oltre 0,03** |
| D | Language ID / Banking77 (in dominio) | 0,208 / 0,575 | 0,231 / 0,614 | +0,023 / +0,038 | guadagni significativi |
| D | le altre 9 righe | | | AG News +0,005, Emotion +0,016, FinPhrase +0,031, SIB-200 inglese +0,059, nessuna perdita significativa | |

**Limiti rigidi:** ECE entro 0,05 (0,040 e 0,033). Il modello int8 dinamico pesa 124.288.004 byte di pesi più
1.308.215 di grafo, cioè 125,6 MB, byte per byte come ft6 e ft7; il `-int8.tar.gz` da 210 MB contiene sia
l'int8 solo pesi sia l'int8 dinamico. Il Pi non è
misurato, perché il candidato è respinto prima, e architettura e dimensione sono identiche a ft6.

**S1MB english-v1** (una misura, configurazione del §4.1; [metrics_s1mb_edge32k-ft8.csv](../results/fase2/metrics_s1mb_edge32k-ft8.csv)).
Per la condizione (ii) **ft6 è stato misurato per la prima volta sulle 21 prove Score**, con lo stesso adattatore
([metrics_s1mb_edge32k-ft6_score.csv](../results/fase2/metrics_s1mb_edge32k-ft6_score.csv)).

| Famiglia | ft6 | ft7 | ft8 | Bekko 68M |
|---|---:|---:|---:|---:|
| Choice | 9,75 | 9,44 | **11,20** | 51,62 |
| Noul | 0,69 | 2,49 | 1,72 | 42,91 |
| Score | 0,89 | 1,75 | 1,73 | 26,85 |
| Decisioni fallite | 0 su 26.269 ¹ | 0 | 0 | |

¹ 21.121 decisioni nella misura del 13 ottobre, senza Score; Score misurata a parte il 16 ottobre: 5.148
decisioni, zero fallite.

### Verdetto: anche ft8 respinto, `edge32k-ft6` resta principale e chiude la fase 2

- **Non regressione: violata due volte.** Nella famiglia A c'è la stessa perdita del giro G sulla sì/no del test
  esteso (−0,011). Nella famiglia D, SMS Spam perde 0,056, oltre lo 0,03 ammesso per l'unica riga tollerata.
  B e C sono rispettate.
- **Guadagno: mancato.** Le Score sono su 21/21 con zero decisioni fallite. Ma le vie di guadagno non tengono:
  (i) la Choice con le opzioni nella lingua guadagna +0,007, non significativo; (ii) la Score di S1MB sale di
  +0,84 punti contro il +1 richiesto; (iii) Choice e Noul salgono di +1,45 e +1,03 contro i +5 richiesti.

**Cosa insegnano i due giri insieme.**

1. **La perdita della famiglia A non viene dalla diluizione.** Con le famiglie del giro F a piena forza resta
   identica: −0,010 nel giro G, −0,011 nel giro H. Dipende da quello che i due giri hanno in comune: addestramento
   a 1.024 token, sì/no lunghe di pertinenza e di Circa, Score, Choice con molte opzioni. Un altro giro non
   dovrebbe cambiare di nuovo la miscela, ma togliere una cosa per volta.
2. **I tetti per fonte hanno recuperato quasi tutta la famiglia D.** SMS Spam passa da −0,192 a −0,056. Emotion,
   AG News e Language ID passano da perdite significative a guadagni.
3. **Su S1MB, con queste fonti e a questa dimensione, il formato porta 1–2 punti per famiglia**, non i 30–40 che
   separano Skejja da Bekko 68M. Le prove di S1MB chiedono capacità (ragionamento, pertinenza graduata) che i
   nostri dati non insegnano. Il prodotto realistico resta quello stretto: azioni, comandi e smistamento in 5
   lingue, con probabilità calibrate, sul Pi.
4. **Il costo del giro G era il padding, non il formato:** 0,42 s per passo contro 8–10, a parità di lunghezza e
   di opzioni.

## 9. Limiti di questo documento

- Gli obiettivi del §2 sono scelte nostre, non soglie di un banco: servono a decidere prima, non a dichiarare un
  risultato.
- Il budget è deciso su una scheda sola, senza dissipatore e con rallentamento termico in corso in due giri su
  tre: le latenze lunghe sono limiti inferiori e andrebbero rifatte con un dissipatore. La RAM, che è ciò su cui
  si è deciso, non è toccata dal rallentamento.
- La regola del §3 non protegge da quello che non misuriamo: il test v0 è ancora scritto dall'agente, le
  etichette di ampiezza sono di un solo insegnante e nessuna persona le ha riviste a campione (limiti del §7 di
  fase2_ampiezza.md, tutti ancora aperti).
- I numeri di S1MB restano misurati da noi con un adattatore nostro, non inviati alla classifica. Un invio si fa
  solo con il consenso dell'utente.
- I dati del §5 hanno tre squilibri dichiarati: l'inglese pesa quasi metà delle richieste, le relazioni vere
  vengono quasi tutte da Circa (una fonte sola, scritta su un dominio ristretto) e i livelli intermedi delle
  Score sono poco rappresentati. Nessuna persona ha ancora riletto un campione delle etichette.
- I file grandi del giro G (pool da 40 MB e richieste da 160 MB) restano sulla workstation: in git ci sono gli
  script, i riepiloghi e le etichette dell'insegnante compresse. Si rigenerano con i comandi del §5, a parità di
  seme e di revisioni delle fonti.
