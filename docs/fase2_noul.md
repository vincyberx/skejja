# Skejja — Fase 2, punto 1: domande sì/no (Noul) sulle intenzioni mai viste

_6 ottobre 2026. I numeri della fase 1 qui sono quelli PyTorch fp32 (Choice 0,869): il report della fase 1 cita in prima pagina l'ONNX int8 (Choice 0,873). Tutti i numeri vengono da [results/fase2/](../results/fase2/) e sono misurati su Mac (Apple Silicon, CPU), modelli PyTorch fp32, con il protocollo di `bench/skejja_measure.py`. Nessun numero sul Raspberry Pi in questo documento._

## Verdetto (aggiornato il 6 ottobre 2026 con il giro C)

> **Aggiornamento del 7 ottobre 2026:** il candidato principale è ora `edge32k-ft5` (giro E, con i controlli sulle azioni degli agenti): vedi [fase2_agente.md](fase2_agente.md). I numeri di ft3 qui sotto restano validi; ft5 è leggermente sotto ft3 sulle intenzioni mai viste.

**Nuovo candidato principale: `edge32k-ft3`** (giro C: dati della fase 2 più descrizioni nella lingua della frase). Rispetto al modello della fase 1 migliora tutto ciò che misuriamo, tranne la calibrazione senza esempi, che resta il problema aperto.

1. **Senza esempi**, sul test esteso delle intenzioni mai viste (3.655 domande, 301 frasi in 5 lingue): AUROC **0,839** [0,819–0,860] contro 0,804 della fase 1 (differenza appaiata **+0,034** [+0,024; +0,045]; contro i negativi fratelli 0,774 contro 0,726). L'ECE però è ancora **0,143**.
2. **Sulle frasi con etichetta pulita** (circa il 31% delle etichette MASSIVE delle mai viste è discutibile, vedi sotto): AUROC **0,902** [0,884–0,918]. L'obiettivo di discriminazione è raggiunto anche senza esempi; quello di calibrazione no (ECE 0,122).
3. **Con la calibrazione per domanda** (`Skejja.calibrate_question`, σ = 0,5):
   - **64 esempi in inglese: ECE 0,032, AUROC 0,904. Entrambi gli obiettivi raggiunti**;
   - 32 esempi: ECE 0,048 (obiettivo raggiunto), AUROC 0,890 (non ancora);
   - pool completo (circa 320 esempi): ECE 0,030, AUROC 0,923;
   - sulle etichette pulite, con 64 esempi: ECE 0,041, AUROC 0,942.
4. **Choice e intenzioni viste:** Choice 0,878 (fase 1: 0,869), sulle mai viste 0,772 (0,748); sì/no sulle viste 0,983 (0,982). Divario fra inglese e lingua peggiore 4,2 punti.
5. **Opzioni nella lingua della frase** (descrizioni del benchmark tradotte con OPUS-MT, solo per il test): Choice 0,848, contro 0,798 della fase 1. Ma resta sotto le opzioni inglesi (0,878), e il divario fra le lingue sale a 10,8 punti (tedesco 0,788), sopra il limite di 8.

In sintesi: per una domanda nuova senza esempi le probabilità vanno dichiarate non calibrate; con circa 64 esempi etichettati gli obiettivi si raggiungono. Il test originale (60 frasi) ha intervalli di ±0,1 e non serve per decidere.

### Risultati del primo giro (prima del controllo delle etichette)

Il nuovo modello `edge32k-ft2` migliora in modo misurabile ma piccolo:

- test esteso: AUROC **0,819** [0,798–0,841] contro 0,804 della fase 1; differenza appaiata **+0,015** [+0,003; +0,026];
- solo contro i negativi dello stesso ambito, il caso difficile: **0,751** contro 0,726 (differenza appaiata +0,026 [+0,009; +0,040]);
- test originale: 0,774 [0,670–0,870] contro 0,711, intervallo troppo largo per contare;
- calibrazione: ECE 0,166 con la Platt globale; 0,091 con una Platt fittata sulle domande mai viste di `calib_noul.jsonl` (per famiglia di domande);
- intenzioni viste e Choice al livello della fase 1: AUROC sì/no sulle viste 0,983 (0,982), Choice 0,861 (0,869), Choice sulle mai viste 0,745 (0,748).

## Cosa non andava: la diagnosi

Ricalcolando il modello della fase 1 con il nuovo script `training/eval_noul.py`:

1. **Il test "mai viste" originale è troppo piccolo.** Le 300 domande sono 60 frasi tradotte in 5 lingue: l'intervallo bootstrap per frase dell'AUROC è 0,597–0,817. Da qui il **test esteso** `bench/data/test_noul_unseen_ext.jsonl` (`bench/build_unseen_ext.py`): tutte le altre frasi del test MASSIVE dei 4 ambiti esclusi, disgiunte dal test originale, con intervallo di circa ±0,02.
2. **Il problema sono i negativi dello stesso ambito.** Sul test esteso, contro negativi di altri ambiti l'AUROC è già 0,987; contro intenzioni sorelle (order/query, post/query, query/ticket/traffic/taxi) scende a 0,726. Il modello riconosce l'argomento, ma non l'azione richiesta.
3. **Una parte delle coppie difficili dipende dalle etichette MASSIVE**, che non sempre corrispondono alle descrizioni brevi del benchmark. Esempi dal test: "order pizza for delivery" è etichettata takeaway_query ("ask about a food delivery order"); "olly i need directions to madison street" è etichettata transport_query ("ask about public transport").
4. Una nota sul formato: in `test_noul.jsonl` il campo `intent` è l'intenzione **della domanda**, non quella vera della frase (che si ricava dal file Choice con gli stessi id). La split "mai viste" della fase 1 è quindi "domanda su un'intenzione mai vista". I nuovi file hanno anche i campi `gold` e `neg_type`.

## Cosa è stato fatto

- **Checkpoint PyTorch ricostruiti** (`training/rebuild_checkpoints.py`). Zero-shot a 32k: dal GLiClass edge originale, con le righe degli embedding prese per contenuto del token; identico all'originale (differenza 0,0) sulle 191+191 frasi con la stessa tokenizzazione. Fine-tuned della fase 1: pesi importati dal bundle ONNX fp32, differenza massima sui logit 1,3·10⁻⁵ (Choice) e 2,1·10⁻⁵ (Noul). Sono in `releases/checkpoints/`, fuori da git ([rebuild_check.json](../results/fase2/rebuild_check.json)).
- **Dati sì/no esterni, solo CC BY** (`bench/build_external.py`): CLINC150, Banking77 e MINDS-14 (testo nativo in it/es/de/fr/en), per 23.540 frasi di addestramento. Sono state scartate 30 intenzioni CLINC affini agli ambiti esclusi (trasporti, cibo a domicilio e ristoranti, cucina, voli, hotel e viaggi), cioè 4.500 frasi; la lista è nel codice.
- **Descrizioni** per 260 intenzioni (`bench/data/ext/descriptions.json`), con 2 parafrasi ciascuna per le 238 usate in addestramento, più 13 coppie quasi duplicate escluse come negativi. Le ha scritte l'agente e vanno riviste. Le descrizioni delle 10 intenzioni MASSIVE mai viste non compaiono mai in addestramento: lo verifica un assert nello script.
- **Traduzione automatica** con OPUS-MT (`training/translate_ext.py`): 21.593 frasi inglesi × 4 lingue, in una cache versionata.
- **Sviluppo e calibrazione su domini esterni tenuti fuori**: due domini CLINC interi, lavoro e auto (22 intenzioni con molte coppie sorelle), in `dev_noul_ext.jsonl` (1.603 domande per lingua) e `calib_noul_ext.jsonl` (2.137 per lingua). Il test MASSIVE non è stato usato per nessuna scelta.
- **Addestramento** (`training/train_skejja2.py`, da riga di comando): per ogni frase un positivo, un negativo fratello e, con probabilità 0,5, un negativo di un altro gruppo compatibile; parafrasi al 50%; mix MASSIVE 40%, CLINC 25%, Banking77 25%, MINDS-14 10%; distillazione leggera dallo zero-shot (peso 0,2). In tutto 59.389 esempi Noul e 24.000 Choice, 2.606 passi, 85 minuti per giro su CPU. Due giri: A dallo zero-shot, B dal fine-tuned della fase 1.
- **Scelta** con l'AUROC sull'intero set di sviluppo ([selezione_dev.csv](../results/fase2/selezione_dev.csv)): fase 1 0,903, A-best 0,931, **B-best 0,938 (scelto)**. Il guadagno sul set di sviluppo (+0,035) è molto più grande di quello sulle intenzioni MASSIVE escluse (+0,015): i domini CLINC tenuti fuori sono più facili e non rappresentano bene le coppie MASSIVE.

## Tabella di confronto

Fonte: [fase2_noul_confronto.csv](../results/fase2/fase2_noul_confronto.csv). Platt globale = protocollo della fase 1 (fittata su `calib_noul.jsonl`). Tra parentesi quadre: intervallo bootstrap al 95% per frase.

| | Zero-shot 32k | Fine-tuned fase 1 | ft2 (giro B) | ft2-A (giro A) | **ft3 (giro C, scelto)** |
|---|---|---|---|---|---|
| AUROC sviluppo esterno (CLINC lavoro+auto, domande inglesi) | — | 0,903 | 0,938 | 0,931 | 0,934 |
| Noul AUROC, tutte | 0,694 | 0,959 | 0,964 | 0,952 | 0,964 |
| Noul AUROC, domande su intenzioni viste | 0,690 | 0,982 | 0,983 | 0,973 | 0,983 |
| Noul AUROC, mai viste, test originale (300) | 0,727 [0,619–0,831] | 0,711 [0,597–0,817] | 0,774 [0,670–0,870] | 0,783 [0,679–0,874] | 0,786 [0,682–0,876] |
| Noul AUROC, mai viste, test esteso (3.655) | 0,716 [0,688–0,743] | 0,804 [0,785–0,828] | 0,819 [0,798–0,841] | 0,826 [0,804–0,848] | **0,839 [0,819–0,860]** |
| … solo negativi dello stesso ambito | 0,647 | 0,726 | 0,751 | 0,763 | **0,774** |
| … solo negativi di altri ambiti | 0,876 | 0,987 | 0,977 | 0,971 | 0,989 |
| ECE test esteso, Platt globale | 0,023 | 0,151 | 0,166 | 0,157 | 0,143 |
| ECE test esteso, Platt su calib esterno | 0,101 | 0,180 | 0,171 | 0,140 | 0,129 |
| ECE test esteso, Platt per famiglia di domande | 0,036 | 0,083 | 0,091 | 0,097 | 0,088 |
| ECE domande su intenzioni viste, Platt globale | 0,047 | 0,032 | 0,044 | 0,039 | 0,042 |
| Choice, accuratezza (tutte / mai viste) | 0,542 / 0,590 | 0,869 / 0,748 | 0,861 / 0,745 | 0,848 / 0,731 | **0,878 / 0,772** |

Nei giri A e B ft2-A aveva numeri sul test leggermente migliori di ft2, ma non è stato scelto perché la regola fissata prima era il set di sviluppo; la differenza fra i due sta dentro gli intervalli. ft3 è stato scelto fra i checkpoint del giro C con la media dell'AUROC di sviluppo con domande inglesi e nella lingua della frase ([selezione_dev_runC.csv](../results/fase2/selezione_dev_runC.csv): C-best 0,938, C-last 0,937). Diventa il candidato principale perché sul test supera la fase 1 su tutte le misure, e non solo sul set di sviluppo, dove ft2 era alla pari (0,941).

**Calibrazione per famiglia di domande** ([metrics_extra_famiglia_domande.csv](../results/fase2/metrics_extra_famiglia_domande.csv)): una Platt fittata solo sulle 270 domande di `calib_noul.jsonl` che riguardano intenzioni mai viste in addestramento. Le frasi sono diverse da quelle del test, ma le domande sono le stesse 10: corrisponde a un uso reale in cui si hanno alcuni esempi etichettati per la domanda nuova. Dimezza l'ECE (0,166 → 0,091 per ft2), ma resta sopra 0,05. Lo zero-shot è ben calibrato (0,023–0,036) solo perché le sue probabilità sono poco informative (AUROC 0,716). Una Platt fittata su domini esterni tenuti fuori (CLINC lavoro+auto) **non** trasferisce la calibrazione alle intenzioni MASSIVE.

**AUROC per domanda**, ft2, test esteso, negativi dello stesso ambito: social_query 0,988, transport_traffic 0,965, transport_ticket 0,958, transport_taxi 0,899, takeaway_order 0,662, transport_query 0,485, takeaway_query 0,479, social_post 0,387. Le due domande cooking non sono valutabili sul test esteso, perché le frasi cooking_query del test MASSIVE sono tutte già nel test originale. Le quattro coppie peggiori sono proprio quelle descritte al punto 3 della diagnosi. ft3: social_query 0,993, transport_traffic 0,969, transport_ticket 0,945, transport_taxi 0,921, takeaway_order 0,725, social_post 0,520, transport_query 0,455, takeaway_query 0,411.


## Controllo delle etichette MASSIVE

File: [unseen_label_audit.csv](../bench/data/annotations/unseen_label_audit.csv); metriche in [metrics_etichette_pulite.csv](../results/fase2/metrics_etichette_pulite.csv). Ho controllato 359 frasi inglesi dei 4 ambiti esclusi (gli id valgono per tutte e 5 le lingue), leggendo solo testo ed etichetta, senza punteggi del modello: 249 `ok`, 39 `desc`, 38 `amb`, 18 `wrong`, 15 `none`. Le cause principali:

- **transport_query** in MASSIVE significa soprattutto indicazioni stradali e navigazione ("directions to disney land", "how do i get home"). La descrizione del benchmark, "ask about public transport", copre solo gli orari dei treni (19 frasi `desc`).
- **takeaway_query** contiene soprattutto domande del tipo "questo locale fa consegne o asporto?". La descrizione, "ask about a food delivery order", presuppone un ordine esistente (20 frasi `desc`). Le etichette sono anche incoerenti: "can i order take out from licari's" è *order*, mentre "can i get take-out from pf changs" è *query*.
- **cooking**: tutte le 72 frasi controllate sono etichettate *recipe*, ma 17 sono domande di cucina ("how long should i boil an egg", "difference between bake and broil").
- **social_post** in MASSIVE è per lo più "lamentati con un'azienda su Twitter". La descrizione calza, ma 16 frasi sono vaghe ("tell comcast i hate them") e 4 fuori tema.

| test esteso mai viste | tutte le frasi (3.655) | solo etichette ok (2.705) |
|---|---|---|
| zero-shot 32k, AUROC | 0,716 | 0,795 |
| fase 1, AUROC | 0,804 [0,785–0,828] | 0,872 [0,852–0,892] |
| ft2, AUROC | 0,819 [0,798–0,841] | 0,897 [0,880–0,913] |
| ft2-A, AUROC | 0,826 [0,804–0,848] | 0,904 [0,887–0,921] |
| **ft3, AUROC** | 0,839 [0,819–0,860] | **0,902 [0,884–0,918]** |

Limite: un solo annotatore (l'agente), un solo passaggio; va rivisto a campione da una persona. Il benchmark v1 resta invariato.

## Calibrazione con pochi esempi per domanda

Implementazione: `runtime/skejja_infer.py`, `Skejja.calibrate_question(domanda, [(testo, 0/1), ...])` e `fit_platt_map`. È una Platt stimata per massimo a posteriori, con prior gaussiano centrato sulla calibrazione globale (σ = 0,5), calcolata in numpy con Newton-Raphson smorzato. Esperimento: `training/fewshot_calib.py`, risultati in [metrics_fewshot.csv](../results/fase2/metrics_fewshot.csv).

**Correzione del 7 ottobre 2026.** La prima versione di `fit_platt_map` (Newton senza ricerca del passo) divergeva per alcune domande, con pendenze 36–47 invece di circa 1. Questo gonfiava l'ECE e aveva falsato anche la scelta di σ (0,3). Ora il passo viene dimezzato finché l'obiettivo non scende; il risultato coincide con scipy (BFGS) sui casi che divergevano. Tutti i numeri qui sotto sono stati ricalcolati con la versione corretta.

- **Esempi**: dallo split *validation* di MASSIVE (`bench/data/calib_noul_fewshot.jsonl`, `bench/build_fewshot_calib.py`), mai dal test; metà positivi e metà negativi, frasi distinte, in inglese. Si misura sul test esteso e sull'originale. 30 estrazioni per ogni k; si riporta la media.
- **σ** scelto sul set di sviluppo esterno (domande CLINC tenute fuori) con k = 16: 0,5 per tutti i modelli fine-tuned. Su quel set l'ECE della fase 1 scende da 0,072 (globale) a 0,026 (k = 16); quella di ft3 da 0,033 a 0,020. A k = 64 risale (0,048 e 0,053): il set ha 30 frasi per domanda, quindi a k = 64 non resta quasi niente da valutare.

| test esteso mai viste | k = 0 | 4 | 8 | 16 | 32 | 64 | pool completo |
|---|---|---|---|---|---|---|---|
| fase 1: ECE | 0,151 | 0,129 | 0,103 | 0,067 | 0,045 | 0,032 | 0,041 |
| fase 1: AUROC complessiva | 0,804 | 0,814 | 0,842 | 0,856 | 0,875 | 0,893 | 0,912 |
| ft2: ECE / AUROC | 0,166 / 0,819 | 0,143 / 0,817 | 0,116 / 0,839 | 0,086 / 0,850 | 0,068 / 0,866 | 0,044 / 0,892 | 0,043 / 0,911 |
| **ft3: ECE** | 0,143 | 0,123 | 0,100 | 0,071 | 0,048 | **0,032** | 0,030 |
| **ft3: AUROC complessiva** | 0,839 | 0,846 | 0,869 | 0,878 | 0,890 | **0,904** | 0,923 |
| ft3, solo etichette ok: ECE / AUROC | 0,122 / 0,902 | 0,106 / 0,907 | 0,088 / 0,925 | 0,065 / 0,930 | 0,043 / 0,939 | 0,041 / 0,942 | 0,033 / 0,947 |
| ft3, test originale (60 frasi): ECE / AUROC | 0,215 / 0,786 | 0,195 / 0,790 | 0,177 / 0,805 | 0,143 / 0,814 | 0,114 / 0,826 | 0,110 / 0,823 | 0,084 / 0,836 |

L'AUROC complessiva cresce perché la calibrazione per domanda allinea le scale delle diverse domande. Dentro una singola domanda l'ordinamento non cambia: la trasformazione è monotona.

**Cosa significa per il prodotto.** Per una domanda nuova senza esempi, le probabilità vanno dichiarate "non calibrate" (ECE circa 0,14). Con 32 esempi etichettati l'ECE scende sotto 0,05; con 64 si raggiungono entrambi gli obiettivi sul test esteso. Il runtime deve sapere quanti esempi ha ricevuto ogni domanda e dirlo; la fase 3 (conformal prediction, astensione) parte da qui. Sul test originale (60 frasi) l'ECE resta sopra 0,1 anche con 64 esempi, ma lì l'intervallo è di circa ±0,1.

## Giro C: descrizioni nella lingua della frase

Motivazione: nel test fuori da MASSIVE ([fase2_ood_v0.md](fase2_ood_v0.md)), con frasi italiane, le opzioni scritte in italiano davano +24 punti nella Choice rispetto alle opzioni inglesi.

- **Dati:** descrizioni e parafrasi delle 238 intenzioni di addestramento tradotte con OPUS-MT (beam 4) in `bench/data/ext/descriptions_mt.json`; ci sono anche le 22 intenzioni tenute fuori, usate solo per lo sviluppo. Le 10 intenzioni MASSIVE mai viste non ci sono.
- **Addestramento:** `train_skejja2.py --p_desc_native 0.5`. Per le frasi non inglesi, metà degli esempi hanno domanda e opzioni nella lingua della frase (tutte le opzioni di un esempio nella stessa lingua). Si parte dal fine-tuned della fase 1, con gli stessi parametri del giro B; 2.608 passi su 10 thread.
- **Misura con opzioni native su MASSIVE** (`eval_noul.py --native`): le 59 descrizioni del benchmark tradotte in `bench/data/intent_descriptions_mt.json`, solo per la valutazione. Calibrazione rifittata sui file calib_* con le stesse traduzioni.

| MASSIVE, opzioni/domande nella lingua della frase | Fase 1 | ft2 | **ft3** |
|---|---|---|---|
| Choice, tutte | 0,798 | 0,81 | **0,848** |
| Choice, mai viste | 0,690 | 0,659 | **0,714** |
| Choice per lingua: it / es / de / fr (en 0,882–0,895) | 0,812 / 0,835 / 0,722 / 0,730 | 0,812 / 0,845 / 0,725 / 0,782 | 0,855 / 0,858 / 0,788 / 0,845 |
| Sì/no, tutte, AUROC | 0,904 | 0,926 | **0,956** |
| Sì/no, mai viste (test esteso), AUROC | 0,782 | 0,792 | **0,809** |

Fonti: [metrics_native_edge32k-ft.csv](../results/fase2/metrics_native_edge32k-ft.csv), [metrics_native_edge32k-ft2.csv](../results/fase2/metrics_native_edge32k-ft2.csv), [metrics_edge32k-ft3.csv](../results/fase2/metrics_edge32k-ft3.csv).

Sintesi: le descrizioni tradotte in addestramento recuperano gran parte della perdita con opzioni native (Choice da 0,798 a 0,848; francese da 0,730 a 0,845) e migliorano anche le misure con opzioni inglesi. Su MASSIVE le opzioni tradotte automaticamente restano comunque peggiori di quelle inglesi (0,848 contro 0,878), soprattutto in tedesco. Nel test v0, con opzioni italiane scritte a mano, succede il contrario. Le traduzioni automatiche delle descrizioni brevi sono spesso goffe (forme imperative, "fragen Sie nach…"), quindi descrizioni scritte bene da una persona dovrebbero dare di più.

## Limiti

- **Solo Mac, solo fp32 PyTorch** per la qualità. Bundle ONNX di ft3 ([export_check_edge32k-ft3.json](../results/fase2/export_check_edge32k-ft3.json), 300+300 item): fp32 differenza massima 2,2·10⁻⁵, accordo 100%; int8 solo pesi, accordo Choice 0,990 e Noul 0,993; int8 dinamico 0,920 e 0,943. I bundle ONNX di ft2 sono stati verificati contro torch sui primi 300 item Choice e 300 Noul del test ([export_check_edge32k-ft2.json](../results/fase2/export_check_edge32k-ft2.json)). fp32: differenza massima sui logit 3,0·10⁻⁵, accordo 100%. int8 solo pesi: accordo sull'argmax Choice 0,993 e sulle decisioni Noul 0,993. int8 dinamico (batch 1): 0,960 e 0,957 (fase 1: 0,9275 e 0,937 sul test completo). La qualità int8 non è ricalcolata sul test completo. Nessuna misura sul Pi.
- **Un solo seed** per giro, nessuna ricerca degli iperparametri.
- I negativi del test sono stati costruiti con le etichette MASSIVE: dove l'etichetta è discutibile, un modello che "ha ragione" viene contato come errore. Il controllo delle etichette (sezione dedicata) è di un solo annotatore.
- Traduzioni automatiche greedy: i nomi propri a volte vengono tradotti (es. "Fifth Third").
- Le descrizioni esterne sono state scritte dall'agente: vanno riviste prima di pubblicare i pesi, e vanno verificati i termini d'uso (CLAUDE.md, sezione Licenze).

## Prossimi passi proposti

1. **Revisione umana a campione** del controllo delle etichette (ad esempio 50 frasi `desc`/`amb`/`wrong`).
2. **Nel runtime e nell'API**: registrare quanti esempi ha ricevuto ogni domanda e restituire un indicatore "calibrata / non calibrata"; salvare e ricaricare le calibrazioni per domanda.
3. **Test fuori da MASSIVE con opzioni in italiano** (punti 2 e 3 della fase 2): domotica reale, controlli sulle azioni degli agenti, smistamento email; lì si verifica anche la calibrazione per domanda.
4. Più intenzioni sorelle native multilingue, se servirà migliorare l'AUROC senza esempi (MTOP è CC BY-SA: escluso per ora).

## Riproduzione

```sh
HF_HUB_DISABLE_XET=1 python training/rebuild_checkpoints.py --ft_bundle <bundle fp32 fase 1 estratto> --out releases/checkpoints
python bench/build_unseen_ext.py --raw data/massive --download --out bench/data
python bench/build_external.py prepare --download --raw data/ext_raw
python training/translate_ext.py            # riusa bench/data/ext/mt_cache.jsonl se presente
python bench/build_external.py items
python training/train_skejja2.py --init releases/checkpoints/edge32k-ft --out releases/checkpoints/run_B \
    --teacher releases/checkpoints/edge32k-zs --kd_w 0.2 --threads 5 --lr 2e-5 --head_lr 6e-5
python training/eval_noul.py --model releases/checkpoints/run_B/best --name edge32k-ft2 \
    --calib_ext bench/data/calib_noul_ext.jsonl --out results/fase2/metrics_edge32k-ft2.csv
python training/translate_ext.py --descriptions                       # -> bench/data/ext/descriptions_mt.json
python training/train_skejja2.py --init releases/checkpoints/edge32k-ft --out releases/checkpoints/run_C \
    --teacher releases/checkpoints/edge32k-zs --kd_w 0.2 --threads 10 --lr 2e-5 --head_lr 6e-5 --p_desc_native 0.5
python bench/build_native_options.py                                  # -> bench/data/intent_descriptions_mt.json
python training/eval_noul.py --model releases/checkpoints/run_C/best --name edge32k-ft3 --native \
    --calib_ext bench/data/calib_noul_ext.jsonl --out results/fase2/metrics_edge32k-ft3.csv
python bench/build_fewshot_calib.py --raw data/massive --out bench/data
python training/fewshot_calib.py --model edge32k-ft2 --model edge32k-ft --model edge32k-ft2A --model edge32k-zs --model edge32k-ft3
python training/export_bundle.py --ckpt releases/checkpoints/run_C/best --name edge32k-ft3 --out_dir releases \
    --calib_ext bench/data/calib_noul_ext.jsonl
```
