# API System One di Skejja

_Fase 3, prima parte (17 ottobre 2026). Codice: [runtime/skejja/](../runtime/skejja/), test: [tests/](../tests/)._

Skejja espone lo stesso endpoint dei System One in cloud, `POST /v1/systemone`. Un client scritto per Jev
funziona cambiando solo l'indirizzo. Il contratto è ricostruito dalla documentazione pubblica di Jev (pagine
_primitives_ e _confidence_ di docs.typesafe.ai) e dall'harness S1MB (MIT, adattatore `typesafe.py`), senza
chiamare il servizio: il suo contratto d'uso vieta di usarlo per sviluppare un prodotto concorrente
([confronto_concorrenti.md](confronto_concorrenti.md)).

## Installazione e avvio

```sh
pip install -e ".[server]"            # dal repository; su PyPI non è ancora pubblicato
skejja --version
skejja serve --port 8080                                            # modello predefinito, ascolta solo su 127.0.0.1
skejja serve <bundle> --port 8080                                   # oppure la cartella di un bundle estratto
SKEJJA_API_KEY=... skejja serve --host 0.0.0.0                      # dalla rete: sempre con una chiave
skejja run richiesta.json                                           # una richiesta, senza server
skejja download [edge32k-ft6-int8|edge32k-ft6-fp32]                 # solo scaricare, stampa la cartella
```

Senza bundle si usa `edge32k-ft6-int8`, scaricato al primo uso (208 MB) e controllato con il suo SHA-256
([runtime/skejja/models.py](../runtime/skejja/models.py)). Cache: `$SKEJJA_HOME`, altrimenti
`$XDG_CACHE_HOME/skejja` o `~/.cache/skejja`. `SKEJJA_MODEL_URL` sostituisce l'indirizzo (mirror, o `file://` per
un archivio locale) ma non l'hash. **Finché i pesi non sono pubblicati** l'indirizzo predefinito (Hugging Face,
`vincyber/skejja-edge32k-ft6`, non ancora pubblicato) risponde 401: si usa un bundle estratto o
`SKEJJA_MODEL_URL=file://$PWD/releases/v0.1.0/skejja-edge32k-ft6_onnx-int8.tar.gz`.

Da Python, senza server:

```python
from skejja import Skejja, SystemOne
engine = SystemOne(Skejja("/tmp/sk/skejja-edge32k-ft6_onnx"), name="skejja-edge32k-ft6")
engine.run({"state": "...", "questions": {...}})
```

Il pacchetto dipende solo da `numpy`, `onnxruntime` e `tokenizers`; il server (FastAPI e uvicorn) è l'extra
`server`. Il runtime resta il file unico `runtime/skejja_infer.py`, quello che il kit del Raspberry copia nei
bundle: il pacchetto `skejja` gli sta accanto e lo importa.

## Docker

```sh
docker build -t skejja .
docker run --rm -p 127.0.0.1:8080:8080 -v /percorso/skejja-edge32k-ft6_onnx-int8:/model:ro \
    -e SKEJJA_API_KEY=... -e SKEJJA_NAME=skejja-edge32k-ft6 skejja
# oppure: scaricato al primo avvio in un volume
docker run --rm -p 127.0.0.1:8080:8080 -v skejja-cache:/cache -e SKEJJA_BUNDLE=edge32k-ft6-int8 skejja
```

Il modello non è nell'immagine (circa 460 MB senza modello): si monta in `/model`. Variabili: `SKEJJA_THREADS`
(4), `SKEJJA_MAX_TOKENS` (512), `SKEJJA_NAME`, `SKEJJA_API_KEY`. Dentro il container il server ascolta su
`0.0.0.0`, quindi l'esposizione la decide `-p`: meglio `127.0.0.1:...`, oppure sempre con una chiave. Con i
bundle esportati fino al 17 ottobre 2026 serviva anche `--user "$(id -u):$(id -g)"`: i loro `.onnx.data` erano
leggibili solo dal proprietario (onnxruntime falliva con `SystemError : 13`); quelli di `releases/v0.1.0/` sono
leggibili da tutti. Provata il 25 ottobre sulla workstation (Ubuntu 24.04 in WSL2, Docker 29.1.3) dal commit
`967d54c` con [scripts/docker_test.sh](../scripts/docker_test.sh): immagine di 465 MB, `skejja --version`; bundle v0.1.0
montato in `/model` **senza `--user`** (il container gira come `skejja`, uid 999): health, 401 senza chiave, risposte
corrette, healthcheck «healthy»; scaricamento al primo avvio nel volume `/cache` (da `file://`), secondo avvio dalla
cache senza riscaricare; archivio sbagliato rifiutato. Esito in
[docker_v0.1.0_workstation.txt](../results/fase3/docker_v0.1.0_workstation.txt).

## Server MCP (per Claude e altri agenti)

```sh
pip install -e ".[mcp]"
skejja mcp /tmp/sk/skejja-edge32k-ft6_onnx                 # stdio: il client lo avvia, nessuna porta aperta
claude mcp add skejja -- skejja mcp /tmp/sk/skejja-edge32k-ft6_onnx   # esempio di registrazione in Claude Code
```

Strumenti: `choose` (una Choice: testo e opzioni {nome: descrizione}), `yes_no` (una noul, con
`calibration` = `question` o `global`), `systemone` (una richiesta completa nel formato sotto) e
`calibrate_question` (calibrazione per domanda con esempi). Le istruzioni del server dicono all'agente quando
usarlo e quando no: messaggi brevi, comandi e intenzioni sì; ragionamento, documenti lunghi e molte opzioni no.
SDK MCP 2.x (`mcp>=2.3`).

## Contratto

Richiesta:

```json
{
  "model": "qualsiasi (ignorato: il server ha un solo modello)",
  "state": "Ciao, puoi spegnere le luci della cucina? Grazie",
  "questions": {
    "azione":   {"type": "choice", "instructions": "What does the user want?",
                 "criteria": {"luci_on": "turn on the lights", "luci_off": "turn off the lights",
                              "musica": "play music", "meteo": "ask about the weather"}},
    "spegnere": {"type": "noul", "instructions": "Does the user want to turn off the lights?"},
    "cortesia": {"type": "score", "criteria": ["rude", "neutral", "polite"]}
  }
}
```

Risposta:

```json
{
  "model": "skejja-edge32k-ft6:edge-v32k-ft6.fp32.onnx",
  "answers": {
    "azione":   {"type": "choice", "choice": "luci_off",
                 "probabilities": {"luci_on": 0.006, "luci_off": 0.992, "musica": 0.001, "meteo": 0.002},
                 "confidence": 0.989},
    "spegnere": {"type": "noul", "noul": 0.923},
    "cortesia": {"type": "score", "score": 1.446, "probabilities": {"0": 0.115, "1": 0.323, "2": 0.562},
                 "legend": {"0": "rude", "1": "neutral", "2": "polite"}, "confidence": 0.169}
  },
  "usage": {"input_tokens": 96, "output_tokens": 0},
  "skejja": {"max_tokens": 512, "truncated": [], "calibration": {"spegnere": "global"}}
}
```

(risposta reale del server sul Mac il 17 ottobre, arrotondata qui a tre decimali: il server restituisce i valori completi.)

| Tipo | `criteria` | Risposta |
|---|---|---|
| `choice` | `{nome: descrizione}`, da 2 a 255 opzioni | `choice` (la più probabile), `probabilities` per nome, `confidence` = (p_max − 1/n) / (1 − 1/n) |
| `noul` | facoltativo, `{"true": ..., "false": ...}` | `noul`, la probabilità del sì |
| `score` | lista dei livelli dal più basso al più alto, da 2 a 10 | `score` = Σ p·livello, `probabilities` con chiavi `"0"`, `"1"`, …, `legend`, `confidence` = max(0, 1 − Σ pᵢ·\|i − m\| / MAD_unif) |

`instructions` e le descrizioni possono essere testo, oggetti o liste. Le formule di confidenza sono quelle
pubblicate da Jev, verificate con l'esempio della sua pagina Score in
[tests/test_systemone.py](../tests/test_systemone.py).

Errori: HTTP 400 con `{"error": {"type": "invalid_request_error", "message": ..., "param": "questions.q.type"}}`;
HTTP 401 se il server ha una chiave (`SKEJJA_API_KEY`) e la richiesta non porta `Authorization: Bearer <chiave>`.
`GET /health` restituisce lo stato e il nome del modello.

Il blocco `skejja` è in più rispetto a Jev, e i client di Jev lo ignorano: `truncated` elenca le domande
accorciate, `calibration` dice per ogni noul se la probabilità è calibrata sulla domanda (`question`) o solo con
la calibrazione globale del bundle (`global`).

## Calibrazione per domanda (in più rispetto a Jev)

Su una domanda nuova, senza esempi, le probabilità della noul sono poco affidabili (sezione sotto). Con esempi
etichettati la calibrazione migliora molto, e il server la ricorda:

```sh
curl -s localhost:8080/v1/skejja/calibrations -H 'Content-Type: application/json' -d '{
  "question": {"type": "noul", "instructions": "Does the user want to turn on the lights?"},
  "examples": [{"state": "accendi la luce in cucina", "answer": true},
               {"state": "spegni la luce", "answer": false}, ...]}'
```

Da quel momento ogni noul con la stessa domanda (stesse `instructions` e stessi `criteria`) usa la sua
calibrazione, e `skejja.calibration` lo dice. Servono almeno 8 esempi con risposte sia sì sia no; ne sono
consigliati 32–64 (con 64 esempi per domanda e il prior predefinito, σ = 0,5, ft6 arriva a ECE 0,047 sul set di
sviluppo esterno: [metrics_fewshot.csv](../results/fase2/metrics_fewshot.csv)). È la stessa Platt MAP ancorata
alla calibrazione globale di `Skejja.calibrate_question` ([fase2_noul.md](fase2_noul.md)). **Gli esempi vanno dati nelle
proporzioni reali** (se nella realtà i sì sono uno su dieci, anche negli esempi): sulle richieste vere a un assistente vocale,
esempi metà sì e metà no portano a dire sì troppo spesso e peggiorano le domande rare
([real_requests.md](real_requests.md) §2). Una calibrazione vale
solo per il `.onnx` con cui è stata stimata, perché la scala dei logit cambia da una variante all'altra.
`GET /v1/skejja/calibrations` le elenca; con `--calibrations calibrazioni.json` si salvano in un file e
sopravvivono ai riavvii.

## Come Skejja legge una domanda (resa fissata)

Skejja ha un solo ingresso, `<<LABEL>>opzione…<<SEP>>testo`, senza posto per istruzioni o prompt di sistema. La
resa è la stessa dell'adattatore S1MB ([bench/s1mb_adapter/skejja.py](../bench/s1mb_adapter/skejja.py)) ed è
fissata in [runtime/skejja/systemone.py](../runtime/skejja/systemone.py):

- **choice e score:** le descrizioni diventano le etichette, nell'ordine dato. Il nome di un'opzione arriva al
  modello solo se la descrizione manca (con gli `_` sostituiti da spazi); oggetti e liste diventano JSON.
- **noul:** con `criteria`, l'etichetta è la descrizione di `true`; senza, l'etichetta è la domanda stessa
  (`instructions`), come in `Skejja.noul`.
- **testo:** istruzioni, una riga vuota, poi lo stato (JSON compatto se non è testo).
- **ingressi lunghi:** oltre `--max-tokens` (predefinito 512) il testo si accorcia testa e coda (60/40) con il
  segno `[…]`, e la risposta elenca le domande accorciate in `skejja.truncated` (campo in più, che i client di
  Jev ignorano). Il limite di 512 viene dal Pi 4: a 511 token la RAM di picco di ft6 int8 dinamico è 246 MB, a
  1.023 token 398 MB, sopra l'obiettivo di 300 MB
  ([costo_lunghezza_opzioni_pi4.csv](../results/fase2/costo_lunghezza_opzioni_pi4.csv)).

## Differenze da Jev e limiti da dichiarare

- **Un solo modello per server:** il campo `model` della richiesta è accettato e ignorato.
- **Descrizioni delle opzioni in inglese:** il modello è addestrato così; le opzioni nella lingua della frase
  rendono meno ([fase2_ampiezza.md](fase2_ampiezza.md)). Lo stato invece può essere in italiano, spagnolo,
  tedesco, francese o inglese.
- **Probabilità delle sì/no generiche non calibrate senza esempi:** su domande mai viste ft6 ha ECE 0,150
  (AUROC 0,840) ([metrics_fewshot.csv](../results/fase2/metrics_fewshot.csv)). Per le domande che contano va
  usata la calibrazione per domanda (sezione sopra); `skejja.calibration` dice quale delle due è stata applicata.
- **Ampiezza:** ragionamento, implicazione fra due testi e Choice con molte opzioni sono deboli
  ([confronto_concorrenti.md](confronto_concorrenti.md), [fase2_giro_G.md](fase2_giro_G.md)).
- **Probabilità non arrotondate** (Jev le arrotonda a due decimali).

## Passi successivi della fase 3

- astensione (risposta «non so» sotto una soglia, con garanzie tipo conformal prediction);
- Choice più veloci con le opzioni codificate una volta sola;
- scheda del modello ([model_card.md](model_card.md)) e prima release pubblica (v0.1).
