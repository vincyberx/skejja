# Adattatore Skejja per S1MB

S1MB (System One Mosaic Benchmark) è il banco pubblico con classifica per i System One:
[codice](https://github.com/hotchpotch/S1MB) (MIT), [dati inglesi](https://huggingface.co/datasets/hotchpotch/S1MB-english-v1),
[risultati](https://huggingface.co/datasets/hotchpotch/s1mb-result), [classifica](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard).
Qui teniamo la nostra parte, perché il resto è un repository di altri: l'adattatore e la modifica alla riga di
comando. Scelte di mappatura e limiti sono in [docs/fase2_ampiezza.md](../../docs/fase2_ampiezza.md) §4.

| File | Dove va |
|---|---|
| `skejja.py` | `evaluator/src/s1mb/adapters/skejja.py` del checkout di S1MB |
| `cli_skejja.patch` | `git apply` su `evaluator/src/s1mb/cli.py` (aggiunge `skejja` fra gli adattatori e le sue opzioni) |

Dal giro G l'adattatore supporta anche la **Score**: i livelli della scala diventano le opzioni nell'ordine
memorizzato e il valore atteso è Σ p·valore, preso dai campi `value` della prova (`Skejja.score` nel runtime).
Il patch della riga di comando aggiunge `--skejja-training-max-len` (serve solo a contare gli ingressi più
lunghi della lunghezza di addestramento del bundle: 192 fino a ft6, il budget del giro G da ft7 in poi). Se un
checkout ha già la versione precedente del patch, prima `git checkout -- evaluator/src/s1mb/cli.py` e poi
`git apply` di quella nuova.

Provato con S1MB al commit `16a23df96d4a593ff20e11d5b230f43838b036a6` e il dataset `S1MB-english-v1`
alla revisione `6aa5f39753f98eef604a2af6a61f41670c030f28`.

```sh
# dal checkout di S1MB, cartella evaluator/
cp .../bench/s1mb_adapter/skejja.py src/s1mb/adapters/skejja.py
git apply .../bench/s1mb_adapter/cli_skejja.patch
mkdir -p /tmp/sk && tar xzf releases/skejja-edge32k-ft5_onnx-fp32.tar.gz -C /tmp/sk
s1mb run --adapter skejja --category english-v1 \
    --model /tmp/sk/skejja-edge32k-ft5_onnx-fp32 \
    --skejja-runtime <repo>/runtime --revision ft5-giroE-7ottobre2026 \
    --skejja-stats stats_ft5.json --skejja-model-id skejja/edge32k-ft5 \
    --skejja-threads 5 --run-id edge32k-ft5-v3
```

Due cose dell'ambiente, non del banco: serve `HF_HUB_DISABLE_XET=1` (il backend Xet di Hugging Face non è
raggiungibile dal nostro sandbox), e il pacchetto `s1mb` deve essere installato perché l'harness legge la propria
versione con `importlib.metadata` (`pip install --target ./libs --no-deps ./evaluator` e `PYTHONPATH=./libs`
bastano).

**Prima di inviare risultati alla classifica, chiedere all'utente** (è una pull request su un dataset pubblico;
vedi `CLAUDE.md`). La politica sul test di S1MB vieta di scegliere configurazione, soglie o checkpoint guardando
i punteggi del test: la configurazione va fissata prima, come è stato fatto qui.
