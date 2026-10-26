# Skejja — kit di benchmark per Raspberry Pi (fase 1)

Misura **latenza, RAM e qualità** dei modelli decisionali di Skejja direttamente sul Raspberry.
Serve solo `onnxruntime`, `tokenizers` e `numpy` (niente torch).

## Requisiti
- Raspberry Pi OS **a 64 bit** (Bookworm o più recente). Controlla con `uname -m`: deve dire `aarch64`.
  Se dice `armv7l`, reinstalla Raspberry Pi OS Lite 64-bit.
- Python 3.11 o più recente (Bookworm ha già 3.11).
- Circa 1,2 GB liberi sulla scheda SD.
- Connessione internet solo per il primo `pip install`.

## Avvio
```sh
tar xzf skejja-pi-kit.tar.gz
cd skejja-pi-kit
./run_pi.sh                 # tutti i modelli che stanno nella RAM, 1 e 4 thread
```
Opzioni utili:
```sh
./run_pi.sh --list                          # elenco dei modelli e RAM stimata
./run_pi.sh --models ft32k-int8,L8-int8     # solo alcuni modelli
./run_pi.sh --quality-n 2000                # qualità sull'intero test set (lento su Pi 3)
./run_pi.sh --low-mem                       # meno RAM (arena di onnxruntime disattivata)
```
### Banco di costo (giro G: ingressi lunghi e molte opzioni)
```sh
./run_pi.sh --cost --models ft6-int8dyn --threads 1,4            # 10 punti x 2 thread, ~15-25 min su Pi 4
./run_pi.sh --cost --models ft6-int8dyn --threads 1 --low-mem    # stessa griglia con l'arena disattivata
```
I `results/*.json` tornano sul Mac e diventano tabella e figura con:
```sh
python bench/costo_formato.py --in <file1.json> <file2.json> --macchina pi4 \
    --out results/fase2/costo_lunghezza_opzioni_pi4.csv --fig results/fase2/costo_formato_pi4.png \
    --unisci results/fase2/costo_lunghezza_opzioni_mac.csv
```
Nei file del repository non entrano nomi host né indirizzi: la tabella tiene solo l'etichetta della macchina
(`pi4`, `pi3`, `mac`) e il modello della scheda letto da `/proc/device-tree/model`.

Misura latenza e RAM di picco per **lunghezza dell'ingresso** (64, 256, 512, 1.024 token) e **numero di
opzioni** (8, 20, 50): serve a decidere il budget del giro G prima di addestrare. Ogni punto gira in un processo
nuovo, così la RAM di picco è quella del punto. Una Score con 5 livelli costa come una Choice con 5 opzioni,
quindi la colonna a 8 opzioni la copre. Se un punto esce con `killed_signal_9`, la RAM non è bastata: è un
risultato, va riportato.

Durata misurata su Pi 3: circa 2 h 40 min per i 7 modelli della fase 1 (giro 2, 9.703 s) e circa 57 min per i 2 modelli ft5 (giro 3, 3.395 s), quindi circa 3,5 ore per tutti. Su Pi 4 attesa meno della metà.

Consigli per numeri puliti:
- chiudi gli altri servizi (desktop, Home Assistant, ecc.) durante la misura;
- usa un alimentatore adeguato: il kit registra temperatura e `vcgencmd get_throttled`;
- sul Pi 3 (1 GB) il baseline `gliclass-edge-w8e4a4` (~650 MB) viene saltato in automatico se la RAM non basta.

## Modelli nel kit
| Chiave | Cosa è | RAM stimata (Mac) |
|---|---|---|
| `L8-int8` | 8 strati, vocabolario 32k, fine-tuned, int8 weight-only | ~140 MB |
| `L8-int8dyn` | come sopra, int8 dinamico (riserva) | ~140 MB |
| `ft32k-int8` | 22 strati, vocabolario 32k, fine-tuned, int8 weight-only — **candidato principale** | ~190 MB |
| `ft32k-int8dyn` | come sopra, int8 dinamico (riserva) | ~180 MB |
| `zs32k-int8` | 22 strati, vocabolario 32k, senza fine-tuning | ~190 MB |
| `ft32k-fp32` | candidato principale in fp32 (riferimento) | ~310 MB |
| `gliclass-edge-w8e4a4` | GLiClass edge originale (vocabolario 256k) | ~650 MB |
| `ft5-int8` / `ft5-int8dyn` | fase 2, giro E | ~190 / ~180 MB |
| `ft6-int8` / `ft6-int8dyn` | fase 2, giro F — **candidato principale**, stessa dimensione di ft5 | ~190 / ~180 MB |

Il kit del giro G (`MODELS="ft6 ft5" ./kit/make_kit.sh`) contiene solo ft5 e ft6: 390 MB invece di 1 GB.

Le varianti "int8 weight-only" usano un kernel (MatMulNBits) che sui Mac sfrutta istruzioni
(dot-product ARMv8.2) assenti sui Cortex-A53 (Pi 3) e A72 (Pi 4): per questo ci sono anche
le varianti int8 dinamiche e fp32. Il kit registra se la CPU ha `asimddp`.

## Cosa mandare indietro
Alla fine trovi `results/<nome-host>_<data>.json` (e un `.csv`). Manda il file **.json**:
contiene modello del Raspberry, sistema operativo, RAM, temperatura e tutti i risultati.
Il kit legge `vcgencmd get_throttled` per tutto il giro (all'inizio e alla fine di ogni modello e ogni 30 s).
Se a fine giro stampa `throttled = true`, il Pi era in sottotensione o rallentato e le latenze non valgono:
cambia alimentatore o cavo e ripeti.

## Prova manuale
```sh
.venv/bin/python models/ft32k-int8/skejja_infer.py models/ft32k-int8 "accendi la luce del salotto" "turn on the lights|turn off the lights|play music"
.venv/bin/python models/ft32k-int8/skejja_infer.py models/ft32k-int8 "accendi la luce" "Does the user want to turn on the lights?"
```
Su una CPU senza dot-product (Pi 3 e Pi 4) `skejja_infer.py` usa da solo il file int8 dinamico, più veloce lì;
`--int8dyn no` forza l'int8 solo pesi, `--int8dyn yes` il dinamico. Il file usato è stampato come `modello: ...`.
