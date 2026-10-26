# Misure su Raspberry Pi 3

Dispositivo: Raspberry Pi 3 Model B Rev 1.2, Raspberry Pi OS 64 bit (`aarch64`), 905 MB di RAM,
Python 3.13.5, onnxruntime 1.30.0. La CPU non ha le istruzioni dot-product (`has_dotprod: false`).

## Giro 1 — `pi3_20261005-120245` (alimentazione insufficiente)

Kit `releases/skejja-pi-kit.tar.gz` con le opzioni predefinite (`./run_pi.sh`), dal 5 ottobre 2026 alle 09:05 circa
fino alle 12:02. Il nome dell'host è stato tolto dal nome dei file.
SHA-256 del kit (quello della release `fase1` su GitHub):
`c8a50531878793780e253d3a3b2c8c1920290199f101b5b09d925b5d97a93007`.

**Latenze non affidabili.** Il JSON registra `get_throttled` una sola volta, all'avvio (`0x50000`:
sottotensione e rallentamento avvenuti in passato). Durante il giro, controllandolo a mano, è risultato:

| Ora | `get_throttled` | Significato |
|---|---|---|
| 09:28 | `0x50005` | sottotensione e rallentamento **in corso** |
| 10:14 | `0x50005` | sottotensione e rallentamento in corso |
| 11:04 | `0x50005` | sottotensione e rallentamento in corso |
| 12:15 (a giro finito) | `0x50000` | solo storico |

Con la sottotensione il Pi 3 abbassa la frequenza della CPU, quindi le latenze di questo giro vanno lette come
limite pessimistico. Qualità (accuratezza, AUROC) e RAM non dipendono dall'alimentazione e restano valide.

## Giro 2 — `pi3_20261006-111423` (alimentazione ancora insufficiente)

6 ottobre 2026, kit nuovo con le opzioni predefinite (`./run_pi.sh`): misure dalle 08:32 circa alle 11:14 (9.703 s).
Alimentatore USB-C da 20 W (5 V, 3 A) con cavo USB-C → micro-USB. Il nome dell'host è stato tolto dal nome dei file.

**Partito senza il requisito `0x0`, per scelta dell'utente.** Prima del lancio `get_throttled` dava `0x50000`:
il kernel aveva registrato una sottotensione durante l'avvio (`Undervoltage detected` alle 08:19:42, `Voltage
normalised` alle 08:19:48). Il giro è stato lanciato lo stesso, sapendo che sarebbe uscito marcato se la
sottotensione fosse tornata.

**Esito: `"throttled": true`.** In 169 campioni di `get_throttled` su 353 erano attivi i bit 0x1 o 0x4. Primo campione
cattivo dopo 330 s. Valori osservati: `0x50000`, `0x50005`, `0x70000`, `0x70002`, `0x70005` (`0x70002`: frequenza
limitata senza sottotensione; non conta nel criterio, che guarda solo i bit 0x1 e 0x4).

Quota di campioni in sottotensione durante ciascun modello (dal campo `throttled_samples` del JSON; un modello
copre latenza e, solo con 4 thread, anche qualità):

| Modello | 1 thread | 4 thread |
|---|---|---|
| `L8-int8` | 0/12 | 21/22 |
| `L8-int8dyn` | 1/6 | 5/9 |
| `ft32k-int8` | 1/28 | 49/53 |
| `ft32k-int8dyn` | 0/11 | 5/18 |
| `zs32k-int8` | 0/29 | 23/44 |
| `ft32k-fp32` | 0/14 | 31/33 |
| `gliclass-edge-w8e4a4` | 1/27 | 32/45 |

Con un solo core attivo l'alimentazione quasi sempre regge; con quattro core no. Qualità e RAM, come nel giro 1,
non dipendono dall'alimentazione.

Kit ricostruito il 5 ottobre 2026 con `./kit/make_kit.sh`, **non** caricato nella release su GitHub.
SHA-256 (verificato sul Pi prima del giro 2): `e3840d7622913d5fee7010cf4cef426e801b26b6d57e89b10ff4589e26797cab`.
Modelli, dati, protocollo e metriche sono gli stessi del giro 1. Cambia solo questo:
- `bench_pi.py` registra `get_throttled` all'inizio e alla fine di ogni modello e ogni 30 s, salva i valori nel JSON
  (`throttled_samples`) e segna il giro `"throttled": true` se un valore ha attivi i bit 0x1 o 0x4;
- `skejja_infer.py` usa da solo il file `*.int8dyn.onnx` sulle CPU ARM senza dot-product (`--int8dyn` forza la scelta).

## Giro 3 — `pi3_20261008-093657` (fase 2, candidato ft5)

8 ottobre 2026, kit `releases/skejja-pi-kit-fase2.tar.gz` (SHA-256, verificato sul Pi:
`6c386a2442d47c46e0e4ecc273b5b08da815a8eade5b6484ae8431112d670ccb`), comando
`./run_pi.sh --models ft5-int8dyn,ft5-int8`: misure dalle 08:40 circa alle 09:36 (3.395 s).
Stesso alimentatore e cavo del giro 2. All'avvio il kernel ha registrato due sottotensioni (08:24:47 e 08:25:13,
rientrate in pochi secondi); prima del lancio `get_throttled` dava `0x50000`.

**Esito: `"throttled": true`** (80 campioni su 123; valori `0x50000` e `0x50005`). Stesso schema del giro 2:

| Modello | 1 thread | 4 thread |
|---|---|---|
| `ft5-int8dyn` | 0/11 | 24/25 |
| `ft5-int8` | 1/29 | 54/56 |

Le latenze a 1 thread sono quindi confrontabili con quelle a 1 thread del giro 2; quelle a 4 thread no.
Qualità e RAM non dipendono dall'alimentazione.
