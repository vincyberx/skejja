# Misure su Raspberry Pi 4

Analisi: [ANALISI.md](ANALISI.md).

Dispositivo: Raspberry Pi 4 Model B Rev 1.5, 4 GB di RAM, Raspberry Pi OS Lite 64 bit (Debian 13 trixie, `aarch64`),
Python 3.13.5, onnxruntime 1.31.0 (dal JSON; il Pi 3 usava la 1.30.0). La CPU non ha le istruzioni dot-product (`has_dotprod: false`), come il Pi 3.
Sistema installato il 9 ottobre 2026 (immagine `2026-10-06-raspios-trixie-arm64-lite`), senza dissipatore né ventola.

## Giro 1 — `pi4_20261009-115526` (tutti i modelli)

9 ottobre 2026, kit `releases/skejja-pi-kit-fase2.tar.gz` (SHA-256, verificato sul Pi:
`6c386a2442d47c46e0e4ecc273b5b08da815a8eade5b6484ae8431112d670ccb`), comando `./run_pi.sh --models all`:
tutti i 9 modelli a 1 e 4 thread, 18 misure, tutte riuscite. Durata delle misure 4.575 s (fine alle 11:55).
Il nome dell'host è stato tolto dal nome dei file.

**Alimentazione: pulita.** Prima del lancio `get_throttled` dava `0x0`; il kit riporta `"throttled": false`
(nessun campione su 190 con i bit 0x1 o 0x4).

**Temperatura: al limite.** Il JSON registra anche il limite di temperatura, che il criterio del kit non conta:

- con 1 thread nessun campione con il limite attivo;
- con 4 thread il limite di temperatura (bit 0x8) è attivo in 2–5 campioni per modello
  (`throttled_samples`: valori `0x80000` e `0x80008`); controllando a mano la CPU era fra 79,8 e 80,8 °C, sempre a
  1,8 GHz nei momenti controllati;
- alla fine del giro (t = 4.561 s, durante `ft5-int8dyn` a 4 thread) compaiono i bit storici 0x20000 e 0x40000
  (frequenza limitata e rallentamento avvenuti): un rallentamento breve, avvenuto fra due campioni e quindi non
  visto dal criterio del kit.

Le latenze a 1 thread sono quindi pulite; quelle a 4 thread sono misurate vicino al limite termico, e l'ultima
(`ft5-int8dyn` a 4 thread) può includere un breve rallentamento. Qualità e RAM non dipendono dalla temperatura.
