# baselines/ — traccia baseline (fase 1)

Misure zero-shot di Laya multilingue e GLiClass multilang edge, export ONNX e quantizzazione
(int8 dinamico e weight-only `w8e4a4`). `skejja_ortlib.py` è il runner senza torch dei bundle baseline.

Dati: `baseline_lib.py`, `ort_quality.py` e `lat_child.py` leggono da `bench/data/` (in fase 1 leggevano dall'archivio di Claude Science; i cinque file sono identici byte per byte, verificato con SHA-256 il 16 ottobre 2026). La variabile `SKEJJA_DATA` sostituisce la cartella. Uso da riga di comando nella docstring di ogni script. Metodo completo: `docs/methods_baseline.md`.
