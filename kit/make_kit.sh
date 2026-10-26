#!/usr/bin/env bash
# Rebuild releases/skejja-pi-kit.tar.gz from the model bundles in releases/ and bench/data/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; R="$ROOT/releases"; OUT="$(mktemp -d "${TMPDIR:-/tmp}/skejja-kit.XXXXXX")"; K="$OUT/skejja-pi-kit"
mkdir -p "$K/models" "$K/data" "$K/results"
# MODELS="ft6 ft5" ./kit/make_kit.sh  -> kit snello con i soli bundle elencati (il giro di costo del giro G usa ft6)
WANT="${MODELS:-}"
ex() { case " $WANT " in " ") ;; *" $2 "*) ;; *) return 0;; esac
  t="$(mktemp -d "${TMPDIR:-/tmp}/skejja-ex.XXXXXX")"; tar xzf "$R/$1" -C "$t"; d="$(find "$t" -mindepth 1 -maxdepth 1 -type d | head -1)"; mkdir -p "$K/models/$2"; cp -R "$d"/. "$K/models/$2/"; rm -rf "$t"; }
ex skejja-edge32k-ft_onnx-int8.tar.gz ft32k-int8
ex skejja-edge32k-ft_onnx-fp32.tar.gz ft32k-fp32
ex skejja-edge32k-L8-ft_onnx-int8.tar.gz ft32k-L8
ex skejja-edge32k-zs_onnx-int8.tar.gz zs32k
ex skejja_baseline_gliclass-multilang-edge_onnx-w8e4a4.tar.gz gliclass-edge-w8e4a4
[ -f "$R/skejja-edge32k-ft5_onnx-int8.tar.gz" ] && ex skejja-edge32k-ft5_onnx-int8.tar.gz ft5
[ -f "$R/skejja-edge32k-ft6_onnx-int8.tar.gz" ] && ex skejja-edge32k-ft6_onnx-int8.tar.gz ft6
rm -f "$K"/models/zs32k/*int8dyn*
for d in "$K"/models/*/; do [ -f "$d/skejja_infer.py" ] && cp "$ROOT/runtime/skejja_infer.py" "$d/skejja_infer.py"; done
cp "$ROOT/bench/data/test_choice.jsonl" "$ROOT/bench/data/test_noul.jsonl" "$K/data/"
for f in bench_pi.py run_pi.sh requirements.txt README_IT.md NOTICE models.json; do cp "$ROOT/kit/$f" "$K/"; done
chmod +x "$K/run_pi.sh" "$K/bench_pi.py"
KIT="${KIT_NAME:-skejja-pi-kit}"   # es. KIT_NAME=skejja-pi-kit-fase2 ./kit/make_kit.sh
tar czf "$R/$KIT.tar.gz" -C "$OUT" skejja-pi-kit && rm -rf "$OUT"
echo "Creato $R/$KIT.tar.gz"
