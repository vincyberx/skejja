#!/bin/bash
# Prova dell'immagine Docker di Skejja v0.1.0 sulla workstation. Uso: bash docker_test.sh <cartella di lavoro>
set -u
W="$1"; cd "$W" || exit 1
D="sudo docker"; KEY="chiave-di-prova"
REQ='{"state":"puoi spegnere le luci della cucina?","questions":{"a":{"type":"choice","criteria":{"on":"turn on the lights","off":"turn off the lights","music":"play music"}},"b":{"type":"noul","instructions":"Does the user want to turn off the lights?"}}}'
ok() { echo "OK   $*"; }; ko() { echo "FAIL $*"; FAIL=1; }; FAIL=0
wait_health() { for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 1; done; return 1; }

echo "== build"; $D build -q -t skejja:0.1.0 src >/dev/null && ok "build" || { ko "build"; exit 1; }
$D image ls skejja:0.1.0 --format 'immagine {{.Size}}'
v=$($D run --rm skejja:0.1.0 skejja --version); [ "$v" = "skejja 0.1.0" ] && ok "--version: $v" || ko "--version: $v"

echo "== A: bundle montato in /model, senza --user"
rm -rf model && mkdir model && tar xzf skejja-edge32k-ft6_onnx-int8.tar.gz -C model
ls -l model/skejja-edge32k-ft6_onnx-int8/*.onnx.data
$D rm -f sk-a >/dev/null 2>&1
$D run -d --name sk-a -p 127.0.0.1:18081:8080 -v "$W/model/skejja-edge32k-ft6_onnx-int8:/model:ro" \
   -e SKEJJA_API_KEY=$KEY -e SKEJJA_NAME=skejja-edge32k-ft6 skejja:0.1.0 >/dev/null
$D exec sk-a id
wait_health 18081 && ok "health" || { ko "health"; $D logs sk-a | tail -20; }
c=$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:18081/v1/systemone -H 'content-type: application/json' -d "$REQ")
[ "$c" = 401 ] && ok "senza chiave: 401" || ko "senza chiave: $c"
r=$(curl -s -X POST localhost:18081/v1/systemone -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -d "$REQ")
echo "$r"
echo "$r" | python3 -c 'import json,sys; d=json.load(sys.stdin)["answers"]; assert d["a"]["choice"]=="off" and d["b"]["noul"]>0.5' \
  && ok "risposte corrette" || ko "risposte"
echo "attendo il primo healthcheck (start-period 30 s, intervallo 30 s)"; sleep 65
h=$($D inspect -f '{{.State.Health.Status}}' sk-a); [ "$h" = healthy ] && ok "healthcheck: $h" || ko "healthcheck: $h"
$D rm -f sk-a >/dev/null

echo "== B: scaricamento al primo avvio in un volume /cache"
$D volume rm -f skejja-cache-test >/dev/null 2>&1
run_b() { $D rm -f sk-b >/dev/null 2>&1
  $D run -d --name sk-b -p 127.0.0.1:18082:8080 -v skejja-cache-test:/cache \
     -v "$W/skejja-edge32k-ft6_onnx-int8.tar.gz:/src/bundle.tar.gz:ro" \
     -e SKEJJA_BUNDLE=edge32k-ft6-int8 -e SKEJJA_MODEL_URL=file:///src/bundle.tar.gz skejja:0.1.0 >/dev/null; }
run_b
wait_health 18082 && ok "health (dopo lo scaricamento)" || { ko "health B"; $D logs sk-b | tail -20; }
$D logs sk-b 2>&1 | grep -q "^scarico edge32k-ft6-int8" && ok "primo avvio: scaricato" || ko "primo avvio: niente scaricamento nei log"
r=$(curl -s -X POST localhost:18082/v1/systemone -H 'content-type: application/json' -d "$REQ")
echo "$r" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["answers"]["a"]["choice"]=="off"; print(d["model"])' \
  && ok "risposta B" || ko "risposta B"
run_b
wait_health 18082 && ok "secondo avvio: health" || ko "secondo avvio"
$D logs sk-b 2>&1 | grep -q "^scarico" && ko "secondo avvio: scaricato di nuovo" || ok "secondo avvio: usa la cache"
$D rm -f sk-b >/dev/null; $D volume rm -f skejja-cache-test >/dev/null

echo "== C: hash sbagliato rifiutato"
printf 'non sono un modello' > finto.tar.gz
o=$($D run --rm -v "$W/finto.tar.gz:/src/bundle.tar.gz:ro" -e SKEJJA_BUNDLE=edge32k-ft6-int8 \
     -e SKEJJA_MODEL_URL=file:///src/bundle.tar.gz skejja:0.1.0 2>&1); echo "$o" | tail -1
echo "$o" | grep -qE "byte|SHA-256" && ok "archivio sbagliato rifiutato" || ko "archivio sbagliato"
echo "== esito: $([ $FAIL = 0 ] && echo TUTTO OK || echo CI SONO ERRORI)"
