#!/bin/sh
# Mede a velocidade do código SPU traduzido contra os nativos EDGE (modo de checagem: os dois
# rodam sobre o mesmo estado; ciclos do cntvct por iteração). Uso: scripts/chk-edge.sh <nome> [s]
# CP/BACKEND/EXE/EXTRA como no rapido.sh.
here=$(cd "$(dirname "$0")/.." && pwd)
n=${1:-chk}; s=${2:-40}
CP=${CP:-work/saves/pedro-cp1} BACKEND=${BACKEND:-vknat} \
EXTRA="GOW_EDGE_NATIVE=3 GOW_EDGE_73E0=3 GOW_EDGE_F538=3 GOW_EDGE_D640=3 GOW_EDGE_D4DC=3 GOW_EDGE_A880=3 GOW_EDGE_6AC8=3 GOW_EDGE_O11430=3 $EXTRA" \
    "$here/scripts/rapido.sh" "$s" "$n" > /dev/null 2>&1
r=$(ls -td "$here"/work/runs/*-"$n" | head -1)
for k in edge-check edge-73e0-check edge-f538-check edge-d640-check edge-a880-check edge-6ac8-check edge-o11430-check; do
    grep "\[$k\]" "$r/run.log" | grep "tsc: native" | tail -2 | sed -E "s/.*(D92C|D4DC)?.*tsc: native ([0-9.]+) M, lifted ([0-9.]+) M \(([0-9.]+) vs ([0-9.]+) per.*/$k \1 nativo \4  traduzido \5  (por iteração)/; s/^.*\] (D92C|D4DC):.*tsc: native ([0-9.]+) M, lifted ([0-9.]+) M \(([0-9.]+) vs ([0-9.]+).*/$k \1 nativo \4  traduzido \5/"
done
grep -h "differ" "$r/run.log" | grep -E "check\]" | grep -vE " 0 differ" | head -3
