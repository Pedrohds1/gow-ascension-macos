#!/bin/sh
# Diagnóstico de uma configuração na cena da Megera (rapido.sh): ftrace de 5 s em t=FT (padrão 170,
# a luta estável; em t≈80–160 s a cena ainda envia texturas novas), um `sample` de 8 s logo antes e,
# com STATS=1, as estatísticas do motor e do backend a cada 10 s (inflam o andador).
#   scripts/diag.sh <nome> <binário> ["VAR=1 ..."]     saída em work/diag/<nome>.*
# Leitura: scripts/cadeia.py work/diag/<nome>.ftr ; scripts/perfil.py work/diag/<nome>.sample.txt ;
#          grep "\[eng-stats\]\|\[vk-draw\]\|\[vbl-lat\]" no run.log da execução.
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here" || exit 1
nm=$1; bin=$2; ex=$3
case "$bin" in /*) ;; *) bin="$here/$bin" ;; esac
mkdir -p work/diag
ft=${FT:-170}; st=""; [ "${STATS:-0}" = 1 ] && st="ENG_PHASE_STATS=10 VK_DRAW_PROF=10"
EXE="$bin" BACKEND=vknat EXTRA="$ex PS3_FTRACE=$ft:5:$here/work/diag/$nm.ftr $st" \
    nice -n 5 scripts/rapido.sh $((ft - 40)) "diag-$nm" > "work/diag/$nm.out" 2>&1 &
drv=$!
out="work/logs/rapido-diag-$nm.out"
until grep -q "^pid" "$out" 2>/dev/null || ! kill -0 $drv 2>/dev/null; do sleep 1; done
pid=$(grep -m1 "^pid" "$out" | awk '{print $2}')
run=$(ls -td work/runs/*-diag-"$nm" 2>/dev/null | head -1)
until grep -q "heartbeat\] t=$((ft - 10))s" "$run/run.log" 2>/dev/null || ! kill -0 $drv 2>/dev/null; do sleep 1; done
[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && sample "$pid" 8 -file "$here/work/diag/$nm.sample.txt" >/dev/null 2>&1
wait $drv
cp "$run/run.log" "work/diag/$nm.run.log" 2>/dev/null
cat "work/diag/$nm.out"
