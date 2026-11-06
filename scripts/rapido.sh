#!/bin/sh
# Teste rápido: do boot direto à cena da Megera (a mais pesada do primeiro nível) pelo
# "Continuar Jogo" do save de checkpoint, e mede o fps lá. ~2,5 min por medição.
#
#   scripts/rapido.sh [segundos-medidos] [nome]     (padrão: 60 s, "rapido")
#   BACKEND=vkfence EXTRA="VAR=1" scripts/rapido.sh  build e variáveis, como no teste.sh
#
# Botões (PAD_FILE): ✕ pula a cinemática, ✕ Jogador Individual, ✕ Perfil 1,
# ✕ Continuar Jogo (já vem selecionado). Save: work/saves/checkpoint1 (gravado pelo jogo
# em t≈180 s de um teste longo; o "Continuar" carrega a cena da Megera em ~15 s).
here=$(cd "$(dirname "$0")/.." && pwd)
secs=${1:-60}; name=${2:-rapido}
pad="$here/work/pad-$name.txt"; : > "$pad"
press() { echo 0x4000 > "$pad"; }
wait_s() { end=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt $end ]; do sleep 0.5; done; }
hb() { grep "\[heartbeat\]" "$log" | tail -1 | sed -E 's/.*t=([0-9]+)s flips=([0-9]+).*/\1 \2/'; }

SAVE=${CP:-work/saves/checkpoint1} BACKEND=${BACKEND:-vkfence} \
EXTRA="PAD_FILE=$pad GOW_HEARTBEAT=5 SPU_RELOC_OFF=0 SPURS_PASS_SLEEP_US=50 $EXTRA" \
    "$here/scripts/teste.sh" $((secs + 130)) "$name" > "$here/work/logs/rapido-$name.out" 2>&1 &
drv=$!
run=""
while [ -z "$run" ]; do wait_s 1; run=$(ls -td "$here"/work/runs/*-"$name" 2>/dev/null | head -1); done
log="$run/run.log"
until grep -q "cellVdec\] Open" "$log" 2>/dev/null || ! kill -0 $drv 2>/dev/null; do wait_s 1; done
wait_s 3;  press     # pula a cinemática
wait_s 10; press     # Jogador Individual
wait_s 10; press     # Perfil 1
wait_s 9;  press     # Continuar Jogo
wait_s 25            # carregamento e entrada na cena
set -- $(hb); t0=$1; f0=$2
wait_s "$secs"
set -- $(hb); t1=$1; f1=$2
pid=$(grep -m1 "^pid" "$here/work/logs/rapido-$name.out" | awk '{print $2}')
[ -n "$pid" ] && kill "$pid" 2>/dev/null
wait $drv 2>/dev/null
if [ -n "$t0" ] && [ -n "$t1" ] && [ "$t1" -gt "$t0" ]; then
    awk -v a="$f0" -v b="$f1" -v s="$t0" -v e="$t1" 'BEGIN { printf "fps na cena (t=%d..%ds): %.1f\n", s, e, (b - a) / (e - s) }'
else
    echo "sem medida (o jogo travou ou não chegou à cena?)"
fi
echo "$run"
