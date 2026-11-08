#!/bin/sh
# Robô sem janela e sem som, com controle passo a passo depois da luta:
#   1. boot e menus até "Continuar Jogo" (como o rapido.sh);
#   2. ataques em ciclo por LUTA segundos (padrão 90: a luta-tutorial da Megera);
#   3. fica parado, e quem dirige é o scripts/passo.sh (analógico e botões por arquivo).
# O jogo roda TOTAL segundos (padrão 1500) e o teste.sh o encerra no fim.
#
#   scripts/robo-nav.sh [nome]
#   CP=... HDD1_DE=... EXE=... EXTRA="..." LUTA=90 TOTAL=1500 scripts/robo-nav.sh nav1
here=$(cd "$(dirname "$0")/.." && pwd)
name=${1:-nav}
luta=${LUTA:-90}; total=${TOTAL:-1500}
pad="$here/work/pad-$name.txt"; : > "$pad"
stick="$here/work/stick-$name.txt"; echo "128 128 128 128" > "$stick"
press() { echo "$1 ${2:-40}" > "$pad"; }
wait_s() { end=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt $end ]; do sleep 0.5; done; }

SAVE=${CP:-work/saves/checkpoint1} BACKEND=${BACKEND:-vknat} \
EXTRA="PAD_FILE=$pad PAD_STICK_FILE=$stick GOW_HEARTBEAT=5 SPU_RELOC_OFF=0 SPURS_PASS_SLEEP_US=50 PS3RECOMP_VK_DUMP_EVERY=30 $EXTRA" \
    "$here/scripts/teste.sh" "$total" "$name" > "$here/work/logs/robo-$name.out" 2>&1 &
drv=$!
run=""
while [ -z "$run" ]; do wait_s 1; run=$(ls -td "$here"/work/runs/*-"$name" 2>/dev/null | head -1); done
echo "$run" > "$here/work/nav-atual.txt"
log="$run/run.log"
until grep -q "cellVdec\] Open" "$log" 2>/dev/null || ! kill -0 $drv 2>/dev/null; do wait_s 1; done
wait_s 3;  press 0x4000    # pula a cinemática
wait_s 10; press 0x4000    # Jogador Individual
wait_s 10; press 0x4000    # Perfil 1
wait_s 9;  press 0x4000    # Continuar Jogo
wait_s 25                  # carregamento
end=$(( $(date +%s) + luta ))
while [ "$(date +%s)" -lt $end ] && kill -0 $drv 2>/dev/null; do
    for m in 0x8000 0x8000 0x1000 0x2000; do press $m 4; sleep 0.2; done
done
echo "luta encerrada; controle manual pelo scripts/passo.sh" >> "$here/work/logs/robo-$name.out"
wait $drv 2>/dev/null
