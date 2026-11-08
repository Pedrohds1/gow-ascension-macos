#!/bin/sh
# Robô sem janela e sem som: do boot ao "Continuar Jogo" (como o rapido.sh) e depois
# aperta ataques em ciclo (quadrado, triângulo, bola) para vencer a luta da Megera e
# seguir até a cena seguinte, a que quebrava. Frames a cada 10 s em <run>/frames.
#
#   scripts/robo-megera.sh [segundos de luta] [nome]      (padrão: 600 s, "robo")
#   CP=work/saves/checkpoint1 HDD1_DE=<cache> EXE=<binário> EXTRA="..." scripts/robo-megera.sh
here=$(cd "$(dirname "$0")/.." && pwd)
secs=${1:-600}; name=${2:-robo}
pad="$here/work/pad-$name.txt"; : > "$pad"
press() { echo "$1 ${2:-40}" > "$pad"; }
wait_s() { end=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt $end ]; do sleep 0.5; done; }

SAVE=${CP:-work/saves/checkpoint1} BACKEND=${BACKEND:-vknat} \
EXTRA="PAD_FILE=$pad GOW_HEARTBEAT=5 SPU_RELOC_OFF=0 SPURS_PASS_SLEEP_US=50 $EXTRA" \
    "$here/scripts/teste.sh" $((secs + 130)) "$name" > "$here/work/logs/robo-$name.out" 2>&1 &
drv=$!
run=""
while [ -z "$run" ]; do wait_s 1; run=$(ls -td "$here"/work/runs/*-"$name" 2>/dev/null | head -1); done
log="$run/run.log"
until grep -q "cellVdec\] Open" "$log" 2>/dev/null || ! kill -0 $drv 2>/dev/null; do wait_s 1; done
wait_s 3;  press 0x4000    # pula a cinemática
wait_s 10; press 0x4000    # Jogador Individual
wait_s 10; press 0x4000    # Perfil 1
wait_s 9;  press 0x4000    # Continuar Jogo
wait_s 25                  # carregamento
# Luta: quadrado, quadrado, triângulo, bola (agarrar / QTE), em ciclo.
end=$(( $(date +%s) + secs ))
while [ "$(date +%s)" -lt $end ] && kill -0 $drv 2>/dev/null; do
    for m in 0x8000 0x8000 0x1000 0x2000; do
        press $m 4
        sleep 0.2
    done
done
wait $drv 2>/dev/null
echo "$run"
