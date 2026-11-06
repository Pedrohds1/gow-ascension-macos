#!/bin/sh
# Leva o jogo do boot ao primeiro nível sozinho e mede lá dentro.
#
#   scripts/ir-nivel.sh [segundos-no-nível] [nome]      (padrão: 60 s, "bench")
#   EXTRA="VAR=1" scripts/ir-nivel.sh ...               variáveis a mais para o jogo
#   SAMPLE=1 scripts/ir-nivel.sh ...                    `sample` de 5 s no meio do nível
#
# Parte do save work/saves/menu-principal (menus de primeiro uso e cinemática já
# vistos), aperta ✕ na sequência: pula a cinemática, Jogador Individual, Perfil 1,
# Sim (cria o perfil), Novo jogo, Normal. Usa scripts/teste.sh (headless, mudo,
# save copiado, mata só o próprio PID). No fim imprime o fps no nível (flips do
# heartbeat), a memória e o caminho da execução.
here=$(cd "$(dirname "$0")/.." && pwd)
secs=${1:-60}; name=${2:-bench}
pad="$here/work/pad-$name.txt"; : > "$pad"
press() { echo 0x4000 > "$pad"; }
wait_s() { end=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt $end ]; do sleep 0.5; done; }

SAVE=work/saves/menu-principal \
EXTRA="PAD_FILE=$pad GOW_HEARTBEAT=5 PS3RECOMP_METAL_FRAME_DUMP_EVERY=60 $EXTRA" \
    "$here/scripts/teste.sh" $((secs + 260)) "$name" > "$here/work/logs/ir-nivel-$name.out" 2>&1 &
drv=$!
run=""
while [ -z "$run" ]; do wait_s 1; run=$(ls -td "$here"/work/runs/*-"$name" 2>/dev/null | head -1); done
log="$run/run.log"
alive() { kill -0 $drv 2>/dev/null; }
until grep -q "cellVdec\] Open" "$log" 2>/dev/null || ! alive; do wait_s 1; done
wait_s 3;  press            # pula a cinemática de abertura
wait_s 10; press            # Jogador Individual
wait_s 10; press            # Perfil 1: Criar Perfil
wait_s 7;  press            # Sim
wait_s 10; press            # Novo jogo
wait_s 10; press            # Normal
t_menu=$(grep "\[heartbeat\]" "$log" | tail -1 | sed -E 's/.*t=([0-9]+)s.*/\1/')
# No nível: o primeiro Close é o da cinemática pulada; o segundo, o do vídeo que
# cobre o carregamento do nível, depois do qual o motor desenha a cena.
until [ "$(grep -c "cellVdec\] Close" "$log" 2>/dev/null)" -ge 2 ] || ! alive; do wait_s 1; done
start=$(grep "\[heartbeat\]" "$log" | tail -1 | sed -E 's/.*t=([0-9]+)s.*/\1/')
echo "no nível a partir de t=${start}s (menus até t=${t_menu}s)"
if [ -n "$SAMPLE" ]; then
    wait_s $((secs / 2))
    pid=$(grep -m1 "^pid" "$here/work/logs/ir-nivel-$name.out" | awk '{print $2}')
    [ -n "$pid" ] && sample "$pid" 5 -mayDie -file "$run/sample.txt" >/dev/null 2>&1
    [ -n "$pid" ] && vmmap --summary "$pid" > "$run/vmmap.txt" 2>/dev/null
    wait_s $((secs / 2))
else
    wait_s "$secs"
fi
pid=$(grep -m1 "^pid" "$here/work/logs/ir-nivel-$name.out" | awk '{print $2}')
[ -n "$pid" ] && kill "$pid" 2>/dev/null
wait $drv 2>/dev/null
# fps no nível: flips por segundo entre o início e o fim da janela medida
awk -v s="$start" '/\[heartbeat\]/ { t=$2; sub("t=","",t); sub("s","",t); f=$3; sub("flips=","",f);
     if (t+0 >= s+10 && first=="") { first=f; t0=t } if (t+0 >= s+10) { last=f; t1=t; mem=$7 } }
     END { if (t1>t0) printf "fps no nível (t=%d..%ds): %.1f   %s\n", t0, t1, (last-first)/(t1-t0), mem; else print "sem janela de medida" }' "$log"
echo "$run"
