#!/bin/sh
# Vigia a execução mais recente com o nome dado: se os flips do heartbeat ficarem
# parados por 3 batidas seguidas (15 s), tira um `sample` de 5 s do processo em
# <run>/trava-sample.txt e sai. Uso: scripts/vigia-trava.sh <nome> [segundos-máx]
here=$(cd "$(dirname "$0")/.." && pwd)
name=$1; max=${2:-900}
end=$(( $(date +%s) + max ))
run=""
while [ -z "$run" ] && [ "$(date +%s)" -lt $end ]; do sleep 2; run=$(ls -td "$here"/work/runs/*-"$name" 2>/dev/null | head -1); done
same=0; last=""
while [ "$(date +%s)" -lt $end ]; do
    sleep 5
    f=$(grep "\[heartbeat\]" "$run/run.log" 2>/dev/null | tail -1 | sed -E 's/.*flips=([0-9]+).*/\1/')
    [ -n "$f" ] || continue
    if [ "$f" = "$last" ]; then same=$((same + 1)); else same=0; fi
    last=$f
    if [ $same -ge 3 ]; then
        pid=$(grep -m1 "^pid" "$here/work/logs/ir-nivel-$name.out" 2>/dev/null | awk '{print $2}')
        kill -0 "$pid" 2>/dev/null || { echo "processo terminou (fim do teste, não travou)"; exit 0; }
        [ -n "$pid" ] && sample "$pid" 5 -mayDie -file "$run/trava-sample.txt" >/dev/null 2>&1
        echo "travou com flips=$f; amostra em $run/trava-sample.txt"; exit 0
    fi
    pgrep -qf "teste.sh .* $name" || { echo "execução terminou sem travar"; exit 0; }
done
echo "tempo esgotado"
