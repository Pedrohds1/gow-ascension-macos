#!/bin/sh
# Acompanha uma sessão do jogar.sh sem tocar no jogo: eventos do log (travamento, checkpoint,
# vídeo, queda) e, a cada 2 min, CPU, memória e se o RSX ainda está recebendo desenhos.
here=$(cd "$(dirname "$0")/.." && pwd)
log="$here/saves/jogador/run.log"
pid=$(pgrep -x MyGameRecomp | head -1)
[ -n "$pid" ] || { echo "jogo não está rodando"; exit 0; }
echo "monitorando pid $pid"
tail -n 0 -F "$log" 2>/dev/null | grep --line-buffered -E "FIFO resync|bad word|semaphore context dma|SAVE complete for 'BCES01741-PROFDATA|cellVdec\] Open|SIGSEGV|SIGBUS|SIGILL|Abort|assert|FATAL|unhandled|\[heartbeat\]" | \
  awk '/\[heartbeat\]/ { if (++n % 12) next } /SAVE complete/ { print "[checkpoint] o jogo salvou o progresso"; fflush(); next } { print; fflush() }' &
tp=$!
subs() { grep "draw batching" "$log" | tail -1 | sed -E 's/.* in ([0-9]+) submissions.*/\1/'; }
prev=$(subs); same=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    kill -0 "$pid" 2>/dev/null || break
    cur=$(subs)
    if [ "$cur" = "$prev" ]; then same=$((same + 1)); else same=0; fi
    ps -o %cpu=,rss= -p "$pid" | awk -v s="$same" '{ printf "[processo] CPU %.0f%%, memória %.1f GB%s\n", $1, $2/1048576, (s >= 2 ? ", SEM DESENHOS NOVOS HÁ 4+ MIN (possível travamento)" : ""); fflush() }'
    prev=$cur
done
kill $tp 2>/dev/null
echo "o jogo fechou (pid $pid); últimas linhas:"
grep -vE "^\[taskset\]|^\[spurs-pm\]" "$log" | tail -5 | cut -c1-200
