#!/bin/sh
# fps do jogo (game_frame do heartbeat) numa janela de tempo, por execução — para comparar A/Bs
# sem a transição de entrada na cena (t≈80–110 s no rapido.sh, que varia com o carregamento).
#   scripts/fps-janela.sh <de> <até> <padrão de work/runs> ...      ex.: 115 190 '*-ln-*'
de=$1; ate=$2; shift 2
cd "$(dirname "$0")/.." || exit 1
for pat in "$@"; do
  for r in $(ls -d work/runs/$pat 2>/dev/null); do
    grep "\[heartbeat\]" "$r/run.log" | sed -E 's/.*t=([0-9]+)s flips=[0-9]+ game_frame=([0-9]+).*/\1 \2/' |
      awk -v a=$de -v b=$ate -v n="$(basename $r | sed -E 's/^[0-9]+-[0-9]+-//')" \
        '$1==a {g0=$2} $1==b {g1=$2} END { if (g0 && g1) printf "%-14s %5.1f\n", n, (g1-g0)/(b-a); else printf "%-14s  (sem janela)\n", n }'
  done
done
