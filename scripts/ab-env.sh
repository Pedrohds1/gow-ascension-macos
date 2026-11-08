#!/bin/sh
# A/B intercalado por variáveis de ambiente: scripts/ab-env.sh <prefixo> <rodadas> "<nome>=<EXTRA>" ...
cd "$(dirname "$0")/.." || exit 1
pre=$1; rounds=$2; shift 2
i=1
while [ $i -le $rounds ]; do
    for cfg in "$@"; do
        nm=${cfg%%=*}; ex=${cfg#*=}
        r=$(BACKEND=vknat EXTRA="$ex" nice -n 5 scripts/rapido.sh 60 "$pre-$nm-$i" 2>&1 | head -1)
        echo "$nm rodada $i: $r"
    done
    i=$((i + 1))
done
