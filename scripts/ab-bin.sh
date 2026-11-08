#!/bin/sh
# A/B intercalado por executável: scripts/ab-bin.sh <prefixo> <rodadas> "<nome>=<binário>[|EXTRA]" ...
# Antes de cada execução espera a carga cair abaixo de CARGA_MAX (8) por até 60 s.
cd "$(dirname "$0")/.." || exit 1
pre=$1; rounds=$2; shift 2
carga() { sysctl -n vm.loadavg | awk '{print int($2)}'; }
i=1
while [ $i -le $rounds ]; do
    for cfg in "$@"; do
        nm=${cfg%%=*}; rest=${cfg#*=}; bin=${rest%%|*}; ex=""
        case "$rest" in *"|"*) ex=${rest#*|} ;; esac
        case "$bin" in /*) ;; *) bin="$PWD/$bin" ;; esac   # o teste.sh roda o jogo de dentro da pasta da execução
        # o jogo do Pedro aberto (nome exato MyGameRecomp: .app ou jogar.sh) pausa os testes
        while pgrep -x MyGameRecomp >/dev/null; do sleep 30; done
        # espera a carga da execução anterior baixar (até 60 s; o mediaanalysisd do macOS
        # sozinho segura a carga acima de 6 com o Mac ocioso, então não dá para esperar por ela)
        w=0; while [ "$(carga)" -ge ${CARGA_MAX:-8} ] && [ $w -lt 12 ]; do sleep 5; w=$((w + 1)); done
        r=$(BACKEND=vknat EXE="$bin" EXTRA="$ex" nice -n 5 scripts/rapido.sh 60 "$pre-$nm-$i" 2>&1 | head -1)
        echo "$nm rodada $i: $r (carga depois: $(carga))"
    done
    i=$((i + 1))
done
