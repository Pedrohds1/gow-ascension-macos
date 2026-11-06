#!/bin/zsh
# Menu principal sem janela: grava cada quadro apresentado e conta os que saem sem a interface.
#   scripts/menu_pisca.sh <nome> ["VAR=1 ..."]      MANTER=1 guarda os quadros em work/menu-<nome>
here=${0:A:h:h}; cd $here || exit 1
n=${1:?nome}; extra=$2
dir=$here/work/menu-$n
[ -d "${dir:?}" ] && rm -rf "${dir:?}"
mkdir -p "$dir"; pad=$here/work/pad-menu-$n.txt; : > $pad
SAVE=work/saves/pedro-cp4 BACKEND=vknat EXTRA="PAD_FILE=$pad GOW_HEARTBEAT=5 SPU_RELOC_OFF=0 SPURS_PASS_SLEEP_US=50 PS3RECOMP_VK_DUMP=$dir/f%05u.ppm PS3RECOMP_VK_DUMP_EVERY=1 $extra" \
  scripts/teste.sh 75 menu-$n > work/logs/menu-$n.out 2>&1 &
sleep 2; run=$(ls -td work/runs/*-menu-$n | head -1)
for i in $(seq 1 150); do grep -q "cellVdec\] Open" $run/run.log 2>/dev/null && break; sleep 1; done
sleep 3; echo 0x4000 > $pad
wait
.venv/bin/python - "$dir" <<'PY'
import sys, glob, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/f*.ppm"))
def ld(f):
    b = open(f, "rb").read(); return np.frombuffer(b[-1280*720*3:], np.uint8).reshape(720, 1280, 3)
fs = fs[int(len(fs) * 0.6):]
# Quadro sem pós-processamento/interface: a faixa de baixo (onde ficam o menu e o logo)
# muito mais clara ou mais escura que a mediana; mede a diferença média para a mediana.
band = np.stack([ld(f)[600:720].astype(np.int16) for f in fs])
med = np.median(band, axis=0)
dev = np.abs(band - med).mean(axis=(1, 2, 3))
m = float(np.median(dev))
bad = [i for i, v in enumerate(dev) if v > max(8.0, 3 * m)]
print(f"quadros do menu: {len(fs)}, desvio mediano {m:.1f}, quadros fora do padrão: {len(bad)} {bad[:12]}")
PY
[ -n "$MANTER" ] || rm -rf "${dir:?}"
