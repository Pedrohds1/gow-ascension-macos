#!/bin/sh
# Um passo de condução no jogo do robo-nav.sh: segura o analógico esquerdo em (lx, ly)
# por N segundos, solta, e salva o quadro atual em work/nav-quadro.png.
#
#   scripts/passo.sh <lx> <ly> <segundos> [botão] [rx ry]
#   lx/ly 0-255 (128 = centro; ly 0 = para frente, 255 = para trás; lx 0 = esquerda)
#   botão: máscara do PAD_FILE (0x8000 quadrado, 0x1000 triângulo, 0x2000 bola,
#          0x4000 xis, 0x0800 R1, 0x0400 L1) apertada a cada 0,5 s durante o passo
#   rx ry: analógico direito (câmera), padrão 128 128
here=$(cd "$(dirname "$0")/.." && pwd)
run=$(cat "$here/work/nav-atual.txt")
name=$(basename "$run" | sed 's/^[0-9-]*-//')
stick="$here/work/stick-$name.txt"; pad="$here/work/pad-$name.txt"
lx=${1:-128}; ly=${2:-128}; secs=${3:-1}; btn=${4:-}; rx=${5:-128}; ry=${6:-128}
echo "$lx $ly $rx $ry" > "$stick"
end=$(( $(date +%s) + secs ))
while [ "$(date +%s)" -lt $end ]; do
    [ -n "$btn" ] && echo "$btn 4" > "$pad"
    sleep 0.5
done
echo "128 128 128 128" > "$stick"
sleep 1.5
"$here/.venv/bin/python" -I -c "
from PIL import Image
Image.open('$run/frames/ultimo.ppm').convert('RGB').resize((960, 540)).save('$here/work/nav-quadro.png')
" 2>/dev/null
grep -h heartbeat "$run/run.log" | tail -1 | sed -E 's/.*(t=[0-9]+s).*(game_frame=[0-9]+).*/\1 \2/'
