#!/bin/sh
# Olha a execução mais recente de scripts/teste.sh: converte o último quadro
# capturado em work/runs/<run>/frames/agora.png e mostra o fim do heartbeat e
# dos botões apertados. Uso: scripts/olhar.sh [nome-da-execução]
here=$(cd "$(dirname "$0")/.." && pwd)
run=$(ls -td "$here"/work/runs/*${1:+-$1} 2>/dev/null | head -1)
[ -n "$run" ] || { echo "nenhuma execução"; exit 1; }
if [ -f "$run/frames/ultimo.ppm" ]; then
    ffmpeg -loglevel error -y -i "$run/frames/ultimo.ppm" "$run/frames/agora.png"
fi
grep "\[heartbeat\]" "$run/run.log" | tail -2
grep -E "\[cellPad\] PAD_FILE|\[cellSaveData\] (Auto)?Save" "$run/run.log" | tail -4
echo "$run/frames/agora.png"
