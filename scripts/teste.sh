#!/bin/sh
# Teste isolado do GoW:A no Mac (regras do Pedro): Metal sem janela (nada rouba o
# foco), mudo (o dispositivo toca silêncio, o ritmo do áudio não muda), save numa
# pasta só desta execução, e no fim mata só o PID que este script abriu.
#
#   scripts/teste.sh [segundos] [nome]          (padrão: 90 s, "boot")
#   EXTRA="VAR=1 OUTRA=2" scripts/teste.sh ...  variáveis a mais para o jogo
#   SAVE=work/saves/apos-menus scripts/teste.sh ... começa de uma cópia desse save
#                                               (gamedata/ e hdd0/; o original fica intacto)
#
# Saída em work/runs/<data>-<nome>/: run.log, frames/tNNN.png (a cada 10 s).
here=$(cd "$(dirname "$0")/.." && pwd)
secs=${1:-90}; name=${2:-boot}
out="$here/work/runs/$(date +%Y%m%d-%H%M%S)-$name"
mkdir -p "$out/hdd0" "$out/hdd1" "$out/frames" "$out/gamedata"
if [ -n "$SAVE" ]; then
    case "$SAVE" in /*) src=$SAVE ;; *) src="$here/$SAVE" ;; esac
    cp -R "$src/." "$out/" || exit 1
fi

iso="$here/../God of War - Ascension (Europe, Australia) (En,Fr,De,Es,It,Nl,Pt,Pl,Ru,El,Tr).iso"
if [ ! -d "$here/work/disc/PS3_GAME" ]; then
    hdiutil attach -readonly -nobrowse -mountpoint "$here/work/disc" "$iso" >/dev/null || exit 1
fi

# BACKEND=vk: o build Vulkan/MoltenVK (build/gow-mac-vk), com os shaders do jogo;
# BACKEND=vknat: o mesmo com os nativos EDGE via SIMDe (build/gow-mac-vk-nat).
if [ "$BACKEND" = vk ] || [ "$BACKEND" = vknat ] || [ "$BACKEND" = vkpromo ] || [ "$BACKEND" = vkfence ]; then
    exe="$here/build/gow-mac-vk/MyGameRecomp"
    [ "$BACKEND" = vknat ] && exe="$here/build/gow-mac-vk-nat/MyGameRecomp"
    # vkpromo: os nativos e os registros da SPU promovidos a locais (scripts/promover_spu.py)
    [ "$BACKEND" = vkpromo ] && exe="$here/build/gow-mac-vk-promo/MyGameRecomp"
    # vkfence: os nativos e a barreira de memória em cada DMA da SPU (spu_dma.h)
    [ "$BACKEND" = vkfence ] && exe="$here/build/gow-mac-vk-fence/MyGameRecomp"
    # EXE=<binário>: um executável guardado (A/B entre versões do mesmo build)
    [ -n "$EXE" ] && exe="$EXE"
    gfx="PS3RECOMP_VK_GUEST_PROGRAMS=1 PS3RECOMP_SPIRV_CACHE=$here/work/spirv-cache PS3RECOMP_VK_DUMP=$out/frames/ultimo.ppm PS3RECOMP_VK_DUMP_EVERY=60"
else
    exe="$here/build/gow-mac/MyGameRecomp"
    gfx="PS3RECOMP_METAL_HEADLESS=1 PS3RECOMP_METAL_FRAME_DUMP=$out/frames/ultimo.ppm"
fi
cd "$out" || exit 1
env PS3_VFS_ROOT="$here/work/disc" PS3_HDD0_ROOT="$out/hdd0" PS3_HDD1_ROOT="$out/hdd1" \
    PS3_GAMEDATA_ROOT="$out/gamedata" $gfx \
    PS3_MUTE=1 $EXTRA \
    "$exe" "$here/work/input/EBOOT.elf" > "$out/run.log" 2>&1 &
pid=$!
echo "pid $pid  log $out/run.log"

i=0
while [ $i -lt "$secs" ] && kill -0 $pid 2>/dev/null; do
    sleep 1; i=$((i + 1))
    if [ $((i % 10)) -eq 0 ] && [ -f "$out/frames/ultimo.ppm" ]; then
        ffmpeg -loglevel error -y -i "$out/frames/ultimo.ppm" "$out/frames/t$(printf %03d $i).png"
    fi
done
if kill -0 $pid 2>/dev/null; then
    kill $pid; sleep 1; kill -9 $pid 2>/dev/null
    echo "encerrado após ${secs}s"
else
    wait $pid; echo "saiu sozinho (código $?) após ${i}s"
fi
# O cache de sistema do jogo (hdd1, ~450 MB) fica para trás em toda execução: 118 testes
# encheram 64 GB. O log, os PNGs e o save (gamedata/hdd0) ficam; MANTER_HDD1=1 guarda tudo.
[ -z "$MANTER_HDD1" ] && rm -rf "$out/hdd1"
echo "$out"
