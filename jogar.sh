#!/bin/sh
# God of War: Ascension (PS3, BCES-01741) recompilado nativo para o Mac — para jogar.
#
#   ./jogar.sh              janela 1280x720, Vulkan sobre MoltenVK, 60 Hz
#   TELA_CHEIA=0 ./jogar.sh janela em vez de tela cheia (padrão: tela cheia; para sair use ⌘Q).
#   UPSCALE=0 ./jogar.sh    sem o ampliador (volta ao bilinear); NITIDEZ=0..1 (padrão 0,5);
#                           HIDPI=0 deixa o macOS ampliar a janela de 1280x720.
#   ALTO_FPS=1 ./jogar.sh   experimental: vblank a 125 Hz e o piso do tempo de quadro do
#                           jogo em 1/120 s (GOW_POKE), para passar de 60 com a velocidade
#                           certa. Validado só no menu (~85 fps); no nível a CPU ainda
#                           segura abaixo de 60.
#
# Controles: DualSense/controle pelo SDL; sem controle, teclado com a janela em foco:
#   setas = direcional, Z = ✕, X = ○, A = □, S = △, Q/W = L1/R1, 1/2 = L2/R2,
#   Enter ou Esc = START, Tab = SELECT. Para sair, feche a janela (ou ⌘Q).
# Save em saves/jogador (na primeira vez, copiado do save que já passou pelos menus
# de primeiro uso e pela cinemática; ✕ pula a abertura). Log em saves/jogador/run.log.
here=$(cd "$(dirname "$0")" && pwd)
iso="${ISO:-$here/../God of War - Ascension (Europe, Australia) (En,Fr,De,Es,It,Nl,Pt,Pl,Ru,El,Tr).iso}"   # ISO=<caminho> para outro lugar
exe="${EXE:-$here/build/gow-mac-vk-nat/MyGameRecomp}"
save="$here/saves/jogador"

[ -x "$exe" ] || { echo "falta o executável: $exe"; exit 1; }
if [ ! -d "$here/work/disc/PS3_GAME" ]; then
    mkdir -p "$here/work/disc"
    hdiutil attach -readonly -nobrowse -mountpoint "$here/work/disc" "$iso" >/dev/null || exit 1
fi
if [ ! -d "$save" ]; then
    mkdir -p "$save"
    [ -d "$here/work/saves/menu-principal" ] && cp -R "$here/work/saves/menu-principal/." "$save/"
fi
mkdir -p "$save/hdd0" "$save/hdd1" "$save/gamedata" "$here/work/spirv-cache"

fps=""
[ -n "$ALTO_FPS" ] && fps="GOW_VBLANK_MS=8 GOW_POKE=905D30,78;81AF14,42EFC28F"

# Cache de disco da FIOS (hdd1): um bloco estragado nele quebrou a cena do Orkos em toda
# sessão (2026-11-08). Depois de cada sessão, um vigia confere cada bloco contra o disco e
# conserta o que estiver errado (scripts/cache_fios.py, log em saves/jogador/cache_fios.log).
# Se a conferência anterior ainda roda, espera ela terminar antes de abrir o jogo.
trava="$save/.cache_fios.pid"
if [ -f "$trava" ] && kill -0 "$(cat "$trava" 2>/dev/null)" 2>/dev/null; then
    echo "conferindo o cache da sessão anterior; o jogo abre em seguida..."
    while kill -0 "$(cat "$trava" 2>/dev/null)" 2>/dev/null; do sleep 1; done
fi
if [ -z "$SEM_CONFERIR_CACHE" ] && [ -x "$here/.venv/bin/python" ]; then
    jogo=$$
    nohup sh -c "while kill -0 $jogo 2>/dev/null; do sleep 5; done
        echo \$\$ > '$trava'
        nice -n 15 '$here/.venv/bin/python' -I '$here/scripts/cache_fios.py' '$save' --consertar > '$save/cache_fios.log' 2>&1
        rm -f '$trava'" >/dev/null 2>&1 &
fi

cd "$save" || exit 1
exec env PS3_VFS_ROOT="$here/work/disc" PS3_HDD0_ROOT="$save/hdd0" PS3_HDD1_ROOT="$save/hdd1" \
    PS3_GAMEDATA_ROOT="$save/gamedata" \
    PS3RECOMP_VK_GUEST_PROGRAMS=1 PS3RECOMP_SPIRV_CACHE="$here/work/spirv-cache" RSX_PIPE_CATALOG="$here/work/pipe-catalog.bin" PS3RECOMP_VK_WINDOW=1 \
    PS3RECOMP_VK_FULLSCREEN=${TELA_CHEIA:-1} PS3RECOMP_VK_HIDPI=${HIDPI:-1} PS3RECOMP_VK_UPSCALE=${UPSCALE:-1} PS3RECOMP_VK_SHARP=${NITIDEZ:-0.5} PS3_TITLE="God of War: Ascension" \
    SPU_RELOC_OFF=0 SPURS_PASS_SLEEP_US=50 GOW_HEARTBEAT=5 $fps \
    "$exe" "$here/work/input/EBOOT.elf" > "$save/run.log" 2>&1
