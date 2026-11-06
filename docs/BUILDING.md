# Building and playing on macOS

Tested on an Apple M4 Max with macOS 27. Any Apple Silicon Mac should work; heavier scenes need a
fast CPU (the SPU work is the bottleneck).

**You need your own copy of the game.** This guide never downloads game data or keys.

## 1. Dependencies

```sh
brew install cmake ninja ccache python@3.12 sdl2-compat molten-vk vulkan-loader vulkan-headers \
             glslang spirv-tools spirv-cross simde ffmpeg
python3.12 -m venv .venv && .venv/bin/pip install capstone pycryptodome pillow
```

## 2. The disc and the executable

1. Put your disc image next to this folder, or point `ISO=` at it. The European ISO of God of War:
   Ascension (BCES-01741) is what was tested. Its PSARC files are already decrypted.
2. Mount it read-only:
   `hdiutil attach -readonly -nobrowse -mountpoint work/disc "<your ISO>"`.
3. Decrypt your `PS3_GAME/USRDIR/EBOOT.BIN` (SELF) to an ELF. For example, run
   `rpcs3 --decrypt` with your console's keys installed. Save it as `work/input/EBOOT.elf`.

## 3. Translate the code

```sh
# PPU: functions, imports, lifted C
.venv/bin/python ps3recomp/tools/ppu_loader.py work/input/EBOOT.elf -o work/meta/EBOOT
.venv/bin/python ps3recomp/tools/ppu_lifter.py work/input/EBOOT.elf \
    --functions work/meta/EBOOT/EBOOT.functions.json --hle-stubs work/meta/EBOOT/EBOOT.imports.json \
    --code-end 0x62C4CC --chunk-lines 150000 -j 12 -o work/recomp
.venv/bin/python gowa-port/scripts/patch_lift.py work/recomp

# SPU: the four images, the 33 job modules
.venv/bin/python gowa-port/analysis/make_spu_port.py work/input/EBOOT.elf data/spu-images-eu.json work/port ps3recomp
.venv/bin/python gowa-port/analysis/extract_gow_jobmods.py work/input/EBOOT.elf work/spu-jobmods
JOBMOD_EXTRA_FUNCS_JSON=gowa-port/spu/jobmod_extra.json .venv/bin/python ps3recomp/tools/lift_jobmods.py \
    --jobmods work/spu-jobmods --out work/port/spu --register work/port/spu/jobmods_register.c \
    --symbol-prefix gow --merge-chunks

# EDGE code fragments: the placements found while playing (data/edge-placements.c)
.venv/bin/python gowa-port/analysis/lift_edge_frags.py work/input/EBOOT.elf work/port ps3recomp data/edge-placements.c
.venv/bin/python gowa-port/analysis/lift_reloc_blobs.py work/input/EBOOT.elf work/port ps3recomp
```

The code addresses above are for the **European** executable. The US version (BCUS98232) is
shifted: see `gowa-port/docs/BUILDING.md` and translate fixed addresses by byte signature.

## 4. Build

```sh
cmake -S gowa-port/port -B build/gow-mac-vk-nat -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPS3RECOMP_DIR=$PWD/ps3recomp -DRECOMP_DIR=$PWD/work/recomp -DSPU_DIR=$PWD/work/port/spu \
  -DPORT_GENERATE_HLE_NIDS=ON -DPORT_ENABLE_LLE_LIBSRE=OFF -DPS3RECOMP_FFMPEG_RUNTIME=ON \
  -DPS3RECOMP_RSX_VULKAN=ON -DPS3RECOMP_VULKAN_INCLUDE_DIR=/opt/homebrew/include \
  -DGOW_EDGE_NATIVE=ON -DGOW_SPU_FP_NATIVE=ON -DGOW_SIMDE_INCLUDE=/opt/homebrew/include \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DPython3_EXECUTABLE=$PWD/.venv/bin/python \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
ninja -C build/gow-mac-vk-nat -j 10
```

A clean build takes about 2 minutes on an M4 Max with ccache warm. The first run of a new binary is
slow while macOS scans it, so do not benchmark that run.

## 5. Play

```sh
./jogar.sh                # full screen, upscaled; TELA_CHEIA=0 for a window
app/fazer_app.sh          # builds "God of War Ascension.app" with an icon from your disc
```

Options are set through environment variables:
- `UPSCALE=0` turns the upscaler off;
- `NITIDEZ=0..1` sets the sharpening;
- `HIDPI=0`;
- `ALTO_FPS=1` is experimental (above 60 fps).

The save goes to `saves/jogador`; the shader catalog and SPIR-V cache go to `work/`.

## fps drop in a new area

If the frame rate collapses somewhere, the SPUs are probably interpreting EDGE code that has not been
translated yet. To confirm, run `sample <pid> 5`: threads will sit in `spu_interp_run_until`. To fix
it:

```sh
SPU_LSA_GETLOG=10000-3E000 SPU_LSA_GETLOG_EA=10000-820000 SPU_INTERP_MISS=1 ./jogar.sh
# play through the slow scene, quit, then:
cp saves/jogador/run.log work/coleta.log
.venv/bin/python gowa-port/analysis/lift_edge_frags.py work/input/EBOOT.elf work/port ps3recomp \
    data/edge-placements.c work/coleta.log
cp work/port/spu/edgefrags_register.c data/edge-placements.c     # keep the new placements
cmake -S gowa-port/port -B build/gow-mac-vk-nat && ninja -C build/gow-mac-vk-nat
```

Please send the new `data/edge-placements.c` as a pull request so everyone benefits.

## Diagnostics

- Headless benchmark from a save: `CP=<save folder> scripts/rapido.sh <seconds> <name>`.
- Capture every render pass of one frame while playing: `./jogar-captura.sh`, stop at the scene,
  then `touch work/captura/gatilho`. Each pass is written to `work/captura` as PPM, with depth,
  stencil and per-draw state.
- Switches to test a hypothesis without rebuilding:
  - `VK_ENG_ASYNC=0` (synchronous shader builds);
  - `RSX_POLY_OFFSET=0`;
  - `RSX_FP_PREC=0` / `RSX_FP_ALIAS=0` (fragment-program fixes);
  - `RSX_PIPE_KEY_RAW=1`;
  - `RSX_DBT_EMULATE=0`;
  - `VK_ENG_SHARE_MODULES=0`.
