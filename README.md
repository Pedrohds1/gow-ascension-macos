# God of War: Ascension — native macOS port (Apple Silicon)

**[Leia em português](README.pt-BR.md)**

God of War: Ascension (PS3, BCES-01741, Europe) **statically recompiled to native arm64 code** and
running on macOS. No emulator and no Wine: the PPU code is translated to C and compiled for the
Apple CPU, the SPU programs (including the EDGE geometry jobs) are translated per image, and the RSX
GPU is reimplemented on Vulkan through MoltenVK (Metal).

**Status: playable from start to the areas tested, natively on macOS, with correct graphics and no
shader-compilation stutter.** Played for long sessions on an M4 Max.

> This repository contains **no game data**. You need your own legally obtained copy of the game
> (disc image) and must decrypt your own `EBOOT.BIN`. Everything derived from the game (the
> recompiled code, shader caches, saves) is generated locally and is git-ignored.

## Test machine

| | |
|---|---|
| Computer | MacBook Pro 14" (Mac16,6) |
| Chip | Apple M4 Max — 14-core CPU (10 performance + 4 efficiency), 32-core GPU |
| Memory | 36 GB unified |
| OS | macOS 27.0.1 |
| Graphics stack | MoltenVK 1.4.2 + Vulkan loader 1.4.363, SDL2 (sdl2-compat 2.32), glslang 16.6 |
| Display | 3024×1964 Retina; game renders at 1280×720, upscaled (Catmull-Rom + adaptive sharpening) |

All numbers below were measured on this machine, on mains power.

## Performance

Frame rate by scene (the game's own frame counter, 50–90 s windows):

| Scene | fps |
|---|---|
| Main menu | 60 (locked); ~74 uncapped |
| Start of the first level | 46–54 |
| Megaera fight | **~60** (59.9 in A/B; 61.1 in its steady part; 59.7 over 10 minutes) |
| Bridge / prison area | 42–44 |
| Rooftop fight (heaviest scene measured) | 22–27 |
| Harem illusion scene | ~30 (was ~3.6 before translating the EDGE code fragments it uses) |

The Megaera fight was re-measured on 2026-11-08 (fixed window of the fight, headless, interleaved
A/B). The other scenes are from earlier builds; they have not been re-measured since the work below.

How the Megaera scene got from unplayable to ~60 fps:

| Step | fps |
|---|---|
| First boot to the level (SPU code interpreted) | 3.5 |
| EDGE code fragments translated (443 placements) | 20 |
| SPU float in NEON under FPCR round-to-zero + flush-to-zero | 33 |
| EDGE native kernels via SIMDe where they beat the translated code | 35 |
| NEON helpers for SPU operations, 3-instruction float range check | ~41 |
| Main-thread park spin capped to the next vblank tick | 43.7 |
| Extra SPURS lanes for the job lists the main thread waits on | 50.8 |
| New textures no longer wait for the GPU; larger draw batches; dense SPU coherence table | ~52 |
| Live-reservation mirror, lock-line backoff, adaptive texture hash | ~56 |
| SPU lock-line lock striped by line | 57.6 |
| Relocatable SPU code copies recognized by content (out of the interpreter); thread-locals cached per SPU context | ~60 |

Shader compilation (the stutter on entering a new area or doing something for the first time):

| | Before | After |
|---|---|---|
| Pipelines a long session creates | ~8,200 (table full after ~13 min → broken image) | ~900–1,450 |
| Worst stall when a new pipeline appears | 98 ms + 190 ms on the draw | 0.4 ms (compiled on worker threads) |
| Precompiling the known pipelines at boot | 7.2 s | 0.12 s on the game thread; the rest in the background |
| Same shader pair in many pipelines | recompiled by Metal each time | one shared module (8.8 → 0.4 ms) |

The game is CPU-bound (SPU work), not GPU-bound. In the Megaera fight the median frame is under
16.7 ms; the ~14% of frames that run past 20 ms are bound by the EDGE geometry jobs, which the game
confines to four SPUs.

## What was built or fixed for macOS

See **[docs/CHANGES.md](docs/CHANGES.md)** for the full list. In short:

- **arm64 correctness:**
  - torn 16-byte SPU writes, which caused rare hangs where the FIFO walker read data as commands;
  - memory ordering on the FIFO and on SPU DMA;
  - SPU reservation race;
  - `_setjmp`, TLS and the clock.
- **Performance:**
  - translated EDGE code fragments (1,720 placements);
  - NEON SPU helpers and the native SPU float path on arm64;
  - SPURS kernel latency;
  - per-region choice between native and translated EDGE code.
- **Graphics (RSX → Vulkan/MoltenVK):**
  - depth bounds test emulated (Metal has none);
  - shadow cascades and comparison direction;
  - polygon offset (no more shadow acne);
  - 3D textures (colour-grading LUT);
  - blend constants;
  - fragment-program precision and R/H register aliasing;
  - 3,072-entry texture cache.
- **Shader stutter:**
  - canonical pipeline keys;
  - on-disk pipeline catalog;
  - shared shader modules;
  - asynchronous pipeline compilation;
  - SPIR-V disk cache.
- **Mac integration:**
  - window on the main thread;
  - DualSense and keyboard through SDL;
  - full screen with HiDPI upscaler;
  - `.app` bundle with icon.
- **Tooling:**
  - on-demand capture of every render pass (colour, depth, stencil, per-draw state);
  - SPU interpreter-miss logging;
  - quick headless benchmark from a save.

## Repository layout

| Path | What |
|---|---|
| `ps3recomp/` | [sp00nznet/ps3recomp](https://github.com/sp00nznet/ps3recomp) v0.17.0 + the GoW:A patch + all macOS/arm64 changes (MIT) |
| `gowa-port/` | [MaChInEgUn3/God-of-War-Ascension-Recompiled](https://github.com/MaChInEgUn3/God-of-War-Ascension-Recompiled) + macOS changes (MIT) |
| `scripts/` | test, benchmark and analysis scripts |
| `app/` | builds `God of War Ascension.app` (icon generated from your own disc's artwork) |
| `data/` | addresses only: SPU images in the EU EBOOT and the EDGE code-fragment placements to translate from your EBOOT |
| `jogar.sh` | launcher (window, controller, save, caches) |
| `docs/` | build guide, change list, technical notes |

## Building and playing

See **[docs/BUILDING.md](docs/BUILDING.md)**. Summary: Homebrew dependencies → mount your ISO →
decrypt your EBOOT → translate PPU and SPU code → CMake/Ninja → `./jogar.sh` or
`app/fazer_app.sh`.

Controls: DualSense/controller via SDL, or keyboard (arrows, Z=✕ X=○ A=□ S=△, Q/W=L1/R1,
1/2=L2/R2, Enter/Esc=START, Tab=SELECT). ⌘Q quits.

## Known issues

- Character "replicas" drawn on some walls; water reflection not right yet.
- The depth-of-field background can look blocky.
- New areas can still drop fps until their EDGE code fragments are translated (see
  `docs/BUILDING.md`, "fps drop in a new area"). Shader compilation never freezes the game, but an
  object may appear a few frames late the first time.

## Contributing

Forks and pull requests are welcome — especially for the open graphics issues, performance (the SPU
side is the bottleneck), and other PS3 titles on the same base. The technique notes in
[docs/TECHNICAL-NOTES.md](docs/TECHNICAL-NOTES.md) explain how each problem was found.

## Credits and license

- **ps3recomp** by Ned Heller / sp00nznet — MIT.
- **God of War: Ascension Recompiled** by MaChInEgUn3 and contributors — MIT. The base of this
  port: game patch, job-module extraction, EDGE native kernels.
- Reference ports for Windows: arnaldolima1/gow-ascension-pc and otavix010/gow-ascension-pc (fragment
  program and SPU fixes ported from the latter).
- macOS port: Pedro Henrique ([@Pedrohds1](https://github.com/Pedrohds1)), developed with Claude
  (Anthropic).

Our own files are MIT ([LICENSE](LICENSE)); `ps3recomp/` and `gowa-port/` keep their own licenses.
God of War is a trademark of Sony Interactive Entertainment. This project is not affiliated with or
endorsed by Sony; it contains no copyrighted game code or assets.
