# Changes made for macOS / Apple Silicon

Everything below is in this repository. The upstream bases are ps3recomp v0.17.0 (a6d76c8) with
MaChInEgUn3's GoW:A patch, and God-of-War-Ascension-Recompiled at 1d6f168. The original commit ids
of the development branches (`macos-port`) are given for reference.

## ps3recomp — runtime

### arm64 correctness

| Change | Why | Commit |
|---|---|---|
| SPU PUT on arm64: the FIFO "park" word written last, with a release store | a 16-byte `memcpy` is two 8-byte stores that another core can see out of order; the RSX FIFO walker saw the park released and stale words after it, read data as commands and hung after minutes of play | 2f6d88b9 |
| Memory barrier on every SPU DMA transfer; acquire loads in the FIFO walker | ARM's weaker memory ordering | 5541a667, 50d1b996, 64cac67e |
| SPU reservation cleared by CAS only on the same line (`resv_valid = line | 1`) | lost-reservation race that dropped a job kick | e17a8cdb |
| `_setjmp` for SPU exits; FIFO `getenv` read once | Darwin's `setjmp` saves the signal mask (`sigprocmask` hot in profiles) | 3e1ceb71 |
| Runtime builds on arm64: thread-local storage, spins (`isb sy`), SPU clock from `cntvct_el0` | x86-only code paths | 84236d67 |
| SPU per-thread caches in one TLS record | Mach-O TLS access (`_tlv_get_addr`) was hot | deaf366c |
| POSIX stub of `ps3_sampler_start` outside the `_WIN32` block | link error on macOS | 0922b7a6 |

### SPU performance

| Change | Effect | Commit |
|---|---|---|
| `SPU_FP_NATIVE` on arm64: SPU float in NEON under FPCR round-toward-zero + flush-to-zero | ~20 → ~33 fps in the Megaera scene | 150b13d6 |
| NEON helpers where only x86 paths existed (128-bit LS access, `shufb`, compares, estimates, conversions) | +9% at a checkpoint; Megaera 35 → ~41 fps | 4a285756 |
| SPU float range check in one compare | −14 to −21% cycles in translated code | 9a04266d |
| SPURS profiler outside Windows; DMA log counter saturates | diagnosis | cb2dbf77, 2b3d6ea7 |

### RSX on Vulkan / MoltenVK

| Change | Symptom it fixed | Commit |
|---|---|---|
| Vulkan on macOS through the Khronos loader + MoltenVK | the Metal backend drew wrong surfaces in-game | 8e8abf6a |
| Depth bounds test emulated in the pixel program (Metal has none) | rectangular light blocks, white glare and "combs" on terrain | 316bbcc7, cae2ae93 |
| Sun shadow cascades read from part of a shared zeta in the RSX texture's scale | rectangles of shadow on the bridge | b6664729 |
| RSX shadow comparison is `texel OP reference` (inverse of GL) | wrong shadows | 94e2eab2 |
| Border colour of depth textures | shadow-map edges | 11967875 |
| `CONSTANT_COLOR`/`CONSTANT_ALPHA` blend factors and `SET_BLEND_COLOR` | bloom layers added at full weight | 7e849bdf |
| Fragment program: R/H register aliasing (H(2n) = R(n).xy as fp16), RSX precision modifiers, DP3 chains, NaN guard by bits (from otavix010's Windows port) | Kratos' skin grey and scaly, tattoo missing | 0ab6238c |
| **Polygon offset (depth bias) as dynamic state** | dotted grid ("shadow acne") over every sunlit surface | f02e8b44 |
| **3D textures** (image, per-slice upload, 3D view, split descriptor layout, 1×1×1 dummy) | whole scene yellow-green: the 32×32×32 colour-grading LUT was sampled as its first (blue-less) 2D slice | fe37b497 |
| Engine texture cache 1,024 → 3,072 entries | texture thrash after ~10 min | fd84b455 |
| Flip through the queue with little room in a title's own callback block | flip stalls | a620cab4 |
| Presentation upscaler: bicubic Catmull-Rom + adaptive sharpening, HiDPI | 720p image on a Retina display | 2731ac4e |

### Shader / pipeline stutter

| Change | Effect | Commit |
|---|---|---|
| On-disk pipeline catalog (`RSX_PIPE_CATALOG`): every pipeline is recorded and rebuilt at the next boot; HLSL hash checked on first use | known pipelines ready before gameplay | b214bbb6 |
| One `VkShaderModule` per distinct program text | MoltenVK caches MSL + Metal library per module: boot 7.2 → 1.3 s, new variant 8.8 → 0.4 ms | b214bbb6 |
| Pipeline key over the **canonical** render state (state of disabled stages zeroed) | ~8,200 pipelines → ~900; the 8,192-entry table no longer fills (broken image after ~13 min) | c2f44ee6 |
| **Asynchronous pipeline builds** (`VK_ENG_ASYNC=N`, default 2): glslang + MoltenVK/Metal on worker threads; draws whose pipeline is still compiling are skipped | worst stall 98 + 190 ms → 0.4 ms; boot precompile 4.9 s → 0.12 s on the game thread | 93072ab3 |

### Input, audio, window

| Change | Commit |
|---|---|
| Keyboard in the SDL pad backend (macOS/Linux) | b57a87bf |
| Esc maps to START instead of closing the window | 332ce067 |
| Controllers initialised on the main thread and read under `SDL_LockJoysticks` | 79d6a813 |
| `PS3_MUTE` plays silence at the device's pace | 6a901856 |

### Diagnostics

| Tool | Commit |
|---|---|
| Render-pass capture on a trigger file (`RSX_ENGINE_PASS_DUMP_TRIGGER`): colour, exact depth (`.f32`), stencil, per-draw state and depth bounds | f24f40dd, 55ed9ffd |
| `RSX_DBT_STATS`, `VK_SCISSOR_TINY`, batch-flush diagnostics | b2a95b06, 8c9a94ad, 703e1781 |
| Metal engine: delayed release of depth snapshots (fixed a 24 GB leak) | ab33cac1 |

## gowa-port — the game port

| Change | Commit |
|---|---|
| Port builds and links on macOS arm64 (EU BCES-01741); title settings were under `#ifdef _WIN32` | f86bca8 |
| EDGE native kernels via SIMDe, native SPU float, guest pages reopened | 2e0f052 |
| `GOW_PEEK` outside Windows; `GOW_POKE` for the frame-time floor (>60 fps experiments) | 7e62e69 |
| Window on the main thread, game on its own thread; `GOW_QOS` | d4b7d0a |
| Each EDGE region in its fastest form (native or translated), chosen by a cycle-accurate check mode | 3547fab, 54e899f |
| Four EDGE lanes on arm64; `GOW_EDGE_BODY_LIFT` | f6ed1e1 |
| vram-guard veto counter | 4fab706 |

## This repository

- `jogar.sh`: launcher (ISO mount, save folder, shader caches, upscaler, controller).
- `app/`: `.app` bundle with an icon built from your own disc's artwork.
- `data/edge-placements.c`: the 1,720 EDGE fragment placements found while playing.
- `scripts/`: headless benchmark from a save (`rapido.sh`), profiling summaries, catalog analysis
  (`catvar.py`), EDGE check mode, hang watcher.
