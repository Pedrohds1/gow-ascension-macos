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
| SPU reservation cleared by CAS only on the same line (`resv_valid = line \| 1`) | lost-reservation race that dropped a job kick | e17a8cdb |
| `_setjmp` for SPU exits; FIFO `getenv` read once | Darwin's `setjmp` saves the signal mask (`sigprocmask` hot in profiles) | 3e1ceb71 |
| Runtime builds on arm64: thread-local storage, spins (`isb sy`), SPU clock from `cntvct_el0` | x86-only code paths | 84236d67 |
| SPU per-thread caches in one TLS record | Mach-O TLS access (`_tlv_get_addr`) was hot | deaf366c |
| POSIX stub of `ps3_sampler_start` outside the `_WIN32` block | link error on macOS | 0922b7a6 |

### File system and HLE

| Change | Why | Commit |
|---|---|---|
| `[fs-race]`: lseek/transfer race detector (report only), `FS_LOCK` around lseek/read/write | one 1 MiB block of a save's FIOS disk cache had been written at the wrong position by two threads sharing an fd: a zone loaded without scenery or sound (the Orkos scene). Keeping a position per thread was tried and reverted: FIOS writes each block in 64 KB pieces alternating threads on the shared position | ef3b5b55 |
| `cellFsAioWrite`, `cellFsReadWithOffset`, `cellFsWriteWithOffset` | imported by the game; they returned OK without doing anything | ef3b5b55 |
| Unresolved NIDs reported once per NID | a 40-line global cap was spent at boot | ef3b5b55 |

### SPU performance

| Change | Effect | Commit |
|---|---|---|
| `SPU_FP_NATIVE` on arm64: SPU float in NEON under FPCR round-toward-zero + flush-to-zero | ~20 → ~33 fps in the Megaera scene | 150b13d6 |
| NEON helpers where only x86 paths existed (128-bit LS access, `shufb`, compares, estimates, conversions) | +9% at a checkpoint; Megaera 35 → ~41 fps | 4a285756 |
| SPU float range check in one compare | −14 to −21% cycles in translated code | 9a04266d |
| SPURS profiler outside Windows; DMA log counter saturates | diagnosis | cb2dbf77, 2b3d6ea7 |
| **Lock-line lock striped by line** (`SPU_LOCKLINE_STRIPES`, 16 in the port): one-line transactions take their line's stripe; blocks and the HLE take stripes in ascending order | the hottest function on the EDGE threads and the main thread; Megaera fight 55.3 → 57.6 fps | 6ae347bb |
| Lock-line: exponential backoff after a lost exchange; per-context mirror of the live reservation (Dekker pairing) instead of walking every context; dense holder table; relaxed mirror scan | +1.3, +0.9 and +0.4 fps | 6ae347bb |
| **Relocatable SPU code recognized by content** on a dispatch miss (word index per blob, whole-blob check), up to 4 resident copies per blob | the BatchJobList job keeps several copies of one blob in local store; only one was registered, the others ran in the interpreter (~1.9 s of every 20 s, on a list the main thread waits for) | 6ae347bb |
| Per-thread dispatch state cached in the SPU context (`SPU_TLS_CACHE`), checked against the stack range | every translated function return paid a `_tlv_get_addr` call | 6ae347bb |
| Inline fast path for `spu_drain_call`; overlay signatures looked up by hash; SPURS clock read only when a statistic asks | `clock_gettime(CLOCK_MONOTONIC)` goes through `gettimeofday` on macOS (+0.4 fps) | 6ae347bb |
| Constant targets of computed branches (`SPU_LIFT_CONST_BI=1`, off by default) | no gain measured: the loops it converts are audio, off the critical path | bea71d89 |

### PPU

| Change | Why | Commit |
|---|---|---|
| VMX in NEON for `vaddfp`/`vsubfp`/`vmaddfp`/`vnmsubfp` and friends (`PPU_VMX_SCALAR=1` reverts) | scalar helpers were ~7% of the main thread; 6 million vectors checked bit for bit | 6ae347bb |
| `reserve_*` of the PPU context on their own cache line (runtime header and lifter template) | every SPU notify read them next to cr/xer/lr, which the running thread writes constantly | 6ae347bb |

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
| **New textures without GPU waits**: layout transition recorded in the open batch, 3D textures rewritten in place, deferred destruction keyed by submission number | entering the fight sends thousands of new textures; the backend waited for the GPU ~1,100 times in 5 s (and the 3D colour LUT was recreated ~40 times a second); ~+2 fps | 6ae347bb |
| Sparse texture hash in 4 chains with prefetch; adaptive full-hash period (`RSX_TEXHASH_ADAPT`) | change detection read ~1 GB/s; +0.9 fps | 6ae347bb |

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
| `SPU_LOCKLINE_PROF`: lock-line traffic per line, all-stripe callers, cost per notify stage | 6ae347bb |
| `SPU_INTERP_DUMPAT=<img>:<pc>`: dump local store and registers at the first interpreter entry there | 6ae347bb |
| `GCM_VBL_LAT`, `GCM_VBL_THREAD`, lean FIFO path; ftrace names the caller of each GPU wait | 6ae347bb |
| Tests: `runtime/spu/tests/test_spu_lockline_stripes.c` (striped lock under 11 threads), `tests/ppu_vmx_neon_test.c` | 6ae347bb |
| `PAD_STICK_FILE`: analog sticks read from a file (driving the character in headless tests) | ef3b5b55 |

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
| **SPURS lanes for the job lists the main thread waits on** (`SPURS_LANES` default on arm64) | fc4cecf |
| Park spin capped to the next vblank tick (`GCM_PARK_TICK_CAP=0` reverts); defaults `SPU_LOCKLINE_STRIPES=16`, `RSX_TEXHASH_ADAPT=3`, `VK_DRAW_BATCH_MAX=512` | fc4cecf |
| Five missing function entries of one job module (`spu/jobmod_extra.json`) | fc4cecf |

## This repository

- `jogar.sh`: launcher (ISO mount, save folder, shader caches, upscaler, controller).
- `app/`: `.app` bundle with an icon built from your own disc's artwork.
- `data/edge-placements.c`: the 1,720 EDGE fragment placements found while playing.
- `scripts/`: headless benchmark from a save (`rapido.sh`), profiling summaries, catalog analysis
  (`catvar.py`), EDGE check mode, hang watcher.
- `scripts/` (Nov 8): interleaved A/B of builds or variables (`ab-bin.sh`, `ab-env.sh`), fps over a
  fixed window (`fps-janela.sh`), normal vs. long frames and per-workload concurrency from a
  trace (`quadros.py`, `concorrencia.py`), trace + sample in one run (`diag.sh`), callers of a
  sampled function (`quem_chama.py`, `pais.py`), headless driving (`robo-megera.sh`, `robo-nav.sh`,
  `passo.sh`), FIOS cache check and repair (`cache_fios.py`).
- `jogar.sh` checks the save's FIOS disk cache after each session and repairs a bad block from the
  disc.
