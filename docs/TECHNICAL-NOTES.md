# Technical notes — how the problems were found

Lessons from bringing God of War: Ascension to macOS. Most of them apply to any PS3 title on
ps3recomp, and the GPU ones to any console port that translates shaders at runtime for
Vulkan/MoltenVK or Metal.

## Method that worked

- **Visual bug: capture the frame pass by pass before guessing.** Two guesses (fragment precision,
  the upscaler's sharpening) cost two rounds of relaunching the game. The pass capture showed the
  guilty pass at first sight. For the dotted floor it was the sun shadow mask; for the green scene,
  only the final pass, which read a 32×32 texture from main memory.
- **Suspect RSX state the backend ignores.** Three different defects had the same root: depth bounds
  test, polygon offset, 3D textures. When a scene looks wrong, list the registers it uses and check
  that the backend implements each one.
- **One switch per fix.** Every correction has an environment variable that turns it off
  (`RSX_POLY_OFFSET=0`, `VK_ENG_ASYNC=0`, `RSX_FP_PREC=0`, ...). The player relaunches at the same
  spot and says whether the defect is gone, without a rebuild.
- **When the automated test cannot reach a scene, the player plays to it** with the capture or the
  code collection on. Headless runs load the last checkpoint only.
- **Measure what the CPU is doing before optimising:** `sample <pid> 5` plus a per-thread summary
  (`scripts/perfil.py`). It is read-only and safe while someone plays.

## arm64 memory ordering

- **Torn 16-byte writes.** On x86, a 16-byte copy is one store; on arm64, `memcpy` makes it two
  8-byte stores, and another core may observe them in either order.
  - The SPU writes RSX FIFO segments with a "park" jump as its first word. The FIFO walker saw the
    park released while the next word was still stale, and read a float of the viewport as a
    command. The result was a "FIFO resync" and a hang minutes later.
  - Fix: write the body first and the park word last, with a release store.
  - The reliable warning sign in logs was an init command (`semaphore context dma`) executing
    mid-level.
- Barriers on every SPU DMA and acquire loads in the FIFO walker are cheap on the M4 and remove a
  whole class of "impossible" states.

## SPU performance

- **Interpreter hot = untranslated code.** EDGE jobs DMA small code fragments from the executable
  into local store at varying addresses. Each (source address, local-store address) pair must be
  translated on its own (`lift_edge_frags.py`). Log them with `SPU_LSA_GETLOG` and
  `SPU_INTERP_MISS`.
  - Megaera: 3.5 → 20 fps.
  - Harem scene: ~3.6 → ~30 fps.
- **NEON beats the generic paths.** Native SPU float under FPCR round-to-zero + flush-to-zero, NEON
  helpers for shuffles, compares and conversions, and a 3-instruction float range check together
  took the Megaera scene from 20 to ~41 fps.
- **x86 native kernels through SIMDe can lose to translated code.** Compare region by region with a
  cycle-accurate check mode, not by fps.
- **No gain on the M4** (dropped): SPU register promotion (context round-trips are cheap in L1),
  `-O3`, raising thread QoS.

## Reaching ~60 fps in the Megaera fight

~41 → ~60 fps over two days (A/B 59.9; steady part 61.1; 10-minute run 59.7).

- **Measure the right phase.** The old benchmark window included a light transition (65–83 fps)
  whose end moves with loading speed, so changes that only sped up loading looked like gains.
  Measure a fixed window of the fight and look at the frames.
- **Give threads only to what the main thread waits on.** The game registers every SPURS
  workload with a contention of one SPU. Extra lanes for the job lists the main thread waits on
  took the fight from 43.7 to 50.8 fps; lanes for the other lists did nothing or worse.
- **No GPU waits inside a frame.** Texture creation waited for a submit just for a layout
  transition, and every release waited for all batches in flight.
- **Profile a lock per line before striping it.** The SPU lock-line lock was the hottest function
  once the lanes were up. Most of its traffic was on different lines, so striping by line paid
  off (+2.3 fps).
- **On ARM, a load-acquire per entry serializes a scan.** Read relaxed after the seq_cst fence
  and acquire only on a match.
- **A field other threads read must not share a cache line with fields its owner writes all the
  time** (the PPU `reserve_addr` next to cr/xer/lr).
- **An interpreter hot spot outside a module's image is runtime-loaded code.** Find where it
  came from before touching the lifter. A local-store dump at the first miss, compared with the
  executable, showed a relocatable blob that was already translated, at a second placement the
  runtime did not know. Recognizing blobs by content took ~2 s of every 20 s out of the
  interpreter.
- **Mach-O thread-locals are calls** (`_tlv_get_addr`). Cache the address in a per-thread
  context and check it against the stack range.
- **Dropped, measured:** handing the SPU back to EDGE when it has priority (`SPURS_POLL_SELECT`,
  56 → 34 fps), an 8 ms vblank (the gain was in the transition, not the fight), MoltenVK's
  asynchronous submits, a vblank delivery thread.
- Still open: ~14% of the frames run past 20 ms. They are bound by the EDGE geometry jobs, which
  the game confines to four SPUs.

## Shader stutter (pipelines)

The first time each shader and state combination appeared, the frame paid the whole host compile:
glslang, then MoltenVK's MSL translation and the Metal compiler, 30–190 ms each. A new area brings a
hundred of them. Fixed in order of impact:

1. **Canonical pipeline key.** Registers keep the last value written even when their stage is off,
   so the key carried junk: blend factors with blend off, alpha function with the test off, stencil
   with stencil off, back-face stencil without two-sided mode, depth function without the depth
   test, cull face with culling off. One session made ~8,200 pipelines from ~900 real ones and
   filled the 8,192-entry table after ~13 minutes, which broke the image. Zero only what every
   backend ignores when the stage is off.
2. **Asynchronous builds.** `pipeline_create` stores the text and returns a handle; 2 worker threads
   build modules and topology variants, and a draw whose pipeline is still compiling is skipped
   (`pipeline_ready`).
   - The render pass and layouts are created beforehand on the FIFO thread, so the workers only
     read them.
   - Each pipeline has a single owner (atomic exchange), and release takes ownership first.
3. **On-disk catalog.** Every pipeline is recorded and rebuilt in the background at boot. The HLSL
   hash is checked on first use, so a stale catalog costs one stall, never a wrong shader. Because
   FNV-1a is invertible, keys could be re-derived after the key format changed, without losing the
   catalog.
4. **Shared shader modules.** MoltenVK caches its translation and Metal library per module; one
   module per pipeline recompiled the same program for each of them.
5. **SPIR-V disk cache** for glslang.

Measuring trap: macOS keeps its own Metal shader cache, so a "new" shader compiles cheaply in
repeated headless tests. The real cost only shows on a machine that has never seen it.

## RSX state that a backend must honour

| State | Without it |
|---|---|
| Depth bounds test (Metal has none: emulate in the pixel program) | deferred lights fill their whole scissor (rectangles), sky passes over geometry |
| Polygon offset / depth bias | shadow acne: a dotted grid over every sunlit surface |
| 3D textures | the colour-grading LUT is read as a 2D slice: one dominant colour |
| Shadow comparison `texel OP ref` (inverse of GL) | inverted or missing shadows |
| Fragment-program precision and H/R register aliasing | wrong skin shading, missing details |
| Blend constants (`SET_BLEND_COLOR`) | post-processing layers at full weight |
