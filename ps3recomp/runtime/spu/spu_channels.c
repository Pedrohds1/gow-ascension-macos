/*
 * ps3recomp - SPU channel + indirect-branch runtime glue
 *
 * Implements the externs the SPU lifter (tools/spu_lifter.py) emits:
 *   - spu_rdch / spu_rchcnt / spu_wrch : SPU channel access. MFC channels are
 *     routed to the DMA engine (spu_dma.h); mailboxes, signal notification,
 *     events and the decrementer use the spu_context channel fields.
 *   - spu_indirect_branch : resolves ctx->pc to a lifted spu_func_* via a
 *     registry that generated code populates by calling spu_recomp_register().
 *
 * The MFC engine state is kept per spu_context here (spu_context.h does not
 * embed one), in a small lazily-populated registry.
 */

#include "spu_dma.h"
#include "spu_coherency.h" /* lock-line lock + PPU-write coherence bitmap */
#include "spu_helpers.h"   /* spu_splat_u32 / spu_ls_read128 (SMC microstep) */
#include "spu_lockstep.h"
#include "spu_interp.h"    /* spu_lifted_fn / spu_lifted_lookup -- defined below */
#include "ps3emu/ps3_ftrace.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
/* Darwin's setjmp saves the signal mask: a sigprocmask syscall per SPU run, ~7% of
 * each EDGE SPU worker in GoW:A's first level (the macOS side of the Windows
 * unwinding cost spu_jmp_no_unwind removes). _setjmp/_longjmp leave the mask alone,
 * which is all these jumps need: they never cross a signal handler. */
#if defined(__APPLE__)
#  define SPU_SETJMP(e)     _setjmp(e)
#  define SPU_LONGJMP(e, v) _longjmp((e), (v))
#else
#  define SPU_SETJMP(e)     setjmp(e)
#  define SPU_LONGJMP(e, v) longjmp((e), (v))
#endif
#ifndef _WIN32
#include <sched.h>   /* sched_yield */
#include "../platform/win32_backtrace.h"   /* GetModuleHandleA ([decpc]) */
#else
#include <intrin.h>  /* __rdtsc (spu_host_ns) */
#endif
#include <time.h>
#include "../platform/win32_compat.h"
#include "ps3emu/log_first.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The SPU decrementer ticks at the PS3 timebase, 79.8 MHz -- the same clock
 * sys_time_get_timebase_frequency reports to the PPU. Titles calibrate real
 * delays against it, so the rate has to be right, not merely non-zero. */
#define SPU_DECREMENTER_HZ  79800000ull

static uint64_t spu_host_ns_qpc(void);

/* Host nanoseconds for the decrementer, read on every `rdch $ch8`. Titles spin on
 * that channel as a delay: GoW:A's EDGE geometry job backs off 2400 reads between
 * polls of the RSX ring label, ~2 million reads a second. Through QPC plus the
 * split divides each read cost ~60 ns, so one backoff took ~150 us against a few
 * on hardware, and that was 15% of every EDGE kernel thread. The TSC (invariant
 * on every x86-64 host this runs on) is scaled to ns with a 32.32 multiplier
 * calibrated against QPC over the first 50 ms; until then QPC answers. */
static uint64_t spu_host_ns(void)
{
#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
    /* SPU_HOST_NS=qpc: QPC on every read, as before the TSC path. */
    static volatile int s_mode = -1;
    if (s_mode < 0) { const char* e = getenv("SPU_HOST_NS"); s_mode = (e && e[0] == 'q') ? 1 : 0; }
    if (s_mode) return spu_host_ns_qpc();
    static volatile uint64_t s_mul;          /* ns per TSC tick, 32.32 */
    static volatile uint64_t s_tsc0, s_ns0;
    const uint64_t mul = s_mul;
    if (mul) {
        const unsigned __int128 d = (unsigned __int128)(__rdtsc() - s_tsc0) * mul;
        /* Measured against QPC once a minute over 18 min: it gains ~45 us a
         * minute (0.75 ppm). */
        return s_ns0 + (uint64_t)(d >> 32);
    }
    static volatile LONG s_cal;              /* 0 none, 1 started */
    static uint64_t s_c_tsc, s_c_ns;
    const uint64_t ns = spu_host_ns_qpc();
    const uint64_t t = __rdtsc();
    if (InterlockedCompareExchange(&s_cal, 1, 0) == 0) { s_c_tsc = t; s_c_ns = ns; }
    else if (s_c_ns && ns - s_c_ns >= 50000000ull && t > s_c_tsc) {
        static volatile LONG s_done;
        if (InterlockedCompareExchange(&s_done, 1, 0) == 0) {
            s_tsc0 = t; s_ns0 = ns;
            s_mul = (uint64_t)((double)(ns - s_c_ns) * 4294967296.0 / (double)(t - s_c_tsc));
        }
    }
    return ns;
#elif defined(__aarch64__)
    /* arm64: the generic timer, readable from EL0 and common to every core.
     * Measured on an M4 Max: 0.6 ns a read against 16.4 ns for clock_gettime,
     * the arm64 side of the TSC path above. 32.32 scale, 128-bit product. */
    static uint64_t s_mul;
    if (!s_mul) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); s_mul = (1000000000ull << 32) / f; }
    uint64_t c; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(c));
    return (uint64_t)(((unsigned __int128)c * s_mul) >> 32);
#else
    return spu_host_ns_qpc();
#endif
}

static uint64_t spu_host_ns_qpc(void)
{
#ifdef _WIN32
    static LARGE_INTEGER s_freq;
    LARGE_INTEGER c;
    if (!s_freq.QuadPart) QueryPerformanceFrequency(&s_freq);
    QueryPerformanceCounter(&c);
    /* Split the divide to keep the multiply from overflowing: QPC counters are
     * large enough that (count * 1e9) wraps a u64 within hours of uptime. */
    return (uint64_t)(c.QuadPart / s_freq.QuadPart) * 1000000000ull
         + (uint64_t)(c.QuadPart % s_freq.QuadPart) * 1000000000ull
           / (uint64_t)s_freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

/* ---------------------------------------------------------------------------
 * Clean SPU job abort (longjmp). The `br .` halt idiom and other terminal
 * spins can't be escaped by setting a status flag (lifted code never checks
 * it), so spu_halt() longjmps back to spu_run_with_halt() in the dispatcher.
 * -----------------------------------------------------------------------*/
#if defined(_MSC_VER)
#  define SPU_TLS __declspec(thread)
#else
#  define SPU_TLS __thread
#endif
/* Called after an SPU atomic commits a 128-byte line, with the lock-line lock
 * released. cellSpurs sets it so a PPU blocked in cellSpursEventFlagWait wakes
 * when a task sets the flag, instead of on its next 2 ms poll. */
void (*g_spu_line_commit_hook)(uint32_t line) = 0;
/* Title hook (port glue), separate from the slot cellSpurs owns: called after a
 * committed PUTLLC/PUTLLUC with the committing SPU context, outside the lock. */
void (*g_spu_line_commit_title_hook)(uint32_t line, spu_context* ctx) = 0;
/* spu_workload.c: wakes tasks parked in WAIT_SIGNAL on that line's taskset. */
void spu_taskset_line_committed(uint32_t line);
/* SPU_LS_WATCH=0x1BE80[,0x927D80...]: watch up to 4 16-byte LS lines (reads
 * and writes) in every SPU image. See spu_ls_watch_hit2 for the fast path. */
#define SPU_WATCH_MAX 4
int g_spu_ls_watch_n = -1;
static unsigned s_spu_ls_watch[SPU_WATCH_MAX];
unsigned* spu_ls_watch_list(int* out_n)
{
    if (g_spu_ls_watch_n < 0) {
        int n = 0; const char* e = getenv("SPU_LS_WATCH");
        while (e && *e && n < SPU_WATCH_MAX) {
            s_spu_ls_watch[n++] = (unsigned)strtoul(e, (char**)&e, 0) & ~0xFu;
            while (*e == ',' || *e == ' ') e++;
        }
        g_spu_ls_watch_n = n;
    }
    *out_n = g_spu_ls_watch_n;
    return s_spu_ls_watch;
}
void spu_ls_watch_slow(uint32_t lsa, int is_write, const uint8_t* p, uint32_t pc, uint32_t lr)
{
    int n; spu_ls_watch_list(&n);
    if (!n) return;
    uint32_t a = lsa & (SPU_LS_MASK & ~0xFu);
    for (int i = 0; i < g_spu_ls_watch_n; i++) {
        if (s_spu_ls_watch[i] == a) {
            fprintf(stderr, "[spu-watch %s 0x%05X pc=0x%05X lr=0x%05X] "
                "%02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                is_write ? "WR" : "rd", a, pc, lr,
                p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7],
                p[8],p[9],p[10],p[11], p[12],p[13],p[14],p[15]);
            fflush(stderr);
            break;
        }
    }
}

static SPU_TLS jmp_buf s_spu_halt_env;
static SPU_TLS int     s_spu_halt_armed = 0;

/* Call right after a setjmp returned 0. On 64-bit Windows longjmp unwinds every
 * host frame between it and its setjmp (RtlUnwindEx, a function-table lookup
 * and a virtual unwind per frame), several microseconds per jump; with the
 * buffer's Frame zeroed it restores the registers directly. These jumps --
 * a halt back to spu_run_with_halt, an iret back to the drain loop that took
 * the interrupt -- only cross C frames of this runtime and of lifted code,
 * which have no destructors or __finally blocks to run. In God of War:
 * Ascension the unwinding was ~5% of its EDGE SPU time. SPU_LONGJMP_UNWIND=1
 * keeps the unwinding jump. */
void spu_jmp_no_unwind(void* env)
{
#if defined(_WIN64) && defined(_MSC_VER)
    static int s_unwind = -1;
    if (s_unwind < 0) s_unwind = getenv("SPU_LONGJMP_UNWIND") ? 1 : 0;
    if (!s_unwind) ((_JUMP_BUFFER*)env)->Frame = 0;
#else
    (void)env;
#endif
}

/* A non-local return replaces the old host call chain without changing
 * any architectural registers, local store, or pending guest work. */
void spu_restart_dispatch(spu_context* ctx)
{
    g_spu_trampoline_fn = 0;
    if (s_spu_halt_armed) SPU_LONGJMP(s_spu_halt_env, 2);
    fprintf(stderr, "[spu] non-local return outside execution driver at 0x%05X\n", ctx->pc);
    ctx->status = SPU_STATUS_STOPPED_BY_HALT;
}

/* SPU->PPU outbound-mailbox delivery hook. The SPU writing WrOutMbox /
 * WrOutIntrMbox must wake PPU code blocked on the SPURS event queue bound to
 * the SPU thread group (e.g. cellSpursInitialize). lv2_register.c installs a
 * handler that maps spu_group_id -> connected event queue and pushes an event.
 * NULL until installed (plain SPU jobs with no PPU listener stay a no-op). */
int (*g_spu_user_event_hook)(spu_context*, uint32_t) = NULL;
void (*g_spu_out_mbox_hook)(uint32_t group_id, uint32_t spu_id,
                            int is_intr, uint32_t value) = 0;

/* Every MFC PUT an SPU completes, if anyone is listening. Installed by the GCM
 * layer (libs/video/cellGcmSys.c), which needs it to find the RSX pushbuffer:
 * a title whose SPUs build the command stream never tells libgcm where that
 * stream is, so the only evidence is the DMA that writes it. NULL until
 * installed, so a title with no SPU-built FIFO pays nothing. */
void (*g_spu_put_hook)(uint32_t ea, uint32_t size) = 0;

/* ---------------------------------------------------------------------------
 * SPU job capture: SPU_JOB_CAPTURE=<dir>
 *
 * Records single SPURS policy-module jobs with everything they read from
 * outside their local store, so a job can later be re-run in isolation and its
 * output compared (docs/EDGE_NATIVE.md, "Capture"). Off unless the env is set;
 * when off, the hot paths pay one load of g_spu_jobcap_on and a predictable
 * branch.
 *
 *   SPU_JOB_CAPTURE_WIDS=<w>[,<w>...]  workloads to capture (default 15,16,17)
 *   SPU_JOB_CAPTURE_N=<n>              jobs in total, all threads (default 100)
 *   SPU_JOB_CAPTURE_T0=<s>             start after <s> seconds, counted from the
 *                                      first SPU dispatch (default 0)
 *
 * Job start: the dispatch where g_spu_job_entries counts one (a policy context
 * branching from its own code, link 0xA00..0x3FFF, into staged job code at pc
 * >= 0x4000), taken right before the lifted entry runs, so resident_ovl is the
 * overlay that dispatch resolved.
 * Job end, first of:
 *   - the job's outermost `bi $r0` back to the link recorded at start: either
 *     the entry bracket's spu_drain_call returning there at the bracket's
 *     host_depth (spu_job_drain_return; the usual way, since the PM enters
 *     with a lifted `bisle` bracket) or a dispatch to pc == link. Dispatches
 *     into other PM addresses during the job are JobApi calls (GetBuffer etc.
 *     at 0x2F30/0x1700/...) and stay inside the capture; ending at the first
 *     pc < 0x4000 cut every job at its first JobApi call;
 *   - spu_halt on the context (module exit to the kernel, or any halt);
 *   - the next job start on the same host thread (the return was a host-frame
 *     return and never reached the dispatcher).
 * Measured on GoW:A EDGE (wids 16/17, overlay 225, entry 0xAD98, link 0x3368):
 * the job never DISPATCHES to its link; it returns there through the bracket's
 * drain. Version 1 captures missed that and ran on through the PM's tail (the
 * dispatch to 0x2F5C, the completion GETLLAR/PUTLLC at 0x35E0, a poll, the
 * exit to the kernel at 0x9C0). That tail runs in two PM host frames above
 * the bracket (host_depth 3 at job start); a replay would have to synthesize
 * them with their return pcs, which it does not, so the job proper ends at
 * the return.
 * Capture state is per host thread: the SPURS kernel threads run concurrently,
 * each with at most one open capture, and every hook runs on the thread that
 * runs the job.
 *
 * File <dir>/job_<wid>_<seq>.bin (seq is global, 1..N), little-endian. The
 * structs are in spu_jobcap.h (spu_jobcap_header / _state / _event).
 *   header (header_size bytes):
 *      0  char[8]  magic "SPUJCAP\0"     8  u32 version (2; 1 lacks the state)
 *     12  u32 header_size               16  u32 ls_size (0x40000)
 *     20  u32 gpr_count (128)           24  u32 wid (LS[0x1DF])
 *     28  i32 image_id                  32  i32 resident_ovl
 *     36  i32 resident_task             40  u32 spu_id
 *     44  i32 policy_mode               48  u32 entry pc
 *     52  u32 link (r0 at entry)        56  u32 seq
 *     60  u32 ms since capture init     64  u32 code_slots (S)
 *     68  S x {u32 lsa, u32 size, u32 source_ea, i32 image_id}  resident_code[]
 *     68+16S  spu_jobcap_state (version 2): the context fields lifted code and
 *        the drain loop read without a channel read -- int_enable, event
 *        status/mask, srr0, the MFC staging registers, tag mask/status and the
 *        engine's tag_completed, the reservation, the parked DMA lists, a saved
 *        interrupt register file, host_depth (informational)
 *   LS snapshot at job start: ls_size bytes, guest (big-endian) bytes
 *   GPRs at job start: gpr_count x 16 bytes, the u128 as the context stores it
 *     (four host-order u32, _u32[0] = SPU word 0, the preferred slot)
 *   events, until job end, each a 32-byte record + payload:
 *      u32 type, u32 payload_len, u32 pc (ctx->pc, may lag inside lifted code),
 *      u32 a, u32 b, u32 c, u32 d, u32 index (0-based event number)
 *     1 GET      a=cmd b=ea c=lsa d=size, payload = the bytes read from guest
 *                memory at transfer time (every DMA-list element is its own
 *                event, with the list's base command)
 *     2 PUT      a=cmd b=ea c=lsa d=size, payload = the bytes written (SDCRZ is
 *                a PUT of zeros with c=0)
 *     3 GETLLAR  a=cmd b=ea c=lsa, payload = the 128-byte line
 *     4 PUTLLC   a=cmd b=ea c=lsa d=1 committed / 0 failed, payload = the line
 *                the SPU offered
 *     5 PUTLLUC  a=cmd b=ea c=lsa d=1, payload = the line written
 *     6 RDCH     a=channel, payload = the 16-byte u128 returned (same form as
 *                the GPRs)
 *     7 POLL     a=r3 answer of the HLE selectWorkload service, payload = LS
 *                0x100..0x17F after the service refreshed it
 *     8 PMCALL   a=target pc (< 0x4000) b=link: a dispatch into PM code while
 *                the job runs (JobApi), no payload
 *     9 IRQ      (v2) an interrupt taken (spu_drain.c spu_take_interrupt):
 *                a=srr0 (the interrupted pc) b=event_status c=event_mask
 *                d=trampoline hops since the previous event, no payload. A
 *                peer SPU or the PPU raising Lr sets event_status at an
 *                arbitrary moment; the hop count is where replay re-raises it
 *     10 RCHCNT  (v2) a=channel b=count returned c=event_status after it
 *                (rchcnt SPU_RdEventStat polls the reservation), no payload
 *    15 END      a=reason (1 return to link, 2 halt, 3 next job start)
 *                b=pc c=r0 d=resident_ovl, payload = GPRs then LS at job end
 * Not recorded: PPU stores landing in guest memory between the job's own
 * transfers (only what the job read is recorded).
 * Captures hold game data: keep them under work/, never commit them.
 * -------------------------------------------------------------------------*/
int g_spu_jobcap_on_v = -1;        /* -1 env not read yet, 0 off, 1 on */
int g_spu_jobhop_on_v = 0;
int g_spu_jobret_on_v = 0;
static char s_jc_dir[260];
static uint32_t s_jc_wids = (1u << 15) | (1u << 16) | (1u << 17);
static LONG s_jc_max = 100;
static unsigned long long s_jc_t0_ms, s_jc_init_ms;
static volatile LONG s_jc_seq;
static volatile LONG s_jc_full;
static LONG s_jc_every = 1;      /* SPU_JOB_CAPTURE_EVERY: keep one job in k */
static int s_jc_img = 0;         /* SPU_JOB_CAPTURE_IMG: only jobs whose resident module image is this one */
static volatile LONG s_jc_cand;

typedef struct {
    spu_context* ctx;            /* NULL = no capture open on this thread */
    FILE*        f;
    uint32_t     link, wid, seq, nev;
    uint32_t     hops;           /* trampoline hops since the last event (IRQ records) */
    uint32_t     depth;          /* host_depth at job start: the entry bracket's */
    uint32_t     ntype[16];
} spu_jobcap;
static SPU_TLS spu_jobcap t_jc;

static void spu_jobcap_init(void)
{
    static SRWLOCK lk = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lk);
    if (g_spu_jobcap_on_v < 0) {
        const char* d = getenv("SPU_JOB_CAPTURE");
        if (d && *d) {
            snprintf(s_jc_dir, sizeof s_jc_dir, "%s", d);
            CreateDirectoryA(s_jc_dir, NULL);   /* an existing dir is fine */
            const char* w = getenv("SPU_JOB_CAPTURE_WIDS");
            if (w && *w) {
                s_jc_wids = 0;
                while (*w) {
                    char* p; const long v = strtol(w, &p, 10);
                    if (p == w) break;
                    if (v >= 0 && v < 32) s_jc_wids |= 1u << v;
                    if (*p != ',') break;
                    w = p + 1;
                }
            }
            const char* n = getenv("SPU_JOB_CAPTURE_N");
            if (n && *n) s_jc_max = atol(n);
            const char* ev = getenv("SPU_JOB_CAPTURE_EVERY");
            if (ev && atol(ev) > 1) s_jc_every = atol(ev);
            const char* im = getenv("SPU_JOB_CAPTURE_IMG");
            if (im && *im) s_jc_img = atoi(im);
            const char* t = getenv("SPU_JOB_CAPTURE_T0");
            s_jc_t0_ms = t && *t ? (unsigned long long)atoll(t) * 1000ull : 0;
            s_jc_init_ms = GetTickCount64();
            fprintf(stderr, "[jobcap] capturing %ld jobs (1 in %ld) of wids mask 0x%08X after %llu s -> %s\n",
                    (long)s_jc_max, (long)s_jc_every, s_jc_wids, s_jc_t0_ms / 1000ull, s_jc_dir);
            /* Statics first: other threads test the flag without the lock. */
            MemoryBarrier();
            g_spu_jobhop_on_v = 1;
            g_spu_jobret_on_v = 1;
            g_spu_jobcap_on_v = 1;
        } else {
            g_spu_jobcap_on_v = 0;
        }
    }
    ReleaseSRWLockExclusive(&lk);
}

static void jc_ev(uint32_t type, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                  const void* p, uint32_t n)
{
    const uint32_t h[8] = { type, n, (uint32_t)t_jc.ctx->pc & SPU_LS_MASK, a, b, c, d, t_jc.nev };
    fwrite(h, 4, 8, t_jc.f);
    if (n) fwrite(p, 1, n, t_jc.f);
    t_jc.nev++;
    t_jc.hops = 0;
    t_jc.ntype[type & 15]++;
}

static void jc_close(uint32_t reason)
{
    spu_context* ctx = t_jc.ctx;
    const uint32_t h[8] = { JC_END, (uint32_t)(sizeof ctx->gpr + SPU_LS_SIZE),
                            (uint32_t)ctx->pc & SPU_LS_MASK, reason,
                            (uint32_t)ctx->pc & SPU_LS_MASK, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                            (uint32_t)ctx->resident_ovl, t_jc.nev };
    fwrite(h, 4, 8, t_jc.f);
    fwrite(ctx->gpr, 1, sizeof ctx->gpr, t_jc.f);
    fwrite(ctx->ls, 1, SPU_LS_SIZE, t_jc.f);
    fclose(t_jc.f);
    fprintf(stderr, "[jobcap] job_%u_%u.bin end=%u pc=0x%05X ev=%u get=%u put=%u llar=%u "
            "putllc=%u putlluc=%u rdch=%u rchcnt=%u poll=%u pmcall=%u irq=%u\n",
            t_jc.wid, t_jc.seq, reason, (uint32_t)ctx->pc & SPU_LS_MASK, t_jc.nev,
            t_jc.ntype[JC_GET], t_jc.ntype[JC_PUT], t_jc.ntype[JC_GETLLAR],
            t_jc.ntype[JC_PUTLLC], t_jc.ntype[JC_PUTLLUC], t_jc.ntype[JC_RDCH],
            t_jc.ntype[JC_RCHCNT], t_jc.ntype[JC_POLL], t_jc.ntype[JC_PMCALL],
            t_jc.ntype[JC_IRQ]);
    memset(&t_jc, 0, sizeof t_jc);
}

/* At a job start (see above). Ends any capture still open on this thread. */
static void spu_jobcap_open(spu_context* ctx, uint32_t link)
{
    if (t_jc.ctx) jc_close(JC_END_NEXTJOB);
    const uint32_t wid = ctx->ls[0x1DF];
    if (s_jc_full || wid >= 32 || !((s_jc_wids >> wid) & 1u)) return;
    const unsigned long long ms = GetTickCount64() - s_jc_init_ms;
    if (ms < s_jc_t0_ms) return;
    /* One job module only (a workload runs several, and a stride can alias onto one kind). */
    if (s_jc_img && ctx->resident_ovl != s_jc_img) return;
    /* A stride spreads the sample over many frames and job kinds; without it
     * N jobs come from one burst of one workload. */
    if (s_jc_every > 1 && (InterlockedIncrement(&s_jc_cand) - 1) % s_jc_every) return;
    const LONG seq = InterlockedIncrement(&s_jc_seq);
    if (seq > s_jc_max) {
        if (!s_jc_full) fprintf(stderr, "[jobcap] %ld jobs captured, done\n", (long)s_jc_max);
        s_jc_full = 1;
        return;
    }
    char path[300];
    snprintf(path, sizeof path, "%s/job_%u_%ld.bin", s_jc_dir, wid, (long)seq);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[jobcap] cannot create %s\n", path); return; }
    setvbuf(f, NULL, _IOFBF, 1u << 20);
    static SPU_TLS spu_jobcap_header hd;   /* ~2.7 KB: off the lifted code's stack */
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, "SPUJCAP", 8);
    hd.version = SPU_JOBCAP_VERSION; hd.header_size = (uint32_t)sizeof hd; hd.ls_size = SPU_LS_SIZE;
    hd.gpr_count = 128; hd.wid = wid;
    hd.image_id = ctx->image_id; hd.resident_ovl = ctx->resident_ovl;
    hd.resident_task = ctx->resident_task; hd.spu_id = ctx->spu_id;
    hd.policy_mode = ctx->policy_mode; hd.entry_pc = (uint32_t)ctx->pc & SPU_LS_MASK;
    hd.link = link; hd.seq = (uint32_t)seq; hd.ms = (uint32_t)ms;
    hd.code_slots = SPU_RESIDENT_CODE_SLOTS;
    for (unsigned s = 0; s < SPU_RESIDENT_CODE_SLOTS; ++s) {
        hd.code[s].lsa = ctx->resident_code[s].lsa;
        hd.code[s].size = ctx->resident_code[s].size;
        hd.code[s].source_ea = ctx->resident_code[s].source_ea;
        hd.code[s].image_id = ctx->resident_code[s].image_id;
    }
    spu_jobcap_state* st = &hd.st;
    st->int_enable = ctx->int_enable; st->event_status = ctx->event_status;
    st->event_mask = ctx->event_mask; st->srr0 = ctx->srr0;
    st->decrementer = ctx->decrementer; st->status = ctx->status;
    st->stop_code = ctx->stop_code; st->host_depth = ctx->host_depth;
    st->mfc_lsa = ctx->mfc_lsa; st->mfc_eah = ctx->mfc_eah; st->mfc_eal = ctx->mfc_eal;
    st->mfc_size = ctx->mfc_size; st->mfc_tag = ctx->mfc_tag;
    st->mfc_tag_mask = ctx->mfc_tag_mask; st->mfc_tag_status = ctx->mfc_tag_status;
    spu_jobcap_mfc_state(ctx, &st->mfc_tag_completed, &st->mfc_queue_count, 0);
    st->resv_ea = ctx->resv_ea; st->resv_valid = (uint32_t)ctx->resv_valid;
    st->atomic_stat = ctx->atomic_stat;
    st->list_stall_mask = ctx->list_stall_mask; st->list_stall_stat = ctx->list_stall_stat;
    st->irq_saved = (uint32_t)ctx->irq_saved; st->irq_resume_pc = ctx->irq_resume_pc;
    st->module_img_a00 = ctx->module_img_a00; st->steps = ctx->steps;
    for (unsigned t = 0; t < 32; ++t) {
        st->list_stall_elem_lsa[t] = ctx->list_stall_elem_lsa[t];
        st->list_stall_remaining[t] = ctx->list_stall_remaining[t];
        st->list_stall_dest_lsa[t] = ctx->list_stall_dest_lsa[t];
        st->list_stall_cmd[t] = ctx->list_stall_cmd[t];
        st->list_stall_ea_base[t][0] = (uint32_t)ctx->list_stall_ea_base[t];
        st->list_stall_ea_base[t][1] = (uint32_t)(ctx->list_stall_ea_base[t] >> 32);
    }
    memcpy(st->resv_line, ctx->resv_line, sizeof st->resv_line);
    memcpy(st->irq_gpr, ctx->irq_gpr, sizeof st->irq_gpr);
    fwrite(&hd, 1, sizeof hd, f);
    fwrite(ctx->ls, 1, SPU_LS_SIZE, f);
    fwrite(ctx->gpr, 1, sizeof ctx->gpr, f);
    memset(&t_jc, 0, sizeof t_jc);
    t_jc.ctx = ctx; t_jc.f = f; t_jc.link = link; t_jc.wid = wid; t_jc.seq = (uint32_t)seq;
    t_jc.depth = ctx->host_depth;
}

/* Every completed MFC transfer (spu_dma.h mfc_do_transfer), when capture is on. */
void spu_jobcap_xfer(spu_context* spu, uint32_t cmd, uint64_t ea, uint32_t lsa,
                     uint32_t size, const void* bytes)
{
    if (t_jc.ctx != spu) return;
    const int put = (cmd & 0xFFFFu) == MFC_SDCRZ_CMD || !mfc_is_get(cmd);
    jc_ev(put ? JC_PUT : JC_GET, cmd, (uint32_t)ea, lsa, size, bytes, size);
}

/* Every trampoline hop (spu_drain.c spu_hop) while g_spu_jobhop_on: an IRQ
 * record says how many hops after the previous event the interrupt was taken. */
void spu_job_hop(spu_context* ctx)
{
    if (t_jc.ctx == ctx) t_jc.hops++;
    if (g_spu_jobrep_on > 0) spu_jobrep_hop(ctx);
}

/* An interrupt taken (spu_drain.c spu_take_interrupt, after srr0 is set). */
void spu_job_irq(spu_context* ctx)
{
    if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
        jc_ev(JC_IRQ, ctx->srr0 & SPU_LS_MASK, ctx->event_status, ctx->event_mask,
              t_jc.hops, NULL, 0);
    if (g_spu_jobrep_on > 0) spu_jobrep_irq(ctx);
}

/* The PM enters a job through a lifted `bisle` bracket (host_depth++,
 * dispatch, spu_drain_call(link)), so the job's outermost `bi $r0` usually
 * comes back as that drain returning at the link, never as a dispatch to it.
 * Same bracket depth as at job start: the job has returned. */
/* SPURS_PM_PROF job timer: from the job start to the entry bracket's drain
 * return, the job proper without the PM's own work around it. */
SPU_THREAD_LOCAL long long g_spu_jobt_ticks;
SPU_THREAD_LOCAL unsigned g_spu_jobt_done;
SPU_THREAD_LOCAL unsigned g_spu_jobt_hist[SPU_JOBT_BUCKETS];
static SPU_TLS struct { spu_context* ctx; uint32_t link, depth; long long t0; } t_jt;
static void spu_jobt_start(spu_context* ctx, uint32_t link)
{
#if 1   /* SPURS_PM_PROF timers: portable via win32_compat */
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    t_jt.ctx = ctx; t_jt.link = link; t_jt.depth = ctx->host_depth; t_jt.t0 = t.QuadPart;
    g_spu_jobret_on_v = 1;
#else
    (void)ctx; (void)link;
#endif
}

void spu_job_drain_return(spu_context* ctx, uint32_t return_pc)
{
#if 1   /* SPURS_PM_PROF timers: portable via win32_compat */
    if (t_jt.ctx == ctx && (return_pc & SPU_LS_MASK) == t_jt.link && ctx->host_depth == t_jt.depth) {
        static long long s_qf;
        if (!s_qf) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_qf = f.QuadPart; }
        LARGE_INTEGER t; QueryPerformanceCounter(&t);
        const long long dt = t.QuadPart - t_jt.t0;
        long long b = dt * 100000 / s_qf;   /* 10 us buckets */
        if (b >= SPU_JOBT_BUCKETS) b = SPU_JOBT_BUCKETS - 1;
        g_spu_jobt_ticks += dt; g_spu_jobt_done++; g_spu_jobt_hist[b]++;
        t_jt.ctx = NULL;
    }
#endif
    if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx && (return_pc & SPU_LS_MASK) == t_jc.link &&
        ctx->host_depth == t_jc.depth)
        jc_close(JC_END_RETURN);
}

void spu_halt(spu_context* ctx)
{
    if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx) jc_close(JC_END_HALT);
    if (g_spu_jobrep_on > 0) spu_jobrep_halt(ctx);
    /* Register/context dump at halt-asserts: the WWS job's parameter guard
     * (heqi 0x1650 in job_wws_7BD900) checks (r12 bit0) & (r9 16-aligned) &
     * (r15 != 0) -- job descriptor/ABI values our execute path must supply.
     * Dump the low registers + the job's parameter area so a firing assert
     * names exactly which input is wrong. */
    { static int _n = 0;
      if (_n < 6 && ctx->status == SPU_STATUS_STOPPED_BY_HALT) { _n++;
        fprintf(stderr, "[spu-halt] img=%d pc=0x%05X regs:", ctx->image_id,
                ctx->pc & SPU_LS_MASK);
        for (int r = 3; r <= 16; r++)
            fprintf(stderr, " r%d=%08X", r, ctx->gpr[r]._u32[0]);
        fprintf(stderr, " r80=%08X r85=%08X\n",
                ctx->gpr[80]._u32[0], ctx->gpr[85]._u32[0]);
        fflush(stderr);
      } }
    if (s_spu_halt_armed) { s_spu_halt_armed = 0; SPU_LONGJMP(s_spu_halt_env, 1); }
}

/* Diagnostic: dump the taskset-policy scheduler's working tables (LS 0x2700..)
 * at func_00000E60 entry, to reverse why it computes "no runnable task". Env
 * YDKJ_E60. Called from the lifted taskset policy. */
void spu_dbg_e60(spu_context* ctx)
{
    static int s_e = -1; if (s_e < 0) s_e = getenv("YDKJ_E60") ? 1 : 0;
    if (!s_e) return;
    static int _d = 0; if (!PS3_LOG_FIRST(_d, 6)) return;
    const uint8_t* L = ctx->ls;
    #define RD(o) (((uint32_t)L[(o)]<<24)|((uint32_t)L[(o)+1]<<16)|((uint32_t)L[(o)+2]<<8)|L[(o)+3])
    fprintf(stderr, "[E60] r20=%08X r24=%08X | run2700=%08X ready2710=%08X pend2720=%08X en2730=%08X sig2740=%08X wait2750=%08X x2770=%08X\n",
            ctx->gpr[20]._u32[0], ctx->gpr[24]._u32[0],
            RD(0x2700), RD(0x2710), RD(0x2720), RD(0x2730), RD(0x2740), RD(0x2750), RD(0x2770));
    #undef RD
    fflush(stderr);
}

/* SPU `stop` / stop-and-signal. A real SPU stop HALTS the core; the previous
 * lifted emission only did `status=...; return;`, which unwound ONE frame and
 * let the caller's service loop keep running -- so a SPURS policy doing
 * stop-and-signal in a loop spun millions of times instead of yielding. Halt the
 * host SPU thread (longjmp to spu_run_with_halt) so the job stops AT the stop,
 * with ctx->status + the outbound mailbox value available for the dispatcher to
 * service. Env YDKJ_STOP_NOHALT restores the old (looping) behavior for A/B. */
void spu_stop(spu_context* ctx)
{
    /* A lifted `stop` sets status and RETURNS to its caller -- for a SPURS
     * policy this is stop-and-signal: it returns up into the kernel/policy
     * service loop, which continues (i.e. the SPU "resumes" past the stop). That
     * resume-by-return behavior is correct; the previous spin was caused by the
     * lack of (a) a PPU listener for the outbound mailbox and (b) mailbox
     * backpressure -- both handled in the channel write path, not here. So by
     * default DO NOT halt. Env YDKJ_STOP_HALT forces a hard halt (longjmp) for
     * A/B experiments (it makes the kernel terminate, which the game restarts). */
    static int s_halt = -1;
    if (s_halt < 0) s_halt = getenv("YDKJ_STOP_HALT") ? 1 : 0;
    ctx->status = SPU_STATUS_STOPPED_BY_STOP;
    if (s_halt) spu_halt(ctx);
}

/* Run a lifted SPU entry with a halt landing pad. Returns 1 if the job halted
 * (via spu_halt), 0 if it returned normally. */
/* SPU_FP_NATIVE (spu_helpers.h): lifted single-precision ops are native
 * instructions that need round-toward-zero, the SPU's rounding, so every run is
 * made under it and the caller's mode is put back on the way out. A halt's
 * longjmp restores the MXCSR saved by the setjmp below, which is already RZ.
 * SPU_FP_CHECK=1: compare every such op with the exact path. */
int g_spu_fp_check = -1;
void spu_fp_check(int op, u128 a, u128 b, u128 c, u128 r)
{
    u128 e;
    switch (op) {
    case 0:  e = spu_fa_ref(a, b); break;
    case 1:  e = spu_fs_ref(a, b); break;
    case 2:  e = spu_fm_ref(a, b); break;
    case 3:  e = spu_fma_ref(a, b, c); break;
    case 4:  e = spu_fms_ref(a, b, c); break;
    default: e = spu_fnms_ref(a, b, c); break;
    }
    static volatile LONG64 s_n, s_bad;
    const LONG64 n = InterlockedIncrement64(&s_n);
    if (memcmp(&e, &r, sizeof e)) {
        const LONG64 k = InterlockedIncrement64(&s_bad);
#if defined(__aarch64__)
        uint64_t fp_ctl; __asm__ volatile("mrs %0, fpcr" : "=r"(fp_ctl));   /* arm64's MXCSR */
#else
        const unsigned fp_ctl = _mm_getcsr();
#endif
        if (k <= 40)
            fprintf(stderr, "[spu-fp] op %d mismatch (mxcsr 0x%04X): a %08X %08X %08X %08X b %08X %08X %08X %08X"
                    " c %08X %08X %08X %08X got %08X %08X %08X %08X want %08X %08X %08X %08X\n",
                    op, (unsigned)fp_ctl, a._u32[0], a._u32[1], a._u32[2], a._u32[3], b._u32[0], b._u32[1], b._u32[2],
                    b._u32[3], c._u32[0], c._u32[1], c._u32[2], c._u32[3], r._u32[0], r._u32[1], r._u32[2],
                    r._u32[3], e._u32[0], e._u32[1], e._u32[2], e._u32[3]);
    }
    if ((n & 0x3FFFFFF) == 0)
        fprintf(stderr, "[spu-fp] %lld ops checked, %lld mismatches\n", (long long)n, (long long)s_bad);
}

int spu_run_with_halt(void (*entry)(spu_context*), spu_context* ctx)
{
    int halted = 0;
    if (g_spu_fp_check < 0) { const char* e = getenv("SPU_FP_CHECK"); g_spu_fp_check = e && *e == '1'; }
#if defined(SPU_FP_NATIVE) && defined(__aarch64__)
    /* arm64: FPCR at round toward zero with flush-to-zero, the MXCSR mode below
     * (spu_helpers.h, SPU_FPCR_RZFZ). */
    const uint64_t fp_csr = spu__fpcr_get();
    spu__fpcr_set(fp_csr | SPU_FPCR_RZFZ);
#elif defined(SPU_FP_NATIVE)
    const unsigned fp_csr = _mm_getcsr();
#if SPU_FP_DAZ
    _mm_setcsr(fp_csr | 0x6000u | 0x8040u);         /* RC = 11: toward zero; FTZ, DAZ (spu_helpers.h) */
#else
    _mm_setcsr(fp_csr | 0x6000u);                   /* RC = 11: toward zero */
#endif
#endif
    /* Job-args probe: image-1 (the WWS job binary) entries arrive at the crt
     * with r3/r4 zeroed in SOME runs while the jm2 runner sets them -- count
     * and stamp every image-1 run here to find the argless caller. */
    if (ctx->image_id == 1) {
        static int _n = 0;
        if (_n < 12) { _n++;
            fprintf(stderr, "[rwh] #%d img=1 ctx=%p r3=%08X r4=%08X entry=%p\n",
                    _n, (void*)ctx, ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0],
                    (void*)entry);
            fflush(stderr); }
    }
    /* Also discard any saved register file keyed to this context pointer:
     * contexts are stack locals, so the address recurs and a stale save from
     * an earlier run would be restored over this job's arguments. */
    { extern void spu_irq_regs_forget(spu_context*); spu_irq_regs_forget(ctx); }
    ctx->steps = 0;   /* fresh run: step 0 is the entry */
    if (g_spu_direct < 0) spu_direct_init();   /* PS3_SPU_DIRECT, spu_context.h */
    /* The host stack is empty here by definition, so no lifted caller frame
     * is live. A persistent context (a SPURS policy module re-entered every
     * scheduling quantum) otherwise carries the depth its last quantum
     * halted at: two frames per quantum for an idle job manager, and the
     * recursion guard halted the SPU at 2000 after a few minutes of idling. */
    ctx->host_depth = 0;
    ctx->irq_frame = 0;
    ctx->drain_ret_pc = 0;
    /* Nest-safe: a job run synchronously from inside another SPU's execution
     * re-enters here on the same host thread. The halt target is per-thread,
     * so save the outer one and put it back on the way out -- otherwise the
     * outer run's next halt longjmps into this (by then dead) frame, and a
     * stack-local job context skips its coherency unregister (GH3: crashes in
     * spu_coh_notify_write on a freed stack). */
    jmp_buf outer_env; const int outer_armed = s_spu_halt_armed;
    if (outer_armed) memcpy(outer_env, s_spu_halt_env, sizeof(jmp_buf));
    s_spu_halt_armed = 1;
    g_spu_trampoline_fn = 0;                        /* no stale transfer pending */
    /* Lockstep gate (env SPU_LOCKSTEP, default off): join the round-robin
     * ring and BLOCK until this ctx holds the run token, so only one lifted SPU
     * executes at a time. No-op when unarmed. The thread-local halt env above
     * makes a token pause/resume mid-run safe. */
    yz_lockstep_register(ctx);
    switch (SPU_SETJMP(s_spu_halt_env)) {
    case 1:
        halted = 1;
        g_spu_trampoline_fn = 0;
        break;
    case 2:
        /* A one-way guest stack reset invalidates every lifted host caller.
         * Re-enter its target with the same guest state on the driver stack. */
        ctx->host_depth = 0;
        g_spu_trampoline_fn = 0;
        spu_indirect_branch(ctx);
        SPU_DRAIN(ctx);
        break;
    default:
        spu_jmp_no_unwind(s_spu_halt_env);
        /* SPU_DRAIN trampoline model: the top-level entry runs until its first
         * cross-function tail transfer, which sets g_spu_trampoline_fn and
         * returns; the drain loop re-enters each queued target until the SPU
         * halts (stop -> longjmp) or the trampoline empties. Nested brsl/bisl
         * calls drain inside their own call brackets (see the lifter). */
        entry(ctx);
        /* The switch is read once: this runs after every module run, and an
         * uncached getenv here was ~3% of GoW:A's EDGE SPU time. */
        { static int _e = 0, s_aw = -1;
          if (s_aw < 0) s_aw = getenv("SPU_ARGWATCH") ? 1 : 0;
          if (s_aw && _e < 8) { _e++;
              fprintf(stderr, "[argwatch] after entry(): img=%d pc=0x%05X "
                      "r2=0x%08X r3=0x%08X r4=0x%08X\n", ctx->image_id,
                      (uint32_t)ctx->pc & SPU_LS_MASK, ctx->gpr[2]._u32[0],
                      ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0]);
              fflush(stderr); } }
        SPU_DRAIN(ctx);
    }
    yz_lockstep_unregister(ctx);   /* leave the ring; hand the token onward */
    s_spu_halt_armed = outer_armed;
    if (outer_armed) memcpy(s_spu_halt_env, outer_env, sizeof(jmp_buf));
#if defined(SPU_FP_NATIVE) && defined(__aarch64__)
    spu__fpcr_set(fp_csr);
#elif defined(SPU_FP_NATIVE)
    _mm_setcsr(fp_csr);
#endif
    return halted;
}

/* ===========================================================================
 * One private MFC engine per live virtual context. Persistent policy contexts
 * outnumber physical SPUs even when their scheduling quanta run sequentially.
 * Registry nodes have stable addresses and survive a quantum, including stalled
 * lists and tag state. The registry lock protects lookup/publication/removal;
 * the context's single execution owner protects its engine after lookup.
 * Release is permitted only after that owner has stopped using the context.
 * ===========================================================================*/
/* Live contexts, not machine SPUs: a SPURS title keeps a persistent policy-module
 * context per (workload, lane) and runs tasks/jobs on their own contexts besides
 * (GoW:A: 19 workloads plus batch tasks overflowed a fixed table of 8 right after
 * boot), so the registry is unbounded. */
typedef struct spu_mfc_slot {
    spu_context* ctx;
    mfc_engine mfc;
    struct spu_mfc_slot* next;
} spu_mfc_slot;

static spu_mfc_slot* s_mfc_slots;
static SRWLOCK s_mfc_claim_lock = SRWLOCK_INIT;
/* Bumped under the writer lock whenever a node is freed. A thread's cached node
 * (mfc_for) is trusted only while no node has been freed since it was cached:
 * a released context's node is gone, and a new context can reuse its address. */
static volatile long s_mfc_free_gen;

/* SPU_PCSAMPLE=<secs>: every <secs>, print image and pc of every context
 * holding an MFC slot -- i.e. every live SPU that has done DMA. Answers
 * "which SPU task never returned, and where is it" without a debugger. */
static DWORD WINAPI spu_pcsample_thread(LPVOID p)
{
    const DWORD ms = (DWORD)(uintptr_t)p;
    for (;;) {
        Sleep(ms);
        int i = 0;
        AcquireSRWLockShared(&s_mfc_claim_lock);
        for (spu_mfc_slot* slot = s_mfc_slots; slot; slot = slot->next, i++) {
            spu_context* c = slot->ctx;
            if (c) fprintf(stderr, "[pcsample] slot=%d img=%d spu=0x%X pc=0x%05X lr=0x%05X"
                                   " r3=0x%08X r81=0x%08X r84=0x%X r13=0x%08X r14=0x%08X\n",
                           i, c->image_id, c->spu_id, (unsigned)(c->pc & SPU_LS_MASK),
                           (unsigned)(c->gpr[0]._u32[0] & SPU_LS_MASK), c->gpr[3]._u32[0],
                           c->gpr[81]._u32[0], c->gpr[84]._u32[0], c->gpr[13]._u32[0], c->gpr[14]._u32[0]);
        }
        ReleaseSRWLockShared(&s_mfc_claim_lock);
        fflush(stderr);
    }
}

static mfc_engine* mfc_for(spu_context* ctx)
{
    /* Decide once. mfc_for runs on every channel op (eight or nine per DMA), and
     * an unset SPU_PCSAMPLE used to be looked up on each of them: getenv takes
     * the CRT's environment lock, so busy SPU workers queued on it (~3x slower
     * EDGE jobs in GoW:A). */
    { static volatile LONG started = 0; const char* e;
      if (!started && !InterlockedExchange(&started, 1) && (e = getenv("SPU_PCSAMPLE")))
          CreateThread(NULL, 0, spu_pcsample_thread, (LPVOID)(uintptr_t)(atoi(e) * 1000), 0, NULL); }
    /* Per-thread cache of the last lookup: mfc_for runs on every channel op, so
     * the shared lock and list walk are worth skipping when the same context asks
     * again. The node cannot be freed under a live lookup of its own context (only
     * its stopped owner releases it); any other free invalidates the cache. */
    /* One record, one TLS lookup: on macOS each separate thread-local is a _tlv_get_addr call. */
    static SPU_THREAD_LOCAL struct { spu_context* ctx; spu_mfc_slot* slot; long gen; } t_mc;
    if (t_mc.slot && t_mc.ctx == ctx && t_mc.gen == s_mfc_free_gen)
        return &t_mc.slot->mfc;

    AcquireSRWLockShared(&s_mfc_claim_lock);
    const long gen = s_mfc_free_gen;
    for (spu_mfc_slot* slot = s_mfc_slots; slot; slot = slot->next) {
        if (slot->ctx == ctx) {
            ReleaseSRWLockShared(&s_mfc_claim_lock);
            t_mc.ctx = ctx; t_mc.slot = slot; t_mc.gen = gen;
            return &slot->mfc;
        }
    }
    ReleaseSRWLockShared(&s_mfc_claim_lock);

    /* A concurrent miss may publish while the read lock is released. Recheck
     * under the writer lock before creating this context's private engine. */
    AcquireSRWLockExclusive(&s_mfc_claim_lock);
    for (spu_mfc_slot* slot = s_mfc_slots; slot; slot = slot->next) {
        if (slot->ctx == ctx) {
            t_mc.ctx = ctx; t_mc.slot = slot; t_mc.gen = s_mfc_free_gen;
            ReleaseSRWLockExclusive(&s_mfc_claim_lock);
            return &slot->mfc;
        }
    }
    spu_mfc_slot* slot = (spu_mfc_slot*)malloc(sizeof(*slot));
    if (!slot) {
        fprintf(stderr, "[SPU] cannot allocate private MFC engine for context %p\n", (void*)ctx);
        fflush(stderr);
        abort(); /* Shared fallback state would silently corrupt DMA completion. */
    }
    slot->ctx = ctx;
    mfc_engine_init(&slot->mfc);
    slot->next = s_mfc_slots;
    s_mfc_slots = slot;
    t_mc.ctx = ctx; t_mc.slot = slot; t_mc.gen = s_mfc_free_gen;
    ReleaseSRWLockExclusive(&s_mfc_claim_lock);
    return &slot->mfc;
}

void spu_mfc_release(spu_context* ctx)
{
    if (!ctx) return;
    AcquireSRWLockExclusive(&s_mfc_claim_lock);
    for (spu_mfc_slot** link = &s_mfc_slots; *link; link = &(*link)->next) {
        if ((*link)->ctx != ctx) continue;
        spu_mfc_slot* slot = *link;
        *link = slot->next;
        s_mfc_free_gen++;
        free(slot);
        break;
    }
    ReleaseSRWLockExclusive(&s_mfc_claim_lock);
}

/* SPU_JOB_CAPTURE / SPU_JOB_REPLAY: the engine's tag state belongs to a job's
 * starting state (a list parked at a stall keeps its tag incomplete). */
void spu_jobcap_mfc_state(spu_context* ctx, uint32_t* tag_completed,
                          uint32_t* queue_count, int set)
{
    mfc_engine* m = mfc_for(ctx);
    if (set) { m->tag_completed = *tag_completed; m->queue_count = *queue_count; }
    else     { *tag_completed = m->tag_completed; *queue_count = m->queue_count; }
}

/* Exit safety net for contexts unwound off a departing thread's stack.
 * Pointer comparisons suffice: the abandoned context need not be readable. */
void spu_mfc_forget_range(uintptr_t lo, uintptr_t hi)
{
    AcquireSRWLockExclusive(&s_mfc_claim_lock);
    for (spu_mfc_slot** link = &s_mfc_slots; *link;) {
        spu_mfc_slot* slot = *link;
        uintptr_t owner = (uintptr_t)slot->ctx;
        if (owner >= lo && owner < hi) {
            *link = slot->next;
            s_mfc_free_gen++;
            free(slot);
        } else {
            link = &slot->next;
        }
    }
    ReleaseSRWLockExclusive(&s_mfc_claim_lock);
}

/* ===========================================================================
 * Atomic reservation (GETLLAR / PUTLLC / PUTLLUC) -- real lock-line semantics
 *
 * Multiple SPU kernel threads (the SPURS workload runtime runs several SPUs on
 * one shared lock-free queue) issue GETLLAR/PUTLLC on the SAME 128-byte lines.
 * Without honoring the reservation, two SPUs both "claim" the same slot and the
 * queue corrupts (observed: the 2nd claim returns garbage [1,1,1,1] -> the SPU
 * traps). PUTLLC must FAIL when the line changed since GETLLAR. We implement the
 * compare-and-swap under one global lock across all SPU host threads.
 * ===========================================================================*/
extern uint8_t* vm_base;

/* The spinlock guarding all atomic line ops now lives in spu_coherency.c as
 * spu_lockline_lock/unlock. It used to be a file-static here, which serialized
 * SPU against SPU but left the PPU free to store into a line in the middle of
 * a PUTLLC's compare-and-commit -- the update the SPU was about to make would
 * be written over, and the SPU would never learn the line had changed. Same
 * lock, same critical sections, now shared with the PPU store paths. */

/* Total PUTLLC attempts (all SPUs). The PM flow trace (spurs_policy.c) reads
 * the delta across one policy run to find the run that performed a claim. */
volatile unsigned g_spu_putllc_count = 0;
/* PUTLLCs that hit the WATCHED sync line (g_barrier_sync_watch) -- isolates the
 * work-run on the stalled job queue from the dozen idle jobmanager instances. */
volatile unsigned g_spu_putllc_sync_hit = 0;

/* Lowest LS address treated as "not job code". Jobs load at 0 and
 * their stacks top out well below this; a branch at or above it with no
 * lifted code is a runaway, not a service call. */
#define SPU_JM2_KERNEL_BASE 0x15000u

/* Link-register value handed to a SPURS job so its final `bi $r0` lands
 * somewhere we recognise. The real job manager passes a return address into
 * itself; we have no manager, so we pass this and treat arriving here as job
 * completion. Chosen above the context/descriptor we park near the top of LS. */
#define SPU_JOB_RETURN_LS   0x3FF00u

/* Returns 1 if `cmd` is an atomic line op and was handled here, else 0. */
static int spu_mfc_atomic(spu_context* ctx, uint32_t cmd)
{
    /* Classify FIRST. Every MFC_Cmd write lands here, and only the switch at
     * the bottom used to filter -- so the uncommitted-EA guard below reported
     * ORDINARY DMA as `[spu-atomic]`. A plain put (0x20) to a garbage EA read
     * as an atomic spin, which is a genuinely misleading place to start
     * debugging from. Non-atomic commands belong to the DMA engine. */
    if (cmd != MFC_GETLLAR_CMD && cmd != MFC_PUTLLC_CMD &&
        cmd != MFC_PUTLLUC_CMD && cmd != MFC_PUTQLLUC_CMD)
        return 0;

    uint32_t ea  = ctx->mfc_eal & ~(uint32_t)(MFC_ATOMIC_LINE - 1);
    uint32_t lsa = ctx->mfc_lsa & SPU_LS_MASK;
    uint8_t* ls  = &ctx->ls[lsa];
    uint8_t* mem = vm_base + ea;

    /* SPU_JOB_REPLAY: the line and the PUTLLC verdict come from the transcript;
     * no guest memory, reservation registry or commit hooks. */
    if (g_spu_jobrep_on > 0 && spu_jobrep_atomic(ctx, cmd, ea, lsa)) return 1;

    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_n = 0; static uint32_t s_lastea = 0;
        if ((++s_n % 2000000) == 0 || ea != s_lastea) {
          if ((s_n % 2000000) == 0)
            fprintf(stderr, "[atomcnt] %llu atomic ops; last cmd=0x%X ea=0x%08X\n",
                    (unsigned long long)s_n, cmd, ea);
          s_lastea = ea; } } }
    /* SPU_ATOMTRACE=1 keeps the old 40-line cap; =<N> raises it. A busy image
     * (FMOD) exhausts 40 before a quieter one issues its first atomic, which
     * reads as "that image never does atomics" when it simply never got a line. */
    { static int s_at = -1;
      if (s_at < 0) { const char* e = getenv("SPU_ATOMTRACE");
                      int v = e ? atoi(e) : 0; s_at = e ? (v > 1 ? v : 40) : 0; }
      if (s_at) { static int _a=0; if (PS3_LOG_FIRST(_a, s_at))
        fprintf(stderr, "[atom] cmd=0x%02X ea=0x%08X (img=%d)\n", cmd, ea, ctx->image_id); } }
    /* cri task (img22) atomic on the taskset: dump the loaded bitset line so we can
     * see if the task reads MY taskset (0x4005E000) with my READY bit, or elsewhere. */
    { static int s_ct=-1; if(s_ct<0) s_ct=getenv("SPU_ATOMTRACE")?1:0;
      if(s_ct && ctx->image_id==22 && cmd==0xD0 && mfc_ea_range_committed(ea,16)) {
        static int _c=0; if(PS3_LOG_FIRST(_c, 24)){
          uint8_t* m=vm_base+ea;
          #define BW(o) (((uint32_t)m[o]<<24)|((uint32_t)m[o+1]<<16)|((uint32_t)m[o+2]<<8)|m[o+3])
          fprintf(stderr,"[cri-atom] GETLLAR ea=0x%08X line[0..0x30]: %08X %08X %08X %08X | %08X %08X %08X %08X | %08X %08X %08X %08X\n",
            ea, BW(0),BW(4),BW(8),BW(0xC), BW(0x10),BW(0x14),BW(0x18),BW(0x1C), BW(0x20),BW(0x24),BW(0x28),BW(0x2C));
          #undef BW
        } } }
    /* YDKJ_CRI_R4: dump the CellSpursTaskset bitsets when the policy atomically
     * touches my taskset (0x0F000000), to watch the task-activation state machine
     * (why task0 isn't selected+first-run). running@0 ready@0x10 pending@0x20
     * enabled@0x30 signalled@0x40 waiting@0x50 (each 16B; word0 = MSB, task0=bit127). */
    { static int s_td = -1; if (s_td < 0) s_td = (getenv("YDKJ_CRI_CHAIN") && getenv("SPU_ATOMTRACE")) ? 1 : 0;
      if (s_td && ea >= 0x0F000000u && ea < 0x0F001900u) {
        extern uint8_t* vm_base;
        static int _t=0; if (vm_base && PS3_LOG_FIRST(_t, 24)) {
            uint8_t* t = vm_base + 0x0F000000u;
            #define TW(o) (((uint32_t)t[o]<<24)|((uint32_t)t[o+1]<<16)|((uint32_t)t[o+2]<<8)|t[o+3])
            fprintf(stderr, "[tset] %s run=%08X rdy=%08X pnd=%08X ena=%08X sig=%08X wait=%08X | wid=%08X last=%02X\n",
                    cmd==0xD0?"GET":cmd==0xB4?"PUT":"?", TW(0x00), TW(0x10), TW(0x20), TW(0x30), TW(0x40), TW(0x50), TW(0x74), t[0x73]);
            #undef TW
        }
      } }

    /* Guard atomic line ops against an uncommitted/garbage EA (e.g. a SPURS
     * policy computing a lock-line address from an incomplete instance context).
     * Same rationale as the DMA EA guard: a bad guest atomic must not segfault
     * the host. GETLLAR returns a zeroed line (no reservation); PUTLLC fails. */
    if (!mfc_ea_range_committed(ea, MFC_ATOMIC_LINE)) {
        static int s_w = 0;
        if (PS3_LOG_FIRST(s_w, 16))
            fprintf(stderr, "[spu-atomic] cmd=0x%X ea=0x%08X pc=0x%05X img=%d uncommitted -- skipped\n",
                    cmd, ea, (uint32_t)ctx->pc & SPU_LS_MASK, ctx->image_id);
        if (cmd == MFC_GETLLAR_CMD) {
            memset(ls, 0, MFC_ATOMIC_LINE);
            ctx->resv_ea = ea; ctx->resv_valid = 0; ctx->atomic_stat = 4;
        } else {
            ctx->atomic_stat = 1;   /* PUTLLC failure (line "moved") */
        }
        return 1;
    }

    /* SPU_ATOM_EA=<hex>: log every lock-line atomic touching that EA`s 128B line.
     * SPU_ATOM_FULL=1 additionally dumps 32 bytes of the line before and after.
     *
     * Built into ONE buffer and written with a single fprintf: several SPU host
     * threads and the PPU all write stderr, so a dump split across many fprintf
     * calls comes out shredded and unreadable exactly when it matters. */
    /* SPU_ATOM_EA=<ea>[,<ea>...] (up to 4 lines); SPU_ATOM_PUTONLY=1 skips GETLLAR
     * (a policy module polling its list line would otherwise flood the log). */
    { static int s_ae = -1, s_po; static uint32_t s_aea, s_aex[4];
      if (s_ae < 0) { const char* e = getenv("SPU_ATOM_EA");
        s_aea = e ? (uint32_t)strtoul(e, 0, 16) & ~127u : 0; s_ae = s_aea ? 1 : 0;
        for (int k = 0; e && k < 4; k++) { s_aex[k] = (uint32_t)strtoul(e, 0, 16) & ~127u;
            e = strchr(e, ','); if (e) e++; }
        s_po = getenv("SPU_ATOM_PUTONLY") ? 1 : 0;
        if (s_ae == 1) { extern uint32_t g_barrier_sync_watch;
            g_barrier_sync_watch = s_aea; } /* arm PUTLLC OK/FAIL verdict log */ }
      const uint32_t aline = (uint32_t)ea & ~127u;
      if (s_ae == 1 && (aline == s_aex[0] || aline == s_aex[1] || aline == s_aex[2] || aline == s_aex[3]) &&
          !(s_po && cmd == MFC_GETLLAR_CMD)) {
          static int _n = 0, s_cap = -1;
          /* SPU_ATOM_EA_MAX=<n>: lines to print (default 48, 0 = unlimited). */
          if (s_cap < 0) { const char* m = getenv("SPU_ATOM_EA_MAX"); s_cap = m ? atoi(m) : 48; }
          if (!s_cap || PS3_LOG_FIRST(_n, s_cap)) {
              extern uint8_t* vm_base;
              const uint8_t* r = vm_base + ((uint32_t)ea & ~127u);
              static int s_af = -1;
              if (s_af < 0) { const char* f = getenv("SPU_ATOM_FULL"); s_af = f ? (atoi(f) > 1 ? 2 : 1) : 0; }
              char buf[1024]; int p = 0;
              /* lr (gpr[0]) as well as pc: these atomics sit in generic helpers,
               * so pc names the HELPER and only lr names the real caller. */
              p += snprintf(buf + p, sizeof buf - p,
                            "[atom-ea] cmd=0x%X img=%d pc=0x%05X lr=0x%05X ea=0x%08X",
                            cmd, ctx->image_id, (uint32_t)ctx->pc & SPU_LS_MASK,
                            ctx->gpr[0]._u32[0] & SPU_LS_MASK, (uint32_t)ea);
              int nb = s_af == 2 ? 128 : s_af ? 64 : 8;   /* 64 covers +0x30 pendingRecv; =2 the whole line */
              p += snprintf(buf + p, sizeof buf - p, " RAM=");
              for (int i = 0; i < nb && p < (int)sizeof buf - 4; i++)
                  p += snprintf(buf + p, sizeof buf - p, "%02X%s", r[i],
                                (i % 4 == 3) ? " " : "");
              if (cmd == MFC_PUTLLC_CMD) {
                  p += snprintf(buf + p, sizeof buf - p, " STORE=");
                  for (int i = 0; i < nb && p < (int)sizeof buf - 4; i++)
                      p += snprintf(buf + p, sizeof buf - p, "%02X%s", ls[i],
                                    (i % 4 == 3) ? " " : "");
              }
              fprintf(stderr, "%s\n", buf); fflush(stderr);
          }
      } }
    /* Bink sync-line atomic trace (armed by the PPU-side producer probe). */
    { extern uint32_t g_barrier_sync_watch;
      uint32_t b = g_barrier_sync_watch;
      if (b && ea >= (b & ~127u) && ea < ((b + 0xC0 + 127) & ~127u)) {
          static int _n = 0;
          if (PS3_LOG_FIRST(_n, 64))
              fprintf(stderr, "[sync-atomic] %s img=%d pc=0x%05X ea=0x%08X\n",
                      cmd == MFC_GETLLAR_CMD ? "GETLLAR" :
                      cmd == MFC_PUTLLC_CMD ? "PUTLLC" : "PUTLLUC",
                      ctx->image_id, (uint32_t)ctx->pc, ea);
      } }
    /* SPU_ATOM_WIDS=<s>: per SPURS workload (kernel wklCurrentId, LS 0x1DF), the
     * lock lines its atomics touch most, every <s> seconds. Workloads that drain
     * one shared job list show the same line; separate lists show their own. */
    { static int s_aw = -1;
      if (s_aw < 0) { const char* e = getenv("SPU_ATOM_WIDS"); s_aw = e ? atoi(e) : 0; }
      if (s_aw > 0) {
          enum { AW_N = 1024 };
          static struct { uint32_t line; uint16_t wid; uint32_t get, put; } s_t[AW_N];   /* put: PUTLLC, PUTLLUC and PUTQLLUC */
          static SRWLOCK s_lk = SRWLOCK_INIT;
          static unsigned long long s_last;
          extern unsigned long long ps3_ms_now(void);
          const uint32_t line = (uint32_t)ea & ~127u;
          const uint16_t wid = ctx->ls[0x1DF];
          AcquireSRWLockExclusive(&s_lk);
          unsigned h = ((line >> 7) * 2654435761u ^ wid * 40503u) & (AW_N - 1);
          for (unsigned n = 0; n < AW_N; n++, h = (h + 1) & (AW_N - 1)) {
              if (!s_t[h].line) { s_t[h].line = line; s_t[h].wid = wid; }
              if (s_t[h].line == line && s_t[h].wid == wid) {
                  if (cmd == MFC_GETLLAR_CMD) s_t[h].get++; else s_t[h].put++;
                  break;
              }
          }
          const unsigned long long now = ps3_ms_now();
          if (!s_last) s_last = now;
          if (now - s_last >= (unsigned long long)s_aw * 1000ull) {
              s_last = now;
              for (unsigned i = 0; i < AW_N; i++)
                  if (s_t[i].line && s_t[i].get + s_t[i].put >= 100)
                      fprintf(stderr, "[atom-wid] wid=%u line=0x%08X getllar=%u put=%u\n",
                              s_t[i].wid, s_t[i].line, s_t[i].get, s_t[i].put);
              memset(s_t, 0, sizeof s_t);
          }
          ReleaseSRWLockExclusive(&s_lk);
      } }
    switch (cmd) {
    case MFC_GETLLAR_CMD:
        /* Lockstep tick: a guest GETLLAR..PUTLLC poll loop is intra-function
         * (all gotos), so it never crosses the SPU_DRAIN tick site -- without a
         * tick here a polling SPU would hold the run token forever while the
         * peer that must WRITE the line waits for it (canersaka ticks the
         * GETLLAR fast+slow paths for exactly this reason). */
        yz_lockstep_tick(ctx);
        /* A registered holder re-reading an already marked line -- every
         * policy-module poll -- publishes its reservation and copies without
         * the lock; any write the copy misses ends that reservation (see
         * spu_coh_reserve_fast). Anything else takes the lock as before. */
        const int llar_fast = spu_coh_reserve_fast(ctx, ea);
        if (!llar_fast) {
            spu_lockline_lock();
            /* Tell the PPU store paths this line is live, so a store into it goes
             * through the lock and raises SPU_EVENT_LR here instead of landing
             * unannounced. Nothing else about this transaction changes. */
            spu_coh_reserve(ctx, ea);
        }

        /* ONE read of guest memory, then the snapshot from that copy.
         *
         * This used to memcpy from `mem` twice -- once to the local store, once
         * to resv_line -- and a PPU store landing BETWEEN the two reads gave the
         * SPU a stale line and a fresh snapshot. That is a permanent deadlock,
         * and it is the one this port spent a session chasing: the SPU compares
         * its (stale) copy, finds produced == consumed, and sleeps on
         * MFC_LLR_LOST_EVENT; spu_resv_lost_poll then compares memory against a
         * snapshot that ALREADY has the new value, finds no difference, and
         * never raises the event. Measured at a freeze: mirror 0x9E, snapshot
         * 0x9F, memory 0x9F, and the SPU polling rchcnt 1.6 billion times.
         *
         * Hardware cannot produce that state: GETLLAR is a single atomic
         * 128-byte read, so the reserved data and the reservation come from the
         * same instant. Copying the snapshot from `ls` restores that property,
         * and a store that lands after the read now leaves BOTH stale -- which
         * is what makes the lost-reservation event fire. */
        memcpy(ls, mem, MFC_ATOMIC_LINE);              /* line -> local store */
        /* Snapshot from `ls`, NOT a second read of `mem`. Hardware GETLLAR is a
         * single atomic 128-byte read, so the reserved data and the reservation
         * come from the same instant. Reading guest memory twice lets a PPU store
         * land between them, leaving the SPU a STALE line and a FRESH snapshot --
         * and then both halves of the wait fail: the SPU compares its stale copy,
         * sees produced == consumed and sleeps on MFC_LLR_LOST_EVENT, while the
         * reservation poll compares memory against a snapshot that already holds
         * the new value, finds no difference, and never raises the event. Nobody
         * wakes anybody. Snapshotting from `ls` makes a later store leave BOTH
         * stale, which is exactly what makes the lost-reservation event fire. */
        memcpy(ctx->resv_line, ls, MFC_ATOMIC_LINE);   /* snapshot for compare */
        /* MFC_RdAtomicStat after a GETLLAR reads 4 (GETLLAR complete), not 0:
         * Havok's SPU allocator (GH3 collide task, LS 0xA66C) re-issues the
         * GETLLAR until that bit is set, and spun forever on 0. PUTLLUC is 2. */
        /* resv_valid holds (line | 1), not 1: a notifier clears only the reservation
         * of the line it wrote, with a CAS (spu_coherency.c, the lost-kick race) */
        if (!llar_fast) {
            ctx->resv_ea = ea;
            __atomic_store_n(&ctx->resv_valid, (int)((ea & ~127u) | 1u), __ATOMIC_RELAXED);
        }   /* the fast path published it already */
        ctx->atomic_stat = 4;
        ctx->dbg_getllar++;
        /* A GETLLAR re-issued millions of times on one line is an SPU spin-wait
         * that nobody is releasing (a ticket lock, a counter barrier). Say so
         * once per line with the line's head, so the stuck word is visible
         * without a per-function probe. */
        { static SPU_THREAD_LOCAL uint32_t t_ea; static SPU_THREAD_LOCAL unsigned long t_n;
          if (ea != t_ea) { t_ea = ea; t_n = 0; }
          else if (++t_n == (unsigned long)({ static long s_sn = -1;
                       if (s_sn < 0) { const char* e = getenv("SPU_SPIN_N"); s_sn = e ? atol(e) : 4000000; }
                       s_sn; })) {   /* SPU_SPIN_N=<n>: report threshold (default 4M) */
              const uint8_t* m = vm_base + ea;
              fprintf(stderr, "[spu-spin] tid=0x%X img=%d pc=0x%05X lr=0x%05X GETLLAR ea=0x%08X x4M; line:"
                              " %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                      (unsigned)ctx->spu_id, ctx->image_id, (unsigned)(ctx->pc & SPU_LS_MASK), ctx->gpr[0]._u32[0] & SPU_LS_MASK, ea,
                      m[0],m[1],m[2],m[3], m[4],m[5],m[6],m[7], m[8],m[9],m[10],m[11], m[12],m[13],m[14],m[15]);
              /* Who is spinning: LS code around pc (disassemble offline) + low GPRs. */
              { static int s_once = 0;
                if (s_once++ < 2) {
                    uint32_t pc0 = ((uint32_t)ctx->pc & SPU_LS_MASK & ~3u);
                    uint32_t lo = pc0 >= 0x80 ? pc0 - 0x80 : 0;
                    fprintf(stderr, "[spu-spin]   code@0x%05X:", lo);
                    for (uint32_t o = lo; o < lo + 0xC0 && o + 3 < SPU_LS_SIZE; o += 4)
                        fprintf(stderr, " %02X%02X%02X%02X", ctx->ls[o], ctx->ls[o+1], ctx->ls[o+2], ctx->ls[o+3]);
                    fputc(10, stderr);
                    for (int g = 0; g < 16; g++)
                        fprintf(stderr, "[spu-spin]   r%-2d %08X %08X %08X %08X%c", g,
                                ctx->gpr[g]._u32[0], ctx->gpr[g]._u32[1], ctx->gpr[g]._u32[2], ctx->gpr[g]._u32[3], 10);
                } }
          } }
        if (!llar_fast) spu_lockline_unlock();
        /* `ls` is this SPU's own copy, so recording it outside the lock is the
         * line exactly as the GETLLAR read it. */
        if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
            jc_ev(JC_GETLLAR, cmd, ea, lsa, 0, ls, MFC_ATOMIC_LINE);
        /* SPU_LLARWATCH=<hex EA>: every GETLLAR of that line, with the LSA it
         * used and the first word as it lands in BOTH places.
         *
         * The deadlock this settles: at a freeze the reservation snapshot holds
         * the fresh produced value (0xC9) while the SPU's local-store mirror
         * still holds the stale one (0xC8) -- and the compare reads the mirror.
         * Both memcpys above copy from the same `mem`, so that can only happen
         * if `lsa` was not where the SPU asked. Reading ctx->mfc_lsa from a
         * later poll cannot prove it (by then the SPU has issued other DMA and
         * overwritten the field, which is exactly the compare-two-moments
         * mistake that has cost this port ten retractions). So log it HERE, at
         * the GETLLAR, with the word that actually landed. */
        { static int s_lw2 = -2; static uint32_t s_lwea;
          if (s_lw2 == -2) { const char* e6 = getenv("SPU_LLARWATCH");
                             s_lw2 = e6 ? 1 : 0;
                             s_lwea = e6 ? (uint32_t)strtoul(e6, 0, 16) & ~127u : 0u; }
          if (s_lw2 && (ea & ~127u) == s_lwea) {
              static unsigned long ln;
              /* A histogram of the LSAs this line's GETLLARs use, not a
               * sample of them. resv_line has exactly one writer -- this
               * GETLLAR -- and it copies to &ls[mfc_lsa] from the same source,
               * so a snapshot that is fresh while LS 0x10800 is stale can only
               * mean some GETLLAR ran with a DIFFERENT mfc_lsa. Sampling every
               * 256th hit would miss exactly those. Count them all and print
               * every LSA seen. */
              { static uint32_t seen[8]; static unsigned long cnt[8]; static int ns;
                int f = -1;
                for (int z = 0; z < ns; z++) if (seen[z] == lsa) { f = z; break; }
                if (f < 0 && ns < 8) { f = ns; seen[ns] = lsa; cnt[ns] = 0; ns++; }
                if (f >= 0) cnt[f]++;
                if ((ln % 4096) == 0) {
                    fprintf(stderr, "[llar-lsa] %lu GETLLARs on 0x%08X;", ln, ea);
                    for (int z = 0; z < ns; z++)
                        fprintf(stderr, " lsa=0x%05X x%lu", seen[z], cnt[z]);
                    fprintf(stderr, "\n"); fflush(stderr);
                } }
              if (++ln <= 8 || (ln % 256) == 0) {
                  const uint8_t* q = (const uint8_t*)ls;
                  const uint32_t landed = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16)
                                        | ((uint32_t)q[2] << 8) | q[3];
                  const uint8_t* m8 = (const uint8_t*)mem;
                  const uint32_t frommem = ((uint32_t)m8[0] << 24) | ((uint32_t)m8[1] << 16)
                                         | ((uint32_t)m8[2] << 8) | m8[3];
                  const uint8_t* mir = ctx->ls + 0x10800u;
                  const uint32_t mirw = ((uint32_t)mir[0] << 24) | ((uint32_t)mir[1] << 16)
                                      | ((uint32_t)mir[2] << 8) | mir[3];
                  fprintf(stderr, "[llar] n=%lu spu%u pc=0x%05X lsa=0x%05X"
                                  " ea=0x%08X mem=0x%08X landed=0x%08X"
                                  " mirror[0x10800]=0x%08X r90=0x%08X\n",
                          ln, (unsigned)(ctx->spu_id & 7u),
                          (uint32_t)ctx->pc & SPU_LS_MASK, lsa, ea,
                          frommem, landed, mirw, ctx->gpr[90]._u32[0]);
                  fflush(stderr);
              }
          } }
        return 1;

    case MFC_PUTLLC_CMD:
        g_spu_putllc_count++;   /* run-scoped delta read by the PM flow trace */
        { extern uint32_t g_barrier_sync_watch;
          uint32_t b = g_barrier_sync_watch;
          if (b && (ea & ~127u) == (b & ~127u)) g_spu_putllc_sync_hit++; }
        spu_lockline_lock();
        if (ctx->resv_valid && ctx->resv_ea == ea &&
            memcmp(mem, ctx->resv_line, MFC_ATOMIC_LINE) == 0) {
            { extern void spu_lockguard_check(spu_context*, uint32_t, const uint8_t*, uint32_t, const char*);
              spu_lockguard_check(ctx, ea, ls, MFC_ATOMIC_LINE, "PUTLLC"); }
            memcpy(mem, ls, MFC_ATOMIC_LINE);          /* commit local store */
            /* A committing PUTLLC is a line write like any other, so every
             * PEER reservation on it is lost and its SPU takes SPU_EVENT_LR.
             * Silent, this is the same lost update the PPU half was added to
             * close, with an SPU on the writing side: a peer is never told to
             * re-read, so it goes on polling a line it believes it still owns,
             * and a peer parked on RdEventStat sleeps through the commit it
             * was waiting for.
             *
             * Our own reservation is dropped FIRST so the notify does not
             * raise a self-LR: hardware CONSUMES the reservation on a
             * successful PUTLLC, it does not report it lost. Already under the
             * lock-line lock, which is what spu_coh_notify_write expects. */
            ctx->resv_valid = 0;
            { static int s_w = -1;
              if (s_w < 0) s_w = getenv("SPU_PUTLLC_WHY") ? 1 : 0;
              if (s_w) { static unsigned long long n;
                  if ((++n % 100000) == 1)
                      /* ...and the first four words of the line being written.
                       * A succeeding PUTLLC that repeats millions of times is a
                       * livelock, and the only useful question is what the SPU
                       * keeps seeing there -- the count alone cannot answer it. */
                      { uint32_t w[8] = {0,0,0,0,0,0,0,0};
                        for (int _i = 0; _i < 8; _i++) {
                            uint32_t t = 0;
                            memcpy(&t, (const unsigned char*)ctx->ls + ((lsa + _i*4) & SPU_LS_MASK), 4);
                            w[_i] = __builtin_bswap32(t);
                        }
                        fprintf(stderr, "[putllc-ok] %llu: img=%d ctx=%p ea=0x%08X"
                                        " line=%08X %08X %08X %08X %08X %08X %08X %08X\n",
                                n, ctx->image_id, (void*)ctx, ea,
                                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]); } } }
            spu_coh_notify_write(ea);
            ctx->atomic_stat = 0;                      /* PUTLLC_SUCCESS */
        } else {
            ctx->atomic_stat = 1;                      /* PUTLLC_FAILURE -> retry */
            /* SPU_PUTLLC_WHY=1: which of the three conditions failed.
             * A PUTLLC retry loop that never succeeds is a livelock, and
             * "failed" alone cannot distinguish a lost reservation from a line
             * another agent is rewriting -- opposite causes, opposite fixes. */
            { static int s_w = -1;
              if (s_w < 0) s_w = getenv("SPU_PUTLLC_WHY") ? 1 : 0;
              if (s_w) {
                  static unsigned long long n_noresv, n_ea, n_line;
                  if (!ctx->resv_valid)            n_noresv++;
                  else if (ctx->resv_ea != ea)     n_ea++;
                  else                             n_line++;
                  static unsigned long long total;
                  if ((++total % 500000) == 0) {
                      fprintf(stderr, "[putllc-why] %llu fails: no-reservation=%llu"
                                      " ea-mismatch=%llu line-changed=%llu"
                                      " (last ea=0x%08X img=%d resv_ea=0x%08X"
                                      " getllar=%llu)\n",
                              total, n_noresv, n_ea, n_line, ea, ctx->image_id,
                              ctx->resv_ea, (unsigned long long)ctx->dbg_getllar);
                      fflush(stderr);
                  }
              } }
        }
        { extern uint32_t g_barrier_sync_watch;
          uint32_t b = g_barrier_sync_watch;
          if (b && ea >= (b & ~127u) && ea < ((b + 0xC0 + 127) & ~127u)) {
              static int _n = 0;
              if (PS3_LOG_FIRST(_n, 96) && ctx->atomic_stat == 0) {
                  uint32_t sn = ((uint32_t)ctx->ls[0x1C8]<<24)|((uint32_t)ctx->ls[0x1C9]<<16)|
                                ((uint32_t)ctx->ls[0x1CA]<<8)|ctx->ls[0x1CB];
                  extern uint8_t* vm_base;
                  const uint8_t* r = vm_base + b + 0x70;   /* row 3 (sync+0x40+16*3) */
                  fprintf(stderr, "[sync-atomic] PUTLLC-OK spuNum=%u ea=+0x%X lanes@row3={%u %u %u %u %u %u %u}\n",
                          sn, ea - b,
                          (r[2]<<8)|r[3],(r[4]<<8)|r[5],(r[6]<<8)|r[7],(r[8]<<8)|r[9],
                          (r[10]<<8)|r[11],(r[12]<<8)|r[13],(r[14]<<8)|r[15]);
              }
          } }
        /* SPU_ATOM_EA: the [atom-ea] line above prints every ATTEMPT, before
         * the compare. Say which of them committed, or a retried failure reads
         * as a lost update. */
        { static int s_v = -1; static uint32_t s_vx[4];
          if (s_v < 0) { const char* e = getenv("SPU_ATOM_EA"); s_v = e ? 1 : 0;
                         for (int k = 0; e && k < 4; k++) { s_vx[k] = (uint32_t)strtoul(e, 0, 16) & ~127u;
                             e = strchr(e, ','); if (e) e++; } }
          const uint32_t vl = ea & ~127u;
          if (s_v && (vl == s_vx[0] || vl == s_vx[1] || vl == s_vx[2] || vl == s_vx[3]))
              fprintf(stderr, "[atom-ea] PUTLLC %s lr=0x%05X ea=0x%08X\n",
                      ctx->atomic_stat ? "FAIL" : "OK",
                      ctx->gpr[0]._u32[0] & SPU_LS_MASK, ea); }
        ctx->resv_valid = 0;                           /* reservation consumed */
        spu_lockline_unlock();
        if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
            jc_ev(JC_PUTLLC, cmd, ea, lsa, ctx->atomic_stat == 0, ls, MFC_ATOMIC_LINE);
        if (ctx->atomic_stat == 0) {
            spu_taskset_line_committed(ea & ~127u);
            if (g_spu_line_commit_hook) g_spu_line_commit_hook(ea & ~127u);
            if (g_spu_line_commit_title_hook) g_spu_line_commit_title_hook(ea & ~127u, ctx);
        }
        return 1;

    case MFC_PUTLLUC_CMD:
    case MFC_PUTQLLUC_CMD:
        spu_lockline_lock();
        { extern void spu_lockguard_check(spu_context*, uint32_t, const uint8_t*, uint32_t, const char*);
          spu_lockguard_check(ctx, ea, ls, MFC_ATOMIC_LINE, "PUTLLUC"); }
        memcpy(mem, ls, MFC_ATOMIC_LINE);              /* unconditional store */
        /* Unconditional, so it invalidates EVERY reservation on the line --
         * this SPU's included, which is why the notify runs before the
         * bookkeeping below rather than after it. A peer left holding a
         * reservation here would commit a PUTLLC against a snapshot this store
         * has already overwritten. */
        spu_coh_notify_write(ea);
        ctx->resv_valid = 0; ctx->atomic_stat = 2;   /* PUTLLUC complete */
        spu_lockline_unlock();
        if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
            jc_ev(JC_PUTLLUC, cmd, ea, lsa, 1, ls, MFC_ATOMIC_LINE);
        spu_taskset_line_committed(ea & ~127u);
        if (g_spu_line_commit_hook) g_spu_line_commit_hook(ea & ~127u);
        if (g_spu_line_commit_title_hook) g_spu_line_commit_title_hook(ea & ~127u, ctx);
        return 1;

    default:
        return 0;
    }
}

static int channel_is_mfc(uint32_t ch)
{
    switch (ch) {
    case MFC_WrMSSyncReq: case MFC_RdTagMask:  case MFC_LSA:
    case MFC_EAH:         case MFC_EAL:         case MFC_Size:
    case MFC_TagID:       case MFC_Cmd:         case MFC_WrTagMask:
    case MFC_WrTagUpdate: case MFC_RdTagStat:   case MFC_RdListStallStat:
    case MFC_WrListStallAck: case MFC_RdAtomicStat:
        return 1;
    default:
        return 0;
    }
}

/* ===========================================================================
 * Channel write
 * ===========================================================================*/
/* Wall time this host thread spent in channel writes (MFC commands = the DMA
 * copies) and channel reads (incl. tag/event waits), QPC ticks. Only measured
 * under SPURS_PM_PROF, which reports them per workload as m% / c%. */
SPU_THREAD_LOCAL long long g_spu_wrch_ticks, g_spu_rdch_ticks;
SPU_THREAD_LOCAL long long g_spu_rdch_ch_ticks[32];   /* the same reads, per channel (0-31) */
static SPU_THREAD_LOCAL uintptr_t t_rdch_ra;           /* lifted caller of spu_rdch (SPU_DECPC) */
static int s_spu_chprof = -1;
static inline int spu_chprof_on(void) {
    if (s_spu_chprof < 0) s_spu_chprof = getenv("SPURS_PM_PROF") ? 1 : 0;
    return s_spu_chprof;
}
static void spu_wrch_impl(spu_context* ctx, uint32_t channel, u128 value);
static u128 spu_rdch_impl(spu_context* ctx, uint32_t channel);
void spu_wrch(spu_context* ctx, uint32_t channel, u128 value)
{
    /* The MFC parameter channels only latch a value in the context
     * (mfc_channel_write); the command write is what reaches the engine. Skip
     * the impl, mfc_for and the switch for them -- five or six of the eight or
     * nine writes of every DMA -- unless SPU_CHHIST is counting channel writes.
     * SPURS_PM_PROF keeps the fast path, so its wrch time leaves these out. */
    { static int s_fast = -1;
      if (__builtin_expect(s_fast < 0, 0)) s_fast = !getenv("SPU_CHHIST");
      if (s_fast) {
          const uint32_t v = value._u32[0];
          switch (channel) {
          case MFC_LSA:       ctx->mfc_lsa = v;         return;
          case MFC_EAH:       ctx->mfc_eah = v;         return;
          case MFC_EAL:       ctx->mfc_eal = v;         return;
          case MFC_Size:      ctx->mfc_size = v;        return;
          case MFC_TagID:     ctx->mfc_tag = v & 0x1F;  return;
          case MFC_WrTagMask: ctx->mfc_tag_mask = v;    return;
          default: break;
          }
      } }
#if 1   /* SPURS_PM_PROF timers: portable via win32_compat */
    if (spu_chprof_on()) {
        LARGE_INTEGER t0, t1; QueryPerformanceCounter(&t0);
        spu_wrch_impl(ctx, channel, value);
        QueryPerformanceCounter(&t1);
        g_spu_wrch_ticks += t1.QuadPart - t0.QuadPart;
        return;
    }
#endif
    spu_wrch_impl(ctx, channel, value);
}
u128 spu_rdch(spu_context* ctx, uint32_t channel)
{
    t_rdch_ra = (uintptr_t)__builtin_return_address(0);
    u128 r;
    /* SPU_JOB_REPLAY: the recorded value, never the live read (the decrementer
     * reads the host clock, RdEventStat can block on an event nobody raises). */
    if (g_spu_jobrep_on > 0 && spu_jobrep_rdch(ctx, channel, &r)) return r;
#if 1   /* SPURS_PM_PROF timers: portable via win32_compat */
    if (spu_chprof_on()) {
        LARGE_INTEGER t0, t1; QueryPerformanceCounter(&t0);
        r = spu_rdch_impl(ctx, channel);
        QueryPerformanceCounter(&t1);
        g_spu_rdch_ticks += t1.QuadPart - t0.QuadPart;
        g_spu_rdch_ch_ticks[channel & 31] += t1.QuadPart - t0.QuadPart;
    } else
#endif
    r = spu_rdch_impl(ctx, channel);
    /* Channel reads are where a job sees time and events (decrementer, tag
     * status, event status): record each value for SPU_JOB_CAPTURE. */
    if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
        jc_ev(JC_RDCH, channel, 0, 0, 0, &r, sizeof r);
    return r;
}
static void spu_wrch_impl(spu_context* ctx, uint32_t channel, u128 value)
{
    /* SPU_CHHIST=1: which channels a job actually touches, and how often. A
     * worker that never issues an MFC command is either not being handed its
     * work descriptor or is waiting on a channel we never satisfy; the channel
     * mix distinguishes those. */
    { static int s_c = -1; if (s_c < 0) { const char* e = getenv("SPU_CHHIST"); s_c = e ? (atoi(e) > 0 ? atoi(e) : 2000) : 0; }
      if (s_c) { static unsigned long long w[128]; static unsigned long long n;
          w[channel & 127]++;
          if ((++n % (unsigned long long)s_c) == 0) { fprintf(stderr, "[chw] %llu writes:%c", n, 10);
              for (int i = 0; i < 128; i++) if (w[i])
                  fprintf(stderr, "   wrch ch%-3d %llu%c", i, w[i], 10); } } }
    uint32_t v = value._u32[0];  /* channel writes use the preferred slot */

    if (channel_is_mfc(channel)) {
        /* Atomic line ops (GETLLAR/PUTLLC/...) need real reservation semantics,
         * not the plain GET/PUT the DMA engine would do. */
        if (channel == MFC_Cmd && spu_mfc_atomic(ctx, v))
            return;
        mfc_channel_write(mfc_for(ctx), ctx, channel, v);
        /* MFC tag-status-update EVENT producer (channel-stall milestone). Our
         * DMA completes synchronously, so the instant the program arms tag
         * notification (MFC_WrTagUpdate with a non-zero mode = any/all), the
         * tags in MFC_WrTagMask are already complete -- pend the tag event
         * (bit 0, MFC_TAG_STATUS_UPDATE_EVENT) when the SPU has it enabled, and
         * wake any host thread blocked in rdch SPU_RdEventStat. This is the
         * long-missing producer for event_status (previously only ever cleared
         * via WrEventAck -> RdEventStat waits could never complete). */
        if (channel == MFC_WrTagUpdate && v != 0 && (ctx->event_mask & 0x1u)) {
            SPU_EVENT_SET(ctx, 0x1u);
            spu_ch_wake(ctx);
        }
        return;
    }

    switch (channel) {
    case SPU_WrOutMbox:
        /* A raw SPU's outbound mailbox is one entry deep and wrch stalls while
         * it is full. Buffering instead let two replies queue up, and a PPU
         * that tests bit 0 of the count -- Tornado Outbreak's boot handshake,
         * func_000D5A10 -- read count 2 as empty and spun forever (an
         * intermittent boot hang). Raw SPUs only: their reader is the PPU
         * polling the MMIO window, whereas SPURS-thread mailboxes are drained
         * by paths that do not wake a stalled writer. A raw SPU's local store
         * is the guest window, not its own buffer. */
        if (ctx->ls != ctx->ls_store)
            while (ctx->ch_out_mbox.count >= SPU_MBOX_DEPTH && ctx->status != SPU_STATUS_STOPPED)
                Sleep(0);
        spu_channel_write(&ctx->ch_out_mbox, v);
        /* SPU_DBG_MBOX=1: depth after the write. The image-1 SPUs publish their
         * LS work buffer here (LS 0x11D4) and then poll it, and the guest sets
         * class-2 mask 0x3 -- which excludes the plain-mailbox bit 0x10 -- so
         * this write raises no interrupt and the PPU must POLL to see it. If the
         * depth climbs and stays, nobody is polling and the message is stranded.
         * That is the difference between "the PPU is slow" and "the PPU never
         * looks", which no other probe here distinguishes. */
        { static int _d = -1; if (_d < 0) _d = getenv("SPU_DBG_MBOX") ? 1 : 0;
          if (_d) { static unsigned long _n = 0; if (++_n <= 24)
            fprintf(stderr, "[spu-outmbox] spu=%X wrote 0x%08X depth=%u\n",
                    ctx->spu_id, v, (unsigned)ctx->ch_out_mbox.count); } }
        { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_MBOXTRACE") ? 1 : 0;
          if (s_t) fprintf(stderr, "[spu-mbox] OUT  grp=0x%X spu=0x%X val=0x%08X\n",
                           ctx->spu_group_id, ctx->spu_id, v); }
        /* Plain mailbox data is consumed by the following interrupt request. */
        break;
    case SPU_WrOutIntrMbox:
        { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_MBOXTRACE") ? 1 : 0;
          if (s_t) fprintf(stderr, "[spu-mbox] INTR grp=0x%X spu=0x%X val=0x%08X\n",
                           ctx->spu_group_id, ctx->spu_id, v); }
        if (g_spu_user_event_hook && g_spu_user_event_hook(ctx, v)) break;
        spu_channel_write(&ctx->ch_out_intr_mbox, v);
        if (g_spu_out_mbox_hook) g_spu_out_mbox_hook(ctx->spu_group_id, ctx->spu_id, 1, v);
        break;
    case SPU_WrDec:          ctx->decrementer = v;
                             ctx->dec_base_ns = spu_host_ns();              break;
    case SPU_WrEventMask:    ctx->event_mask = v;                           break; /* WrEventMask */
    case SPU_WrEventAck:
        /* Under the lock-line lock, because every producer of the LR bit sets
         * it under that lock: the PPU coherent store, and a peer SPU's PUTLLC,
         * PUTLLUC or plain PUT. A bare read-modify-write here can read
         * event_status, have a concurrent |= SPU_EVENT_LR land in between, and
         * write the stale value back. The edge is then gone, and an SPU that
         * has just acknowledged the events it read goes back to sleep believing
         * it still holds a reservation it has already lost, which is the parked
         * SPURS kernel this whole mechanism exists to wake. */
        spu_lockline_lock();
        SPU_EVENT_CLEAR(ctx, v);
        /* A stall-and-notify nobody has read yet stays pending. We run a
         * resumed list inline, inside the MFC_WrListStallAck that resumes it,
         * so its next stall can land BEFORE the handler acknowledges the event
         * it was taken for; on hardware the MFC gets there later, after the
         * ack. Acking then cleared that new edge while the list stayed parked
         * (list_stall_stat set, Sn clear). */
        if ((v & 0x2u) && ctx->list_stall_stat)
            SPU_EVENT_SET(ctx, 0x2u);
        spu_lockline_unlock();
        break;
    case SPU_WrSRR0:         ctx->srr0 = v;                                 break;
    default:
        /* Unknown / unhandled channel write -- ignore (matches a no-op SPU). */
        break;
    }
}

/* ===========================================================================
 * Channel-stall contract (faithful-adopt, from canersaka's fork).
 *
 * A blocking spu_rdch on an empty read channel parks the SPU host thread on a
 * per-SPU CV until a producer (mailbox/signal write, event raise) calls
 * spu_ch_wake -- never fabricating a value. Under SPU_LOCKSTEP the wait
 * releases the run token first (block_begin) so a peer SPU that must post the
 * awaited data is never blocked on this ctx. A 10 ms re-poll is the missed-wake
 * safety net. Opt-in (env SPU_CH_BLOCK=1), default OFF while the producers
 * (mailbox/signal writes, MFC tag-status event raise) are being wired -- until
 * then blocking a read whose producer is missing would hang, so default stays
 * legacy non-blocking with zero regression, exactly like the lockstep gate.
 * ===========================================================================*/
/* Set by runtime/spu/spu_raw.c once a raw SPU exists. A raw SPU runs on its own
 * host thread against a PPU that pokes its mailboxes by MMIO, so an empty
 * RdInMbox must PARK, not fabricate a zero -- there is no scheduler above it to
 * retry. Off unless a raw SPU is actually created, so no other port changes. */
int g_spu_force_ch_block = 0;

static int yz_ch_block(void)
{
    static int v = -1;
    if (v < 0) v = getenv("SPU_CH_BLOCK") ? 1 : 0;
    return v || g_spu_force_ch_block;
}

/* Would rdch complete right now? Plain reads + the 10 ms re-poll cover cross-
 * thread visibility (x86 TSO + the wait syscall's barrier); the s43 atomics are
 * a later refinement. */
/* MFC_LLR_LOST_EVENT (0x400) -- the only producer for it.
 *
 * The SPU "wait until another processor touches this variable" idiom is: GETLLAR
 * the 128-byte line, set the event mask to Lr, then block in
 * `rdch SPU_RdEventStat` until the reservation is lost. Nothing ever raised that
 * bit, so an SPU using it waited forever -- ps1_netemu's GPU core (SPU 4) parks
 * at pc=0x0A5E8 with evmask=0x400 exactly here while the PPU spins waiting on the
 * SPU. Deadlock, and the reason the PS1 core never starts.
 *
 * Losing a reservation means the reserved line changed, and GETLLAR already
 * snapshots it, so compare against that snapshot rather than hooking every store.
 * ponytail: polled, not store-hooked -- one 128-byte memcmp per 10 ms poll on an
 * already-blocked SPU, and nothing at all on the PPU store path. The ceiling is
 * one poll of wake latency, and an A-B-A write that restores the same bytes
 * between polls is missed; hook the writers if either ever matters.
 */
static void spu_resv_lost_poll(spu_context* ctx)
{
    if (!(ctx->event_mask & 0x400u) || !ctx->resv_valid) return;
    if (!vm_base || ctx->resv_ea == 0) return;
    if (memcmp(vm_base + ctx->resv_ea, ctx->resv_line, 128) != 0) {
        ctx->resv_valid = 0;
        SPU_EVENT_SET(ctx, 0x400u);
    }
}

static int spu_ch_ready(spu_context* ctx, uint32_t channel)
{
    if (channel == SPU_RdEventStat) spu_resv_lost_poll(ctx);
    switch (channel) {
    case SPU_RdInMbox:      return ctx->rcv_evt_n != 0 || ctx->ch_in_mbox.count != 0;
    case SPU_RdSigNotify1:  return ctx->ch_sig_notify[0].count != 0;
    case SPU_RdSigNotify2:  return ctx->ch_sig_notify[1].count != 0;
    case SPU_RdEventStat:   return (ctx->event_status & ctx->event_mask) != 0;
    default:                return 1;   /* non-blocking channels: always ready */
    }
}

/* Signal the per-SPU wait CV so a blocked spu_rdch re-checks its predicate.
 * MUST be called by every producer AFTER it makes a read predicate true. */
void spu_ch_wake(spu_context* ctx)
{
    if (!ctx) return;
    /* Pass through ch_wait_lock first: spu_ch_wait tests and sleeps under it, so a wake landing between its test and
     * its sleep was lost and the read waited for the 10 ms re-poll (fix from upstream PR #619). SPU_WAKE_LOCK=0 skips
     * it (A/B). */
    static int s_wl = -1;
    if (s_wl < 0) { const char* e = getenv("SPU_WAKE_LOCK"); s_wl = !(e && e[0] == '0'); }
    if (s_wl) {
        AcquireSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
        ReleaseSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
    }
    WakeAllConditionVariable((CONDITION_VARIABLE*)&ctx->ch_wait_cv);
}

/* PS3_FTRACE: when this SPU thread last left a blocking wait. The time between
 * two waits is a busy span, tagged with the lifted image and the SPURS workload
 * id the kernel keeps at LS 0x1DC (SpursKernelContext.wklCurrentId) when it
 * goes idle. Waits shorter than PS3_FTRACE_MIN_US do not split a busy span. */
static __thread uint64_t t_ft_busy0;

static void spu_ft_wait_span(spu_context* ctx, uint32_t channel, uint64_t t0)
{
    const uint64_t t1 = ps3_ft_now();
    if (t1 - t0 < ps3_ft_min_ns()) return;
    if (t_ft_busy0 && t0 > t_ft_busy0) {
        const uint8_t* w = ctx->ls + 0x1DC;
        ps3_ft_span(PS3_FT_SPU_BUSY, ctx->spu_id, (uint32_t)ctx->image_id,
                    ((uint32_t)w[0] << 24) | ((uint32_t)w[1] << 16) | ((uint32_t)w[2] << 8) | w[3],
                    t_ft_busy0, t0);
    }
    ps3_ft_span(PS3_FT_SPU_WAIT, ctx->spu_id, channel, ctx->pc & SPU_LS_MASK, t0, t1);
    t_ft_busy0 = t1;
}

/* Block the calling SPU host thread until `channel` is readable. */
static void spu_ch_wait(spu_context* ctx, uint32_t channel, const char* op)
{
    if (!yz_ch_block() || spu_ch_ready(ctx, channel)) return;
    const uint64_t ft_t0 = PS3_FT_ON() ? ps3_ft_now() : 0;

    { static unsigned long bn = 0; unsigned long n = ++bn;
      if (n <= 50 || (n % 512) == 0)
        fprintf(stderr, "[ch-block] spu=%X pc=0x%05X op=%s ch=%u evstat=0x%X evmask=0x%X\n",
                ctx->spu_id, ctx->pc & SPU_LS_MASK, op, channel,
                ctx->event_status, ctx->event_mask); }

    ctx->status = SPU_STATUS_WAITING_CHANNEL;
    yz_lockstep_block_begin(ctx);          /* release the run token before the OS wait */
    {
        unsigned long long start = GetTickCount64(), next_hb = 2000;
        while (!spu_ch_ready(ctx, channel)) {
            AcquireSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
            if (!spu_ch_ready(ctx, channel))
                SleepConditionVariableSRW((CONDITION_VARIABLE*)&ctx->ch_wait_cv,
                                          (SRWLOCK*)&ctx->ch_wait_lock, 10, 0);
            ReleaseSRWLockExclusive((SRWLOCK*)&ctx->ch_wait_lock);
            unsigned long long waited = GetTickCount64() - start;
            if (waited >= next_hb) {
                fprintf(stderr, "[ch-wait] spu=%X pc=0x%05X ch=%u waited=%llums evstat=0x%X evmask=0x%X resv[valid=%d ea=0x%08X]\n",
                        ctx->spu_id, ctx->pc & SPU_LS_MASK, channel, waited,
                        ctx->event_status, ctx->event_mask,
                        ctx->resv_valid, ctx->resv_ea);
                fflush(stderr);
                next_hb = ((waited / 2000) + 1) * 2000;
            }
        }
    }
    yz_lockstep_block_end(ctx);            /* rejoin the rotation, reacquire the token */
    if (ft_t0) spu_ft_wait_span(ctx, channel, ft_t0);
    ctx->status = SPU_STATUS_RUNNING;
}

/* ===========================================================================
 * Channel read (returns value in the preferred word slot)
 * ===========================================================================*/
static u128 spu_rdch_impl(spu_context* ctx, uint32_t channel)
{
    { static int s_c = -1; if (s_c < 0) { const char* e = getenv("SPU_CHHIST"); s_c = e ? (atoi(e) > 0 ? atoi(e) : 2000) : 0; }
      if (s_c) { static unsigned long long r[128]; static unsigned long long n;
          r[channel & 127]++;
          if ((++n % (unsigned long long)s_c) == 0) { fprintf(stderr, "[chr] %llu reads:%c", n, 10);
              for (int i = 0; i < 128; i++) if (r[i])
                  fprintf(stderr, "   rdch ch%-3d %llu%c", i, r[i], 10); } } }
    /* Block (never fabricate) on an empty producer-fed read channel (opt-in
     * SPU_CH_BLOCK). RdEventStat now has a producer (the MFC tag-status event
     * raise above), but only block it when the SPU has actually enabled events
     * (event_mask != 0) -- a masked-off read must return 0 immediately, not
     * park forever. RdInMbox/RdSigNotify park on their PPU producers. */
    if (channel == SPU_RdInMbox || channel == SPU_RdSigNotify1 || channel == SPU_RdSigNotify2)
        spu_ch_wait(ctx, channel, "rdch");
    else if (channel == SPU_RdEventStat && ctx->event_mask != 0)
        spu_ch_wait(ctx, channel, "rdch");

    uint32_t v = 0;

    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_c[10] = {0}; static uint64_t s_tot = 0;
        int b = (channel==SPU_RdInMbox)?0:(channel==SPU_RdSigNotify1)?1:(channel==SPU_RdSigNotify2)?2:
                (channel==SPU_RdDec)?3:(channel==SPU_RdEventStat)?4:(channel==SPU_RdEventMask)?5:
                (channel==MFC_RdTagStat)?6:(channel==MFC_RdAtomicStat)?7:(channel==SPU_RdMachStat)?8:9;
        s_c[b]++;
        if ((++s_tot % 2000000) == 0)
          fprintf(stderr, "[rdch] InMbox=%llu Sig1=%llu Sig2=%llu Dec=%llu EvStat=%llu EvMask=%llu TagStat=%llu AtomStat=%llu MachStat=%llu other=%llu\n",
            (unsigned long long)s_c[0],(unsigned long long)s_c[1],(unsigned long long)s_c[2],(unsigned long long)s_c[3],
            (unsigned long long)s_c[4],(unsigned long long)s_c[5],(unsigned long long)s_c[6],(unsigned long long)s_c[7],
            (unsigned long long)s_c[8],(unsigned long long)s_c[9]); } }

    if (channel_is_mfc(channel)) {
        v = mfc_channel_read(mfc_for(ctx), ctx, channel);
        return spu_make_preferred_u32(v);
    }

    switch (channel) {
    case SPU_RdInMbox:
        if (ctx->rcv_evt_n > 0) { v = ctx->rcv_evt[ctx->rcv_evt_i++]; ctx->rcv_evt_n--; }
        else v = spu_channel_read(&ctx->ch_in_mbox);
        break;
    case SPU_RdSigNotify1:  v = spu_channel_read(&ctx->ch_sig_notify[0]); break;
    case SPU_RdSigNotify2:  v = spu_channel_read(&ctx->ch_sig_notify[1]); break;
    /* The decrementer must actually DECREMENT. Returning the latched WrDec
     * value verbatim makes every `rdch $ch8` deadline poll spin forever: the
     * elapsed-time term is always zero, so the deadline never passes. LBP's
     * wwsjob SPURS policy module hangs on exactly that at LS 0x2D68:
     *
     *     2d6c:  lqa   $2, 0x1530     ; deadline
     *     2d74:  rdch  $5, $ch8       ; now  <- always read 0
     *     2d78:  sf    $6, $2, $5     ; now - deadline
     *     2d80:  cgti  $5, $6, 0
     *     2d88:  binz  $2, $0         ; leave once it goes positive
     *
     * It never leaves, never reaches its first DMA, and blocks the workload.
     * Wraparound on subtract is deliberate -- it is what hardware does, and the
     * `cgti > 0` test above is written to tolerate it. */
    case SPU_RdDec: {
        /* SPU_DECPC=<n>: every n decrementer reads, the host return addresses (module
         * RVAs; resolve against the link map) that read it most -- which lifted
         * function spins on the clock. */
        { static int s_dp = -1; if (s_dp < 0) { const char* e = getenv("SPU_DECPC"); s_dp = e ? atoi(e) : 0; }
          if (s_dp > 0) {
              enum { NR = 4096 };   /* open-addressed on the return address */
              static uintptr_t ra[NR]; static unsigned long long rc[NR], tot; static SRWLOCK lk = SRWLOCK_INIT;
              const uintptr_t r = t_rdch_ra;
              AcquireSRWLockExclusive(&lk);
              unsigned k = (unsigned)((r >> 2) * 2654435761u) & (NR - 1);
              for (unsigned probe = 0; probe < NR; probe++, k = (k + 1) & (NR - 1))
                  if (!rc[k] || ra[k] == r) { ra[k] = r; rc[k]++; break; }
              if (++tot % (unsigned long long)s_dp == 0) {
                  const uintptr_t mb = (uintptr_t)GetModuleHandleA(NULL);
                  fprintf(stderr, "[decpc] %llu reads:", tot);
                  for (unsigned q = 0; q < NR; q++)
                      if (rc[q] * 100 >= tot) fprintf(stderr, " rva 0x%llX=%llu", (unsigned long long)(ra[q] - mb), rc[q]);
                  fputc('\n', stderr);
              }
              ReleaseSRWLockExclusive(&lk);
          } }
        /* Never armed by a WrDec: stamp the base on first read rather than
         * measuring from epoch 0, which would subtract the host's entire
         * uptime and hand back a wild value to code that has every right to
         * read the decrementer before writing it. */
        if (!ctx->dec_base_ns) ctx->dec_base_ns = spu_host_ns();
        uint64_t elapsed = spu_host_ns() - ctx->dec_base_ns;
        /* Split seconds from the remainder before scaling: elapsed_ns * 79.8M
         * overflows 64 bits 231 s after the base stamp, and a context that
         * never writes its decrementer (the SPURS kernel's) keeps the stamp
         * from boot, so every timed loop on that SPU saw the counter jump
         * forward by ~16 s of ticks at 231 s, 462 s, 693 s... */
#if defined(__SIZEOF_INT128__)
        /* Same split, as a 32.32 multiply: ticks per ns = 0.0798 (no divides on
         * this path, which runs millions of times a second). */
        static const uint64_t k_ticks_per_ns = (uint64_t)((SPU_DECREMENTER_HZ << 32) / 1000000000ull);
        uint32_t ticks   = (uint32_t)(((unsigned __int128)elapsed * k_ticks_per_ns) >> 32);
#else
        const uint64_t secs = elapsed / 1000000000ull, rem = elapsed % 1000000000ull;
        uint32_t ticks   = (uint32_t)(secs * SPU_DECREMENTER_HZ +
                                      (rem * SPU_DECREMENTER_HZ) / 1000000000ull);
#endif
        v = ctx->decrementer - ticks;
        break;
    }
    case SPU_RdEventMask:   v = ctx->event_mask;                        break;
    case SPU_RdEventStat:   spu_resv_lost_poll(ctx); v = ctx->event_status; break;
    /* CBEA 9.8: numeric bit 0 is interrupt-enable, independent of run status.
     * This runtime models nonisolated SPUs, so the isolation bit stays clear. */
    case SPU_RdMachStat:    v = ctx->int_enable & 1u; break;
    case SPU_RdSRR0:        v = ctx->srr0;                              break;
    default:
        v = 0;
        break;
    }
    return spu_make_preferred_u32(v);
}

/* ===========================================================================
 * Channel count (rchcnt) -- how many entries can be read/written right now
 * ===========================================================================*/
static uint32_t spu_rchcnt_impl(spu_context* ctx, uint32_t channel);
uint32_t spu_rchcnt(spu_context* ctx, uint32_t channel)
{
    /* Channel counts depend on live state (rchcnt SPU_RdEventStat polls the
     * reservation against guest memory): record each one for SPU_JOB_CAPTURE,
     * serve it from the transcript under SPU_JOB_REPLAY. */
    uint32_t v;
    if (g_spu_jobrep_on > 0 && spu_jobrep_rchcnt(ctx, channel, &v)) return v;
    v = spu_rchcnt_impl(ctx, channel);
    if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
        jc_ev(JC_RCHCNT, channel, v, ctx->event_status, 0, NULL, 0);
    return v;
}
static uint32_t spu_rchcnt_impl(spu_context* ctx, uint32_t channel)
{
    /* SPU_WHOPOLLS=<n>: rchcnt/park accounting PER SPU, printed every n calls
     * with the pc each SPU is sitting at.
     *
     * SPU_CHHIST sums every SPU into one histogram, which is exactly the wrong
     * shape for a deadlock: it showed rchcnt ch0/ch4 climbing past 1.7 billion
     * while MFC_WrTagUpdate (ch23) stayed frozen, i.e. somebody polls forever
     * and somebody else stopped doing DMA -- without saying whether that is one
     * SPU or two. The freeze under investigation is the PPU-side spin in
     * func_000D2298 waiting for spu4 to advance a counter at 0x002DF008, so
     * which SPU is parked, and on what, is the whole question. */
    { static int s_w = -1;
      if (s_w < 0) { const char* e = getenv("SPU_WHOPOLLS");
        s_w = e ? (atoi(e) > 0 ? atoi(e) : 4000000) : 0; }
      if (s_w) { static unsigned long long c[8][40]; static unsigned long long n;
          static uint32_t lastpc[8], lastst[8], lastmask[8];
          static uint32_t lastsig[8][2], lastmb[8], lastrea[8];
          static int lastrv[8], lastrd[8];
          static uint32_t lastline[8][8], lastls[8][8];
          static uint32_t lastlsa[8], lastr90[8];
          const unsigned sp = (unsigned)(ctx->spu_id & 7u);
          if (channel < 40u) c[sp][channel]++;
          lastpc[sp] = (uint32_t)ctx->pc & SPU_LS_MASK;
          lastst[sp] = ctx->event_status; lastmask[sp] = ctx->event_mask;
          lastsig[sp][0] = ctx->ch_sig_notify[0].count;
          lastsig[sp][1] = ctx->ch_sig_notify[1].count;
          lastmb[sp] = ctx->ch_in_mbox.count;
          lastrv[sp] = ctx->resv_valid; lastrea[sp] = ctx->resv_ea;
          lastlsa[sp] = ctx->mfc_lsa & SPU_LS_MASK;
          lastr90[sp] = ctx->gpr[90]._u32[0];
          lastrd[sp] = (ctx->resv_valid && vm_base && ctx->resv_ea)
                     ? (memcmp(vm_base + ctx->resv_ea, ctx->resv_line, 128) != 0)
                     : -1;
          { static int s_ld2 = -2; static uint32_t s_lo2;
            if (s_ld2 == -2) { const char* e5 = getenv("SPU_LSDUMP");
                               s_ld2 = e5 ? 1 : 0;
                               s_lo2 = e5 ? (uint32_t)strtoul(e5,0,16) : 0u; }
            if (s_ld2 && ctx->ls)
                for (int z = 0; z < 8; z++) {
                    const uint8_t* l8 = ctx->ls + ((s_lo2 + z * 4) & SPU_LS_MASK);
                    lastls[sp][z] = ((uint32_t)l8[0] << 24) | ((uint32_t)l8[1] << 16)
                                  | ((uint32_t)l8[2] << 8) | l8[3];
                } }
          for (int z = 0; z < 8; z++) {
              const uint8_t* b8 = ctx->resv_line + z * 4;
              lastline[sp][z] = ((uint32_t)b8[0] << 24) | ((uint32_t)b8[1] << 16)
                              | ((uint32_t)b8[2] << 8) | b8[3];
          }
          if ((++n % (unsigned long long)s_w) == 0) {
              fprintf(stderr, "[whopolls] %llu rchcnt calls\n", n);
              for (unsigned q = 0; q < 8; q++) {
                  int any = 0;
                  for (unsigned k = 0; k < 40; k++) if (c[q][k]) any = 1;
                  if (!any) continue;
                  /* The channel state each poll is actually testing. Counts
                   * alone cannot say whether a poll returns 0 or 1, and that is
                   * the difference between "the SPU is not being told" and "the
                   * SPU is told and ignores it". */
                  fprintf(stderr, "   spu%u pc=0x%05X ev[st=%08X mask=%08X]"
                                  " sig[%u %u] inmbox=%u", q, lastpc[q],
                          lastst[q], lastmask[q], lastsig[q][0], lastsig[q][1],
                          lastmb[q]);
                  /* The reservation is the other half of an evmask=0x400 wait:
                   * spu_resv_lost_poll returns immediately unless resv_valid,
                   * so a cleared reservation is indistinguishable from "nobody
                   * wrote the line" in event_status alone -- and they are
                   * different bugs. diff says whether the line has actually
                   * changed under the snapshot right now. */
                  /* mfc_lsa and r90 alongside the reservation. GETLLAR
                   * copies the line to &ls[mfc_lsa] and to resv_line from the
                   * same source, so a snapshot that disagrees with the local
                   * store can only mean mfc_lsa was not what the SPU asked for
                   * -- and the SPU computes both that LSA and the address its
                   * compare reads from r90. Printing the two together is what
                   * decides it. */
                  fprintf(stderr, " resv[v=%d ea=0x%08X diff=%d]"
                                  " mfc_lsa=0x%05X r90=0x%08X",
                          lastrv[q], lastrea[q], lastrd[q],
                          lastlsa[q], lastr90[q]);
                  /* And the line itself. "The reservation is intact and the
                   * line has not changed" still does not say what the SPU is
                   * waiting FOR; the words do. Snapshot side, so it is exactly
                   * what the SPU last read. */
                  /* SPU_LSDUMP=<hex LS offset>: 8 words of this SPU's local
                   * store. The reserved line says what the SPU can SEE; its LS
                   * says what it BELIEVES. For the spu4 deadlock those are the
                   * two numbers to compare -- its consumed counter is staged at
                   * LS 0x10888 (the counter PUT's lsa 0x10880, +8). */
                  { static int s_ld = -2; static uint32_t s_lo;
                    if (s_ld == -2) { const char* e4 = getenv("SPU_LSDUMP");
                                      s_ld = e4 ? 1 : 0;
                                      s_lo = e4 ? (uint32_t)strtoul(e4,0,16) : 0u; }
                    if (s_ld) {
                        fprintf(stderr, " ls[0x%05X:", s_lo);
                        for (int z = 0; z < 8; z++)
                            fprintf(stderr, " %08X", lastls[q][z]);
                        fprintf(stderr, "]");
                    } }
                  if (lastrv[q]) {
                      fprintf(stderr, " line[");
                      for (int z = 0; z < 8; z++)
                          fprintf(stderr, "%s%08X", z ? " " : "", lastline[q][z]);
                      fprintf(stderr, "]");
                  }
                  for (unsigned k = 0; k < 40; k++) if (c[q][k])
                      fprintf(stderr, " ch%u=%llu", k, c[q][k]);
                  fprintf(stderr, "\n");
              }
              fflush(stderr);
          } } }
    /* SPU_CHHIST also covers rchcnt. It used to instrument only wrch/rdch,
     * which is the one place a parked persistent worker is guaranteed NOT to
     * appear: park_on_empty_inmbox halts from inside THIS function, so a
     * worker that polls rchcnt and parks produced a completely empty channel
     * histogram and read as 'never touches a channel'. */
    { static int s_c = -1;
      if (s_c < 0) { const char* e = getenv("SPU_CHHIST");
        s_c = e ? (atoi(e) > 0 ? atoi(e) : 2000) : 0; }
      if (s_c) { static unsigned long long c[128]; static unsigned long long n;
          c[channel & 127]++;
          if ((++n % (unsigned long long)s_c) == 0) {
              fprintf(stderr, "[chc] %llu rchcnt:%c", n, 10);
              for (int i = 0; i < 128; i++) if (c[i])
                  fprintf(stderr, "   rchcnt ch%-3d %llu%c", i, c[i], 10);
              fflush(stderr); } } }
    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_cnt[8] = {0}; static uint64_t s_total = 0;
        int b = (channel==SPU_RdInMbox)?0:(channel==SPU_RdEventStat)?1:(channel==SPU_RdSigNotify1)?2:
                (channel==SPU_RdSigNotify2)?3:(channel==MFC_RdTagStat)?4:(channel==SPU_WrOutMbox)?5:
                (channel==SPU_WrOutIntrMbox)?6:7;
        s_cnt[b]++;
        if ((++s_total % 2000000) == 0)
          fprintf(stderr, "[pollcnt] InMbox=%llu EvStat=%llu Sig1=%llu Sig2=%llu TagStat=%llu OutMbox=%llu OutIntr=%llu other=%llu (ch last=%u)\n",
                  (unsigned long long)s_cnt[0],(unsigned long long)s_cnt[1],(unsigned long long)s_cnt[2],
                  (unsigned long long)s_cnt[3],(unsigned long long)s_cnt[4],(unsigned long long)s_cnt[5],
                  (unsigned long long)s_cnt[6],(unsigned long long)s_cnt[7], channel); } }
    switch (channel) {
    case SPU_RdInMbox:
        /* Synchronous persistent-worker park: after its handshake the SPU polls
         * the inbound mailbox for PPU commands; if empty and parking is armed,
         * halt (longjmp out of spu_run_with_halt) instead of spinning forever. */
        /* A pending sys_spu_thread_receive_event reply counts as readable
         * mailbox words: the worker checks rchcnt to decide whether its reply
         * has arrived, and parking on an "empty" inbox that actually holds the
         * reply strands it. */
        if (ctx->park_on_empty_inmbox && ctx->rcv_evt_n == 0 &&
            ctx->ch_in_mbox.count == 0) {
            extern void spu_halt(spu_context*);
            spu_halt(ctx);
        }
        return (uint32_t)ctx->rcv_evt_n + ctx->ch_in_mbox.count;           /* readable */
    /* Free slots, CLAMPED. The queue behind each channel is SPU_CHANNEL_CAP
     * deep -- deeper than the hardware depth reported here -- so `count` can
     * exceed the depth and a bare subtraction underflows, telling the SPU it
     * has four billion free slots. That silently converts a bounded mailbox
     * into an unbounded one for any title that polls before writing, which is
     * every SPU-heavy one. */
    case SPU_WrOutMbox:
        return ctx->ch_out_mbox.count >= SPU_MBOX_DEPTH
             ? 0u : (uint32_t)(SPU_MBOX_DEPTH - ctx->ch_out_mbox.count);
    case SPU_WrOutIntrMbox:
        return ctx->ch_out_intr_mbox.count >= SPU_INTR_MBOX_DEPTH
             ? 0u : (uint32_t)(SPU_INTR_MBOX_DEPTH - ctx->ch_out_intr_mbox.count);
    case SPU_RdSigNotify1:   return ctx->ch_sig_notify[0].count;
    case SPU_RdSigNotify2:   return ctx->ch_sig_notify[1].count;
    case MFC_Cmd:            return MFC_QUEUE_DEPTH - mfc_for(ctx)->queue_count;
    /* An enabled event pending, or 0 -- NOT the `default: 1` this used to fall
     * through to. The SPU idiom is `rchcnt SPU_RdEventStat; brnz -> rdch`: it
     * asks whether an event is pending and only commits to the BLOCKING read if
     * the answer is yes. Answering 1 unconditionally lured it into a read that
     * then parked forever, because rdch correctly blocks while
     * (event_status & event_mask) == 0 -- rchcnt and rdch disagreed.
     *
     * ps1_netemu's audio SPU does exactly this at LS 0x8934 and parked at
     * 0x0A5E8 waiting for MFC_LLR_LOST_EVENT on a line nothing was ever going
     * to touch; on hardware it would simply have fallen through to 0x893C and
     * kept working. Same condition as spu_ch_ready's case, including the
     * lost-reservation poll, so the two now agree by construction. */
    case SPU_RdEventStat:
        spu_resv_lost_poll(ctx);
        return (ctx->event_status & ctx->event_mask) != 0;
    case MFC_RdTagStat:      return 1;  /* synchronous: status always ready */
    /* Stall-and-notify status is a single-value channel: count is 1 while a
     * newly-stalled tag is pending to be read, else 0. The wwsjob interrupt
     * handler gates on this (rchcnt $ch25; brz IhcExit) -- returning the old
     * default of 1 made it enter on spurious interrupts with an empty stat,
     * running IhcLoop on a zero mask (clz(0) -> bogus tagId/ack). */
    case MFC_RdListStallStat: return ctx->list_stall_stat ? 1 : 0;
    default:                 return 1;  /* default: channel ready */
    }
}

/* ===========================================================================
 * Indirect-branch dispatch + function registry
 * ===========================================================================*/
typedef void (*spu_fn)(spu_context*);

typedef struct {
    uint32_t addr;
    spu_fn   fn;
    int      image_id;   /* which recompiled image this function belongs to */
} spu_reg_entry;

/* One entry per lifted SPU function across EVERY registered image, and a game
 * that lifts its whole SPU workload set has a lot of them: Yakuza: Dead Souls
 * registers ~170k (cri_audio alone is ~35k, gs_task ~9k, the job binaries ~11k).
 * At 65536 the registry silently truncated every image registered past the cap
 * -- and the SPURS job-chain policy, the job binaries and the Edge geometry
 * task (gs_task) register LAST, so their functions were dropped wholesale and
 * every indirect branch into them fell through to a branch-to-0. Size it for
 * the real workload; the overflow is now loud (see spu_register_function). */
#define SPU_FN_REGISTRY_MAX 262144
static spu_reg_entry s_registry[SPU_FN_REGISTRY_MAX];
static uint32_t s_registry_count = 0;
/* Bumped by every function, task-entry and task-ELF registration: the
 * indirect-branch dispatch cache (t_ibc) empties when it changes. */
static uint32_t s_ibc_epoch = 0;

/* Accessor for the sampling profiler in ppu_loader.cpp. Its address->function
 * map is built from the PPU table alone, which leaves the SPU-lifted bodies as
 * unowned gaps -- and an [entry, next_entry) extent then charges every SPU
 * sample to whichever PPU function happens to precede it. That produced a
 * confident and completely wrong "99% of guest time in one PPU function".
 * Handing the SPU host pointers over lets the map cover both, so the gaps are
 * real entries and the extents mean something. */
uint32_t spu_registry_size(void) { return s_registry_count; }

int spu_registry_entry(uint32_t i, void** host, uint32_t* ls_addr)
{
    if (i >= s_registry_count) return 0;
    if (host)    *host    = (void*)s_registry[i].fn;
    if (ls_addr) *ls_addr = s_registry[i].addr;
    return 1;
}

/* Hash index over the registry. spu_lookup runs on EVERY guest indirect
 * branch -- with the Bink decoder live that is millions of dispatches per
 * second, and the old linear scan (~950 binkspu entries walked per branch)
 * dominated movie playback. Buckets chain in REGISTRATION ORDER (tail
 * append) so the first-registered-match-wins semantics of the linear scan
 * are preserved exactly; the image-id wildcard rules stay in the bucket
 * walk, which is 1-3 entries (same LS addr across overlapping images).
 * Registration is startup-single-threaded; lookups treat the index as
 * read-only. Chain links store index+1 so zero-init means "empty". */
#define SPU_FN_HASH_SIZE 131072   /* power of two, ~= MAX/2 -> load factor ~2 */
static uint32_t s_hash_head[SPU_FN_HASH_SIZE];
static uint32_t s_hash_tail[SPU_FN_HASH_SIZE];
static uint32_t s_hash_next[SPU_FN_REGISTRY_MAX];

static inline uint32_t spu_fn_hash(uint32_t addr)
{
    return ((addr >> 2) * 2654435761u) & (SPU_FN_HASH_SIZE - 1);
}

/* Image currently being registered. SPURS images (kernel/policy/job) overlap in
 * LS, so each registers under a distinct id via spu_begin_image() before calling
 * its (prefixed) spu_recomp_register(). Single-image callers leave it 0. */
static int s_reg_image = 0;
void spu_begin_image(int image_id) { s_reg_image = image_id; }

void spu_register_function(uint32_t addr, spu_fn fn)
{
    if (s_registry_count >= SPU_FN_REGISTRY_MAX) {
        /* Silent truncation here dropped whole late-registered SPU images and
         * cost a multi-round hunt (the branch-to-0 looked like a lifter/overlay
         * bug). Never again: say so, once, loudly, with the number to raise the
         * cap to. */
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                    "[spu] FATAL: SPU function registry full at %u entries "
                    "(SPU_FN_REGISTRY_MAX=%u) -- image %d function 0x%05X and all "
                    "later registrations DROPPED. Raise SPU_FN_REGISTRY_MAX.\n",
                    s_registry_count, (unsigned)SPU_FN_REGISTRY_MAX,
                    s_reg_image, addr);
            fflush(stderr);
        }
        return;
    }
    uint32_t i = s_registry_count;
    s_registry[i].addr = addr;
    s_registry[i].fn = fn;
    s_registry[i].image_id = s_reg_image;
    s_registry_count = i + 1;
    s_ibc_epoch++;
    uint32_t h = spu_fn_hash(addr);
    s_hash_next[i] = 0;
    if (s_hash_head[h] == 0)
        s_hash_head[h] = i + 1;
    else
        s_hash_next[s_hash_tail[h] - 1] = i + 1;
    s_hash_tail[h] = i + 1;
}

/* Report a CROSS-IMAGE match: the registry served a function some other image
 * registered at this address, because one side's image_id was the 0 wildcard.
 *
 * This is a real hazard, reported with evidence by @canersaka (#59): SPURS job
 * chains load job binaries into descriptor-assigned LS slots, so the same binary
 * can sit at different addresses on different rounds. If the lift was fixed-base
 * and the exact match misses, the wildcard can serve ANOTHER image's function at
 * that address -- the job then runs the wrong program end to end, returns
 * cleanly, and the only symptom is something three layers downstream that never
 * happens. Their words: one log line would have saved a day.
 *
 * Logging, not refusing. Refusing the wildcard outright was measured and does
 * NOT survive contact: a dormant task image can register spans overlapping the
 * resident kernel's low LS range, so a legitimate service-to-kernel yield gets
 * falsely rejected. Telling resident from dormant needs residency tracking the
 * registry does not have. So this reports and keeps going.
 *
 * Deduped per (addr, from, to): spu_lookup runs millions of times a second, and
 * a substitution that repeats every dispatch would bury the log it is meant to
 * make readable. */
static void spu_report_cross_image(uint32_t addr, int want, int got)
{
    enum { SEEN_MAX = 64 };
    static struct { uint32_t addr; int want, got; } seen[SEEN_MAX];
    static unsigned n_seen = 0;
    for (unsigned i = 0; i < n_seen; i++)
        if (seen[i].addr == addr && seen[i].want == want && seen[i].got == got)
            return;
    if (n_seen < SEEN_MAX) {
        seen[n_seen].addr = addr; seen[n_seen].want = want; seen[n_seen].got = got;
        n_seen++;
    }
    fprintf(stderr,
            "[spu] CROSS-IMAGE dispatch: LS 0x%05X requested by image %d, served "
            "by image %d's function (id-0 wildcard). If this job was loaded at a "
            "runtime-chosen base, it may be running the WRONG program -- see "
            "issue #59.\n", addr, want, got);
}

static spu_fn spu_lookup_walk(uint32_t addr, int image_id);
spu_fn spu_lookup(uint32_t addr, int image_id)   /* exported: clang-built fast-path dispatch (spu_dispatch_mt.c) needs it */
{
    /* SPU_NOLIFT_IMG=<lo>[-<hi>][,...]: those images have no lifted code, so their
     * blocks run in the interpreter (with SPU_INTERP_UNLIFTED=1) -- an A/B of the
     * lift against the interpreter for one job module. */
    { static int s_nl = -1; static int s_lo[4], s_hi[4];
      if (s_nl < 0) { s_nl = 0; const char* e = getenv("SPU_NOLIFT_IMG");
          while (e && *e && s_nl < 4) { char* p; s_lo[s_nl] = (int)strtol(e, &p, 10);
              s_hi[s_nl] = (*p == '-') ? (int)strtol(p + 1, &p, 10) : s_lo[s_nl]; s_nl++;
              e = (*p == ',') ? p + 1 : NULL; } }
      for (int k = 0; k < s_nl; k++)
          if (image_id >= s_lo[k] && image_id <= s_hi[k]) return NULL; }
    /* Per-thread result cache, misses included: a dispatch asks for the resident
     * task, overlay and base image in turn, and each walk costs two cache misses
     * in the shared index (s_hash_head, s_registry), ~7% of a GoW:A EDGE kernel
     * thread. Entries are stamped with the registry size, so a later
     * registration invalidates them. */
    enum { LC_SIZE = 4096 };
    /* 16-byte entries: address and image share a key word, so the table is
     * 64 KB instead of 96 KB and no entry straddles two cache lines. Keys that
     * do not fit go straight to the walk. */
    static SPU_THREAD_LOCAL struct { uint32_t key; uint32_t gen; spu_fn fn; } t_lc[LC_SIZE];
    if (__builtin_expect(addr >= SPU_LS_SIZE || (uint32_t)image_id >= 0x4000u, 0))
        return spu_lookup_walk(addr, image_id);
    const uint32_t key = addr | (uint32_t)image_id << 18;
    const uint32_t gen = s_registry_count;
    const uint32_t slot = ((addr >> 2) ^ ((uint32_t)image_id * 0x9E37u)) & (LC_SIZE - 1);
    if (t_lc[slot].key == key && t_lc[slot].gen == gen)
        return t_lc[slot].fn;
    spu_fn fn = spu_lookup_walk(addr, image_id);
    t_lc[slot].key = key; t_lc[slot].gen = gen; t_lc[slot].fn = fn;
    return fn;
}

static spu_fn spu_lookup_walk(uint32_t addr, int image_id)
{
    /* Match the context's active image; image_id 0 (context or entry) matches
     * any, for back-compat with single-image contexts. */
    for (uint32_t n = s_hash_head[spu_fn_hash(addr)]; n; n = s_hash_next[n - 1]) {
        const spu_reg_entry* e = &s_registry[n - 1];
        if (e->addr == addr &&
            (image_id == 0 || e->image_id == 0 || e->image_id == image_id)) {
            /* One compare on the return path; the report itself is rare. */
            if (e->image_id != image_id)
                spu_report_cross_image(addr, image_id, e->image_id);
            return e->fn;
        }
    }
    return NULL;
}

/* Which image registered a function at this LS address, or -1 if none did.
 *
 * Deliberately NOT spu_lookup(addr, 0): that call means "dispatch here in a
 * context with no image", so it reports a cross-image substitution for every
 * hit in a real image. This is a question about the registry, not a dispatch,
 * and it runs once per SPU thread start. First match wins, in registration
 * order, so it agrees with which function spu_lookup would actually serve. */
int spu_image_of_function(uint32_t addr)
{
    for (uint32_t n = s_hash_head[spu_fn_hash(addr)]; n; n = s_hash_next[n - 1]) {
        const spu_reg_entry* e = &s_registry[n - 1];
        if (e->addr == addr) return e->image_id;
    }
    return -1;
}

/* Does a lifted function exist at this LS address (any image)? Lets the lv2
 * layer decide between real SPU execution and the PPU-fallback paths. */
int spu_have_function(uint32_t addr)
{
    return spu_image_of_function(addr) >= 0;
}

/* The pure interpreter's fast-path "is this LSA already lifted?" probe
 * (spu_interp.c, rejoin path). It IS the registry lookup, so overlay eviction and
 * self-modifying-code invalidation are honored automatically. Lived in
 * spu_fn_registry.c before the registry was consolidated into this TU. */
spu_lifted_fn spu_lifted_lookup(const spu_context* ctx, uint32_t lsa)
{
    if (!ctx) return (spu_lifted_fn)spu_lookup(lsa, 0);
    /* Same resolution order as spu_indirect_branch: resident code spans, the resident
     * task, the resident overlay, then the context's own image. Checking only the
     * image kept the interpreter running through a whole streamed job module after a
     * single unlifted branch target (GoW:A: one missing entry in a WWS job cost the
     * rest of that job, interpreted). */
    for (unsigned slot = 0; slot < SPU_RESIDENT_CODE_SLOTS; ++slot)
        if (ctx->resident_code[slot].image_id && lsa >= ctx->resident_code[slot].lsa &&
            lsa - ctx->resident_code[slot].lsa < ctx->resident_code[slot].size) {
            const uint32_t canon = ctx->resident_code[slot].canon;
            if (!canon) return (spu_lifted_fn)spu_lookup(lsa, ctx->resident_code[slot].image_id);
            /* Relocatable code: the caller runs the function next, so set its delta now. */
            spu_fn f = spu_lookup(lsa - ctx->resident_code[slot].lsa + canon, ctx->resident_code[slot].image_id);
            if (f) ((spu_context*)ctx)->reloc_delta = (int32_t)(ctx->resident_code[slot].lsa - canon);
            return (spu_lifted_fn)f;
        }
    spu_fn fn = NULL;
    if (ctx->resident_task) fn = spu_lookup(lsa, ctx->resident_task);
    if (!fn && ctx->resident_ovl) fn = spu_lookup(lsa, ctx->resident_ovl);
    if (!fn) fn = spu_lookup(lsa, ctx->image_id);
    return (spu_lifted_fn)fn;
}

/* ---- Swappable code overlays (see spu_context.resident_ovl) --------------
 * A title registers each runtime-streamed overlay's SOURCE content EA with
 * the image id its lifted functions were registered under. The MFC GET path
 * marks that overlay resident in the streaming context; dispatch retries a
 * primary-image miss against the resident overlay's registry. */
typedef struct { uint32_t src_ea; int image_id; uint8_t sig[16]; int has_sig; uint32_t span; uint32_t lsa; uint32_t canon; } spu_ovl_src;
/* 6 FMOD codec/DSP overlays + up to 89 WWS job-code modules (all stream into
 * the same job code buffer at LS 0x4000, dispatched by content signature). */
/* GoW:A's EDGE jobs alone stage 360 (fragment, LSA) placements, one image each. */
#define SPU_OVL_SRC_MAX 2048
static spu_ovl_src s_ovl_src[SPU_OVL_SRC_MAX];
static int s_ovl_src_count = 0;
/* Every MFC GET looks itself up here, so the table is indexed: source-EA entries
 * by a hash of the EA (chains hold index+1, 0 ends), signature entries in a list.
 * Entries are only added at registration time, before the SPUs run. */
#define SPU_OVL_HASH 4096u
static int s_ovl_head[SPU_OVL_HASH];
static int s_ovl_next[SPU_OVL_SRC_MAX];
static int s_ovl_sig[SPU_OVL_SRC_MAX];
static int s_ovl_sig_count = 0;
static inline unsigned spu_ovl_hash(uint32_t ea) { return ((ea >> 4) ^ (ea >> 16)) & (SPU_OVL_HASH - 1u); }

void spu_overlay_register_source(uint32_t content_ea, int image_id)
{
    if (s_ovl_src_count < SPU_OVL_SRC_MAX) {
        s_ovl_src[s_ovl_src_count].src_ea = content_ea;
        s_ovl_src[s_ovl_src_count].image_id = image_id;
        s_ovl_src[s_ovl_src_count].has_sig = 0;
        s_ovl_src[s_ovl_src_count].span = 0;
        s_ovl_src[s_ovl_src_count].lsa = 0;
        s_ovl_src[s_ovl_src_count].canon = 0;
        const unsigned h = spu_ovl_hash(content_ea);
        s_ovl_next[s_ovl_src_count] = s_ovl_head[h];
        s_ovl_head[h] = s_ovl_src_count + 1;
        s_ovl_src_count++;
    }
}

/* Register a bounded code image that may coexist with other streamed images.
 * Its functions must be translated at the local-store addresses used by the title. */
void spu_overlay_register_region(uint32_t content_ea, uint32_t span, int image_id)
{
    if (!span || span > SPU_LS_SIZE || s_ovl_src_count >= SPU_OVL_SRC_MAX) return;
    spu_overlay_register_source(content_ea, image_id);
    s_ovl_src[s_ovl_src_count - 1].span = span;
}

/* The same, for code that is translated once per local-store placement: a fragment
 * the title stages at several addresses (GoW:A's EDGE jobs pack EBOOT code fragments
 * one after another from LS 0x11090) gets one lifted image per (source EA, LSA). */
void spu_overlay_register_region_at(uint32_t content_ea, uint32_t lsa, uint32_t span, int image_id)
{
    if (!lsa || lsa + span > SPU_LS_SIZE) return;
    spu_overlay_register_region(content_ea, span, image_id);
    if (s_ovl_src[s_ovl_src_count - 1].image_id == image_id) s_ovl_src[s_ovl_src_count - 1].lsa = lsa;
}

/* And for position-independent code the title copies to a different LS address at each
 * load (GoW:A's BatchJobList jobs place four EBOOT blobs at a new 16-byte-aligned address
 * every time): lifted ONCE at a canonical base with spu_lifter.py --reloc. A GET of the
 * content at any address makes that copy resident; dispatch then looks the function up at
 * canon + offset and sets ctx->reloc_delta, which the lifted code adds to every address
 * it derives from its own pc. */
void spu_overlay_register_region_reloc(uint32_t content_ea, uint32_t span, uint32_t canon, int image_id)
{
    if (!canon || canon + span > SPU_LS_SIZE) return;
    { static int s_off = -1; if (s_off < 0) { const char* e = getenv("SPU_RELOC_OFF"); s_off = e && e[0] == '1'; }
      if (s_off) return; }   /* SPU_RELOC_OFF=1: leave that code to the interpreter (A/B) */
    spu_overlay_register_region(content_ea, span, image_id);
    if (s_ovl_src[s_ovl_src_count - 1].image_id == image_id) s_ovl_src[s_ovl_src_count - 1].canon = canon;
}

/* Register one-way runtime entries that replace the guest call stack.
 * Matching the resolved function, not just its LS address, avoids affecting
 * unrelated overlays using the same address. Registration precedes execution. */
static struct { uint32_t entry; int image_id; } s_stack_reset[16];
unsigned g_spu_stack_reset_count;   /* spu_drain.c skips the check at 0 */
/* Nonzero at (entry >> 2) & 1023 of every registered entry, so SPU_DIRECT
 * (spu_context.h) can tell with one load that a transfer needs the drain's
 * stack-reset check. Collisions only cost a trampoline step. */
uint8_t g_spu_stack_reset_hint[1024];
void spu_register_stack_reset_entry(uint32_t entry, int image_id)
{
    if (g_spu_stack_reset_count < 16) {
        s_stack_reset[g_spu_stack_reset_count].entry = entry & SPU_LS_MASK;
        s_stack_reset[g_spu_stack_reset_count++].image_id = image_id;
        g_spu_stack_reset_hint[((entry & SPU_LS_MASK) >> 2) & 1023u] = 1;
    }
}

/* Also called before direct trampoline transfers: generated calls can enter
 * a stack-switching routine without going through spu_indirect_branch. */
void spu_check_stack_reset(spu_context* ctx, void (*fn)(spu_context*))
{
    if (!g_spu_stack_reset_count || !ctx->host_depth || !s_spu_halt_armed || ctx->policy_mode) return;
    for (unsigned i = 0; i < g_spu_stack_reset_count; ++i)
        if (ctx->pc == s_stack_reset[i].entry &&
            fn == spu_lookup(ctx->pc, s_stack_reset[i].image_id)) {
            g_spu_trampoline_fn = 0;
            SPU_LONGJMP(s_spu_halt_env, 2);
        }
}

/* SPURS taskset TASK entries (see spu_context.resident_task). A taskset can hold
 * several tasks whose lifts share the SAME LS base -- the co-resident task-code
 * region -- so no LS address identifies which task owns it. The title registers
 * each task's ELF ENTRY point with the image id its functions were registered
 * under; spu_indirect_branch adopts that image the moment the policy branches
 * into the entry from outside the region. This is what lets each such task be
 * registered under its own real image id instead of the id-0 wildcard, so a
 * dormant co-resident task can no longer shadow the one the policy launched. */
#define SPU_TASK_ENTRY_MAX 16
static struct { uint32_t entry; int image_id; } s_task_entry[SPU_TASK_ENTRY_MAX];
static int s_task_entry_count = 0;
void spu_taskset_register_task_entry(uint32_t entry, int image_id)
{
    if (s_task_entry_count < SPU_TASK_ENTRY_MAX) {
        s_task_entry[s_task_entry_count].entry = entry & SPU_LS_MASK;
        s_task_entry[s_task_entry_count].image_id = image_id;
        s_task_entry_count++;
        s_ibc_epoch++;
    }
}
static int spu_taskset_task_image(uint32_t entry)
{
    for (int i = 0; i < s_task_entry_count; i++)
        if (s_task_entry[i].entry == entry) return s_task_entry[i].image_id;
    return 0;
}
/* Resume PCs and even fresh entry PCs overlap between task ELFs. The
 * taskset policy's TaskInfo identifies which ELF it has actually loaded. */
static struct { uint32_t elf_ea; int image_id, policy_image_id; }
    s_task_elf[SPU_TASK_ENTRY_MAX];
static int s_task_elf_count;
void spu_taskset_register_task_elf(uint32_t elf_ea, int image_id, int policy_image_id)
{
    if (s_task_elf_count < SPU_TASK_ENTRY_MAX) {
        s_task_elf[s_task_elf_count].elf_ea = elf_ea & ~7u;
        s_task_elf[s_task_elf_count].image_id = image_id;
        s_task_elf[s_task_elf_count++].policy_image_id = policy_image_id;
        s_ibc_epoch++;
    }
}
static int spu_taskset_resident_image(const spu_context* ctx)
{
    const uint8_t* p = ctx->ls + 0x2794; /* SpursTasksetContext.taskInfo.elf */
    uint32_t elf = (((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                    ((uint32_t)p[2] << 8) | p[3]) & ~7u;
    for (int i = 0; i < s_task_elf_count; ++i)
        if (s_task_elf[i].elf_ea == elf &&
            s_task_elf[i].policy_image_id == ctx->resident_ovl)
            return s_task_elf[i].image_id;
    return 0;
}
/* Lowest LS address of the shared task-code region: the taskset tasks all lift
 * at LS 0x3000 (below it is the SPURS kernel/policy/context, 0x290..0x2FFF). */
#define SPU_TASKSET_TASK_LO 0x3000u

/* Content-signature variant: FMOD COPIES codec overlays to the heap before
 * streaming them into the swap slot, so the source EA is unknowable ahead of
 * time -- match the first 16 bytes of the streamed content instead. */
void spu_overlay_register_sig(const uint8_t sig[16], int image_id)
{
    if (s_ovl_src_count < SPU_OVL_SRC_MAX) {
        s_ovl_src[s_ovl_src_count].src_ea = 0;
        s_ovl_src[s_ovl_src_count].image_id = image_id;
        memcpy(s_ovl_src[s_ovl_src_count].sig, sig, 16);
        s_ovl_src[s_ovl_src_count].has_sig = 1;
        s_ovl_sig[s_ovl_sig_count++] = s_ovl_src_count;
        s_ovl_src_count++;
    }
}

/* Called from the MFC GET path after the copy: ls points at the JUST-COPIED
 * bytes. EA match first (exact, cheap), then content signature for sizeable
 * chunks (overlay bodies are >= 0x500 bytes). */
/* SPU_DUMP_OVL=<dir>[,<lsa>]: write out every distinct code body streamed into
 * LS <lsa> (default 0x5000, the job code buffer), so job bodies the title
 * keeps in its own data can be lifted. A body larger than one DMA arrives as
 * consecutive GETs (contiguous EA and LS); those are appended to the same
 * file. One file per source EA: ovl_<ea>.bin. */
static void spu_ovl_dump(spu_context* ctx, uint32_t ea, uint32_t lsa, const uint8_t* ls, uint32_t size)
{
    static int s_on = -1; static char s_dir[400]; static uint32_t s_lsa = 0x5000;
    if (s_on < 0) { const char* e = getenv("SPU_DUMP_OVL"); s_on = e && *e;
        if (s_on) { snprintf(s_dir, sizeof s_dir, "%s", e);
                    char* c = strchr(s_dir, ','); if (c) { *c = 0; s_lsa = (uint32_t)strtoul(c + 1, 0, 0); } } }
    if (!s_on) return;
    static uint32_t s_seen[256]; static unsigned s_n;
    static SPU_TLS uint32_t t_src, t_next_ea, t_next_lsa;
    FILE* f = NULL; char path[512];
    if (lsa == s_lsa && size >= 256) {
        for (unsigned i = 0; i < s_n; i++) if (s_seen[i] == ea) { t_src = 0; return; }
        if (s_n < 256) s_seen[s_n++] = ea;
        t_src = ea;
        snprintf(path, sizeof path, "%s/ovl_%08X.bin", s_dir, ea);
        f = fopen(path, "wb");
        fprintf(stderr, "[spu-ovl-dump] img=%d body ea=0x%08X size=%u -> %s\n", ctx->image_id, ea, size, path);
    } else if (t_src && ea == t_next_ea && lsa == t_next_lsa) {
        snprintf(path, sizeof path, "%s/ovl_%08X.bin", s_dir, t_src);
        f = fopen(path, "ab");
    } else { t_src = 0; return; }
    if (f) { fwrite(ls, 1, size, f); fclose(f); }
    t_next_ea = ea + size; t_next_lsa = lsa + size;
}

/* SPU_LOCKGUARD=<hex ea>: the word at that EA is a ticket lock (serving,next
 * halfwords). Log any SPU write that moves either halfword by anything other
 * than 0 or +1 -- a stale whole-line store rolling the lock back. `src` is the
 * new bytes for [ea, ea+size). */
void spu_lockguard_check(spu_context* ctx, uint32_t ea, const uint8_t* src, uint32_t size, const char* what)
{
    static int64_t s_g = -2;
    if (s_g == -2) { const char* e = getenv("SPU_LOCKGUARD"); s_g = e ? (int64_t)strtoul(e, 0, 16) : -1; }
    if (s_g < 0 || !vm_base) return;
    uint32_t g = (uint32_t)s_g;
    if (g < ea || g + 4 > ea + size) return;
    const uint8_t* o = vm_base + g; const uint8_t* n = src + (g - ea);
    unsigned os = (o[0] << 8) | o[1], on = (o[2] << 8) | o[3];
    unsigned ns = (n[0] << 8) | n[1], nn = (n[2] << 8) | n[3];
    int bad = !(ns == os || ns == ((os + 1) & 0xFFFF)) || !(nn == on || nn == ((on + 1) & 0xFFFF));
    if (bad) { static int c; if (PS3_LOG_FIRST(c, 32))
        fprintf(stderr, "[lockguard] %s img=%d pc=0x%05X ea=0x%08X size=%u: %04X/%04X -> %04X/%04X\n",
                what, ctx ? ctx->image_id : -1, ctx ? (unsigned)(ctx->pc & SPU_LS_MASK) : 0u,
                ea, size, os, on, ns, nn); }
}

void spu_overlay_note_get(spu_context* ctx, uint32_t ea, const uint8_t* ls, uint32_t size)
{
    uint32_t lsa = (uint32_t)(ls - ctx->ls);
    spu_ovl_dump(ctx, ea, lsa, ls, size);
    /* SPU_LSA_GETLOG=<lo>-<hi> (hex LS): each distinct (lsa, ea, size) GET landing in
     * that local-store range, with the workload and the first words -- which blob
     * is staged where (code a job loads outside its module image). */
    /* SPU_LSA_GETLOG_EA=<lo>-<hi> keeps only sources in that EA range (e.g. the EBOOT). */
    { static int s_gl = -1; static uint32_t s_glo, s_ghi, s_elo, s_ehi = 0xFFFFFFFFu;
      if (s_gl < 0) { const char* e = getenv("SPU_LSA_GETLOG"); s_gl = 0;
          if (e) { char* p; s_glo = (uint32_t)strtoul(e, &p, 16); s_ghi = (*p == '-') ? (uint32_t)strtoul(p + 1, 0, 16) : s_glo; s_gl = 1; }
          const char* x = getenv("SPU_LSA_GETLOG_EA");
          if (x) { char* p; s_elo = (uint32_t)strtoul(x, &p, 16); if (*p == '-') s_ehi = (uint32_t)strtoul(p + 1, 0, 16); } }
      if (s_gl && lsa < s_ghi && lsa + size > s_glo && ea >= s_elo && ea < s_ehi) {
          static uint32_t seen[2048][3]; static int ns; static SRWLOCK lk = SRWLOCK_INIT;
          AcquireSRWLockExclusive(&lk);
          int f = 0;
          for (int k = 0; k < ns; k++) if (seen[k][0] == lsa && seen[k][1] == ea && seen[k][2] == size) { f = 1; break; }
          if (!f && ns < 2048) {
              seen[ns][0] = lsa; seen[ns][1] = ea; seen[ns][2] = size; ns++;
              fprintf(stderr, "[lsa-get] wid=%d img=%d lsa=0x%05X ea=0x%08X size=0x%X data=%02X%02X%02X%02X %02X%02X%02X%02X\n",
                      ctx->policy_mode ? ctx->ls[0x1DF] : -1, ctx->image_id, lsa, ea, size,
                      ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7]);
          }
          ReleaseSRWLockExclusive(&lk);
      } }
    for (uint32_t m = ctx->resident_code_mask; m; m &= m - 1) {
        const unsigned slot = (unsigned)__builtin_ctz(m);
        uint32_t base = ctx->resident_code[slot].lsa;
        uint32_t end = base + ctx->resident_code[slot].size;
        if (lsa < end && lsa + size > base &&
            (lsa < base || ea != ctx->resident_code[slot].source_ea + (lsa - base))) {
            ctx->resident_code[slot].image_id = 0;
            ctx->resident_code_mask &= ~(1u << slot);
        }
    }
    /* The first registered entry that matches wins, as in a scan in order. */
    int best = -1;
    if (size >= 512)
        for (int k = 0; k < s_ovl_sig_count; k++)
            if (memcmp(ls, s_ovl_src[s_ovl_sig[k]].sig, 16) == 0) { best = s_ovl_sig[k]; break; }
    for (int j = s_ovl_head[spu_ovl_hash(ea)]; j; j = s_ovl_next[j - 1]) {
        const spu_ovl_src* o = &s_ovl_src[j - 1];
        if (!o->has_sig && o->src_ea == ea && (!o->lsa || o->lsa == lsa) && (best < 0 || j - 1 < best))
            best = j - 1;
    }
    if (best >= 0) {
        const spu_ovl_src* o = &s_ovl_src[best];
        {
            if (o->span) {
                if (lsa + o->span > SPU_LS_SIZE) return;
                for (unsigned slot = 0; slot < SPU_RESIDENT_CODE_SLOTS; ++slot) {
                    /* A relocatable image takes over its own previous placement's slot: each load
                     * lands somewhere new, and stale copies would fill the slots. */
                    if (ctx->resident_code[slot].image_id && ctx->resident_code[slot].lsa != lsa &&
                        !(o->canon && ctx->resident_code[slot].image_id == o->image_id))
                        continue;
                    ctx->resident_code[slot].lsa = lsa;
                    ctx->resident_code[slot].size = o->span;
                    ctx->resident_code[slot].source_ea = ea;
                    ctx->resident_code[slot].image_id = o->image_id;
                    ctx->resident_code[slot].canon = o->canon;
                    if (o->image_id) ctx->resident_code_mask |= 1u << slot;
                    else ctx->resident_code_mask &= ~(1u << slot);
                    return;
                }
                fprintf(stderr, "[spu-ovl] no free resident code span for image %d\n", o->image_id);
                return;
            }
            if (ctx->resident_ovl != o->image_id) {
                ctx->resident_ovl = o->image_id;
                { static int _n = 0; if (PS3_LOG_FIRST(_n, 32))
                    fprintf(stderr, "[spu-ovl] img=%d streamed overlay src=0x%08X "
                            "-> LS 0x%05X size=%u -> resident ovl image %d%s\n",
                            ctx->image_id, ea, (uint32_t)(ls - ctx->ls), size,
                            ctx->resident_ovl, o->has_sig ? " (sig match)" : ""); }
            }
            return;
        }
    }
}

/* Match a 16-byte module header against the registered content signatures.
 * Returns the overlay image id, or 0 if none. Used by the WWS job-code dispatch
 * path: the job manager streams a module into the code buffer via a DMA LIST
 * (per-element GETs the single-transfer note_get hook doesn't see as one >=512
 * chunk), so resident_ovl is set at branch-in time by matching the buffer head. */
int spu_overlay_match_sig(const uint8_t hdr[16])
{
    for (int i = 0; i < s_ovl_src_count; i++) {
        const spu_ovl_src* o = &s_ovl_src[i];
        if (o->has_sig && memcmp(hdr, o->sig, 16) == 0)
            return o->image_id;
    }
    return 0;
}

/* HLE of the taskset Policy Module's task-syscall entry (LS 0xA70). A SPURS task
 * (e.g. the cri_mpv task, image 22) reads syscallAddr from its SpursTasksetContext
 * (LS 0x27C4) and branches to it to perform a task syscall (EXIT/YIELD/WAIT/POLL).
 * The real kernel has the PM code resident at 0xA70; we don't, so we plant 0xA70 as
 * syscallAddr (in the cri dispatch) and INTERCEPT a branch to it here to HLE the
 * syscall. num = r3&0xF (0x10 bit = the "2" variant), args in r4. Adopted from the
 * JonathanDC64/ps3recomp fork (aaea4158) which uses this to run SPURS tasks clean. */
/* Write the event bits a PPU-side Set owed this wait object into guest memory,
 * at object+0x00 (events) and every object+0x30 slot (pendingRecv).
 *
 * Called from the WAIT_SIGNAL handler immediately before control returns to the
 * guest. Doing it at Set time loses a race: the guest reads object+0x30 the
 * instant it resumes, and a task that took the latched-wake path was never
 * parked for the Set to target. Here the guest demonstrably has not read yet.
 *
 * ponytail: fills all 16 slots rather than the one the guest picked -- the slot
 * index is derived from state we do not model, and a wrong slot delivers
 * nothing. Narrow it if a title ever cares which slot fired. */
static void spu_ef_deliver_owed(spu_context* ctx, uint32_t wobj)
{
    (void)ctx;
    static int s_od = -1;
    if (s_od < 0) s_od = getenv("SPURS_EF_OBJ_DELIVER") ? 1 : 0;
    if (!s_od || !wobj || !vm_base) return;
    extern uint16_t spu_ef_bits_take(uint32_t);
    uint16_t bits = spu_ef_bits_take(wobj);
    if (!bits) return;
    uint8_t* m = vm_base + wobj;
    m[0] = (uint8_t)(bits >> 8); m[1] = (uint8_t)bits;          /* +0x00 events */
    for (uint32_t s = 0; s < 16; s++) {                          /* +0x30 slots */
        m[0x30 + 2*s]     = (uint8_t)(bits >> 8);
        m[0x30 + 2*s + 1] = (uint8_t)bits;
    }
    { static int _n = 0; if (PS3_LOG_FIRST(_n, 8))
        fprintf(stderr, "[spu] delivered owed bits=0x%04X to object 0x%08X \n",
                (unsigned)bits, wobj); }
}

void spu_spurs_taskset_syscall(spu_context* ctx)   /* non-static: also called by the pure interpreter (spu_interp.c) */
{
    uint32_t raw = ctx->gpr[3]._u32[0];
    uint32_t num = raw & 0x0F;
    /* SPU_SYSCALL_IMG=<n>: every syscall of image n, uncapped (the 24-line
     * budget is spent by FMOD's WAIT_SIGNAL loop long before a later task). */
    static int s_si = -2;
    if (s_si == -2) { const char* e = getenv("SPU_SYSCALL_IMG"); s_si = e ? atoi(e) : -1; }
    { static int _n = 0; if (PS3_LOG_FIRST(_n, 24) || ctx->image_id == s_si)
        fprintf(stderr, "[spu] SPURS taskset syscall num=%u (raw=0x%X args=0x%08X) image=%d link/r0=0x%05X wobj@2FDC=0x%02X%02X%02X%02X\n",
                num, raw, ctx->gpr[4]._u32[0], ctx->image_id, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                ctx->ls[0x2FDC], ctx->ls[0x2FDD], ctx->ls[0x2FDE], ctx->ls[0x2FDF]); }
    /* The task-API argument is passed in LOCAL STORE at 0x2FD0, not in r4: the
     * caller does shufb(arg,...) -> stqd 0x2FD0 and only then sets r3 = number
     * (see func_00026DE0 / func_000272AC in image 22). Logging r4 shows caller
     * leftovers and hides what the task actually asked for. */
    { static int s_sa = -1; if (s_sa < 0) s_sa = getenv("YDKJ_SYSCALL_ARG") ? 1 : 0;
      if (s_sa) { static int _n = 0; if (PS3_LOG_FIRST(_n, 40)) {
        const uint8_t* a = &ctx->ls[0x2FD0];
        fprintf(stderr, "[spu] syscall num=%u LS[0x2FD0]=%02X%02X%02X%02X %02X%02X%02X%02X"
                        " %02X%02X%02X%02X %02X%02X%02X%02X img=%d\n",
                num, a[0],a[1],a[2],a[3], a[4],a[5],a[6],a[7],
                a[8],a[9],a[10],a[11], a[12],a[13],a[14],a[15], ctx->image_id);
        fflush(stderr); } } }
    /* NOTE (YDKJ cri_mpv): the cri task's BOOTSTRAP (func_00003040) calls the
     * task-API syscall and EXPECTS IT TO RETURN, then branches to the real task
     * entry (0x3050). Halting on num=0 here kills the task at bootstrap before it
     * runs. So for image 22 we DON'T halt on num=0 -- we return so the bootstrap
     * continues to the decode entry. (A genuine end-of-task EXIT would re-enter and
     * spin; if that happens, gate a real halt after the task has done work.)
     * For non-cri images keep the fork's EXIT=halt semantics. Env YDKJ_CRI_EXIT_HALT
     * forces the old halt behaviour for comparison. */
    /* The cri bootstrap calls EXIT once and EXPECTS IT TO RETURN, then branches to
     * the real task entry -- so image 22 cannot halt on the first one. But a task
     * that has since done work and calls EXIT again means it: returning there makes
     * it re-enter and spin (observed: 6x num=0 from link 0x26E18 once the task is
     * woken). Honour the first call, halt on the rest. Each task runs on its own
     * host thread, so a thread-local count is per task. */
    static _Thread_local int s_exit_seen = 0;
    if (num == 0 && ctx->image_id == 22 && !getenv("YDKJ_CRI_EXIT_HALT")) {
        if (s_exit_seen++ == 0) { ctx->gpr[3]._u32[0] = 0; return; }   /* bootstrap */
        { static int _n = 0; if (PS3_LOG_FIRST(_n, 8))
            fprintf(stderr, "[spu] cri task EXIT #%d -- halting (was spinning)\n",
                    s_exit_seen); }
        ctx->status = SPU_STATUS_STOPPED_BY_STOP;
        spu_halt(ctx);
        return;
    }
    if (num == 0 && (ctx->image_id != 22 || getenv("YDKJ_CRI_EXIT_HALT"))) {
        ctx->status = SPU_STATUS_STOPPED_BY_STOP;
        spu_halt(ctx);          /* longjmp out to spu_run_with_halt; post-run writes exit code */
        return;
    }
    /* WAIT_SIGNAL(2): REAL semantics (RPCS3 spursTasksetProcessSyscall) --
     * consume the task's bit in the guest taskset's `signalled` bitset, or
     * SLEEP this task's host thread until _cellSpursSendSignal /
     * cellSpursEventFlagSet delivers one (the FMOD mixer's flag-A wait path:
     * the task registers its wait slot in the flag struct with atomics, then
     * syscalls WAIT_SIGNAL; the PPU-side Set satisfies the slot and signals).
     * Returning "success" here without waiting made the task spin on empty
     * work state forever -- the LBP boot deadlock. taskset/taskId come from
     * the SpursTasksetContext this runtime planted at LS 0x2700. */
    if (num == 2) {
        uint32_t ts  = ((uint32_t)ctx->ls[0x27BC] << 24) | ((uint32_t)ctx->ls[0x27BD] << 16) |
                       ((uint32_t)ctx->ls[0x27BE] << 8)  |  (uint32_t)ctx->ls[0x27BF];
        uint32_t tid = ((uint32_t)ctx->ls[0x27D4] << 24) | ((uint32_t)ctx->ls[0x27D5] << 16) |
                       ((uint32_t)ctx->ls[0x27D6] << 8)  |  (uint32_t)ctx->ls[0x27D7];
        /* The guest names its wait OBJECT in the task-API argument at LS 0x2FD0
         * (low nibble is flags). It reads its received event bits from
         * object+0x30, so this is both who to wake and where to deliver. */
        uint32_t wobj = (((uint32_t)ctx->ls[0x2FDC] << 24) |
                         ((uint32_t)ctx->ls[0x2FDD] << 16) |
                         ((uint32_t)ctx->ls[0x2FDE] << 8)  |
                          (uint32_t)ctx->ls[0x2FDF]) & ~0xFu;
        /* SPURS_EF_SPU_REPLY=1 -- DIAGNOSTIC BISECT, NOT A FIX.
         *
         * The guest task is supposed to set its SPU->PPU flag (object+0x100)
         * when a work cycle completes; it never does, so the PPU blocks on that
         * flag forever and can never enqueue the next unit of work. That is a
         * cycle: we cannot see what the PPU would do next without breaking it.
         *
         * On RE-ENTRY to WAIT_SIGNAL for the same object the previous cycle has
         * finished, so set the flag on the task`s behalf and watch where the PPU
         * goes. This FABRICATES guest state -- the repo`s own history says faked
         * success is the most expensive kind of bug (see the unresolved-NID note
         * in ppu_hle.cpp), so it stays opt-in and must never become the default.
         * If the PPU advances, the answer is what the task must do to earn it. */
        { static int s_rp = -1;
          if (s_rp < 0) { const char* e = getenv("SPURS_EF_SPU_REPLY");
                          s_rp = e ? atoi(e) : 0; }   /* 2 = reply on EVERY entry */
          if (s_rp && wobj) {
              static uint32_t seen[8]; static int seen_n = 0;
              int again = 0;
              for (int i = 0; i < seen_n; i++) if (seen[i] == wobj) { again = 1; break; }
              if (s_rp >= 2) again = 1;   /* force: answer "would the PPU move at all?" */
              if (!again) { if (seen_n < 8) seen[seen_n++] = wobj; }
              else {
                  extern void spurs_ef_set_from_spu(uint32_t, uint16_t);
                  spurs_ef_set_from_spu(wobj + 0x100u, 1);
                  static int _n = 0;
                  if (PS3_LOG_FIRST(_n, 8))
                      fprintf(stderr, "[spu] SPU-REPLY: set flag 0x%08X for object "
                                      "0x%08X (cycle complete)\n", wobj + 0x100u, wobj);
              }
          } }

        if (ts) {
            extern int spu_taskset_wait_signal(uint32_t, uint32_t);
            /* Park = OS-level wait on this SPU's host thread. Under the lockstep
             * gate the thread HOLDS the global run token here; parking without
             * releasing it starves every other lifted SPU (observed: FMOD task 1
             * parks in WAIT_SIGNAL holding the token -> the mixer never runs
             * again -> the PPU audio pump blocks forever on flag 0x94F600). */
            /* Publish that this task is parked so a PPU-side event-flag Set can
             * reach it even though it registered no wait slot in the flag. */
            extern void spu_taskset_parked_add(uint32_t, uint32_t, uint32_t);
            extern void spu_taskset_parked_del(uint32_t, uint32_t);
            extern int spu_taskset_consume_wake(uint32_t);
            if (spu_taskset_consume_wake(ts)) {   /* a Set beat us to the park */
                spu_ef_deliver_owed(ctx, wobj);
                ctx->gpr[3]._u32[0] = 0;
                return;
            }
            yz_lockstep_block_begin(ctx);
            spu_taskset_parked_add(ts, tid, wobj);
            spu_taskset_wait_signal(ts, tid);
            spu_taskset_parked_del(ts, tid);
            yz_lockstep_block_end(ctx);
            spu_ef_deliver_owed(ctx, wobj);
        }
        ctx->gpr[3]._u32[0] = 0;
        return;
    }
    /* EXIT(0, cri bootstrap)/YIELD(1)/POLL(3)/RECV_WKL_FLAG(4):
     * report success and resume (return -> lifted caller continues at its link). */
    ctx->gpr[3]._u32[0] = 0;
}

/* ---------------------------------------------------------------------------
 * Micro-interpreter for RUNTIME-GENERATED stub code (SMC).
 *
 * The WWS jobmanager's interrupt handler GENERATES a register save/restore
 * stub above the static image (LS 0x3FEC0: a chain of stqd/lqd plus a `bi`
 * terminator, written at interrupt entry) and calls it. No lift can exist for
 * bytes that only come into being at runtime, so when the indirect dispatch
 * finds no registered function we interpret the LIVE LS bytes directly for
 * the small instruction set such stubs use, and exit back into lifted code at
 * the first branch (trampoline re-dispatch). Returns 1 if it executed to a
 * branch, 0 on an unknown opcode (caller falls through to BRANCH-TO-0). */
static int spu_smc_microstep(spu_context* ctx)
{
    uint32_t pc = ctx->pc & SPU_LS_MASK & ~3u;
    const uint32_t pc_entry = pc;   /* the unresolved branch target (SPU_INTERP_MISS) */
    /* Straight runs of quadword loads and stores (lqa/stqa/lqd/stqd) go through
     * a tight loop: no decode chain, no per-access probe. The WWS handler's
     * generated save and restore stubs are 76 of them and a `bi`, run on every
     * stall-and-notify interrupt -- ~40k stubs a second on GoW:A's serial EDGE
     * chain, ~7% of its time at ~25 ns an instruction the generic way. Only
     * while every LS diagnostic is off (they read 0 once the first access has
     * read their switches); otherwise the generic path below runs them. */
    const int ldst_fast = g_spu_ls_watch_n == 0 && g_spu_ls_probe == 0 &&
                          g_spu_smc_watch == 0 && ctx->image_id != 2;
    for (int steps = 0; steps < 4096; steps++) {
        if (ldst_fast) {
            for (; steps < 4096; steps++, pc = (pc + 4) & SPU_LS_MASK) {
                uint32_t wf;
                memcpy(&wf, ctx->ls + pc, 4);
                wf = SPU_BSWAP32(wf);
                const uint32_t op9f = wf >> 23, op8f = wf >> 24;
                uint32_t a;
                if (op9f == 0x041 || op9f == 0x061)                   /* stqa / lqa: LSA = I16 << 2 */
                    a = ((wf >> 7) & 0xFFFF) << 2;
                else if (op8f == 0x24 || op8f == 0x34)                /* stqd / lqd */
                    a = ctx->gpr[(wf >> 7) & 0x7F]._u32[0] + ((uint32_t)((int32_t)(wf << 8) >> 22) << 4);
                else
                    break;
                uint8_t* p = ctx->ls + (a & SPU_LS_MASK & ~15u);
                u128* r = &ctx->gpr[wf & 0x7F];
                if (op9f == 0x041 || op8f == 0x24) {
                    const uint32_t q[4] = { SPU_BSWAP32(r->_u32[0]), SPU_BSWAP32(r->_u32[1]),
                                            SPU_BSWAP32(r->_u32[2]), SPU_BSWAP32(r->_u32[3]) };
                    memcpy(p, q, 16);
                } else {
                    uint32_t q[4];
                    memcpy(q, p, 16);
                    for (int k = 0; k < 4; k++) r->_u32[k] = SPU_BSWAP32(q[k]);
                }
            }
            if (steps >= 4096) break;
        }
        uint32_t w = ((uint32_t)ctx->ls[pc] << 24) | ((uint32_t)ctx->ls[pc+1] << 16) |
                     ((uint32_t)ctx->ls[pc+2] << 8) | ctx->ls[pc+3];
        uint32_t op11 = w >> 21, op9 = w >> 23, op8 = w >> 24, op7 = w >> 25;
        uint32_t rt = w & 0x7F, ra = (w >> 7) & 0x7F, rb = (w >> 14) & 0x7F;
        int32_t  i10 = (int32_t)(w << 8) >> 22;               /* bits 14-23 sext */
        int32_t  i16 = (int32_t)(int16_t)((w >> 7) & 0xFFFF);
        (void)rb;

        if (op11 == 0x1A8 || op11 == 0x1A9 || op11 == 0x1AA || op11 == 0x1AB) {
            /* bi / bisl / iret / bisled (+E/D interrupt bits) */
            if (op11 == 0x1AB &&                       /* bisled: only on event */
                (ctx->event_status & ctx->event_mask) == 0) { pc += 4; continue; }
            if (w & 0x40000) ctx->int_enable = 1;
            else if (w & 0x80000) ctx->int_enable = 0;
            uint32_t tgt = (op11 == 0x1AA) ? ctx->srr0
                                           : ctx->gpr[ra]._u32[0];
            if (op11 == 0x1A9 || op11 == 0x1AB)
                ctx->gpr[rt] = spu_splat_u32(pc + 4);
            ctx->pc = tgt & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op11 >= 0x128 && op11 <= 0x12B) {
            /* biz / binz / bihz / bihnz rt,ra (+E/D bits, taken-only) */
            uint32_t cv = ctx->gpr[rt]._u32[0];
            int taken;
            switch (op11) {
            case 0x128: taken = (cv == 0); break;                     /* biz  */
            case 0x129: taken = (cv != 0); break;                     /* binz */
            case 0x12A: taken = ((cv & 0xFFFF) == 0); break;          /* bihz */
            default:    taken = ((cv & 0xFFFF) != 0); break;          /* bihnz*/
            }
            if (!taken) { pc += 4; continue; }
            if (w & 0x40000) ctx->int_enable = 1;
            else if (w & 0x80000) ctx->int_enable = 0;
            ctx->pc = ctx->gpr[ra]._u32[0] & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op9 == 0x064 || op9 == 0x060 || op9 == 0x066 || op9 == 0x062) {
            /* br / bra / brsl / brasl */
            uint32_t tgt = (op9 == 0x060 || op9 == 0x062)
                         ? ((uint32_t)i16 << 2)
                         : (pc + ((uint32_t)i16 << 2));
            if (op9 == 0x066 || op9 == 0x062)
                ctx->gpr[rt] = spu_splat_u32(pc + 4);
            ctx->pc = tgt & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op9 == 0x040 || op9 == 0x042 || op9 == 0x044 || op9 == 0x046) {
            /* brz / brnz / brhz / brhnz rt,label (relative) */
            uint32_t cv = ctx->gpr[rt]._u32[0];
            int taken;
            switch (op9) {
            case 0x040: taken = (cv == 0); break;
            case 0x042: taken = (cv != 0); break;
            case 0x044: taken = ((cv & 0xFFFF) == 0); break;
            default:    taken = ((cv & 0xFFFF) != 0); break;
            }
            if (!taken) { pc += 4; continue; }
            ctx->pc = (pc + ((uint32_t)i16 << 2)) & SPU_LS_MASK & ~3u;
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        if (op8 == 0x24) {                                    /* stqd rt,i10(ra) */
            uint32_t a = (ctx->gpr[ra]._u32[0] + ((uint32_t)i10 << 4)) & SPU_LS_MASK & ~15u;
            spu_ls_write128(ctx, a, ctx->gpr[rt]);
            pc += 4; continue;
        }
        if (op8 == 0x34) {                                    /* lqd rt,i10(ra) */
            uint32_t a = (ctx->gpr[ra]._u32[0] + ((uint32_t)i10 << 4)) & SPU_LS_MASK & ~15u;
            ctx->gpr[rt] = spu_ls_read128(ctx, a);
            pc += 4; continue;
        }
        if (op9 == 0x041 || op9 == 0x061) {                   /* stqa / lqa (abs) */
            /* LSA = I16 << 2 (SPU ISA, RI16 form), as in spu_interp.c and
             * the lifter. The old << 4 sent the WWS shim builder's template
             * load (lqa 0xA40, GoW:A) to 0x2900 and its register saves 4x
             * too high. */
            uint32_t a = (((w >> 7) & 0xFFFF) << 2) & SPU_LS_MASK & ~15u;
            if (op9 == 0x041) spu_ls_write128(ctx, a, ctx->gpr[rt]);
            else              ctx->gpr[rt] = spu_ls_read128(ctx, a);
            pc += 4; continue;
        }
        if (op9 == 0x047 || op9 == 0x067) {                   /* stqr / lqr (rel) */
            uint32_t a = (pc + ((uint32_t)i16 << 2)) & SPU_LS_MASK & ~15u;
            if (op9 == 0x047) spu_ls_write128(ctx, a, ctx->gpr[rt]);
            else              ctx->gpr[rt] = spu_ls_read128(ctx, a);
            pc += 4; continue;
        }
        if (op8 == 0x44) {                                    /* xori rt,ra,i10 */
            /* Runtime-generated job stubs toggle their own instruction words.
             * Use the signed RI10 immediate, just like statically lifted xori. */
            ctx->gpr[rt] = spu_xori(ctx->gpr[ra], i10);
            pc += 4; continue;
        }
        if (op8 == 0x1C) {                                    /* ai rt,ra,i10 */
            u128 r = ctx->gpr[ra];
            for (int k = 0; k < 4; k++) r._u32[k] += (uint32_t)i10;
            ctx->gpr[rt] = r; pc += 4; continue;
        }
        if (op11 == 0x040 || op11 == 0x0C0) {                 /* sf / a (word) */
            u128 x = ctx->gpr[ra], y = ctx->gpr[rb], r;
            for (int k = 0; k < 4; k++)
                r._u32[k] = (op11 == 0x040) ? y._u32[k] - x._u32[k]
                                            : x._u32[k] + y._u32[k];
            ctx->gpr[rt] = r; pc += 4; continue;
        }
        if (op9 == 0x081) { ctx->gpr[rt] = spu_splat_u32((uint32_t)i16); pc += 4; continue; }  /* il  */
        if (op9 == 0x082) { ctx->gpr[rt] = spu_splat_u32(((w >> 7) & 0xFFFF) << 16); pc += 4; continue; } /* ilhu */
        if (op9 == 0x0C1) { u128 r = ctx->gpr[rt];            /* iohl */
            for (int k = 0; k < 4; k++) r._u32[k] |= (w >> 7) & 0xFFFF;
            ctx->gpr[rt] = r; pc += 4; continue; }
        if (op7 == 0x21)  { ctx->gpr[rt] = spu_splat_u32((w >> 7) & 0x3FFFF); pc += 4; continue; } /* ila */
        if (op11 == 0x201 || op11 == 0x001) { pc += 4; continue; }  /* nop/lnop */
        if (op11 == 0x002 || op11 == 0x003) { pc += 4; continue; }  /* sync/dsync */
        /* Branch hints, all no-ops for execution: hbr is the 11-bit-opcode RR
         * form (0x1AC), hbra and hbrr are the 7-bit-opcode RI18 form (0x08,
         * 0x09) -- the same op7 group as ila (0x21) above, NOT op9. Testing
         * op9 here never matched either (hbrr 0x12033296 has op9 0x24), so a
         * runtime-generated stub carrying a branch hint decoded to UNKNOWN,
         * the microstep bailed, and the SPU fell into branch-to-0. */
        /* sagemono reached the same fix independently in #166, from a different
         * witness (hbrr 0x1200048C rather than 0x12033296). Same line, same
         * diagnosis, arrived at separately -- which is about as good as
         * corroboration gets for a decode bug. */
        if (op11 == 0x1AC || op7 == 0x08 || op7 == 0x09) { pc += 4; continue; } /* hbr/hbra/hbrr */

        /* Hand off to the real interpreter rather than giving up.
         *
         * This microstepper knows a dozen opcodes -- enough for the
         * runtime-generated stubs it was written for -- and spu_interp.c is a
         * full, selftested SPU interpreter that was already sitting here
         * unused on this path. Bailing on the first ordinary instruction is
         * strictly worse than running it.
         *
         * It matters for overlay code: a SPURS job policy module DMAs its job
         * bodies into local store and branches to them, so there is no lifted
         * body to find and the fallback IS the execution path. Guitar Hero
         * III`s job kernel dies here on an `ilh` at LS 0x6638 -- a perfectly
         * ordinary instruction the interpreter handles -- and with it the
         * decompression job it was running never completes. */
        { static int _n = 0;
          if (PS3_LOG_FIRST(_n, 8))
              fprintf(stderr, "[spu-smc] microstep img=%d pc=0x%05X word 0x%08X not in the"
                      " microstep set -- handing to the interpreter (steps=%d)\n",
                      ctx->image_id, pc, w, steps); }
        ctx->pc = pc;
        /* SPU_INTERP_MISS=1: interpreter time by (workload, image or resident overlay,
         * branch target), top 40 every 20 s -- unresolved branch targets ranked by what
         * interpreting from them costs. A target inside a lifted module's image is a
         * missing function entry (feed it back to the lifter); one past the image is
         * code the module does not contain (runtime-loaded or generated). */
        enum { NM = 4096 };
        static uint32_t mk[NM]; static int mi[NM], mw[NM]; static unsigned mc[NM]; static long long mt[NM];
        static SRWLOCK ml = SRWLOCK_INIT;
        static int s_im = -1; if (s_im < 0) s_im = getenv("SPU_INTERP_MISS") ? 1 : 0;
        int miss_slot = -1;
        if (s_im) {
            uint32_t key = pc_entry;
            int im = (pc_entry >= 0x4000 && ctx->resident_ovl) ? ctx->resident_ovl : ctx->image_id;
            /* Inside resident relocatable code: report its image and the CANONICAL pc
             * (placement-independent), which is the entry its --reloc lift needs. */
            for (uint32_t m = ctx->resident_code_mask; m; m &= m - 1) {
                const unsigned sl = (unsigned)__builtin_ctz(m);
                if (ctx->resident_code[sl].canon && pc_entry >= ctx->resident_code[sl].lsa &&
                    pc_entry - ctx->resident_code[sl].lsa < ctx->resident_code[sl].size) {
                    im = ctx->resident_code[sl].image_id;
                    key = pc_entry - ctx->resident_code[sl].lsa + ctx->resident_code[sl].canon;
                    break;
                }
            }
            const int wid = ctx->policy_mode ? ctx->ls[0x1DF] : -1;
            AcquireSRWLockExclusive(&ml);
            unsigned h = (key * 2654435761u ^ (unsigned)im * 40503u ^ (unsigned)(wid + 1) * 977u) % NM;
            for (unsigned q = 0; q < NM; q++, h = (h + 1) % NM) {
                if (mc[h] && mk[h] == key && mi[h] == im && mw[h] == wid) { mc[h]++; miss_slot = (int)h; break; }
                if (!mc[h]) { mk[h] = key; mi[h] = im; mw[h] = wid; mc[h] = 1; mt[h] = 0; miss_slot = (int)h; break; }
            }
            ReleaseSRWLockExclusive(&ml);
        }
        /* SPU_INTERP_STATS=1: interpreter hand-offs and steps per image, every
         * 20000 hand-offs -- how much of an image runs interpreted, and from where. */
        static int s_is = -1; if (s_is < 0) s_is = getenv("SPU_INTERP_STATS") ? 1 : 0;
        /* Stop at the drain's return point too: it sits mid-function, so it is
         * no lifted entry, and a job's return would otherwise run on through
         * the caller's code (GH3's song-freeze; see drain_ret_pc). */
        const uint32_t return_pc = ctx->drain_ret_pc;
        LARGE_INTEGER t0, t1;
        if (miss_slot >= 0) QueryPerformanceCounter(&t0);
        uint32_t result = spu_interp_run_until(ctx, pc, return_pc);
        if (miss_slot >= 0) {
            QueryPerformanceCounter(&t1);
            static LARGE_INTEGER s_f, s_w0;
            AcquireSRWLockExclusive(&ml);
            mt[miss_slot] += t1.QuadPart - t0.QuadPart;
            if (!s_f.QuadPart) { QueryPerformanceFrequency(&s_f); s_w0 = t1; }
            if (t1.QuadPart - s_w0.QuadPart >= 20 * s_f.QuadPart) {
                long long tot = 0;
                for (int k = 0; k < NM; k++) tot += mt[k];
                fprintf(stderr, "[interp-miss] %.1f ms interpreted in 20 s, top targets:\n",
                        1000.0 * tot / s_f.QuadPart);
                static unsigned char shown[NM];
                memset(shown, 0, sizeof shown);
                for (int r = 0; r < 40; r++) {
                    int bb = -1;
                    for (int k = 0; k < NM; k++)
                        if (mc[k] && !shown[k] && (bb < 0 || mt[k] > mt[bb])) bb = k;
                    if (bb < 0 || !mt[bb]) break;
                    shown[bb] = 1;
                    fprintf(stderr, "[interp-miss]   wid=%d img=%d pc=0x%05X n=%u %.1fms%s\n", mw[bb], mi[bb], mk[bb],
                            mc[bb], 1000.0 * mt[bb] / s_f.QuadPart, spu_lookup(mk[bb], mi[bb]) ? " (lifted entry exists)" : "");
                }
                memset(mc, 0, sizeof mc); memset(mt, 0, sizeof mt);
                s_w0 = t1;
                fflush(stderr);
            }
            ReleaseSRWLockExclusive(&ml);
        }
        if (s_is) {
            extern uint64_t g_spu_interp_steps;
            static volatile long s_calls[64]; static volatile long long s_steps[64];
            static volatile long s_tot; static uint32_t s_lastpc[64];
            unsigned im = (unsigned)ctx->image_id & 63u;
            _InterlockedIncrement(&s_calls[im]); s_lastpc[im] = pc;
            s_steps[im] += (long long)g_spu_interp_steps;
            if (_InterlockedIncrement(&s_tot) % 20000 == 0)
                for (unsigned k = 0; k < 64; k++) if (s_calls[k])
                    fprintf(stderr, "[interp-stats] img=%u handoffs=%ld steps=%lld lastpc=0x%05X\n",
                            k, s_calls[k], s_steps[k], s_lastpc[k]);
        }
        /* A live interpreter rejoin is a transfer, not a host return. Nested
         * drains can repair an empty trampoline, but the top-level driver
         * otherwise ends the run before executing the lifted target.
         * An already-owning drain must retain its own host continuation;
         * stopped states and diagnostic exits keep their existing behavior.
         * (GoW:A: without the rejoin the WWS interrupt handler's shim-builder
         * loop at LS 0xA60 ended the policy-module run -> [pm-end] pc=0xA60, so
         * the stall-and-notify was never acknowledged.) */
        if (ctx->image_id >= 0 && (ctx->status & SPU_STATUS_RUNNING) &&
            (!return_pc || (ctx->pc & SPU_LS_MASK) != return_pc) &&
            spu_lifted_lookup(ctx, ctx->pc)) {
            g_spu_trampoline_fn = spu_indirect_branch;
            return 1;
        }
        return result ? 1 : 0;
    }
    { static int _n = 0; if (PS3_LOG_FIRST(_n, 4))
        fprintf(stderr, "[spu-smc] microstep img=%d runaway (4096 steps from 0x%05X)\n",
                ctx->image_id, ctx->pc & SPU_LS_MASK); }
    return 0;
}

/* The diagnostic bodies below run only when their switch is set. They live out
 * of line so spu_indirect_branch -- entered on every indirect branch of lifted
 * code, millions of times a second -- keeps a small stack frame: their locals
 * (a 400-byte line buffer among them) gave it a ~1 KB frame, eight register
 * saves and a stack-cookie check on every call. */
static __attribute__((noinline)) void spu_ib_cov_note(const spu_context* ctx)
{
    static uint8_t pages[64] = {0};   /* 64 pages * 0x1000 = 256KB LS */
    static uint64_t hits = 0;
    uint32_t pg = (ctx->pc & 0x3FFFF) >> 12;
    int newp = 0;
    if (pg < 64 && !pages[pg]) { pages[pg] = 1; newp = 1; }
    ++hits;
    if (newp || (hits % 200000) == 0) {
        char line[400]; int p = snprintf(line, sizeof line, "[ibcov3] hits=%llu tgt=0x%05X pages:", (unsigned long long)hits, ctx->pc & 0x3FFFF);
        for (int i = 0; i < 64; i++) if (pages[i]) p += snprintf(line+p, sizeof(line)-p, " 0x%X", i<<12);
        fprintf(stderr, "%s\n", line); }
}

/* SPU_IB_STATS=1: indirect branches per 20 s, and the 24 most frequent by
 * (workload, image or resident overlay, target, link register) -- which
 * branches the dispatcher resolves most often, and from where. */
static __attribute__((noinline)) void spu_ib_stats_note(const spu_context* ctx)
{
    enum { NS = 8192 };
    static uint32_t k_pc[NS], k_lr[NS], k_n[NS]; static int k_img[NS], k_wid[NS];
    static SRWLOCK lk = SRWLOCK_INIT;
    static unsigned long long s_tot, s_t0;
    const uint32_t pc = ctx->pc & SPU_LS_MASK, lr = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
    const int img = ctx->resident_ovl ? ctx->resident_ovl : ctx->image_id;
    const int wid = ctx->policy_mode ? ctx->ls[0x1DF] : -1;
    AcquireSRWLockExclusive(&lk);
    s_tot++;
    unsigned h = (pc * 2654435761u ^ lr * 40503u ^ (unsigned)img * 977u ^ (unsigned)(wid + 1) * 131u) % NS;
    for (unsigned q = 0; q < 64; q++, h = (h + 1) % NS) {
        if (k_n[h] && k_pc[h] == pc && k_lr[h] == lr && k_img[h] == img && k_wid[h] == wid) { k_n[h]++; break; }
        if (!k_n[h]) { k_pc[h] = pc; k_lr[h] = lr; k_img[h] = img; k_wid[h] = wid; k_n[h] = 1; break; }
    }
    const unsigned long long now = GetTickCount64();
    if (!s_t0) s_t0 = now;
    if (now - s_t0 >= 20000) {
        fprintf(stderr, "[ib-stats] %llu indirect branches in %.1f s, top targets (wid img pc lr n):\n",
                s_tot, (now - s_t0) / 1000.0);
        for (int r = 0; r < 24; r++) {
            int b = -1;
            for (int k = 0; k < NS; k++) if (k_n[k] && (b < 0 || k_n[k] > k_n[b])) b = k;
            if (b < 0) break;
            fprintf(stderr, "[ib-stats]   w%-3d img %-4d pc 0x%05X lr 0x%05X %u\n",
                    k_wid[b], k_img[b], k_pc[b], k_lr[b], k_n[b]);
            k_n[b] = 0;
        }
        memset(k_n, 0, sizeof k_n);
        s_tot = 0; s_t0 = now;
    }
    ReleaseSRWLockExclusive(&lk);
}

static __attribute__((noinline)) void spu_pm_exitstat_note(const spu_context* ctx, int every)
{
    extern unsigned spu_recent_pcs(uint32_t* out, unsigned max);
    enum { NS = 64 };
    static uint32_t k_wid[NS], k_p1[NS], k_p2[NS], k_r3[NS], k_n[NS];
    static volatile LONG s_lock, s_tot;
    uint32_t pcs[2] = {0, 0};
    spu_recent_pcs(pcs, 2);
    const uint32_t wid = ctx->ls[0x1DF], r3 = ctx->gpr[3]._u32[0];
    while (InterlockedCompareExchange(&s_lock, 1, 0)) SwitchToThread();
    int k = 0;
    for (; k < NS; k++) {
        if (!k_n[k]) { k_wid[k] = wid; k_p1[k] = pcs[0]; k_p2[k] = pcs[1]; k_r3[k] = r3; break; }
        if (k_wid[k] == wid && k_p1[k] == pcs[0] && k_p2[k] == pcs[1] && k_r3[k] == r3) break;
    }
    if (k < NS) k_n[k]++;
    if (++s_tot % every == 0) {
        fprintf(stderr, "[pm-exit] %ld exits; top paths (wid, prev fn, last fn, r3):\n", (long)s_tot);
        for (int r = 0; r < 12; r++) {
            int b = -1;
            for (int q = 0; q < NS; q++) if (k_n[q] && (b < 0 || k_n[q] > k_n[b])) b = q;
            if (b < 0) break;
            fprintf(stderr, "[pm-exit]   %7u  w%-2u %05X -> %05X r3=0x%08X\n",
                    k_n[b], k_wid[b], k_p1[b], k_p2[b], k_r3[b]);
            k_n[b] = 0;
        }
        memset(k_n, 0, sizeof k_n);
    }
    s_lock = 0;
}

static __attribute__((noinline)) void spu_unlifted_branch_report(const spu_context* ctx)
{
    static uint32_t seen[16]; static int n_seen = 0;
    int known = 0;
    for (int i = 0; i < n_seen; i++) if (seen[i] == ctx->pc) { known = 1; break; }
    if (known || n_seen >= 16) return;
    seen[n_seen++] = ctx->pc;
    fprintf(stderr, "[spu] img=%d branched into unlifted LS 0x%05X "
            "(lr=0x%05X) -- ending the job\n",
            ctx->image_id, ctx->pc, ctx->gpr[0]._u32[0] & SPU_LS_MASK);
    { fprintf(stderr, "      last dispatched PCs (oldest first):");
      for (unsigned q = 0; q < 8; q++) {
          unsigned idx = (g_spu_pch_n + q) & 7u;
          if (g_spu_pch_n > q || g_spu_pch[idx]) fprintf(stderr, " 0x%05X", g_spu_pch[idx]);
      }
      fprintf(stderr, "%c", 10); }
    /* Is there real code at the target, or is the pc garbage? Eight
     * words at the target and at the return address separate "the lift
     * missed a function" from "this branch should never have happened". */
    { uint32_t a[2]; a[0] = ctx->pc & SPU_LS_MASK;
      a[1] = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
      for (int k = 0; k < 2; k++) {
          fprintf(stderr, "      LS[0x%05X]:", a[k]);
          for (uint32_t o = 0; o < 32 && a[k] + o + 3 < SPU_LS_SIZE; o += 4)
              fprintf(stderr, " %02X%02X%02X%02X",
                      ctx->ls[a[k]+o], ctx->ls[a[k]+o+1],
                      ctx->ls[a[k]+o+2], ctx->ls[a[k]+o+3]);
          fprintf(stderr, "\n");
      } }
    fflush(stderr);
}

/* Indirect-branch dispatch cache. Most dispatches go to a few targets (in a
 * GoW:A render-chain census one target was 12% of all indirect branches and
 * 24 targets 44%), yet each one walked every check below and up to three
 * spu_lookup probes into a 64 KB table that the lifted code between
 * dispatches evicts. A per-thread table of 512 16-byte entries answers the
 * common case.
 *
 * An entry is only written at the tail of spu_indirect_branch, when the
 * standard lookups found the target and nothing else applied. Its key is what
 * that decision read besides the link register, the resident code slots and
 * local store: pc, image, overlay and policy mode. The probe checks the link
 * register (job entry) and the code slots live; the image-23 and 0xA70 paths,
 * which read local store, are never cached. Any registration -- function, task
 * entry or task ELF -- bumps s_ibc_epoch, which empties the table. PS3_SPU_IBC=0 turns it off, as does
 * any trace switch spu_indirect_branch reads. */
enum { IBC_SIZE = 512 };
static SPU_THREAD_LOCAL struct { uint32_t gen; struct { uint64_t key; spu_fn fn; } e[IBC_SIZE]; } t_ibcs;
static int s_ibc_on = -1;
static __attribute__((noinline)) int spu_ibc_init(void)
{
    const char* e = getenv("PS3_SPU_IBC");
    s_ibc_on = !(e && e[0] == '0') &&
        !getenv("SPU_IBCOV") && !getenv("SPU_IB_STATS") && !getenv("SPU_POLLTRACE") &&
        !getenv("YDKJ_POLTRACE") && !getenv("SPU_JOBTRACE") && !getenv("SPU_JOBEXEC") &&
        !getenv("LBP_HLE_JOBDONE");
    return s_ibc_on;
}
static inline uint32_t spu_ibc_gen(void)
{
    return s_ibc_epoch;
}
static inline int spu_ibc_key(const spu_context* ctx, uint64_t* key, uint32_t* slot)
{
    const uint32_t img = (uint32_t)ctx->image_id, ovl = (uint32_t)ctx->resident_ovl;
    if (img >= 0x1000u || ovl >= 0x1000u) return 0;
    *key = (uint64_t)ctx->pc | (uint64_t)img << 18 | (uint64_t)ovl << 30 |
           (uint64_t)(ctx->policy_mode != 0) << 42 | 1ull << 43;
    *slot = ((ctx->pc >> 2) ^ (img * 0x9E37u) ^ (ovl * 0x85EBu)) & (IBC_SIZE - 1);
    return 1;
}
static inline spu_fn spu_ibc_probe(const spu_context* ctx)
{
    uint64_t key; uint32_t slot;
    if (!spu_ibc_key(ctx, &key, &slot)) return NULL;
    const uint32_t gen = spu_ibc_gen();
    if (__builtin_expect(gen != t_ibcs.gen, 0)) { memset(t_ibcs.e, 0, sizeof t_ibcs.e); t_ibcs.gen = gen; return NULL; }
    return t_ibcs.e[slot].key == key ? t_ibcs.e[slot].fn : NULL;
}
static inline void spu_ibc_fill(const spu_context* ctx, spu_fn fn)
{
    uint64_t key; uint32_t slot;
    if (!spu_ibc_key(ctx, &key, &slot)) return;
    const uint32_t gen = spu_ibc_gen();
    if (gen != t_ibcs.gen) { memset(t_ibcs.e, 0, sizeof t_ibcs.e); t_ibcs.gen = gen; }
    t_ibcs.e[slot].key = key; t_ibcs.e[slot].fn = fn;
}

void spu_indirect_branch(spu_context* ctx)
{
    /* Real SPU bi/bisl mask the target to the 256 KB local store; the high bits
     * of a computed pointer (e.g. a packed handle like 0x7a028803) are ignored.
     * Without this, any indirect branch through such a value fails the lookup
     * and falls into branch-to-0. All lifted funcs live below SPU_LS_SIZE, so
     * masking is a no-op for already-valid targets. */
    ctx->pc &= SPU_LS_MASK;
    /* SPU_JOB_CAPTURE: a captured job ends at its return to the link it was
     * entered with; any other PM address it reaches is a JobApi call. */
    if (g_spu_jobcap_on) {
        if (g_spu_jobcap_on < 0) spu_jobcap_init();
        if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx && ctx->pc < 0x4000) {
            if (ctx->pc == t_jc.link) jc_close(JC_END_RETURN);
            else jc_ev(JC_PMCALL, ctx->pc, ctx->gpr[0]._u32[0] & SPU_LS_MASK, 0, 0, NULL, 0);
        }
    }
    /* SPU_JOB_REPLAY: the same dispatch must be the next recorded event. */
    if (g_spu_jobrep_on > 0 && ctx->pc < 0x4000) spu_jobrep_pmcall(ctx);
    /* Interrupt-return register restore (see spu_drain.c spu_irq_regs_save):
     * the WWS handler's save/restore shim lives at unlifted top-of-LS, so we
     * enforce the preserve-all-registers hardware contract here -- the first
     * interrupts-enabled dispatch at the saved srr0 is the irete. */
    { extern int spu_irq_regs_maybe_restore(spu_context*);
      if (spu_irq_regs_maybe_restore(ctx)) {
          static int s_it = -1; if (s_it < 0) s_it = getenv("SPU_IRQTRACE") ? 1 : 0;
          static int _r = 0;
          if (s_it && PS3_LOG_FIRST(_r, 200))
              fprintf(stderr, "[irq] IRET restore at srr0=0x%05X\n", ctx->pc & SPU_LS_MASK);
      } }
    /* Dispatch cache (see t_ibc). A hit does what the path to the tail below
     * does for it: drop resident_task outside the task region, then the
     * stack-reset check and the call. */
    if (__builtin_expect(s_ibc_on != 0, 1) && (s_ibc_on > 0 || spu_ibc_init()) &&
        !g_spu_jobcap_on && g_spu_jobrep_on <= 0 &&
        (ctx->resident_task == 0 || ctx->pc < SPU_TASKSET_TASK_LO)) {
        spu_fn fn = spu_ibc_probe(ctx);
        if (fn) {
            const uint32_t jl = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
            int plain = !(ctx->policy_mode && ctx->pc >= 0x4000 && ctx->pc < 0x3E000 && jl >= 0xA00 && jl < 0x4000);
            for (uint32_t m = plain ? ctx->resident_code_mask : 0; m; m &= m - 1) {
                const unsigned slot = (unsigned)__builtin_ctz(m);
                if (ctx->pc - ctx->resident_code[slot].lsa < ctx->resident_code[slot].size &&
                    ctx->pc >= ctx->resident_code[slot].lsa) { plain = 0; break; }
            }
            if (plain) {
                ctx->resident_task = 0;
                spu_check_stack_reset(ctx, fn);
#if defined(__clang__)
                __attribute__((musttail)) return fn(ctx);
#else
                fn(ctx);
                return;
#endif
            }
        }
    }
    /* SPURS kernel services (policy-module runs only): the HLE kernel plants
     * these two reserved addresses as exitToKernelAddr / selectWorkloadAddr in
     * the SpursKernelContext (spurs_policy.c). */
    if (ctx->policy_mode) {
        extern SPU_THREAD_LOCAL volatile unsigned g_spurs_pm_polls, g_spurs_pm_exited;
        if (ctx->pc == SPURS_PM_EXIT_TO_KERNEL_LS) {
            /* Module exit: the workload returned to the kernel (drained/yield).
             * Print gated: fires once per policy run = thousands/sec. */
            g_spurs_pm_exited = 1;
            /* SPURS_PM_EXITSTAT=<n>: every n exits, the most common (wid, last two
             * function entries, r3) exit paths -- why a policy module keeps handing
             * the SPU back after about one job. */
            { static int s_es = -1;
              if (s_es < 0) { const char* e = getenv("SPURS_PM_EXITSTAT"); s_es = e ? atoi(e) : 0; }
              if (s_es > 0) spu_pm_exitstat_note(ctx, s_es); }
            { static SPU_THREAD_LOCAL int s_t = -1; if (s_t < 0) s_t = getenv("SPURS_PM_TRACE") ? 1 : 0;
              if (s_t) { static SPU_THREAD_LOCAL unsigned long _n = 0; unsigned long n = ++_n;
                if (n <= 64 || (n & 0xFFF) == 0)
                    fprintf(stderr, "[spurs-pm] exit-to-kernel#%lu (r3=0x%08X polls=%u)\n",
                            n, ctx->gpr[3]._u32[0], g_spurs_pm_polls); } }
            ctx->status = SPU_STATUS_STOPPED_BY_STOP;
            spu_halt(ctx);
            return;
        }
        if (ctx->pc == SPURS_PM_SELECT_WORKLOAD_LS) {
            /* SPU_JOB_REPLAY: the recorded answer (r3, LS 0x100..0x17F). */
            if (g_spu_jobrep_on > 0 && spu_jobrep_poll(ctx)) {
                ++g_spurs_pm_polls;
                ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
                return;
            }
            /* cellSpursModulePoll: report "no contention — keep running".
             * (The host scheduler admits bounded workers; this service does not preempt.) */
            unsigned n = ++g_spurs_pm_polls;
            /* g_spurs_pm_polls restarts at 0 every module run, so cap on a
             * process-wide count (it printed the first polls of EVERY run). */
            { static volatile LONG s_np;
              const LONG np = InterlockedIncrement(&s_np);
              if (np <= 4 || (np % 65536) == 0)
                  fprintf(stderr, "[spurs-pm] poll #%u (total %ld, r3=0x%08X) -> continue\n",
                          n, (long)np, ctx->gpr[3]._u32[0]); }
            ctx->gpr[3] = spu_make_preferred_u32(0);
            /* The real selectWorkload GETLLARs the first CellSpurs line into the
             * kernel context's tempArea (LS 0x100..0x17F) on every call, not only
             * at module entry; jobs that ran since entry may have DMA'd over it,
             * and GoW:A's WWS jobs read flags1 there to look up wids >= 16. */
            { const uint32_t sp = ((uint32_t)ctx->ls[0x1C4] << 24) | ((uint32_t)ctx->ls[0x1C5] << 16) |
                                  ((uint32_t)ctx->ls[0x1C6] << 8) | ctx->ls[0x1C7];
              if (sp) memcpy(ctx->ls + 0x100, vm_base + sp, 0x80);
              /* SPURS_POLL_SELECT=1: a real selection (cellSpurs.c spurs_poll_select). */
              extern int spurs_poll_select_on(void);
              extern uint32_t spurs_poll_select(uint32_t ea, uint32_t wid);
              /* Only while the module is productive: one that polls again without
               * having entered a job since its last poll is idling on an empty list
               * (readyCount alone stays up), and keeping it would hold the SPU
               * from the workloads that feed that list. */
              extern SPU_THREAD_LOCAL unsigned g_spu_job_entries;
              static SPU_THREAD_LOCAL unsigned t_poll_jobs;
              const int productive = g_spu_job_entries != t_poll_jobs;
              t_poll_jobs = g_spu_job_entries;
              if (sp && productive && spurs_poll_select_on())
                  ctx->gpr[3]._u32[0] = spurs_poll_select(sp, ((uint32_t)ctx->ls[0x1DC] << 24) |
                      ((uint32_t)ctx->ls[0x1DD] << 16) | ((uint32_t)ctx->ls[0x1DE] << 8) | ctx->ls[0x1DF]); }
            /* SPURS_PM_POLL_SLICE_US=<us> (0 = off, the old answer): the real
             * selectWorkload(isPoll=1) returns r3 = {selected wid, pollStatus}
             * (RPCS3 spursKernel1SelectWorkload), and a module keeps running only
             * while the selected wid is its own (wklCurrentId @LS 0x1DC; GoW:A's
             * wwsjob PM compares at LS 0x37A4). Answering 0 meant "switch to wid 0",
             * so every other module handed its SPU back after ONE job and waited a
             * full kernel pass for the next. Select the caller's own wid for <us>
             * of its run, then wid 0 (yield) so a module that never drains (GoW:A
             * BrBDblBufWrkld) still shares its kernel thread. */
            { static long long s_slice = -1;
              if (s_slice < 0) {
                  const char* e = getenv("SPURS_PM_POLL_SLICE_US");
                  LARGE_INTEGER f; QueryPerformanceFrequency(&f);
                  s_slice = e ? (long long)f.QuadPart * atoll(e) / 1000000 : 0;
              }
              extern SPU_THREAD_LOCAL unsigned long long g_spurs_pm_t0;
              LARGE_INTEGER q; QueryPerformanceCounter(&q);
              if (s_slice > 0 && (long long)((unsigned long long)q.QuadPart - g_spurs_pm_t0) < s_slice)
                  ctx->gpr[3]._u32[0] = ((uint32_t)ctx->ls[0x1DC] << 24) | ((uint32_t)ctx->ls[0x1DD] << 16) |
                                        ((uint32_t)ctx->ls[0x1DE] << 8) | ctx->ls[0x1DF];
            }
            { extern SPU_THREAD_LOCAL unsigned g_spu_job_entries;
              extern void spurs_pm_hang_note_poll(unsigned jobs, unsigned answer);
              spurs_pm_hang_note_poll(g_spu_job_entries, ctx->gpr[3]._u32[0]); }
            /* The service wrote LS 0x100..0x17F and r3: job inputs a replay
             * has to reproduce. */
            if (g_spu_jobcap_on > 0 && t_jc.ctx == ctx)
                jc_ev(JC_POLL, ctx->gpr[3]._u32[0], 0, 0x100, 0x80, ctx->ls + 0x100, 0x80);
            ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
            return;
        }
    }
    /* Synthetic HLE tasks have no resident taskset policy at 0xA70. Real
     * tasksets use the SAME syscall address, so the API table alone is not
     * evidence that this is an HLE context. LLE tasks must enter Sony's policy
     * to yield/select workloads instead of parking an entire SPU host thread.
     * Image 22 is the legacy standalone HLE CRI task runner. */
    if (ctx->pc == SPURS_TASKSET_PM_SYSCALL_LS) {
        uint32_t sc = ((uint32_t)ctx->ls[0x27C4] << 24) | ((uint32_t)ctx->ls[0x27C5] << 16)
                    | ((uint32_t)ctx->ls[0x27C6] << 8)  | ctx->ls[0x27C7];
        /* A task dispatched standalone (no policy module in this local store)
         * runs on the SpursTasksetContext spu_workload.c plants, whose
         * syscallAddr is this address; nothing is lifted at 0xA70 for it, so
         * the only alternative to the HLE is the synthesised-stop exit below
         * -- which is how LBP's audio tasks (images 6/7) ended on their first
         * syscall and left the loading thread waiting on their event flag. */
        int lifted = spu_lookup(SPURS_TASKSET_PM_SYSCALL_LS, ctx->image_id) ||
            (ctx->resident_task && spu_lookup(SPURS_TASKSET_PM_SYSCALL_LS, ctx->resident_task)) ||
            (ctx->resident_ovl  && spu_lookup(SPURS_TASKSET_PM_SYSCALL_LS, ctx->resident_ovl));
        int standalone = !ctx->policy_mode && sc == SPURS_TASKSET_PM_SYSCALL_LS && !lifted;
        if (ctx->image_id == 22 || standalone ||
            (ctx->policy_mode && sc == SPURS_TASKSET_PM_SYSCALL_LS)) {
            spu_spurs_taskset_syscall(ctx);
            ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
            return;
        }
    }
    /* YDKJ_CRI_R4: the taskset policy entry (LS 0xA00, image 23) writes r4 into
     * SpursTasksetContext.taskset @LS 0x27B8 (per RPCS3 cellSpursSpu.cpp). Our
     * kernel->policy handoff doesn't convey the taskset EA, so the policy DMAs
     * the taskset from garbage -> waiting!=0 -> wrong resume path -> savedContextLr=0.
     * Inject r4 = taskset EA (0x0F000000) at the policy entry dispatch (this is the
     * exact point before the entry reads r4, after the kernel's arg setup). */
    if (ctx->image_id == SPU_RESOLVER_ONLY_IMAGE_ID && !getenv("YDKJ_NO_CRI_R4")) {
        static int s_r4 = -1; if (s_r4 < 0) s_r4 = getenv("YDKJ_CRI_CHAIN") ? 1 : 0;
        if (s_r4) {
            /* Force ctxt->taskset @LS 0x27B8 = the REAL game taskset EA on every
             * image-23 branch, so the policy's atomic reads + context-EA computation
             * use the actual taskset (the r4 handoff sets it to garbage 0x0000FFFF via
             * a path we can't intercept). Was hardcoded 0x0F000000, which mismatched
             * the game's real taskset (0x45F1B000) -> policy DMA'd garbage. */
            extern uint32_t g_ydkj_real_taskset_ea;
            uint32_t ts = g_ydkj_real_taskset_ea ? g_ydkj_real_taskset_ea : 0x0F000000u;
            ctx->ls[0x27B8]=0x00; ctx->ls[0x27B9]=0x00; ctx->ls[0x27BA]=0x00; ctx->ls[0x27BB]=0x00;
            ctx->ls[0x27BC]=(uint8_t)(ts>>24); ctx->ls[0x27BD]=(uint8_t)(ts>>16); ctx->ls[0x27BE]=(uint8_t)(ts>>8); ctx->ls[0x27BF]=(uint8_t)ts;
            if (ctx->pc == 0xA00u) { static int _n=0; if (PS3_LOG_FIRST(_n, 4))
                fprintf(stderr, "[cri-r4] policy entry pc=0xA00: forced ctxt->taskset LS[0x27B8]=0x0F000000\n"); }
        }
    }
    /* SPU_IBCOV: image-3 (Bink SPU) PC-page coverage. Track which 0x1000-byte LS
     * pages the task's indirect branches land in; dump the set periodically. If
     * coverage stays in the kernel/wait region (~0x13xxx) the decode routine never
     * runs; if it spans a wide high range, decode executes but doesn't output. */
    { static int s_cov = -1; if (s_cov < 0) s_cov = getenv("SPU_IBCOV") ? 1 : 0;
      if (s_cov && ctx->image_id == 3) spu_ib_cov_note(ctx); }
    { static int s_ibs = -1; static uint32_t s_ibw;
      if (s_ibs < 0) {
          s_ibs = getenv("SPU_IB_STATS") ? 1 : 0;
          /* SPU_IB_STATS_WIDS=<w>[,<w>...]: count only branches of policy
           * modules running these workloads. */
          for (const char* w = getenv("SPU_IB_STATS_WIDS"); w && *w; ) {
              char* p; const long v = strtol(w, &p, 10);
              if (p == w) break;
              if (v >= 0 && v < 32) s_ibw |= 1u << v;
              if (*p != ',') break;
              w = p + 1;
          }
      }
      if (s_ibs && (!s_ibw || (ctx->policy_mode && ctx->ls[0x1DF] < 32 && ((s_ibw >> ctx->ls[0x1DF]) & 1u))))
          spu_ib_stats_note(ctx); }
    { static int s_t = -1; if (s_t < 0) s_t = getenv("SPU_POLLTRACE") ? 1 : 0;
      if (s_t) { static uint64_t s_n = 0; static uint32_t s_last = 0; static uint64_t s_run = 0;
        if (ctx->pc == s_last) s_run++; else { s_last = ctx->pc; s_run = 1; }
        if ((++s_n % 2000000) == 0)
          fprintf(stderr, "[ibranch] %llu indirect branches; current target=0x%05X run=%llu\n",
                  (unsigned long long)s_n, ctx->pc, (unsigned long long)s_run); } }
    /* Policy-entry trace: the SPURS policy at LS 0xA00 branches on
     * r8 = word at LS[r6] (must be 32 for the path that sets the dispatch ptr
     * LS[0x780]). Log r3..r6 + the word the kernel handed it, to see why the
     * wrong branch is taken. Env YDKJ_POLTRACE. */
    if (ctx->pc == 0xA00u) {
        static int64_t pt=-2; if (pt==-2){ const char* e=getenv("YDKJ_POLTRACE"); pt=e?1:0; }
        if (pt) { static int _p=0; if (PS3_LOG_FIRST(_p, 8)) {
            /* re-lifted policy entry uses r80 (kernel-set context base): r44=LS[r80+0xC0] */
            uint32_t r80=ctx->gpr[80]._u32[0] & SPU_LS_MASK;
            const uint8_t* q = ctx->ls + ((r80+0xC0)&SPU_LS_MASK);
            uint32_t ctxw = ((uint32_t)q[0]<<24)|((uint32_t)q[1]<<16)|((uint32_t)q[2]<<8)|q[3];
            fprintf(stderr, "[POLTRACE] policy@0xA00 r3=%08X r4=%08X r80=%08X  LS[r80+0xC0]=%08X\n",
                ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0], r80, ctxw);
            fflush(stderr);
        } }
    }
    /* Resolve both fresh launches and mid-function resumes from TaskInfo.
     * A scheduler call temporarily leaves the task region, but its return PC
     * is usually not an ELF entry. Prefer the policy's selected ELF so tasks
     * sharing an entry address cannot shadow one another. Keep the legacy
     * entry-only mapping for runners that have not registered ELF metadata. */
    if (ctx->pc < SPU_TASKSET_TASK_LO) {
        ctx->resident_task = 0;
    } else {
        int ti = spu_taskset_resident_image(ctx);
        if (ti && !ctx->resident_task && ctx->host_depth &&
            s_spu_halt_armed && !ctx->policy_mode) {
            /* The policy restored a task's guest registers and branches to
             * its saved PC. Its host frames still belong to the scheduler or
             * an earlier task invocation; none can satisfy this task's return.
             * Resume at depth zero so SPU_RET follows the restored guest link. */
            ctx->resident_task = ti;
            g_spu_trampoline_fn = 0;
            SPU_LONGJMP(s_spu_halt_env, 2);
        }
        if (!ti && !ctx->resident_task) ti = spu_taskset_task_image(ctx->pc);
        if (ti) ctx->resident_task = ti;
    }
    /* The launched task owns the shared task-code region: resolve it there FIRST
     * so a co-resident task at the same LS base cannot shadow it. (resident_task
     * is 0 outside the region, so this only ever fires for genuine task code.) */
    spu_fn fn = NULL;
    int code_owner = 0;
    for (uint32_t m = ctx->resident_code_mask; m; m &= m - 1) {
        const unsigned slot = (unsigned)__builtin_ctz(m);
        if (ctx->pc >= ctx->resident_code[slot].lsa &&
            ctx->pc - ctx->resident_code[slot].lsa < ctx->resident_code[slot].size) {
            code_owner = ctx->resident_code[slot].image_id;
            const uint32_t canon = ctx->resident_code[slot].canon;
            if (canon) {   /* relocatable code (spu_overlay_register_region_reloc) */
                fn = spu_lookup(ctx->pc - ctx->resident_code[slot].lsa + canon, code_owner);
                ctx->reloc_delta = (int32_t)(ctx->resident_code[slot].lsa - canon);
            } else
                fn = spu_lookup(ctx->pc, code_owner);
            break;
        }
    }
    if (!code_owner && ctx->resident_task)
        fn = spu_lookup(ctx->pc, ctx->resident_task);
    /* Resident overlay next: streamed code overwrote that LS range, so its lift
     * is the truth there -- the base image's stale bytes at the same addresses
     * may also be registered (historical junk lifts) and must lose. */
    if (!code_owner && !fn && ctx->resident_ovl) fn = spu_lookup(ctx->pc, ctx->resident_ovl);
    if (!code_owner && !fn) fn = spu_lookup(ctx->pc, ctx->image_id);
    /* Only these lookups' answer goes into the dispatch cache: the fallbacks
     * below read local store or change the overlay. */
    const int ibc_fill = fn && !code_owner && !ctx->resident_task && s_ibc_on > 0 &&
        ctx->image_id != SPU_RESOLVER_ONLY_IMAGE_ID && ctx->pc != SPURS_TASKSET_PM_SYSCALL_LS;
    const spu_fn ibc_fn = fn;
    /* Job starts: a policy module branching from its own code (LS 0xA00-0x3FFF)
     * into staged job code (SPURS_PM_PROF counts them per workload). */
    uint32_t jc_link = 0;   /* SPU_JOB_CAPTURE: nonzero = open a capture before the entry runs */
    { extern SPU_THREAD_LOCAL unsigned g_spu_job_entries;
      const uint32_t jl = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
      if (ctx->policy_mode && ctx->pc >= 0x4000 && ctx->pc < 0x3E000 && jl >= 0xA00 && jl < 0x4000) {
          g_spu_job_entries++;
#if SPU_JOBLAB
          if (spu_chprof_on()) spu_jobt_start(ctx, jl);
#endif
          if (g_spu_jobcap_on > 0) jc_link = jl;
          if (g_spu_jobrep_on > 0) spu_jobrep_jobstart(ctx);
      } }
    /* The job returned through the link register we planted: it is finished.
     * Its outermost frame ends in `bi $r0`, and r0 was 0 -- so without this the
     * return landed on LS 0, which is the job's OWN entry, and it ran a second
     * lap with dead registers: re-reading its parameters through a now-zero
     * context pointer and spinning on a DMA to what were really its own
     * opcodes. The work is already done by the time it returns. */
    if (!ctx->policy_mode && ctx->pc == SPU_JOB_RETURN_LS) {
        static int _n = 0;
        if (PS3_LOG_FIRST(_n, 4))
            fprintf(stderr, "[spurs-job] img=%d returned to the job manager \n",
                    ctx->image_id);
        spu_halt(ctx);
        return;
    }
    /* Branch into high LS with no lifted code there.
     *
     * This was added believing the jobs called a resident SPURS job-manager
     * kernel at 0x16100 / 0x18160 / 0x1A100. That was WRONG. Those "absolute
     * branches" were the disassembler decoding ASCII as instructions: the
     * bytes behind one of them are 30 2C 20 43, i.e. "0, C" from the string
     * "ch0, F:0:400, 100, Channel 0 level, %". Every job embeds the same
     * parameter-description text, which is why the same phantom target
     * appeared in 9 of 12 unrelated images and looked like a shared ABI.
     *
     * The guard is kept because branching into unlifted high local store is
     * still an error worth stopping at rather than executing whatever is
     * there -- but it is a backstop, not a kernel interface. */
    /* SPU exit: the CRT does not branch to a `stop` in the image -- it BUILDS one
     * in memory and jumps to it. ps1_netemu's GPU core ends with
     *
     *     ori $r80,$r3,0 / andi $r80,$r80,255 / iohl $r80,0x2000
     *     stqd $r80,0x10($r1) / sync / ai $r3,$r1,16 / bi $r3
     *
     * i.e. `stop (0x2000 | status)` assembled onto the stack and executed there.
     * That target is a stack address with no lifted code, so the unlifted-branch
     * guard below called a clean exit "branched into unlifted LS 0x3FFB0 -- ending
     * the job" and reported stop_code 0, which reads as a runaway and sent a long
     * chase after phantom memory corruption. Decode the target word instead: a
     * `stop` is opcode 0 in the top 11 bits, so this is unambiguous. */
    if (!fn) {
        uint32_t p0 = ctx->pc & SPU_LS_MASK;
        if (p0 + 3 < SPU_LS_SIZE) {
            uint32_t w = ((uint32_t)ctx->ls[p0] << 24) | ((uint32_t)ctx->ls[p0+1] << 16) |
                         ((uint32_t)ctx->ls[p0+2] << 8) | ctx->ls[p0+3];
            if ((w >> 21) == 0u) {                 /* stop / stopd */
                ctx->stop_code = w & 0x3FFFu;
                static int _n = 0;
                if (PS3_LOG_FIRST(_n, 8))
                    fprintf(stderr, "[spu] img=%d exit: synthesised stop 0x%04X at LS 0x%05X\n",
                            ctx->image_id, ctx->stop_code, p0);
                spu_halt(ctx);
                return;
            }
        }
    }

    if (!fn && !ctx->policy_mode && ctx->pc >= SPU_JM2_KERNEL_BASE) {
        spu_unlifted_branch_report(ctx);
        /* SPU_INTERP_UNLIFTED=1: if real code is present at the target, run it
         * through the interpreter instead of ending the job. No lift can exist
         * for a module the title loads at runtime -- You Don't Know Jack's FMOD
         * mixer relocates a DSP plugin into local store and calls it -- and the
         * interpreter rejoins the compiled path as soon as it reaches a lifted
         * address, which is what the plugin's return does. Opt-in so titles
         * that reach here on a genuinely bad pc keep the loud stop. */
        { uint32_t p = ctx->pc & SPU_LS_MASK;
          uint32_t w0 = ((uint32_t)ctx->ls[p] << 24) | ((uint32_t)ctx->ls[p+1] << 16) |
                        ((uint32_t)ctx->ls[p+2] << 8) | ctx->ls[p+3];
          static int s_iu = -1;
          if (s_iu < 0) { const char* e = getenv("SPU_INTERP_UNLIFTED"); s_iu = e ? 1 : 0; }
          if (s_iu && w0) {
              static int _n = 0;
              if (PS3_LOG_FIRST(_n, 8))
                  fprintf(stderr, "[spu] img=%d interpreting unlifted LS 0x%05X "
                          "(runtime-loaded code)\n", ctx->image_id, p);
              spu_interp_run(ctx, p);
              return;
          } }
        spu_halt(ctx);
        return;
    }
    /* WWS job-code overlay match: the PM (image 2) streams a job module into the
     * code buffer (base = LS[0x1320]) and branches to base+entryOffset. Its 16-byte
     * ila header identifies which of the 89 lifted job modules it is; match it and
     * mark that overlay resident so the module's lifted functions dispatch. The
     * module arrives via a DMA LIST, so the single-GET note_get hook never fired. */
    /* Any WWS job-manager PM, not only LBP's image 2 (GoW:A's is image 10). */
    if (!fn && (ctx->image_id == 2 || ctx->policy_mode) && ctx->pc >= 0x4000 && ctx->pc < 0x3E000) {
        uint32_t codeBase = ((uint32_t)ctx->ls[0x1320] << 24) | ((uint32_t)ctx->ls[0x1321] << 16) |
                            ((uint32_t)ctx->ls[0x1322] << 8) | ctx->ls[0x1323];
        if (codeBase + 16 <= SPU_LS_SIZE) {
            extern int spu_overlay_match_sig(const uint8_t hdr[16]);
            int img = spu_overlay_match_sig(&ctx->ls[codeBase]);
            /* SPU_JOBMOD_LOG=1: each distinct (codeBase, module header) a PM jumps
             * into with no lifted module -- which job modules run interpreted and
             * at which LS base they are staged (the base a lift has to use). */
            { static int s_jl = -1; if (s_jl < 0) s_jl = getenv("SPU_JOBMOD_LOG") ? 1 : 0;
              if (s_jl && !img) {
                  static uint32_t seen[128][2]; static int ns; static SRWLOCK lk = SRWLOCK_INIT;
                  const uint8_t* h = &ctx->ls[codeBase];
                  const uint32_t w0 = (uint32_t)h[0] << 24 | h[1] << 16 | h[2] << 8 | h[3];
                  const uint32_t w1 = (uint32_t)h[4] << 24 | h[5] << 16 | h[6] << 8 | h[7];
                  AcquireSRWLockExclusive(&lk);
                  int f = 0;
                  for (int k = 0; k < ns; k++) if (seen[k][0] == codeBase && seen[k][1] == w0) { f = 1; break; }
                  if (!f && ns < 128) {
                      seen[ns][0] = codeBase; seen[ns][1] = w0; ns++;
                      fprintf(stderr, "[jobmod] img=%d codeBase=0x%05X pc=0x%05X (entry+0x%X) hdr=%08X %08X "
                              "%02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                              ctx->image_id, codeBase, ctx->pc & SPU_LS_MASK, (ctx->pc & SPU_LS_MASK) - codeBase,
                              w0, w1, h[8], h[9], h[10], h[11], h[12], h[13], h[14], h[15],
                              h[16], h[17], h[18], h[19]);
                  }
                  ReleaseSRWLockExclusive(&lk);
              } }
            if (img) {
                if (ctx->resident_ovl != img) {
                    ctx->resident_ovl = img;
                    static int _n = 0; if (PS3_LOG_FIRST(_n, 32))
                        fprintf(stderr, "[spu-ovl] WWS job module resident: codeBase=0x%05X "
                                "pc=0x%05X -> overlay image %d\n", codeBase, ctx->pc & SPU_LS_MASK, img);
                }
                fn = spu_lookup(ctx->pc, img);
            }
        }
    }
    /* WWS job-code fallback: the jobmanager PM (image 2) stages each job's
     * code into an LS buffer and branches into it. The staged blobs are
     * lifted as their own images at their LS residence addresses (image 1 =
     * job_wws_7BD900 at the 0x14400-region buffer); the PM's own table can
     * never contain them. On a policy-module miss inside the job-buffer
     * range, retry against the job image before falling to the interpreter. */
    /* WWS job-code fallback -- but ONLY when real code is staged at the target.
     * A PM job whose code buffer is empty (the {0xA400,0} null-code placeholder)
     * must NOT resolve to image-1's unrelated function that happens to share the
     * file offset: that ran garbage, the job did no work, and its ring slot
     * never advanced -> the loader's job ring deadlocked. An empty buffer falls
     * through to the null-job handling below. */
    if (!fn && ctx->image_id == 2 && ctx->policy_mode &&
        ctx->pc >= 0x4000 && ctx->pc < 0x3E000) {
        uint32_t w0 = ((uint32_t)ctx->ls[ctx->pc] << 24) | ((uint32_t)ctx->ls[ctx->pc+1] << 16) |
                      ((uint32_t)ctx->ls[ctx->pc+2] << 8) | ctx->ls[ctx->pc+3];
        if (w0 != 0)
            fn = spu_lookup(ctx->pc, 1);
    }
    /* Null-code PM job (empty buffer): return control to the dispatcher's own
     * host-call bracket so its post-job continuation runs the completion
     * bookkeeping -- exactly as a real job that did nothing and returned.
     * (Falling to the SMC microstep would log a spurious BRANCH-TO-0 on the
     * 0x00000000 word; this is the same outcome, quiet.) */
    if (!fn && ctx->image_id == 2 && ctx->policy_mode &&
        ctx->pc >= 0x4000 && ctx->pc < 0x3E000 &&
        ((uint32_t)ctx->ls[ctx->pc] | ctx->ls[ctx->pc+1] |
         ctx->ls[ctx->pc+2] | ctx->ls[ctx->pc+3]) == 0) {
        static int _n = 0;
        if (PS3_LOG_FIRST(_n, 8)) {
            /* Ground-truth dump of the WWS code-buffer resolution at dispatch
             * (wwsjob-rosetta offsets). jobEntry = lsaJobCodeBuffer + [buf+0x10];
             * lsaJobCodeBuffer = Gbt_bufferPageNum<<10 from the RunJob command.
             * If the code buffer really is 0x4000 but empty -> the code LOAD is
             * the bug; if lsaJobCodeBuffer should be 0x14400 -> the RunJob
             * buffer-set resolution is. Dump both + the bufferSetArray rows. */
            uint32_t lsaCode = (uint32_t)ctx->ls[0x1330]<<24 | ctx->ls[0x1331]<<16 |
                               ctx->ls[0x1332]<<8 | ctx->ls[0x1333];
            uint32_t runJob  = (uint32_t)ctx->ls[0x12E0]<<24 | ctx->ls[0x12E1]<<16 |
                               ctx->ls[0x12E2]<<8 | ctx->ls[0x12E3];
            const uint8_t* bs = ctx->ls + 0xDF0;   /* g_WwsJob_bufferSetArray */
            fprintf(stderr, "[pm-jobres] pc=0x%05X lsaJobCodeBuffer(0x1330)=0x%08X "
                    "runJobNum(0x12E0)=0x%08X code@14400:%02X%02X%02X%02X\n"
                    "           bufferSetArray[0xDF0]: %02X%02X%02X%02X %02X%02X%02X%02X "
                    "%02X%02X%02X%02X %02X%02X%02X%02X\n",
                    ctx->pc & SPU_LS_MASK, lsaCode, runJob,
                    ctx->ls[0x14400],ctx->ls[0x14401],ctx->ls[0x14402],ctx->ls[0x14403],
                    bs[0],bs[1],bs[2],bs[3], bs[4],bs[5],bs[6],bs[7],
                    bs[8],bs[9],bs[10],bs[11], bs[12],bs[13],bs[14],bs[15]);
        }
        return;   /* enclosing SPU_DRAIN bracket resumes the dispatcher */
    }
    /* JOB-BODY EXECUTION probe: the WWS job manager enters a staged job via
     * `bie pJobCode` (changeloadtorunjob.spu), an INDIRECT branch to
     * lsaJobCodeBuffer + entryOffset. A successful dispatch with the pc inside
     * the job code buffer and a jobmod overlay resident (image ids 200+) is the
     * proof the lifted job module actually runs. Env SPU_JOBEXEC=1. */
    /* SPU_JOBTRACE=1: log EVERY dispatcher entry while a job module (overlay
     * id >= 200) is resident -- the job's cross-function control-flow trail.
     * The pm-flow drain hook can't see a job that performs no channel ops
     * (LBP's frozen job runs as a complete no-op: entry, no DMAs, return);
     * this names its early-out path instead. */
    { static int s_jt = -1;
      if (s_jt < 0) { const char* e = getenv("SPU_JOBTRACE"); s_jt = e ? 1 : 0; }
      if (s_jt && ctx->resident_ovl >= 200) { static int _n = 0;
        uint32_t p = ctx->pc & SPU_LS_MASK;
        /* Full trail, but only start logging once the job body proper begins
         * (pc in the job code buffer >= 0x4000) so the cap isn't spent on the
         * PM code the job calls back into -- and keep logging thereafter. */
        static int _armed = 0;
        if (p >= 0x4000) _armed = 1;
        if (_armed && PS3_LOG_FIRST(_n, 1500)) {
            fprintf(stderr, "[jobtrace] pc=0x%05X lr=0x%05X%s",
                    p, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                    fn ? "" : " (no lift)");
            /* JobApi entries: args r3-r5. Post-job return 0x3308: result r3 +
             * the job's saved GetBufferTag result r80. */
            if (p == 0x2F30 || p == 0x1700 || p == 0x1770 || p == 0x17C8)
                fprintf(stderr, " args r3=%08X.%08X r4=%08X r5=%08X",
                        ctx->gpr[3]._u32[0], ctx->gpr[3]._u32[1],
                        ctx->gpr[4]._u32[0], ctx->gpr[5]._u32[0]);
            if (p == 0x3308 || p == 0x3258)
                fprintf(stderr, " r3=%08X.%08X.%08X.%08X r80=%08X.%08X",
                        ctx->gpr[3]._u32[0], ctx->gpr[3]._u32[1],
                        ctx->gpr[3]._u32[2], ctx->gpr[3]._u32[3],
                        ctx->gpr[80]._u32[0], ctx->gpr[80]._u32[1]);
            /* Mid-GetBufferTag: r8 = the set/buf id bytes computed at 0x2F30
             * entry; if wrong here, something clobbered it ACROSS the
             * GetLogicalBuffer call (interrupt delivery?). */
            if (p == 0x2F6C)
                fprintf(stderr, " r8=%08X r10=%08X r13=%08X r15=%08X int=%d",
                        ctx->gpr[8]._u32[0], ctx->gpr[10]._u32[0],
                        ctx->gpr[13]._u32[0], ctx->gpr[15]._u32[0],
                        ctx->int_enable);
            if (p == 0x3030)   /* final BufferTag in r7 (result) */
                fprintf(stderr, " BufferTag r7=%08X.%08X",
                        ctx->gpr[7]._u32[0], ctx->gpr[7]._u32[1]);
            fprintf(stderr, "\n");
        }
      } }
    /* LBP_HLE_JOBDONE=1: HLE the WWS job COMPLETION. Our lifted SPURS PM never
     * reaches the store/DecrementDependency phase (the multi-SPU barrier never
     * converges), so a job's completion word -- passed by the PPU as param[4],
     * an EA in the 0x470Axxxx pool, embedded in the job's command list at LS
     * ~0xC00 -- is never written, and the PPU JobManagerWorker (guest-fn
     * 0x521BD4) spins on it forever (loading freeze). When a job body actually
     * dispatches (overlay >=200, entry 0x4A40), scan its staged command list
     * for the completion EA and write it nonzero, unblocking the worker. This
     * fakes ONLY the done-signal (not the job's data DMA), so it tests how far
     * the boot gets once loading stops deadlocking. */
    if (ctx->pc == 0x4A40 && ctx->resident_ovl >= 200) {
        static int s_hd = -1;
        if (s_hd < 0) { const char* e = getenv("LBP_HLE_JOBDONE"); s_hd = e ? 1 : 0; }
        if (s_hd) {
            extern uint8_t* vm_base;
            int wrote = 0;
            for (uint32_t o = 0xC00; o + 4 <= 0xE80 && wrote < 4; o += 4) {
                uint32_t w = (ctx->ls[o]<<24)|(ctx->ls[o+1]<<16)|(ctx->ls[o+2]<<8)|ctx->ls[o+3];
                if ((w & 0xFFFF0000u) == 0x470A0000u && vm_base) {
                    /* write nonzero to the completion word in main RAM (BE) */
                    if (vm_base[w]==0 && vm_base[w+1]==0 && vm_base[w+2]==0 && vm_base[w+3]==0) {
                        vm_base[w+3] = 1;
                        static int _n = 0; if (PS3_LOG_FIRST(_n, 24))
                            fprintf(stderr, "[hle-jobdone] wrote completion 0x%08X=1 (job overlay %d)\n",
                                    w, ctx->resident_ovl);
                        wrote++;
                    }
                }
            }
        }
    }
    if (fn && ctx->pc >= 0x4000 && ctx->resident_ovl >= 200) {
        static int s_je = -1;
        if (s_je < 0) { const char* e = getenv("SPU_JOBEXEC"); s_je = e ? 1 : 0; }
        if (s_je) { static int _n = 0; if (PS3_LOG_FIRST(_n, 24))
            fprintf(stderr, "[jobexec] DISPATCH job code pc=0x%05X overlay=%d "
                    "runJobNum=0x%02X%02X%02X%02X\n",
                    ctx->pc & SPU_LS_MASK, ctx->resident_ovl,
                    ctx->ls[0x12E0], ctx->ls[0x12E1], ctx->ls[0x12E2], ctx->ls[0x12E3]);
          /* SPU_JOBEXEC_DUMP=<path>: write the full 256KB LS the first time a job
           * entry (pc==codeBuffer+entryOffset) dispatches, to diff vs the RPCS3
           * oracle SPU4 dump (why our job bails where the oracle's runs). */
          static int s_dumped = 0;
          if (!s_dumped && ctx->pc == 0x4A40) {
              const char* dp = getenv("SPU_JOBEXEC_DUMP");
              if (dp) { FILE* f = fopen(dp, "wb");
                  if (f) { fwrite(ctx->ls, 1, SPU_LS_SIZE, f); fclose(f);
                      fprintf(stderr, "[jobexec] dumped LS at job entry -> %s\n", dp); }
                  s_dumped = 1; } } }
    }
    if (fn) {
        if (ibc_fill && fn == ibc_fn) spu_ibc_fill(ctx, fn);
        /* After the overlay match above, so the header names the overlay this
         * entry resolved to. */
        if (jc_link) spu_jobcap_open(ctx, jc_link);
        spu_check_stack_reset(ctx, fn);
        /* MUSTTAIL: a guest loop that iterates through an indirect branch (the
         * Bink decoder's per-command dispatch does) must not grow the host
         * stack -- a plain call here leaked a resolver+callee frame per
         * iteration and blew the thread stack ~4k iterations into the first
         * really-decoding movie frame (silent 0x80000001 death). */
#if defined(__clang__)
        __attribute__((musttail)) return fn(ctx);
#else
        fn(ctx);
        return;
#endif
    }
    /* wwsjob JOB-CODE entry probe: the PM stages each job's code into a
     * buffer above its static image (0x3700..) and branches into it. No lift
     * exists at those pcs (the code arrives at runtime), so the branch lands
     * here. Log the entry + leading bytes -- the bytes identify WHICH job
     * blob was staged (match against lifted job images for dispatch). */
    /* Staged-job "return to kernel": a jm2-protocol job binary ends by
     * jumping to LS 0 (the jm2 kernel's home on real hardware). Inside the
     * wwsjob PM's LS, address 0 holds the interrupt VECTOR instead -- letting
     * the jump proceed runs the interrupt handler as the job's continuation
     * and wedges the pipeline (lanes freeze mid-queue). The PM called the job
     * with a return link in r0 (0x31C8 sets 0x3258); route the kernel-return
     * there: the PM resumes its post-job flow, which is exactly what the jm2
     * kernel hand-back does on hardware. */
    if (ctx->image_id == 2 && ctx->policy_mode && ctx->pc == 0 &&
        (ctx->gpr[0]._u32[0] & SPU_LS_MASK) >= 0xA00 &&
        (ctx->gpr[0]._u32[0] & SPU_LS_MASK) < 0x3700) {
        static int _n = 0;
        if (PS3_LOG_FIRST(_n, 8))
            fprintf(stderr, "[wws-jobret] staged job returned to kernel (LS 0); "
                    "resuming PM at link 0x%05X\n", ctx->gpr[0]._u32[0] & SPU_LS_MASK);
        ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
        spu_indirect_branch(ctx);      /* re-dispatch at the rewritten pc */
        return;
    }

    /* A SPURS *job* that branches back to LS 0 has RETURNED TO THE JOB MANAGER.
     * That is how a jm2 job signals completion: its crt tail-jumps to the
     * resident manager, which on hardware loads the next job and sets up r3/r4
     * afresh. We load the job at LS 0 and have no manager, so the jump lands on
     * the job's OWN entry and it runs a second lap -- with dead registers.
     *
     * Tokyo Jungle showed exactly that: lap one is correct at every step and
     * does the real work, then the crt jumps to 0 and lap two re-reads its
     * parameters through a now-zero context pointer (hence the reads of LS 0x20
     * and the DMA addresses that were really its own opcodes) and spins forever.
     * The work was already finished before the jump; the second lap is noise.
     *
     * The initial entry is called directly, not through here, so any arrival at
     * LS 0 in this path is a re-entry. Treat it as job end. */
    if (!ctx->policy_mode && ctx->pc == 0 && ctx->image_id > 0) {
        static int _n = 0;
        if (PS3_LOG_FIRST(_n, 8))
            fprintf(stderr, "[spurs-job] img=%d returned to the job manager "
                    "(branch to LS 0) -- job complete\n", ctx->image_id);
        spu_halt(ctx);
        return;
    }
    if (ctx->image_id == 2 &&
        ((ctx->pc >= 0x3700 && ctx->pc < 0x3FE80) ||
         (ctx->pc < 0xA00 && ctx->pc != SPURS_PM_EXIT_TO_KERNEL_LS &&
          ctx->pc != SPURS_PM_SELECT_WORKLOAD_LS))) {
        static int _n = 0;
        if (_n < 12) {
            _n++;
            const uint8_t* p = ctx->ls + (ctx->pc & SPU_LS_MASK);
            fprintf(stderr, "[wws-jobentry] pc=0x%05X lr=0x%05X bytes:"
                    " %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                    ctx->pc & SPU_LS_MASK, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                    p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7],
                    p[8],p[9],p[10],p[11], p[12],p[13],p[14],p[15]);
            /* Companion state: the OTHER load target (the real 0x100-byte
             * loads land at 0x14400 -- possibly the actual job code, with the
             * entry computed from the wrong slot's base) + the job records
             * the entry math reads (0x1320 base block, 0x1440 batch record). */
            { const uint8_t* q = ctx->ls + 0x14400;
              const uint8_t* r1 = ctx->ls + 0x1320;
              const uint8_t* r2 = ctx->ls + 0x1440;
              fprintf(stderr, "[wws-jobentry]   LS14400: %02X%02X%02X%02X %02X%02X%02X%02X"
                      "  LS1320: %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X"
                      "  LS1440: %02X%02X%02X%02X %02X%02X%02X%02X\n",
                      q[0],q[1],q[2],q[3], q[4],q[5],q[6],q[7],
                      r1[0],r1[1],r1[2],r1[3], r1[4],r1[5],r1[6],r1[7],
                      r1[8],r1[9],r1[10],r1[11], r1[12],r1[13],r1[14],r1[15],
                      r2[0],r2[1],r2[2],r2[3], r2[4],r2[5],r2[6],r2[7]); }
            fflush(stderr);
        }
    }
    /* SPU_WWSJOB=1 (GoW:A WWS PM, image 10): log each entry from the policy module
     * (lr below 0x4000) into staged job code (LS >= 0x4000) with the running wid
     * (kernel ctx wklCurrentId @0x1DC) -- which jobs actually start, per workload. */
    { static int s_wj = -1; if (s_wj < 0) s_wj = getenv("SPU_WWSJOB") ? 1 : 0;
      const uint32_t p = ctx->pc & SPU_LS_MASK, lr = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
      /* ...and each call from job code into the PM (JobApi): pc in the PM image,
       * link inside the staged job. The result shows as r3 on the next entry. */
      if (s_wj && ctx->image_id == 10 && ctx->policy_mode && p >= 0xA00 && p < 0x3700 &&
          lr >= 0x4000 && lr < 0x3E000) {
          static int _a[256];
          const uint8_t* L = ctx->ls;
          /* ...and past the first 60, only calls whose r1 differs from the last call
           * from the same site: a stack that drifts between jobs shows up as a step. */
          static uint32_t s_r1[256][64], s_from[256][64]; static int s_drift;
          const unsigned w = L[0x1DF], k = (lr >> 2) & 63, r1 = ctx->gpr[1]._u32[0] & SPU_LS_MASK;
          const int drift = s_from[w][k] == lr && s_r1[w][k] != r1;
          s_from[w][k] = lr; s_r1[w][k] = r1;
          if (_a[w]++ < 60 || (drift && s_drift++ < 2000))
              fprintf(stderr, "[wwsapi] wid=%u r1=0x%05X call pc=0x%05X from=0x%05X r3=%08X.%08X r4=%08X r5=%08X r6=%08X r7=%08X\n",
                      L[0x1DF], ctx->gpr[1]._u32[0] & SPU_LS_MASK, p, lr, ctx->gpr[3]._u32[0], ctx->gpr[3]._u32[1], ctx->gpr[4]._u32[0],
                      ctx->gpr[5]._u32[0], ctx->gpr[6]._u32[0], ctx->gpr[7]._u32[0]);
      }
      if (s_wj && ctx->image_id == 10 && ctx->policy_mode && p >= 0x4000 && p < 0x3E000 && lr < 0x4000) {
          static int _n[256];
          const uint8_t* L = ctx->ls;
          if (_n[L[0x1DF]]++ < 40) {
              fprintf(stderr, "[wwsjob] wid=%u r1=0x%05X enter pc=0x%05X lr=0x%05X r3=%08X r4=%08X r5=%08X "
                      "code=%02X%02X%02X%02X%02X%02X%02X%02X loadJob@12D0=%02X%02X%02X%02X runJob@12E0=%02X%02X%02X%02X "
                      "state@12A0=%02X%02X%02X%02X\n",
                      L[0x1DF], ctx->gpr[1]._u32[0] & SPU_LS_MASK, p, lr, ctx->gpr[3]._u32[0],
                      ctx->gpr[4]._u32[0], ctx->gpr[5]._u32[0],
                      L[p], L[p+1], L[p+2], L[p+3], L[p+4], L[p+5], L[p+6], L[p+7],
                      L[0x12D0], L[0x12D1], L[0x12D2], L[0x12D3], L[0x12E0], L[0x12E1], L[0x12E2], L[0x12E3],
                      L[0x12A0], L[0x12A1], L[0x12A2], L[0x12A3]);
          }
      } }
    /* No lifted function at this PC: it may be RUNTIME-GENERATED code (the
     * WWS jobmanager writes a save/restore stub above its static
     * image and calls it). Interpret the live LS bytes; on success the next
     * branch re-enters lifted code via the trampoline. */
    if (spu_smc_microstep(ctx))
        return;
    /* SPU_INTERP_UNLIFTED=1 covers this site too: code staged at a placement no
     * lift saw (GoW:A's EDGE fragments land at a second set of LS addresses when
     * the WWS manager pipelines jobs) runs in the interpreter, which rejoins the
     * compiled path at the next lifted address, instead of ending the job. */
    { const uint32_t p = ctx->pc & SPU_LS_MASK;
      const uint32_t w0 = ((uint32_t)ctx->ls[p] << 24) | ((uint32_t)ctx->ls[p+1] << 16) |
                          ((uint32_t)ctx->ls[p+2] << 8) | ctx->ls[p+3];
      static int s_iu = -1;
      if (s_iu < 0) s_iu = getenv("SPU_INTERP_UNLIFTED") ? 1 : 0;
      if (s_iu && w0) {
          static int _n = 0;
          if (PS3_LOG_FIRST(_n, 8))
              fprintf(stderr, "[spu] img=%d interpreting unlifted LS 0x%05X (lr 0x%05X)\n",
                      ctx->image_id, p, ctx->gpr[0]._u32[0] & SPU_LS_MASK);
          spu_interp_run(ctx, p);
          return;
      } }
    /* Cap the unresolved-branch log PER IMAGE: a global cap let one noisy
     * image (the FMOD mixer's overlay calls) exhaust it and silently hide
     * every other image's misses -- LBP's loading jobs skipped their command
     * handlers for a whole session without a single log line. */
    { enum { BT0_MAX_IMG = 64, BT0_PER_IMG = 12 };
      static int _bt0[BT0_MAX_IMG];
      unsigned img = (ctx->image_id >= 0 && ctx->image_id < BT0_MAX_IMG)
                     ? (unsigned)ctx->image_id : 0;
      if (PS3_LOG_FIRST(_bt0[img], BT0_PER_IMG)) {
        /* Words at pc-4/pc: a staged WWS job's assert is `stopd` (0x280xxxxx)
         * or `stop` (0x000xxxxx) -- the interpreter halts there and lands here. */
        const uint32_t p = ctx->pc & SPU_LS_MASK & ~3u, q = (p - 4) & SPU_LS_MASK;
        const uint32_t wq = (uint32_t)ctx->ls[q]<<24 | ctx->ls[q+1]<<16 | ctx->ls[q+2]<<8 | ctx->ls[q+3];
        const uint32_t wp = (uint32_t)ctx->ls[p]<<24 | ctx->ls[p+1]<<16 | ctx->ls[p+2]<<8 | ctx->ls[p+3];
        fprintf(stderr, "[SPU] BRANCH-TO-0 unresolved pc=0x%05X image=%d lr=0x%05X "
                "ls[pc-4]=%08X ls[pc]=%08X r3=%08X\n",
                ctx->pc, ctx->image_id, ctx->gpr[0]._u32[0] & SPU_LS_MASK,
                wq, wp, ctx->gpr[3]._u32[0]); } }
    /* One-shot: the FMOD null-handler DSP node carries a PPU descriptor EA at
     * node+0x14 (observed 0x93C3C0). Dump it to identify which plugin/unit
     * type never got its SPU code streamed (env SPU_DSPDESC=<hex ea>). */
    { static int _d = -1; static uint32_t _ea = 0;
      if (_d < 0) { const char* e = getenv("SPU_DSPDESC");
        _ea = e ? (uint32_t)strtoul(e, 0, 16) : 0; _d = _ea ? 1 : 0; }
      if (_d == 1) { _d = 2;
        extern uint8_t* vm_base;
        fprintf(stderr, "[dspdesc] RAM[0x%08X]:", _ea);
        for (int k = 0; k < 0x60; k += 4)
            fprintf(stderr, " %02X%02X%02X%02X", vm_base[_ea+k], vm_base[_ea+k+1],
                    vm_base[_ea+k+2], vm_base[_ea+k+3]);
        fprintf(stderr, "\n"); fflush(stderr); } }
    /* SPU_MISS_DUMP_IMG=<n>: reserve the deep-dump budget for image n's misses
     * (the global 2-shot budget was always consumed by an earlier image's
     * misses, hiding the one under investigation). Unset = old behavior. */
    { static int s_img = -2; static uint32_t s_pcmin = 0;
      if (s_img == -2) { const char* e = getenv("SPU_MISS_DUMP_IMG"); s_img = e ? atoi(e) : -1;
        const char* p = getenv("SPU_MISS_DUMP_PCMIN"); s_pcmin = p ? (uint32_t)strtoul(p,0,0) : 0; }
      static int _n=0;
      uint32_t misspc = ctx->pc & SPU_LS_MASK;
      if ((s_img < 0 || ctx->image_id == s_img) && misspc >= s_pcmin && PS3_LOG_FIRST(_n, 2)) {
        fprintf(stderr, "[SPU] branch-to-0 lr=0x%05X r1=0x%05X\n",
                ctx->gpr[0]._u32[0] & SPU_LS_MASK, ctx->gpr[1]._u32[0] & SPU_LS_MASK);
#ifdef _WIN32
        void* frames[24]; unsigned short fn = RtlCaptureStackBackTrace(0, 24, frames, NULL);
        char* base = (char*)GetModuleHandleA(NULL);
        fprintf(stderr, "[SPU] host bt RVAs:");
        for (unsigned short i = 0; i < fn; i++)
            fprintf(stderr, " 0x%zX", (size_t)((char*)frames[i] - base));
        fprintf(stderr, "\n");
#endif
        /* State-diff oracle: dump full LS + all GPRs at the branch-to-0 so it can
         * be compared byte-for-byte against the RPCS3 savestate LS of the same
         * (cri_mpv) task. Path from YDKJ_SPU_LSDUMP, else ./recomp_spu_ls.bin. */
        const char* dp = getenv("YDKJ_SPU_LSDUMP");
        if (!dp || !*dp) dp = "recomp_spu_ls.bin";
        FILE* lf = fopen(dp, "wb");
        if (lf) { fwrite(ctx->ls, 1, SPU_LS_SIZE, lf); fclose(lf);
                  fprintf(stderr, "[SPU] dumped 256KB LS -> %s\n", dp); }
        fprintf(stderr, "[SPU] image_id=%d  GPR dump (r0..r127, hi64:lo64 of each quadword, preferred slot = _u32[0]):\n", ctx->image_id);
        for (int g = 0; g < 128; g++) {
            fprintf(stderr, " r%-3d=%08X %08X %08X %08X", g,
                    ctx->gpr[g]._u32[0], ctx->gpr[g]._u32[1],
                    ctx->gpr[g]._u32[2], ctx->gpr[g]._u32[3]);
            if ((g & 1) == 1) fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
        /* echo the dispatch chain values the way func_00026DE0 computes them */
        { uint32_t bec0 = ctx->ls[0xBEC0]<<24 | ctx->ls[0xBEC1]<<16 | ctx->ls[0xBEC2]<<8 | ctx->ls[0xBEC3];
          fprintf(stderr, "[SPU] LS[0xBEC0].w0=0x%08X  LS[0x2d4e0:16]=%02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
            bec0,
            ctx->ls[0x2d4e0],ctx->ls[0x2d4e1],ctx->ls[0x2d4e2],ctx->ls[0x2d4e3],
            ctx->ls[0x2d4e4],ctx->ls[0x2d4e5],ctx->ls[0x2d4e6],ctx->ls[0x2d4e7],
            ctx->ls[0x2d4e8],ctx->ls[0x2d4e9],ctx->ls[0x2d4ea],ctx->ls[0x2d4eb],
            ctx->ls[0x2d4ec],ctx->ls[0x2d4ed],ctx->ls[0x2d4ee],ctx->ls[0x2d4ef]); }
    } }
    ctx->status = SPU_STATUS_STOPPED_BY_HALT;
}

/* ===========================================================================
 * Execution trace (for §3 validation: diff vs RPCS3 SPU interpreter)
 *
 * When the lifter is invoked with --trace, every emitted instruction is
 * surrounded by spu_trace_pc(ctx, PC) before execution and spu_trace_rt(
 * ctx, RT) after, for instructions whose destination is the rt slot. The
 * output is one line per event:
 *
 *     <PC-5hex>                          - PC about to execute
 *       r<rt> <hi-64hex> <lo-64hex>      - register written, post-state
 *
 * Direct to stderr by default; call spu_trace_init(path) once at startup
 * to redirect to a file. The format is intentionally minimal and stable
 * so a small converter can line it up against an RPCS3.log SPU trace.
 * ===========================================================================*/
static FILE* s_trace_fp = NULL;

void spu_trace_init(const char* path)
{
    if (!path || !*path) { s_trace_fp = stderr; return; }
    s_trace_fp = fopen(path, "w");
    if (!s_trace_fp) s_trace_fp = stderr;
}

/* Bounded by design: a --trace lift emits one call PER INSTRUCTION, so an image
 * that spins (a persistent SPURS policy module polling its job queue never
 * returns) would otherwise write stderr until the disk fills. The cap keeps the
 * useful part -- the entry path plus enough revolutions to show what the loop
 * tests -- and the tail IS the loop. SPU_TRACE_MAX=0 restores unbounded.
 * SPU_TRACE_FILE redirects off stderr so the trace does not interleave with the
 * boot log. */
/* Trace output is OPT-IN at runtime: a --trace lift bakes the calls into the
 * generated C forever, so a trace-lifted image shipped in a title build (LBP's
 * job kernel, kept from the WWS verification) would otherwise spew the
 * per-instruction log into every boot. Enable with SPU_TRACE=1 (stderr) or
 * SPU_TRACE_FILE=<path>. */
static long long s_trace_left = -1;     /* -1 uninit, -2 unbounded, 0 off */
/* Set when spu_trace_pc suppressed its line (gates below): the paired
 * spu_trace_rt call for the same instruction must also stay silent. */
static SPU_THREAD_LOCAL int s_trace_suppress;

void spu_trace_pc(spu_context* ctx, uint32_t pc)
{
    (void)ctx;
    /* SPU_TRACE_AT_WALL=1: hold the trace until the PPU's job-barrier probe
     * arms g_barrier_sync_watch, so the budget covers the post-ticket claim/
     * stage pass instead of boot-time idle polling. */
    { static int s_aw = -1;
      if (s_aw < 0) s_aw = getenv("SPU_TRACE_AT_WALL") ? 1 : 0;
      if (s_aw) { extern uint32_t g_barrier_sync_watch;
                  if (!g_barrier_sync_watch) { s_trace_suppress = 1; return; } } }
    /* SPU_TRACE_FLOWCTX=1: trace ONLY the run the SPURS_PM_FLOW tracer armed
     * (the stalled queue's spu0 dispatch) -- post-wall idle dispatches of the
     * other 13 jobmanager wids otherwise exhaust the budget in under a second. */
    { static int s_fc = -1;
      if (s_fc < 0) s_fc = getenv("SPU_TRACE_FLOWCTX") ? 1 : 0;
      if (s_fc) { extern void* volatile g_pm_flow_ctx;
                  if (g_pm_flow_ctx != (void*)ctx) { s_trace_suppress = 1; return; } } }
    /* SPU_TRACE_IMG=<n>: trace only contexts running image n (e.g. 1 = the
     * WWS job binary, to catch the crt corrupting the job's r3/r4 args). */
    { static int64_t s_ti = -2;
      if (s_ti == -2) { const char* e = getenv("SPU_TRACE_IMG");
                        s_ti = e ? atoll(e) : -1; }
      if (s_ti >= 0 && ctx->image_id != (int)s_ti) { s_trace_suppress = 1; return; } }
    s_trace_suppress = 0;
    if (s_trace_left < 0) {
        if (s_trace_left == -1) {
            const char* p = getenv("SPU_TRACE_FILE");
            if ((!p || !*p) && !getenv("SPU_TRACE")) { s_trace_left = 0; return; }
            if (p && *p && !s_trace_fp) spu_trace_init(p);
            const char* m = getenv("SPU_TRACE_MAX");
            s_trace_left = (m && *m) ? atoll(m) : 200000;
            if (s_trace_left == 0) s_trace_left = -2;   /* -2 = unbounded */
        }
    }
    if (!s_trace_fp) s_trace_fp = stderr;
    if (s_trace_left == 0) return;
    if (s_trace_left > 0 && --s_trace_left == 0) {
        fprintf(s_trace_fp, "-- SPU_TRACE_MAX reached; trace stopped --\n");
        fflush(s_trace_fp);
        return;
    }
    fprintf(s_trace_fp, "%05X\n", pc & SPU_LS_MASK);
}

void spu_trace_rt(spu_context* ctx, uint32_t rt)
{
    if (s_trace_suppress) return;          /* paired _pc line was gated out */
    if (s_trace_left == 0) return;         /* tracing off/stopped (see _pc) */
    if (!s_trace_fp) s_trace_fp = stderr;
    u128 v = ctx->gpr[rt & 0x7F];
    fprintf(s_trace_fp, "  r%-3u %016llX %016llX\n",
            (unsigned)(rt & 0x7F),
            (unsigned long long)v._u64[0],
            (unsigned long long)v._u64[1]);
}

#ifdef __cplusplus
}
#endif