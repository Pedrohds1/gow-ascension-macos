/*
 * ps3recomp - PPU <-> SPU lock-line coherence
 *
 * See spu_coherency.h for what this is for. The state here is deliberately
 * plain: a bit per 128-byte line, a small table of the SPU contexts that have
 * ever reserved one, and the lock-line spinlock (striped by line with
 * SPU_LOCKLINE_STRIPES; then the bitmap OR is atomic and the holder table has
 * its own lock). Everything that WRITES it runs
 * with the lock-line lock held (the SPU's GETLLAR, the PPU's coherent store),
 * so the read-modify-write on a shared bitmap byte is serialized. Only
 * spu_coh_is_reserved reads without the lock, which is the entire point: it is
 * on every PPU store. The one other unlocked path is a GETLLAR that re-reserves
 * a marked line (spu_coh_reserve_fast); its ordering argument is there.
 */

#include <stdlib.h>
#include "spu_coherency.h"
#include "spu_coh_bitmap.h"

#include <stdint.h>
#include <stdio.h>
#include "ps3emu/log_first.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * The lock-line lock
 *
 * Moved here from spu_channels.c, where it was a file-static and so could
 * serialize SPU against SPU but not SPU against PPU -- which left the PPU free
 * to write a line in the middle of a PUTLLC's compare-and-commit. Same
 * spinlock, same semantics, now shared. _InterlockedExchange is a clang-cl/MSVC
 * intrinsic (no runtime library symbol needed); elsewhere use the C11
 * equivalent, which lowers to the same LL/SC or lock-xchg on every target.
 * ===========================================================================*/
#if defined(_MSC_VER)
#include <intrin.h>
#include <windows.h>   /* SwitchToThread */
static volatile long g_lockline = 0;
/* Test-and-test-and-set: spin on a plain read (no cache-line ping-pong) with a
 * pause, and yield after a long wait. The bare exchange loop let GH3's four
 * idle job-policy pollers saturate the line and delay every other SPU atomic
 * -- FMOD's mixer task among them, 25-40 ms at a time. */
/* Per-thread time spent waiting for the global lock line (QPC ticks); read by
 * the WAIT_SIGNAL log (SPU_WAITLOG) to tell lock contention from task work. */
__declspec(thread) unsigned long long g_spu_lockline_wait_ticks;
__declspec(thread) unsigned long long g_spu_lockline_count;
void spu_lockline_lock(void)
{
    unsigned spins = 0;
    g_spu_lockline_count++;
    if (!_InterlockedExchange(&g_lockline, 1)) return;
    LARGE_INTEGER t0, t1; QueryPerformanceCounter(&t0);
    do {
        do {
            _mm_pause();
            if (++spins >= 4096) { spins = 0; SwitchToThread(); }
        } while (g_lockline);
    } while (_InterlockedExchange(&g_lockline, 1));
    QueryPerformanceCounter(&t1);
    g_spu_lockline_wait_ticks += (unsigned long long)(t1.QuadPart - t0.QuadPart);
}
void spu_lockline_unlock(void) { _InterlockedExchange(&g_lockline, 0); }
/* One lock here: the line and span forms are the whole lock (see the POSIX
 * path for the striped version). */
void spu_lockline_lock_line(uint32_t ea) { (void)ea; spu_lockline_lock(); }
void spu_lockline_unlock_line(uint32_t ea) { (void)ea; spu_lockline_unlock(); }
void spu_lockline_lock_span(uint32_t ea, uint32_t size) { (void)ea; (void)size; spu_lockline_lock(); }
void spu_lockline_unlock_span(uint32_t ea, uint32_t size) { (void)ea; (void)size; spu_lockline_unlock(); }
int spu_lockline_striped(void) { return 0; }
static inline unsigned ll_count(void) { return 1u; }
static void ll_barrier(void) { spu_lockline_lock(); spu_lockline_unlock(); }
static inline void coh_reg_lock(void) {}
static inline void coh_reg_unlock(void) {}
static inline int llp_on(void) { return 0; }
static inline void llp_notify(uint32_t ea, int found, uint64_t tb, uint64_t tf, uint64_t ts) { (void)ea; (void)found; (void)tb; (void)tf; (void)ts; }
static inline uint64_t llp_tick(void) { return 0; }
#else
#include <stdatomic.h>
#include <sched.h>     /* sched_yield */
#include <time.h>      /* SPU_LOCKLINE_PROF */
#include <dlfcn.h>     /* SPU_LOCKLINE_PROF: name the all-stripe callers */
#include "../platform/win32_compat.h"   /* YieldProcessor, QueryPerformanceCounter */
/* The Windows path's test-and-test-and-set, for the same reason: a bare
 * exchange loop keeps the line bouncing between the spinning cores and delays
 * every other SPU atomic behind the pollers.
 *
 * Striped (SPU_LOCKLINE_STRIPES=<n>, a power of two up to 64; 1, the default,
 * is the single process-wide lock this always was). With the SPURS lanes the
 * lock was the hottest function on every EDGE thread and on the main thread,
 * and nearly every holder touches ONE line: an SPU's GETLLAR/PUTLLC/PUTLLUC, a
 * PPU store or stwcx. to a reserved line, a one-line DMA PUT. Those take only
 * their line's stripe (spu_lockline_lock_line), so transactions on different
 * lines stop queueing behind each other, while two on the same line still
 * exclude each other -- all a PUTLLC's compare-and-commit needs. A block over
 * several lines takes its lines' stripes in ascending order
 * (spu_lockline_lock_span), and spu_lockline_lock() -- every caller that names
 * no line, the HLE libraries -- takes all of them in that same order, so it
 * still excludes everything and no two holders can wait on each other in a
 * cycle. The holder table changes under its own leaf lock, and a context
 * leaving it waits for each stripe to pass once (ll_barrier). */
#define LL_STRIPES_MAX 64u
typedef struct { atomic_int v; char pad[128 - sizeof(atomic_int)]; } ll_stripe_t;
static _Alignas(128) ll_stripe_t g_ll[LL_STRIPES_MAX];
static int s_ll_n;                       /* stripes in use, fixed at the first lock */
static int coh_mirror_on(void);
_Thread_local unsigned long long g_spu_lockline_wait_ticks;   /* ns (QPC is ns on POSIX) */
_Thread_local unsigned long long g_spu_lockline_count;
static inline unsigned ll_count(void)
{
    int n = __atomic_load_n(&s_ll_n, __ATOMIC_RELAXED);
    if (__builtin_expect(n > 0, 1)) return (unsigned)n;
    const char* e = getenv("SPU_LOCKLINE_STRIPES");
    int want = e ? atoi(e) : 1, p = 1;
    while (p < want && p < (int)LL_STRIPES_MAX) p <<= 1;
    /* Only the mirror notifier reads nothing but the holder table; the
     * full-list scan (SPU_COH_MIRROR=0) walks an array that grows under the
     * table's lock, so that mode keeps the single lock. */
    if (!coh_mirror_on()) p = 1;
    int z = 0;
    __atomic_compare_exchange_n(&s_ll_n, &z, p, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return (unsigned)__atomic_load_n(&s_ll_n, __ATOMIC_RELAXED);
}
int spu_lockline_striped(void) { return ll_count() > 1u; }

/* SPU_LOCKLINE_PROF=1 (diagnosis): which lines the line locks go to, and who
 * takes the all-stripe lock, printed every 5 s. Counts are approximate. */
#define LLP_N 4096u
#define LLA_N 64u
static struct { uint32_t key, acq, cont, notif, hits; uint64_t tb, tf, ts; } s_llp[LLP_N];
static struct { uintptr_t ra; uint32_t acq, cont; } s_lla[LLA_N];
static int s_llp_on = -1;
static inline int llp_on(void)
{
    if (__builtin_expect(s_llp_on < 0, 0)) s_llp_on = getenv("SPU_LOCKLINE_PROF") ? 1 : 0;
    return s_llp_on;
}
static void llp_table_report(void);
static void llp_report(void)
{
    static unsigned long long s_last;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    const unsigned long long now = (unsigned long long)ts.tv_sec * 1000ull + (unsigned long long)ts.tv_nsec / 1000000ull;
    unsigned long long last = __atomic_load_n(&s_last, __ATOMIC_RELAXED);
    if (last && now - last < 5000ull) return;
    if (!__atomic_compare_exchange_n(&s_last, &last, now, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    if (!last) return;                   /* the first call only starts the clock */
    unsigned long long tot = 0, totc = 0;
    for (unsigned i = 0; i < LLP_N; i++) { tot += s_llp[i].acq; totc += s_llp[i].cont; }
    fprintf(stderr, "[lockline-prof] %u stripe(s), %.1f s: %llu line locks, %llu waited\n",
            ll_count(), (double)(now - last) / 1000.0, tot, totc);
    llp_table_report();
    for (int r = 0; r < 14; r++) {
        int b = -1;
        for (unsigned i = 0; i < LLP_N; i++)
            if (s_llp[i].key && s_llp[i].acq &&
                (b < 0 || s_llp[i].cont > s_llp[b].cont ||
                 (s_llp[i].cont == s_llp[b].cont && s_llp[i].acq > s_llp[b].acq))) b = (int)i;
        if (b < 0) break;
        const double nn = s_llp[b].notif ? (double)s_llp[b].notif : 1.0, tk = 1e9 / 24e6;   /* cntvct: 24 MHz */
        fprintf(stderr, "[lockline-prof]   line 0x%08X stripe %2u: %u locks, %u waited, %u notifies, %u found a holder;"
                        " ns/notify: ppu-scan %.0f fence %.0f mirrors %.0f\n",
                s_llp[b].key & ~127u, (s_llp[b].key >> 7) & (ll_count() - 1u), s_llp[b].acq, s_llp[b].cont,
                s_llp[b].notif, s_llp[b].hits, s_llp[b].tb * tk / nn, s_llp[b].tf * tk / nn, s_llp[b].ts * tk / nn);
        s_llp[b].acq = 0; s_llp[b].cont = 0;
    }
    for (unsigned i = 0; i < LLP_N; i++) { s_llp[i].acq = 0; s_llp[i].cont = 0; s_llp[i].notif = 0; s_llp[i].hits = 0;
                                         s_llp[i].tb = 0; s_llp[i].tf = 0; s_llp[i].ts = 0; }
    for (unsigned i = 0; i < LLA_N; i++) {
        if (!s_lla[i].ra || !s_lla[i].acq) continue;
        Dl_info di; const char* nm = "?"; uintptr_t off = 0;
        if (dladdr((void*)s_lla[i].ra, &di) && di.dli_sname) { nm = di.dli_sname; off = s_lla[i].ra - (uintptr_t)di.dli_saddr; }
        fprintf(stderr, "[lockline-prof]   all-stripe lock from %s+0x%lX: %u, %u waited\n",
                nm, (unsigned long)off, s_lla[i].acq, s_lla[i].cont);
        s_lla[i].acq = 0; s_lla[i].cont = 0;
    }
    fflush(stderr);
}
static int llp_slot(uint32_t ea)
{
    const uint32_t key = (ea & ~127u) | 1u;
    unsigned h = ((ea >> 7) * 2654435761u) >> 20;
    for (unsigned k = 0; k < 32u; k++, h = (h + 1u) & (LLP_N - 1u)) {
        uint32_t cur = __atomic_load_n(&s_llp[h].key, __ATOMIC_RELAXED);
        if (!cur) {
            uint32_t z = 0;
            cur = __atomic_compare_exchange_n(&s_llp[h].key, &z, key, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED) ? key : z;
        }
        if (cur == key) return (int)h;
    }
    return -1;
}
static void llp_line(uint32_t ea, int cont)
{
    const int h = llp_slot(ea);
    if (h >= 0) {
        __atomic_fetch_add(&s_llp[h].acq, 1u, __ATOMIC_RELAXED);
        if (cont) __atomic_fetch_add(&s_llp[h].cont, 1u, __ATOMIC_RELAXED);
    }
    llp_report();
}
/* A notify of ea's line, whether it took a live reservation from anyone, and
 * the counter ticks (24 MHz) of its phases: the PPU reservation scan, the
 * fence, the mirror scan. */
static void llp_notify(uint32_t ea, int found, uint64_t tb, uint64_t tf, uint64_t ts)
{
    const int h = llp_slot(ea);
    if (h < 0) return;
    __atomic_fetch_add(&s_llp[h].notif, 1u, __ATOMIC_RELAXED);
    if (found) __atomic_fetch_add(&s_llp[h].hits, 1u, __ATOMIC_RELAXED);
    __atomic_fetch_add(&s_llp[h].tb, tb, __ATOMIC_RELAXED);
    __atomic_fetch_add(&s_llp[h].tf, tf, __ATOMIC_RELAXED);
    __atomic_fetch_add(&s_llp[h].ts, ts, __ATOMIC_RELAXED);
}
#if defined(__aarch64__)
static inline uint64_t llp_tick(void) { uint64_t v; __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v)); return v; }
#else
static inline uint64_t llp_tick(void) { return 0; }
#endif
static void llp_all(uintptr_t ra, int cont)
{
    unsigned h = (unsigned)((uint32_t)(ra >> 2) * 2654435761u) >> 26;
    for (unsigned k = 0; k < LLA_N; k++, h = (h + 1u) & (LLA_N - 1u)) {
        uintptr_t cur = __atomic_load_n(&s_lla[h].ra, __ATOMIC_RELAXED);
        if (!cur) {
            uintptr_t z = 0;
            cur = __atomic_compare_exchange_n(&s_lla[h].ra, &z, ra, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED) ? ra : z;
        }
        if (cur == ra) {
            __atomic_fetch_add(&s_lla[h].acq, 1u, __ATOMIC_RELAXED);
            if (cont) __atomic_fetch_add(&s_lla[h].cont, 1u, __ATOMIC_RELAXED);
            break;
        }
    }
    llp_report();
}

/* The wait, out of line: the uncontended acquire is one exchange. */
static int s_ll_time = -1, s_ll_bo = -1;
__attribute__((noinline)) static void ll_wait(atomic_int* L)
{
    unsigned spins = 0;
    /* g_spu_lockline_count (thread-local, a call to _tlv_get_addr per use on
     * Mach-O) counts contended acquisitions only. */
    g_spu_lockline_count++;
    /* The wait time is a diagnostic (SPU_LOCKLINE_TIME=1, read by the
     * WAIT_SIGNAL log): two clock reads per contended acquisition were a
     * visible share of the SPU threads' time once the SPURS lanes made the
     * contention common. */
    if (s_ll_time < 0) s_ll_time = getenv("SPU_LOCKLINE_TIME") ? 1 : 0;
    LARGE_INTEGER t0, t1;
    if (s_ll_time) QueryPerformanceCounter(&t0);
    /* Exponential backoff after each lost exchange (SPU_LOCKLINE_BACKOFF=0: off;
     * =<n>: the ceiling in pause spins, default 256). With the SPURS lanes up
     * to seven threads queued on the one lock; on every release they all saw 0
     * at once and all exchanged, each exchange taking the line exclusive in
     * turn -- the holder's next critical section then waited on that storm. A
     * waiter that just lost backs off before it looks again. */
    if (s_ll_bo < 0) { const char* e = getenv("SPU_LOCKLINE_BACKOFF"); int b = e ? atoi(e) : 256; s_ll_bo = b < 0 ? 0 : b; }
    unsigned backoff = 4;
    do {
        do {
            YieldProcessor();
            if (++spins >= 4096) { spins = 0; sched_yield(); }
        } while (atomic_load_explicit(L, memory_order_relaxed));
        if (!atomic_exchange_explicit(L, 1, memory_order_acquire)) break;
        if (s_ll_bo) {
            for (unsigned k = 0; k < backoff; k++) YieldProcessor();
            if (backoff < (unsigned)s_ll_bo) backoff <<= 1;
        }
    } while (1);
    if (s_ll_time) {
        QueryPerformanceCounter(&t1);
        g_spu_lockline_wait_ticks += (unsigned long long)(t1.QuadPart - t0.QuadPart);
    }
}
/* 1 when it had to wait. */
static inline int ll_acquire(atomic_int* L)
{
    if (__builtin_expect(!atomic_exchange_explicit(L, 1, memory_order_acquire), 1)) return 0;
    ll_wait(L);
    return 1;
}
static inline void ll_release(atomic_int* L) { atomic_store_explicit(L, 0, memory_order_release); }
static inline atomic_int* ll_of(uint32_t ea) { return &g_ll[(ea >> 7) & (ll_count() - 1u)].v; }

void spu_lockline_lock(void)
{
    const unsigned n = ll_count();
    int cont = 0;
    for (unsigned i = 0; i < n; i++) cont |= ll_acquire(&g_ll[i].v);
    if (__builtin_expect(llp_on(), 0)) llp_all((uintptr_t)__builtin_return_address(0), cont);
}
void spu_lockline_unlock(void)
{
    for (unsigned i = ll_count(); i-- > 0; ) ll_release(&g_ll[i].v);
}
void spu_lockline_lock_line(uint32_t ea)
{
    const int cont = ll_acquire(ll_of(ea));
    if (__builtin_expect(llp_on(), 0)) llp_line(ea, cont);
}
void spu_lockline_unlock_line(uint32_t ea) { ll_release(ll_of(ea)); }
/* The stripes of the lines in [ea, ea + size) as a mask; every stripe once
 * the span has as many lines as there are stripes. */
static uint64_t ll_span_mask(uint32_t ea, uint32_t size)
{
    const unsigned n = ll_count();
    const uint32_t l0 = ea >> 7;
    const uint32_t l1 = (uint32_t)(((uint64_t)ea + (size ? size : 1u) - 1u) >> 7);
    if (l1 - l0 + 1u >= n) return n >= 64u ? ~0ull : ((1ull << n) - 1ull);
    uint64_t m = 0;
    for (uint32_t l = l0; ; l++) { m |= 1ull << (l & (n - 1u)); if (l == l1) break; }
    return m;
}
void spu_lockline_lock_span(uint32_t ea, uint32_t size)
{
    int cont = 0;
    for (uint64_t m = ll_span_mask(ea, size); m; m &= m - 1u)
        cont |= ll_acquire(&g_ll[__builtin_ctzll(m)].v);       /* ascending */
    if (__builtin_expect(llp_on(), 0)) llp_line(ea, cont);
}
void spu_lockline_unlock_span(uint32_t ea, uint32_t size)
{
    for (uint64_t m = ll_span_mask(ea, size); m; m &= m - 1u)
        ll_release(&g_ll[__builtin_ctzll(m)].v);
}
/* Every stripe free at least once since the call: a notifier that read the
 * holder table before has left its critical section. */
static void ll_barrier(void)
{
    const unsigned n = ll_count();
    for (unsigned i = 0; i < n; i++) { (void)ll_acquire(&g_ll[i].v); ll_release(&g_ll[i].v); }
}
/* The holder table and the context list. A leaf: nothing is locked under it. */
static atomic_int s_coh_reg;
static inline void coh_reg_lock(void) { (void)ll_acquire(&s_coh_reg); }
static inline void coh_reg_unlock(void) { ll_release(&s_coh_reg); }
#endif

/* ===========================================================================
 * The reserved-line bitmap
 *
 * One bit per 128-byte line over the whole 32-bit guest address space: 4 MiB of
 * zero-initialized BSS, of which only the pages around addresses an SPU has
 * actually reserved are ever touched. The reference implementation scopes its
 * bitmap to the one window a single title's SPURS structures live in; this
 * runtime hands out guest memory from several disjoint regions (the ELF image,
 * the sys_memory window at 0x40000000, the raw-SPU local-store windows), and a
 * line outside a hardcoded window would be silently non-coherent, which is the
 * failure this whole file exists to remove.
 * ===========================================================================*/
#define SPU_COH_LINES       (1u << (32 - SPU_COH_LINE_SHIFT))   /* 2^25 lines */
#define SPU_COH_BITMAP_SZ   (SPU_COH_LINES / 8)                 /* 4 MiB */

static unsigned char s_coh_bitmap[SPU_COH_BITMAP_SZ];

/* g_spu_coh_bits (spu_coh_bitmap.h) stays NULL until the first GETLLAR
 * anywhere. That keeps the PPU fast path off the bitmap entirely for a title
 * that runs no SPU code, and keeps every existing test that stores through
 * vm_write* paying one compare against a hot global. Written under the lock;
 * read unlocked, where a stale NULL costs at most one missed event on the very
 * first reservation of the run. */

/* Every live virtual context may retain a reservation between scheduling
 * quanta; physical SPU count is not a bound on this set. All access is under
 * the lock-line lock. Nodes are removed before their context's lifetime ends. */
/* A dense array, not a linked list: the notifiers below run under the
 * lock-line lock and visit every registered context on every PPU store to a
 * reserved line and every committing PUTLLC. With a policy-module context per
 * workload per SPURS lane that is ~80 contexts, and walking malloc'd nodes
 * added a dependent miss per context to every one of those lock holds. */
static spu_context** s_coh_ctxs;
static unsigned s_coh_n, s_coh_cap;

/* The holder table: every registered context keeps one fixed slot, and the
 * slot's mirror holds (line | 1) of the reservation that context last took,
 * 0 once it ended it itself. A notifier reads the mirrors -- a few dense cache
 * lines -- and touches a context only when its mirror names the written line,
 * where it used to read resv_valid out of every registered context (~80 with
 * the SPURS lanes) inside the global lock-line lock. The mirror is a filter in
 * front of the same CAS (spu_coh_take), never a substitute for it:
 *  - the owner stores resv_valid, THEN its mirror (release), then the seq_cst
 *    fence that publishes the reservation; a notifier stores the line, fences,
 *    THEN reads the mirror (acquire). Either the notifier sees the mirror, and
 *    with it resv_valid, or the owner's copy of the line sees the store -- the
 *    same Dekker pair spu_coh_reserve_fast rests on.
 *  - only the owner ever clears its mirror, after its own resv_valid; a
 *    notifier never does, so it cannot erase a reservation taken again since.
 *    A stale mirror costs one failing CAS.
 * SPU_COH_MIRROR=0 scans every registered context as before. */
#define COH_SLOTS 1024u
static spu_context* s_coh_slot_ctx[COH_SLOTS];
static int s_coh_mirror[COH_SLOTS];
static unsigned s_coh_hi;                 /* slots [0, s_coh_hi) may be in use */
static int s_coh_mirror_on = -1;
static int coh_mirror_on(void)
{
    if (s_coh_mirror_on < 0) { const char* e = getenv("SPU_COH_MIRROR"); s_coh_mirror_on = !(e && *e == '0'); }
    return s_coh_mirror_on;
}
static inline int coh_slot_of(const spu_context* ctx)
{
    const int k = ctx->coh_slot1 - 1;
    return (k >= 0 && (unsigned)k < COH_SLOTS &&
            __atomic_load_n(&s_coh_slot_ctx[k], __ATOMIC_RELAXED) == ctx) ? k : -1;
}
void spu_coh_mirror_set(spu_context* ctx, uint32_t ea)
{
    const int k = ctx ? coh_slot_of(ctx) : -1;
    if (k < 0) return;
    /* Only the owner writes its mirror (the table writes 0 only while the
     * context is being registered or dropped), so this read is its own last
     * store. Storing the same value again -- every poll of a line it already
     * names -- still took the cache line away from every notifier scanning the
     * table (the main thread's stores to reserved lines among them). */
    const int v = (int)((ea & ~127u) | 1u);
    if (__atomic_load_n(&s_coh_mirror[k], __ATOMIC_RELAXED) != v)
        __atomic_store_n(&s_coh_mirror[k], v, __ATOMIC_RELEASE);
}
void spu_coh_mirror_clear(spu_context* ctx)
{
    const int k = ctx ? coh_slot_of(ctx) : -1;
    if (k >= 0) __atomic_store_n(&s_coh_mirror[k], 0, __ATOMIC_RELEASE);
}
/* Under the holder-table lock (coh_reg_lock): give ctx a slot (its old one if
 * the table still names it, e.g. after its owner memset it). The notifiers
 * read the table without that lock: s_coh_hi and the slot's context are
 * published before the owner's mirror_set (a release store) names a line. */
static void coh_slot_assign(spu_context* ctx)
{
    if (coh_slot_of(ctx) >= 0) return;
    int k = -1;
    for (unsigned i = 0; i < s_coh_hi; i++) if (s_coh_slot_ctx[i] == ctx) { k = (int)i; break; }
    if (k < 0) {
        for (unsigned i = 0; i < s_coh_hi; i++) if (!s_coh_slot_ctx[i]) { k = (int)i; break; }
        if (k < 0) {
            if (s_coh_hi >= COH_SLOTS) {
                fprintf(stderr, "[spu-coh] holder table full (%u)\n", COH_SLOTS);
                fflush(stderr);
                abort(); /* A missed holder loses invalidation events, including ABA. */
            }
            k = (int)s_coh_hi;
            __atomic_store_n(&s_coh_hi, (unsigned)k + 1u, __ATOMIC_RELAXED);
        }
        __atomic_store_n(&s_coh_mirror[k], 0, __ATOMIC_RELAXED);
        __atomic_store_n(&s_coh_slot_ctx[k], ctx, __ATOMIC_RELAXED);
    }
    ctx->coh_slot1 = k + 1;
}
/* Under the holder-table lock. */
static void coh_slot_free(spu_context* ctx)
{
    for (unsigned i = 0; i < s_coh_hi; i++)
        if (s_coh_slot_ctx[i] == ctx) {
            __atomic_store_n(&s_coh_mirror[i], 0, __ATOMIC_RELAXED);
            __atomic_store_n(&s_coh_slot_ctx[i], NULL, __ATOMIC_RELAXED);
        }
    unsigned hi = s_coh_hi;
    while (hi && !s_coh_slot_ctx[hi - 1]) hi--;
    __atomic_store_n(&s_coh_hi, hi, __ATOMIC_RELAXED);
    ctx->coh_slot1 = 0;
}

#if !defined(_MSC_VER)
/* SPU_LOCKLINE_PROF: the holder table every notifier scans -- its high-water
 * mark, the slots in use, the mirrors naming a line, and which images hold the
 * slots (a context that never unregisters keeps its slot for good). */
static void llp_table_report(void)
{
    const unsigned hi = __atomic_load_n(&s_coh_hi, __ATOMIC_RELAXED);
    unsigned used = 0, named = 0;
    int img[16]; unsigned cnt[16]; int nimg = 0;
    for (unsigned i = 0; i < hi; i++) {
        spu_context* c = __atomic_load_n(&s_coh_slot_ctx[i], __ATOMIC_RELAXED);
        if (__atomic_load_n(&s_coh_mirror[i], __ATOMIC_RELAXED)) named++;
        if (!c) continue;
        used++;
        const int id = c->image_id;
        int k = 0;
        while (k < nimg && img[k] != id) k++;
        if (k == nimg && nimg < 16) { img[nimg] = id; cnt[nimg] = 0; nimg++; }
        if (k < nimg) cnt[k]++;
    }
    fprintf(stderr, "[lockline-prof]   holder table: high-water %u of %u, %u in use, %u mirrors name a line, %u listed;",
            hi, COH_SLOTS, used, named, s_coh_n);
    for (int k = 0; k < nimg; k++) fprintf(stderr, " img%d x%u", img[k], cnt[k]);
    fputc('\n', stderr);
}
#endif

unsigned long g_spu_lr_raise = 0;

/* Called for every PUT-class SPU DMA when set (spu_dma.h mfc_do_transfer); src
 * is the LS source of the transfer. Diagnosis hook, NULL by default. */
void (*g_spu_put_observer)(uint32_t ea, uint32_t size, const uint8_t* src,
                           uint32_t cmd, int img, unsigned wid, uint32_t pc) = 0;

void spu_coh_reserve(spu_context* ctx, uint32_t ea)
{
    /* Read before writing. The kernel reserves the same few lines over and
     * over, and a store of the same value still takes the cache line away
     * from every other core: each PPU store then missed on the pointer and on
     * the bitmap byte. The OR is atomic: with the lock striped, a GETLLAR of
     * a neighbouring line (same byte, another stripe) can mark it meanwhile. */
    uint32_t line = ea >> SPU_COH_LINE_SHIFT;
    const unsigned char bit = (unsigned char)(1u << (line & 7));
    if (!(__atomic_load_n(&s_coh_bitmap[line >> 3], __ATOMIC_RELAXED) & bit))
        __atomic_fetch_or(&s_coh_bitmap[line >> 3], bit, __ATOMIC_RELAXED);
    if (!g_spu_coh_bits)
        g_spu_coh_bits = s_coh_bitmap;

    if (!ctx) return;
    /* coh_listed is set exactly while ctx is in the array (both changed under
     * the holder-table lock), and a listed context keeps its slot: the common
     * case, a registered holder, takes no lock but its line's. */
    if (coh_slot_of(ctx) < 0 || !__atomic_load_n(&ctx->coh_listed, __ATOMIC_RELAXED)) {
        coh_reg_lock();
        coh_slot_assign(ctx);
        int listed = 0;
        for (unsigned i = 0; i < s_coh_n; i++)
            if (s_coh_ctxs[i] == ctx) { listed = 1; break; }
        if (!listed) {
            if (s_coh_n == s_coh_cap) {
                const unsigned cap = s_coh_cap ? s_coh_cap * 2u : 64u;
                spu_context** a = (spu_context**)realloc(s_coh_ctxs, cap * sizeof *a);
                if (!a) {
                    fprintf(stderr, "[spu-coh] cannot register reservation holder %p\n", (void*)ctx);
                    fflush(stderr);
                    abort(); /* A missed holder loses invalidation events, including ABA. */
                }
                s_coh_ctxs = a; s_coh_cap = cap;
            }
            s_coh_ctxs[s_coh_n++] = ctx;
        }
        __atomic_store_n(&ctx->coh_listed, 1, __ATOMIC_RELAXED);
        coh_reg_unlock();
    }
    /* Every locked GETLLAR comes through here, under its line's lock, right
     * before it stores resv_valid: name the line in the mirror now, so no
     * caller can set a reservation the notifiers will not see (they take the
     * same line's lock). */
    spu_coh_mirror_set(ctx, ea);
}

/* Drop a context from the reserving set.
 *
 * There was no way to do this, and the contexts that most need it are
 * TRANSIENT: spu_run_lifted_job_abi runs each job on a STACK-LOCAL
 * spu_context, which registered itself here on its first reservation and then
 * returned. The entry outlived the object, so spu_coh_notify_write walked a
 * dangling pointer -- reading resv_valid, writing event_status and calling
 * spu_ch_wake on reclaimed stack. Undefined behaviour, and observable: Guitar
 * Hero III showed THREE live-looking contexts for one dispatch, stealing each
 * other`s reservations, with every PUTLLC failing "no reservation". */
void spu_coh_unregister(spu_context* ctx)
{
    if (!ctx) return;
    /* A notifier that loaded this pointer just before removal would otherwise
     * go on dereferencing it after the job returned and its stack (or its
     * whole async thread) was gone. GH3 at -O2 crashed that way in
     * spu_coh_notify_write, reading a freed thread stack. With one lock the
     * removal holds it, as every walker does. Striped, a walker holds one
     * stripe and reads only the holder table: the entry is cleared under the
     * table's lock, then every stripe is waited out once, so a notifier that
     * read the entry before it was cleared is done with ctx when this returns. */
    const int striped = ll_count() > 1u;
    if (!striped) spu_lockline_lock();
    coh_reg_lock();
    for (unsigned i = 0; i < s_coh_n; i++)
        if (s_coh_ctxs[i] == ctx) { s_coh_ctxs[i] = s_coh_ctxs[--s_coh_n]; break; }
    coh_slot_free(ctx);
    __atomic_store_n(&ctx->coh_listed, 0, __ATOMIC_RELAXED);
    coh_reg_unlock();
    if (!striped) spu_lockline_unlock();
    else ll_barrier();
    /* The context is going away: give back its MFC engine too. Transient job and
     * task contexts never did, so the engine slots ran out right after boot and
     * every later context shared one engine -- one SPU's parked list then held
     * another SPU's tag "incomplete" (GoW:A: a WWS policy module spun forever on
     * a tag wait that belonged to a different workload). */
    { extern void spu_mfc_release(spu_context*); spu_mfc_release(ctx); }
}

/* Drop every registered context that lies in [lo, hi) -- a host thread's
 * stack, called as that thread exits. A stack-local context still registered
 * then has leaked past its unregister; the next walker would read an unmapped
 * stack (GH3 at the end of a Havok task: spu_coh_notify_write faulting).
 * ponytail: safety net that also names the leak; find the path that skips
 * spu_coh_unregister if it keeps firing. */
void spu_coh_forget_range(uintptr_t lo, uintptr_t hi)
{
    const int striped = ll_count() > 1u;     /* as in spu_coh_unregister */
    if (!striped) spu_lockline_lock();
    coh_reg_lock();
    for (unsigned i = 0; i < s_coh_n;) {
        spu_context* ctx = s_coh_ctxs[i];
        uintptr_t c = (uintptr_t)ctx;
        if (c >= lo && c < hi) {
            static int _n = 0;
            if (PS3_LOG_FIRST(_n, 16))
                fprintf(stderr, "[spu-coh] leaked ctx %p (img=%d) dropped at thread exit\n",
                        (void*)c, ctx->image_id);
            __atomic_store_n(&ctx->coh_listed, 0, __ATOMIC_RELAXED);
            coh_slot_free(ctx);
            s_coh_ctxs[i] = s_coh_ctxs[--s_coh_n];   /* re-examine the one moved here */
        } else {
            i++;
        }
    }
    coh_reg_unlock();
    if (!striped) spu_lockline_unlock();
    else ll_barrier();
}

int spu_coh_is_reserved(uint32_t addr)
{
    return spu_coh_line_reserved(addr);
}

/* A DMA PUT is issued BY an SPU, and hardware does not take that SPU's own
 * reservation away for its own MFC write -- the MFC is part of the same SPE.
 * Our notify had no way to say "everyone but me", so an SPU that reserved a
 * line and then DMA'd a buffer overlapping it killed its own reservation and
 * its next PUTLLC could never succeed. Guitar Hero III's job policy module
 * livelocks exactly there: 16 million atomic ops on one line, every PUTLLC
 * failing for "no reservation", 600k of them self-inflicted.
 *
 * The PUTLLC path already guards this case by dropping its own reservation
 * before notifying; this gives the DMA path the same ability. */
/* ppu_loader.cpp, when linked. The loader is compiled per game, so the host,
 * the fixtures and any build without it have no PPU reservations to break.
 * POSIX: a weak no-op definition the loader's overrides (a weak undefined
 * reference is not enough: ld64 only lets those stay unresolved for dylib
 * symbols). link.exe/lld-link: the linker's own default (see ppu_gcm_pump in
 * ppu_loader.cpp). */
#if defined(_MSC_VER)
void ppu_resv_break_line(uint32_t ea);
void ppu_resv_break_line_absent(uint32_t ea) { (void)ea; }
#pragma comment(linker, "/alternatename:ppu_resv_break_line=ppu_resv_break_line_absent")
#else
__attribute__((weak)) void ppu_resv_break_line(uint32_t ea) { (void)ea; }
#endif

/* ===========================================================================
 * GETLLAR without the lock-line lock
 *
 * Every SPURS policy module polls its workload lines with GETLLAR, so taking
 * the one global lock for each poll serialized every SPU on a 128-byte copy.
 * A poll is almost always a context that already holds a node in s_coh_ctxs
 * re-reading a line that is already marked, and for that case the lock buys
 * nothing the reservation itself does not:
 *
 *   reader (spu_coh_reserve_fast + caller)    writer (any notifier caller)
 *     resv_valid = 0, resv_ea = ea,             store the new line data
 *     resv_valid = 1                            spu_coh_writer_fence()
 *     seq_cst fence                             load resv_valid / resv_ea
 *     copy the line                             if they cover the line:
 *                                                 LR, resv_valid = 0
 *
 * This is the store-buffering (Dekker) shape: with a seq_cst fence between the
 * store and the load on both sides, the two fences are ordered, so at least
 * one side sees the other. Either the reader's copy already holds the write
 * (an ordinary GETLLAR after the store), or the writer sees the reservation
 * and ends it -- the reader may then hold a stale or torn copy, but its
 * PUTLLC fails and it has SPU_EVENT_LR, exactly as for a write landing just
 * after a locked GETLLAR. A writer that loads stale resv_valid/resv_ea is one
 * whose fence came first, so the copy already holds its data; at worst it
 * ends the new reservation spuriously. What the reader gives up is a copy
 * atomic against a writer's multi-store critical section, but such a copy is
 * always followed by that writer's notify, so the reservation is ended.
 *
 * Two preconditions keep every writer of the line on that right-hand side:
 *  - the line is already marked. Marks are never cleared, so every store path
 *    that checks spu_coh_is_reserved takes the notifier for this line.
 *  - ctx is already in s_coh_ctxs (coh_listed). The node was linked under the
 *    lock by an earlier GETLLAR of this same thread, so any writer whose
 *    critical section could overlap this read walks past it.
 * Anything else takes the lock and spu_coh_reserve as before.
 *
 * Writers still hold the lock-line lock, so they stay serialized with each
 * other and with PUTLLC's compare-and-commit; only the reader stopped
 * waiting. resv_valid and resv_ea are accessed atomically on both sides of
 * the race; everything else about a reservation is still only touched by the
 * owning thread or under the lock.
 * ===========================================================================*/
static inline void spu_coh_writer_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

static inline int spu_coh_holds(const spu_context* c, uint32_t line)
{
    return __atomic_load_n(&c->resv_valid, __ATOMIC_RELAXED) &&
           (__atomic_load_n(&c->resv_ea, __ATOMIC_RELAXED) & ~127u) == line;
}

int spu_coh_reserve_fast(spu_context* ctx, uint32_t ea)
{
    if (!ctx || !__atomic_load_n(&ctx->coh_listed, __ATOMIC_RELAXED) ||
        !spu_coh_is_reserved(ea))
        return 0;
    /* Retire any older reservation before retargeting it. The fence is what
     * the argument above rests on. */
    __atomic_store_n(&ctx->resv_valid, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&ctx->resv_ea, ea, __ATOMIC_RELAXED);
    __atomic_store_n(&ctx->resv_valid, (int)((ea & ~127u) | 1u), __ATOMIC_RELAXED);
    spu_coh_mirror_set(ctx, ea);                    /* after resv_valid, before the fence */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return 1;
}

/* Check-and-clear in one step (port of otavix010/gow-ascension-pc ffc6e00, the
 * lost-kick hang). It used to be "if the reservation covers the line: raise
 * LR, ..., resv_valid = 0" with a separate store. A notifier preempted between
 * the LR and the store let the SPU consume the LR, reserve another line (the
 * SPURS one) and sleep; the late store then wiped that NEW reservation with no
 * event, and the SPU never woke -- SPUs lost one by one over long sessions.
 * resv_valid holds (line | 1), so the CAS clears only this line's
 * reservation, and the LR comes after it (at worst one spurious LR). */
static inline int spu_coh_take(spu_context* c, uint32_t line)
{
    int want = (int)(line | 1u);
    if (__atomic_load_n(&c->resv_valid, __ATOMIC_RELAXED) != want) return 0;
    return __atomic_compare_exchange_n(&c->resv_valid, &want, 0, 0,
                                       __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
}

void spu_coh_notify_write_except(uint32_t ea, const void* self)
{
    uint32_t line = ea & ~127u;
    const int prof = __builtin_expect(llp_on(), 0);
    uint64_t t0 = prof ? llp_tick() : 0, t1 = 0, t2 = 0;
    ppu_resv_break_line(ea);
    if (prof) t1 = llp_tick();
    spu_coh_writer_fence();
    if (prof) t2 = llp_tick();
    if (coh_mirror_on()) {
        const int want = (int)(line | 1u);
        int found = 0;
        const unsigned hi = __atomic_load_n(&s_coh_hi, __ATOMIC_RELAXED);
        for (unsigned i = 0; i < hi; i++) {
            /* Relaxed: the seq_cst fence above already orders these loads after
             * the store being reported, and the Dekker argument needs no more
             * (a load sequenced after that fence sees any mirror store made
             * before the owner's fence, if that fence came first). An acquire
             * load per slot (LDAR) made the scan serial -- each miss waited for
             * the one before -- and cost 1.5-8 us a notify (SPU_LOCKLINE_PROF);
             * the acquire, which pairs with the owner's release store so that
             * the slot's context and resv_valid are seen, is only needed on a
             * match. */
            if (__atomic_load_n(&s_coh_mirror[i], __ATOMIC_RELAXED) != want) continue;
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            spu_context* c = __atomic_load_n(&s_coh_slot_ctx[i], __ATOMIC_RELAXED);
            if (!c || (const void*)c == self) continue;
            if (spu_coh_take(c, line)) {
                SPU_EVENT_SET(c, SPU_EVENT_LR);
                g_spu_lr_raise++;
                spu_ch_wake(c);
                found++;
            }
        }
        if (prof) llp_notify(line, found, t1 - t0, t2 - t1, llp_tick() - t2);
        return;
    }
    for (unsigned i = 0; i < s_coh_n; i++) {
        spu_context* c = s_coh_ctxs[i];
        if (i + 4u < s_coh_n) __builtin_prefetch(&s_coh_ctxs[i + 4u]->resv_valid);
        if (!c || (const void*)c == self) continue;
        if (spu_coh_take(c, line)) {
            SPU_EVENT_SET(c, SPU_EVENT_LR);
            g_spu_lr_raise++;
            spu_ch_wake(c);
        }
    }
}

static void coh_lost(spu_context* c, uint32_t line)
{
    /* Reservation lost (cleared by the CAS): its PUTLLC must fail. */
    SPU_EVENT_SET(c, SPU_EVENT_LR);
    /* SPU_PUTLLC_WHY=1: name the agent that killed it. A PUTLLC that
     * always fails for "no reservation" is useless without knowing who
     * took it away -- a peer SPU, or the PPU committing to the line. */
    { static int s_w = -1;
      if (s_w < 0) s_w = getenv("SPU_PUTLLC_WHY") ? 1 : 0;
      if (s_w) { static unsigned long long n;
          if ((++n % 200000) == 1)
              fprintf(stderr, "[resv-lost] %llu: line 0x%08X taken from img=%d ctx=%p\n",
                      n, line, c->image_id, (void*)c); } }
    g_spu_lr_raise++;
    /* The event_status store has to be visible before the wake. The
     * waiter re-polls its predicate on a timeout as well, so a
     * straggling store costs latency and not the wakeup itself. */
    spu_ch_wake(c);
}

void spu_coh_notify_write(uint32_t ea)
{
    uint32_t line = ea & ~127u;
    const int prof = __builtin_expect(llp_on(), 0);
    uint64_t t0 = prof ? llp_tick() : 0, t1 = 0, t2 = 0;
    ppu_resv_break_line(ea);
    if (prof) t1 = llp_tick();
    spu_coh_writer_fence();
    if (prof) t2 = llp_tick();
    if (coh_mirror_on()) {
        const int want = (int)(line | 1u);
        int found = 0;
        const unsigned hi = __atomic_load_n(&s_coh_hi, __ATOMIC_RELAXED);
        for (unsigned i = 0; i < hi; i++) {
            /* Relaxed: the seq_cst fence above already orders these loads after
             * the store being reported, and the Dekker argument needs no more
             * (a load sequenced after that fence sees any mirror store made
             * before the owner's fence, if that fence came first). An acquire
             * load per slot (LDAR) made the scan serial -- each miss waited for
             * the one before -- and cost 1.5-8 us a notify (SPU_LOCKLINE_PROF);
             * the acquire, which pairs with the owner's release store so that
             * the slot's context and resv_valid are seen, is only needed on a
             * match. */
            if (__atomic_load_n(&s_coh_mirror[i], __ATOMIC_RELAXED) != want) continue;
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            spu_context* c = __atomic_load_n(&s_coh_slot_ctx[i], __ATOMIC_RELAXED);
            if (c && spu_coh_take(c, line)) { coh_lost(c, line); found++; }
        }
        if (prof) llp_notify(line, found, t1 - t0, t2 - t1, llp_tick() - t2);
        return;
    }
    for (unsigned i = 0; i < s_coh_n; i++) {
        spu_context* c = s_coh_ctxs[i];
        if (i + 4u < s_coh_n) __builtin_prefetch(&s_coh_ctxs[i + 4u]->resv_valid);
        if (!c) continue;
        if (spu_coh_take(c, line)) coh_lost(c, line);
    }
}

#ifdef __cplusplus
}
#endif
