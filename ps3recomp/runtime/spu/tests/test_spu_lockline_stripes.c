/* Stress test for the striped lock-line lock (SPU_LOCKLINE_STRIPES).
 *
 * Six SPU threads run GETLLAR/PUTLLC increments the way spu_channels.c does,
 * two PPU threads lwarx/stwcx. increments, one PPU thread plain stores, one
 * thread transient (heap) contexts that register, increment and unregister,
 * and one an HLE-style read-modify-write under the all-stripe lock -- all on
 * the same few lines. Every counter has to end exactly at the number of
 * increments its writer saw commit, and no notifier may touch a context after
 * spu_coh_unregister returned.
 *
 * Build (from ps3recomp/):
 *   clang -std=gnu17 -O2 -pthread -I include -I runtime/spu -I runtime/spu/tests \
 *         -o /tmp/test_spu_lockline_stripes runtime/spu/tests/test_spu_lockline_stripes.c
 * Run with SPU_LOCKLINE_STRIPES=1, 4, 16, 64; optional argument: iterations
 * per thread (default 150000). Exit code 0 when every count matches. */
#include "../spu_coherency.c"
#include "../../ppu/ppu_memory.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sched.h>

static uint8_t g_guest_mem[1u << 20] __attribute__((aligned(128)));
uint8_t* vm_base = g_guest_mem;
uint32_t g_ww_lo = 0xFFFFFFFFu, g_ww_hi = 0;
int      g_resv_store_active = 0;
void ppu_resv_break_store(uint64_t ea) { (void)ea; }
void ps3_ww_report_inline(uint32_t a, uint64_t v, int w) { (void)a; (void)v; (void)w; }
static unsigned long g_wakes;
void spu_ch_wake(spu_context* ctx) { (void)ctx; __atomic_fetch_add(&g_wakes, 1, __ATOMIC_RELAXED); }
void spu_mfc_release(spu_context* c) { (void)c; }

#define NLINES 8
#define NSPU   6
#define NPPU   2
static long ITERS = 150000;
static uint32_t line_ea(int i) { return 0x00040000u + (uint32_t)i * 640u; }

/* spu_channels.c MFC_GETLLAR_CMD, minus the local store */
static void getllar(spu_context* ctx, uint32_t ea, uint8_t* ls)
{
    const int fast = spu_coh_reserve_fast(ctx, ea);
    if (!fast) { spu_lockline_lock_line(ea); spu_coh_reserve(ctx, ea); }
    memcpy(ls, vm_base + ea, 128);
    memcpy(ctx->resv_line, ls, 128);
    if (!fast) {
        ctx->resv_ea = ea;
        __atomic_store_n(&ctx->resv_valid, (int)((ea & ~127u) | 1u), __ATOMIC_RELAXED);
        spu_coh_mirror_set(ctx, ea);
        spu_lockline_unlock_line(ea);
    }
}
/* spu_channels.c MFC_PUTLLC_CMD */
static int putllc(spu_context* ctx, uint32_t ea, const uint8_t* ls)
{
    int ok = 0;
    spu_lockline_lock_line(ea);
    if (__atomic_load_n(&ctx->resv_valid, __ATOMIC_RELAXED) && ctx->resv_ea == ea &&
        memcmp(vm_base + ea, ctx->resv_line, 128) == 0) {
        memcpy(vm_base + ea, ls, 128);
        __atomic_store_n(&ctx->resv_valid, 0, __ATOMIC_RELAXED);
        spu_coh_notify_write(ea);
        ok = 1;
    }
    __atomic_store_n(&ctx->resv_valid, 0, __ATOMIC_RELAXED);
    spu_coh_mirror_clear(ctx);
    spu_lockline_unlock_line(ea);
    return ok;
}
static void incr_word(spu_context* ctx, uint32_t ea, int word)
{
    uint8_t ls[128] __attribute__((aligned(16)));
    for (;;) {
        getllar(ctx, ea, ls);
        uint32_t v; memcpy(&v, ls + 4 * word, 4); v++; memcpy(ls + 4 * word, &v, 4);
        if (putllc(ctx, ea, ls)) return;
    }
}

static spu_context g_spu[NSPU];
static long s_spu_ok[NSPU][NLINES], s_ppu_ok[NPPU][NLINES], s_tr_ok[NLINES], s_hle_ok[NLINES];
static uint32_t s_plain_last[NLINES];
static volatile int s_go, s_stop;
static long s_tr_bad;

static uint32_t rnd(uint32_t* r) { *r = *r * 1103515245u + 12345u; return *r >> 16; }

static void* spu_thr(void* p)
{
    const int id = (int)(intptr_t)p; uint32_t r = 77u + (uint32_t)id;
    while (!s_go) sched_yield();
    for (long it = 0; it < ITERS; it++) {
        const int li = (int)(rnd(&r) % NLINES);
        incr_word(&g_spu[id], line_ea(li), 0);
        s_spu_ok[id][li]++;
    }
    return 0;
}
static void* ppu_thr(void* p)
{
    const int id = (int)(intptr_t)p; uint32_t r = 991u + (uint32_t)id;
    ppu_context ppu; ppu_context_init(&ppu);
    while (!s_go) sched_yield();
    for (long it = 0; it < ITERS; it++) {
        const int li = (int)(rnd(&r) % NLINES);
        const uint32_t a = line_ea(li) + 4;
        for (;;) { uint32_t v = ppu_lwarx(&ppu, a); if (ppu_stwcx(&ppu, a, v + 1u)) break; }
        s_ppu_ok[id][li]++;
    }
    return 0;
}
/* Plain PPU stores: each line has to keep the last sequence number written. */
static void* plain_thr(void* p)
{
    (void)p; uint32_t r = 5u, seq = 0;
    while (!s_go) sched_yield();
    while (!s_stop) {
        const int li = (int)(rnd(&r) % NLINES);
        vm_write32(line_ea(li) + 8, ++seq);
        s_plain_last[li] = seq;
    }
    return 0;
}
/* Transient job contexts: register, increment, unregister, then check that no
 * notifier touches the context afterwards. */
static void* trans_thr(void* p)
{
    (void)p; uint32_t r = 31u;
    while (!s_go) sched_yield();
    while (!s_stop) {
        spu_context* c = (spu_context*)calloc(1, sizeof *c);
        spu_context_init(c, 0);
        const int li = (int)(rnd(&r) % NLINES);
        const uint32_t ea = line_ea(li);
        incr_word(c, ea, 3);
        s_tr_ok[li]++;
        uint8_t ls[128];
        getllar(c, ea, ls);                      /* leave a live reservation */
        spu_coh_unregister(c);
        /* After unregister nothing may touch the context: plant a fake
         * reservation on a contended line and give notifiers time to hit it. */
        __atomic_store_n(&c->resv_valid, (int)(ea | 1u), __ATOMIC_RELAXED);
        __atomic_store_n(&c->event_status, 0u, __ATOMIC_RELAXED);
        for (int k = 0; k < 2000; k++) YieldProcessor();
        if (__atomic_load_n(&c->resv_valid, __ATOMIC_RELAXED) != (int)(ea | 1u) ||
            (__atomic_load_n(&c->event_status, __ATOMIC_RELAXED) & SPU_EVENT_LR))
            s_tr_bad++;
        free(c);
    }
    return 0;
}
/* HLE: all-stripe lock, read-modify-write, notify. */
static void* hle_thr(void* p)
{
    (void)p; uint32_t r = 3u;
    while (!s_go) sched_yield();
    while (!s_stop) {
        const int li = (int)(rnd(&r) % NLINES);
        const uint32_t ea = line_ea(li);
        spu_lockline_lock();
        uint32_t v; memcpy(&v, vm_base + ea + 16, 4); v++; memcpy(vm_base + ea + 16, &v, 4);
        spu_coh_notify_write(ea);
        spu_lockline_unlock();
        s_hle_ok[li]++;
        for (int k = 0; k < 200; k++) YieldProcessor();
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1) ITERS = atol(argv[1]);
    for (int i = 0; i < NSPU; i++) { spu_context_init(&g_spu[i], 0); g_spu[i].spu_id = (uint32_t)i; }
    /* Mark the lines first: a line's very first reservation is the known
     * unlocked window of the bitmap design, not what this test is about. */
    { uint8_t ls[128]; for (int li = 0; li < NLINES; li++) getllar(&g_spu[0], line_ea(li), ls); }
    pthread_t t[NSPU + NPPU + 3];
    int n = 0;
    for (int i = 0; i < NSPU; i++) pthread_create(&t[n++], 0, spu_thr, (void*)(intptr_t)i);
    for (int i = 0; i < NPPU; i++) pthread_create(&t[n++], 0, ppu_thr, (void*)(intptr_t)i);
    pthread_t tp, tt, th;
    pthread_create(&tp, 0, plain_thr, 0);
    pthread_create(&tt, 0, trans_thr, 0);
    pthread_create(&th, 0, hle_thr, 0);
    s_go = 1;
    for (int i = 0; i < n; i++) pthread_join(t[i], 0);
    s_stop = 1;
    pthread_join(tp, 0); pthread_join(tt, 0); pthread_join(th, 0);

    int bad = 0; long tot_tr = 0, tot_hle = 0;
    for (int li = 0; li < NLINES; li++) {
        const uint8_t* m = vm_base + line_ea(li);
        uint32_t w0, w1, w2, w3, w4;
        memcpy(&w0, m, 4); memcpy(&w1, m + 4, 4); memcpy(&w2, m + 8, 4); memcpy(&w3, m + 12, 4); memcpy(&w4, m + 16, 4);
        w1 = __builtin_bswap32(w1); w2 = __builtin_bswap32(w2);
        long e0 = 0, e1 = 0;
        for (int i = 0; i < NSPU; i++) e0 += s_spu_ok[i][li];
        for (int i = 0; i < NPPU; i++) e1 += s_ppu_ok[i][li];
        tot_tr += s_tr_ok[li]; tot_hle += s_hle_ok[li];
        const int ok = (long)w0 == e0 && (long)w1 == e1 && w2 == s_plain_last[li] &&
                       (long)w3 == s_tr_ok[li] && (long)w4 == s_hle_ok[li];
        if (!ok) {
            bad++;
            printf("LINE %d WRONG: spu %u/%ld ppu %u/%ld plain %u/%u trans %u/%ld hle %u/%ld\n", li,
                   w0, e0, w1, e1, w2, s_plain_last[li], w3, s_tr_ok[li], w4, s_hle_ok[li]);
        }
    }
    printf("stripes=%u lines=%d spu=%dx%ld ppu=%dx%ld jobs=%ld hle=%ld wakes=%lu touched_after_unregister=%ld -> %s\n",
           ll_count(), NLINES, NSPU, ITERS, NPPU, ITERS, tot_tr, tot_hle, g_wakes, s_tr_bad,
           (bad || s_tr_bad) ? "FAILED" : "OK");
    return (bad || s_tr_bad) ? 1 : 0;
}
