/* spurs_policy.c — run a SPURS workload's policy module on a lifted SPU.
 *
 * The real SPURS kernel dispatches a workload by loading its policy module
 * (a raw SPU binary) at LS 0xA00 and entering it with the ABI of
 * cellSpursModuleEntry(uintptr_t context, uint64_t ea):
 *
 *   r0 = return-to-kernel address     r3 = kernel context LS address (0x100)
 *   r1 = stack (the PM re-loads its   r4 = the workload's u64 data (preferred
 *        own initial stack)                doubleword) — e.g. a joblist EA
 *                                     r5 = poll status
 *
 * The SpursKernelContext fields this fills, per the layout RPCS3 documents
 * (Emu/Cell/Modules/cellSpurs.h, struct SpursKernelContext) and the lifted
 * wwsjob PM verifiably reads. NOTE both pointers are 64-BIT:
 *   0x1C0 be u64 spurs EA           0x1D8 be u32 wklCurrentUniqueId
 *   0x1C8 be u32 spuNum             0x1DC be u32 wklCurrentId
 *   0x1CC be u32 dmaTagId           0x1E0 be u32 exitToKernelAddr
 *   0x1D0 be u64 wklCurrentAddr     0x1E4 be u32 selectWorkloadAddr
 *        (PM base 0xA00, in the LOW word at 0x1D4)
 *
 * exitToKernel/selectWorkload point at two reserved LS addresses below the PM
 * base; spu_indirect_branch intercepts them (ctx->policy_mode) and HLEs the
 * kernel side: select = "no contention, keep running", exit = end of run.
 */
#include "spu_context.h"
#include "spu_coherency.h"
#include "../platform/win32_compat.h"
#include "spu_workload.h"
#include "spu_jobcap.h"     /* SPURS_PM_PROF job timer */
#include "ps3emu/ps3_ftrace.h"
#include "spu_coherency.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ps3emu/log_first.h"

extern int spu_run_with_halt(void (*)(spu_context*), spu_context*);

/* SPURS_PM_HANG_LS=<ls>,<bytes>[;...] (hex): LS ranges printed with a [pm-hang]
 * report and, once g_spurs_pm_enddump is set (the port's stall watchdog sets it
 * through cellGcm_walk_report), at the end of the next few runs of each wid in
 * SPURS_PM_ENDLS_WID=<lo>-<hi> (decimal). */
volatile int g_spurs_pm_enddump = 0;
static void spurs_pm_dump_ls(const uint8_t* L, unsigned wid, const char* tag)
{
    const char* e = getenv("SPURS_PM_HANG_LS");
    while (e && *e) {
        char* p; uint32_t a = (uint32_t)strtoul(e, &p, 16) & SPU_LS_MASK & ~0xFu, n = 0x40;
        if (*p == ',') n = (uint32_t)strtoul(p + 1, &p, 16);
        for (uint32_t r = a; r < a + n && r + 16 <= SPU_LS_SIZE; r += 16)
            fprintf(stderr, "[%s]   wid=%u ls %05X: %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n",
                    tag, wid, r, L[r], L[r+1], L[r+2], L[r+3], L[r+4], L[r+5], L[r+6], L[r+7],
                    L[r+8], L[r+9], L[r+10], L[r+11], L[r+12], L[r+13], L[r+14], L[r+15]);
        e = strchr(p, ';'); if (e) e++;
    }
}

/* SPURS_PM_HANG=<sec>: report policy-module runs that have not returned after
 * <sec> seconds (a stuck run holds its kernel poll thread, so every workload of
 * that SPURS instance stops). Every 2 s while stuck: wid, lane, pc, lr, r3..r5 and
 * the LS words at pc -- a moving pc is a loop, a fixed one a blocked op.
 * Racy reads of the live context by design; diagnostic only. */
#ifdef _WIN32
#include <windows.h>
#define SPURS_HANG_SLOTS 64
static struct { spu_context* volatile ctx; volatile unsigned long long t0; unsigned wid, lane;
                volatile unsigned polls, jobs, answer; } s_hang[SPURS_HANG_SLOTS];
static SPU_THREAD_LOCAL int t_hang_slot = -1;   /* this thread's running module, for the poll note */
/* Poll intercept (spu_channels.c): count this run's polls, the job entries seen at
 * the poll and the answer, so a stuck run shows whether it still polls and why it
 * is kept. */
void spurs_pm_hang_note_poll(unsigned jobs, unsigned answer)
{
    const int i = t_hang_slot;
    if (i < 0) return;
    s_hang[i].polls++; s_hang[i].jobs = jobs; s_hang[i].answer = answer;
}
static int s_hang_sec = -1;
static DWORD WINAPI spurs_pm_hang_thread(LPVOID p)
{
    (void)p;
    for (;;) {
        Sleep(2000);
        unsigned long long now = GetTickCount64();
        for (int i = 0; i < SPURS_HANG_SLOTS; i++) {
            spu_context* c = s_hang[i].ctx;
            if (!c || now - s_hang[i].t0 < (unsigned long long)s_hang_sec * 1000ull) continue;
            uint32_t pc = (uint32_t)c->pc & SPU_LS_MASK & ~3u;
            const uint8_t* L = c->ls;
            fprintf(stderr, "[pm-hang] wid=%u lane=%u running %llus pc=0x%05X lr=0x%05X r3=%08X r4=%08X r5=%08X "
                    "status=0x%X evstat=0x%X evmask=0x%X int=%d stallstat=0x%X srr0=0x%05X "
                    "ls[pc]=%02X%02X%02X%02X %02X%02X%02X%02X\n",
                    s_hang[i].wid, s_hang[i].lane, (now - s_hang[i].t0) / 1000ull, pc,
                    c->gpr[0]._u32[0] & SPU_LS_MASK, c->gpr[3]._u32[0], c->gpr[4]._u32[0],
                    c->gpr[5]._u32[0], (unsigned)c->status, c->event_status, c->event_mask,
                    c->int_enable, (unsigned)c->list_stall_stat, c->srr0 & SPU_LS_MASK,
                    L[pc], L[pc+1], L[pc+2], L[pc+3], L[pc+4], L[pc+5], L[pc+6], L[pc+7]);
            /* Parked stall-and-notify lists: a handler that waits for a stalled tag to
             * complete spins forever if its resumed list parked again. */
            fprintf(stderr, "[pm-hang]   tagmask=0x%X tagstat=0x%X parked=0x%X", c->mfc_tag_mask,
                    c->mfc_tag_status, c->list_stall_mask);
            for (int t = 0; t < 32; t++)
                if (c->list_stall_mask & (1u << t)) {
                    uint32_t el = c->list_stall_elem_lsa[t] & SPU_LS_MASK & ~7u;
                    fprintf(stderr, " | tag %d elem@0x%05X left=%u next=%02X%02X%02X%02X %02X%02X%02X%02X", t, el,
                            c->list_stall_remaining[t], L[el], L[el+1], L[el+2], L[el+3], L[el+4], L[el+5],
                            L[el+6], L[el+7]);
                }
            { extern unsigned spurs_wid_ready_cont(uint32_t ea, uint32_t wid);
              const uint32_t sp = ((uint32_t)L[0x1C4] << 24) | ((uint32_t)L[0x1C5] << 16) |
                                  ((uint32_t)L[0x1C6] << 8) | L[0x1C7];
              const unsigned rc = sp ? spurs_wid_ready_cont(sp, s_hang[i].wid) : 0;
              fprintf(stderr, " | polls=%u jobs=%u answer=0x%X ready=%u curcont=%u", s_hang[i].polls,
                      s_hang[i].jobs, s_hang[i].answer, rc >> 8, rc & 0xFF); }
            fprintf(stderr, "\n");
            /* Once per stuck run: which image is loaded (a job module or overlay, not the
             * policy module, when pc is past it) and its low GPRs, to name the loop. */
            { static unsigned long long s_said[SPURS_HANG_SLOTS];
              if (s_said[i] != s_hang[i].t0) { s_said[i] = s_hang[i].t0;
                fprintf(stderr, "[pm-hang]   img=%d task=%d ovl=%d", c->image_id, (int)c->resident_task,
                        (int)c->resident_ovl);
                for (int g = 0; g < 16; g++)
                    fprintf(stderr, "%s r%d=%08X %08X %08X %08X", g % 4 ? "" : "\n[pm-hang]  ", g,
                            c->gpr[g]._u32[0], c->gpr[g]._u32[1], c->gpr[g]._u32[2], c->gpr[g]._u32[3]);
                fprintf(stderr, "\n");
                /* ...and where its loop runs: 2000 samples of the step pc, ~1 ms apart. */
                enum { NPC = 24 };
                uint32_t pcs[NPC] = {0}, cnt[NPC] = {0};
                for (int k = 0; k < 2000 && s_hang[i].ctx == c; k++) {
                    const uint32_t p = (uint32_t)c->pc & SPU_LS_MASK;
                    int j = 0;
                    while (j < NPC && cnt[j] && pcs[j] != p) j++;
                    if (j < NPC) { pcs[j] = p; cnt[j]++; }
                    Sleep(1);
                }
                fprintf(stderr, "[pm-hang]   loop pcs:");
                for (int j = 0; j < NPC && cnt[j]; j++) fprintf(stderr, " %05X x%u", pcs[j], cnt[j]);
                fprintf(stderr, "\n"); } }
            spurs_pm_dump_ls(L, s_hang[i].wid, "pm-hang");
            fflush(stderr);
        }
    }
}
static void spurs_pm_hang_begin(spu_context* ctx, unsigned wid, unsigned lane)
{
    if (s_hang_sec < 0) { const char* e = getenv("SPURS_PM_HANG"); s_hang_sec = e ? atoi(e) : 0;
        if (s_hang_sec > 0) { HANDLE h = CreateThread(NULL, 0, spurs_pm_hang_thread, NULL, 0, NULL); if (h) CloseHandle(h); } }
    if (s_hang_sec <= 0) return;
    for (int i = 0; i < SPURS_HANG_SLOTS; i++)
        if (!s_hang[i].ctx) { s_hang[i].wid = wid; s_hang[i].lane = lane; s_hang[i].t0 = GetTickCount64();
                              s_hang[i].polls = s_hang[i].jobs = s_hang[i].answer = 0;
                              s_hang[i].ctx = ctx; t_hang_slot = i; return; }
}
static void spurs_pm_hang_end(spu_context* ctx)
{
    if (s_hang_sec <= 0) return;
    t_hang_slot = -1;
    for (int i = 0; i < SPURS_HANG_SLOTS; i++) if (s_hang[i].ctx == ctx) { s_hang[i].ctx = 0; return; }
}
#else
static void spurs_pm_hang_begin(spu_context* c, unsigned w, unsigned l) { (void)c; (void)w; (void)l; }
static void spurs_pm_hang_end(spu_context* c) { (void)c; }
void spurs_pm_hang_note_poll(unsigned jobs, unsigned answer) { (void)jobs; (void)answer; }
#endif

/* Counters the intercepts in spu_channels.c maintain for the CURRENT run
 * (per worker, including the exit classification used by the caller). */
SPU_THREAD_LOCAL volatile unsigned g_spurs_pm_polls = 0;   /* selectWorkload calls this run */
SPU_THREAD_LOCAL volatile unsigned g_spurs_pm_exited = 0;  /* set when exitToKernel was taken */
SPU_THREAD_LOCAL unsigned long long g_spurs_pm_t0 = 0;     /* QPC at this run's start (poll slice) */
SPU_THREAD_LOCAL unsigned g_spu_job_entries = 0;  /* job-code entries on this thread (SPURS_PM_PROF) */
SPU_THREAD_LOCAL volatile unsigned g_wws_batch_gets = 0;   /* loadCommands (LS 0xC00) fetch count, spu_dma.h */

/* Contexts are resident per (instance, workload, physical SPU). Workload IDs
 * from different instances must never share local store or pipeline state.
 * Each slot lock also serializes direct callers of this existing API. */
#define SPURS_PM_INSTANCES 4
/* CellSpurs2 has 32 workload slots (GoW:A uses 19). */
#define SPURS_PM_MAX_WKL 32
struct PmInstance;
struct PmSlot { SRWLOCK lock; spu_context* ctx; uint64_t data; struct PmInstance* owner;
    const uint8_t* pm; uint32_t size, wid; spu_lifted_entry_fn entry; int image_id; };
static struct PmInstance {
    int used, retiring;
    unsigned users, workload_users[SPURS_PM_MAX_WKL];
    uint32_t ea;
    struct PmSlot slots[SPURS_PM_MAX_WKL][6];
} s_pm_instances[SPURS_PM_INSTANCES];
static SRWLOCK s_pm_directory_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_pm_directory_wake;
static struct PmSlot* policy_slot(uint32_t ea, uint32_t wid, uint32_t lane)
{
    struct PmInstance* free_slot = NULL;
    struct PmSlot* slot = NULL;
    AcquireSRWLockExclusive(&s_pm_directory_lock);
    for (unsigned i = 0; i < SPURS_PM_INSTANCES; i++) {
        struct PmInstance* pi = &s_pm_instances[i];
        if (pi->used && pi->ea == ea) {
            if (!pi->retiring) slot = &pi->slots[wid][lane];
            free_slot = NULL; break;
        }
        if (!pi->used && !free_slot) free_slot = pi;
    }
    if (!slot && free_slot) {
        free_slot->used = 1; free_slot->ea = ea;
        slot = &free_slot->slots[wid][lane];
    }
    if (slot) {
        for (unsigned i = 0; i < SPURS_PM_INSTANCES; i++)
            if (s_pm_instances[i].used && s_pm_instances[i].ea == ea) {
                slot->owner = &s_pm_instances[i]; slot->wid = wid;
                slot->owner->users++;slot->owner->workload_users[wid]++; break;
            }
    }
    ReleaseSRWLockExclusive(&s_pm_directory_lock);
    if (slot) AcquireSRWLockExclusive(&slot->lock);
    return slot;
}

static void policy_slot_release(struct PmSlot* slot)
{
    ReleaseSRWLockExclusive(&slot->lock);
    AcquireSRWLockExclusive(&s_pm_directory_lock);
    slot->owner->users--;slot->owner->workload_users[slot->wid]--;
    WakeAllConditionVariable(&s_pm_directory_wake);
    ReleaseSRWLockExclusive(&s_pm_directory_lock);
}

/* Quiescent workload removal must retire its incarnation even when the next
 * registration has identical image and data. Never wait under admission locks:
 * a direct caller that still owns a reference makes removal BUSY instead. */
int spurs_policy_release_workload(uint32_t ea, uint32_t wid)
{
    if (wid >= SPURS_PM_MAX_WKL) return -1;
    AcquireSRWLockExclusive(&s_pm_directory_lock);
    for (unsigned i = 0; i < SPURS_PM_INSTANCES; i++) {
        struct PmInstance* pi = &s_pm_instances[i];
        if (!pi->used || pi->ea != ea) continue;
        if (pi->retiring || pi->workload_users[wid]) {
            ReleaseSRWLockExclusive(&s_pm_directory_lock); return -1;
        }
        for (unsigned n = 0; n < 6; n++) {
            struct PmSlot* slot = &pi->slots[wid][n];
            if (slot->ctx) {
                spu_coh_unregister(slot->ctx);spu_mfc_release(slot->ctx);
                free(slot->ctx);slot->ctx = NULL;
            }
            slot->data = 0;slot->pm = NULL;slot->size = 0;
            slot->entry = NULL;slot->image_id = 0;
        }
        break;
    }
    ReleaseSRWLockExclusive(&s_pm_directory_lock);
    return 0;
}

/* Caller must have joined the instance's scheduler workers. Direct policy
 * callers are serialized by their slot locks before resident resources retire.
 * Keep the zero-initialized lock objects valid when the directory slot recurs. */
void spurs_policy_release_instance(uint32_t ea)
{
    AcquireSRWLockExclusive(&s_pm_directory_lock);
    for (unsigned i = 0; i < SPURS_PM_INSTANCES; i++) {
        struct PmInstance* pi = &s_pm_instances[i];
        if (!pi->used || pi->ea != ea) continue;
        if (pi->retiring) break;
        pi->retiring = 1;
        while (pi->users)
            SleepConditionVariableSRW(&s_pm_directory_wake, &s_pm_directory_lock, INFINITE, 0);
        for (unsigned w = 0; w < SPURS_PM_MAX_WKL; w++) for (unsigned n = 0; n < 6; n++) {
            struct PmSlot* slot = &pi->slots[w][n];
            AcquireSRWLockExclusive(&slot->lock);
            if (slot->ctx) {
                spu_coh_unregister(slot->ctx);
                spu_mfc_release(slot->ctx);
                free(slot->ctx);
                slot->ctx = NULL;
            }
            slot->data = 0; slot->pm = NULL; slot->size = 0; slot->entry = NULL; slot->image_id = 0;
            ReleaseSRWLockExclusive(&slot->lock);
        }
        pi->used = 0; pi->retiring = 0; pi->ea = 0;
        break;
    }
    ReleaseSRWLockExclusive(&s_pm_directory_lock);
}

/* PS3_FTRACE "who" for a host kernel thread: numbered in first-use order. */
#define SPURS_KT_MAX 32
volatile long g_spurs_kt_tid[SPURS_KT_MAX];
volatile long g_spurs_kt_wid[SPURS_KT_MAX] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

int spurs_ft_kidx(void)
{
    static SPU_THREAD_LOCAL int t_ft_k = -1; static volatile LONG s_ft_k;
    if (t_ft_k < 0) t_ft_k = (int)InterlockedIncrement(&s_ft_k) - 1;
    return t_ft_k;
}

int spu_run_policy_module(spu_lifted_entry_fn entry, int image_id,
                          const uint8_t* pm_host, uint32_t pm_size,
                          uint64_t wkl_data, uint32_t wid, uint32_t spurs_ea,
                          uint32_t spu_num)
{
    if (!entry || !pm_host || !pm_size || pm_size > SPU_LS_SIZE - 0xA00)
        return -1;

    /* Reuse one context per thread: this runs per policy dispatch (thousands
     * per second with the SPURS workload churn live), and a fresh 260KB
     * calloc/free each time was allocator + page-zeroing churn. The explicit
     * memset preserves the exact fresh-zero semantics calloc provided. */
    /* PERSISTENT context per (workload, virtual SPU): a real policy module
     * stays RESIDENT on its SPU -- its LS carries continuation state between
     * scheduling quanta (the wwsjob PM caches the last-seen queue line at LS
     * 0x14B0 and its staging rings at 0x12xx; a freshly zeroed context every
     * dispatch made it treat every re-entry as first contact and it could
     * never progress its claim/stage pipeline). Image + context init happen
     * once; re-entries keep LS and only refresh the entry registers and the
     * (possibly game-updated) kernel-context fields. */
    if (wid >= SPURS_PM_MAX_WKL || spu_num >= 6) return -1;
    const uint64_t ft_s0 = PS3_FT_ON() ? ps3_ft_now() : 0;
    struct PmSlot* resident = policy_slot(spurs_ea, wid, spu_num);
    if (!resident) return -1;
    if (ft_s0) {                    /* waited for the workload's resident slot */
        const uint64_t ft_s1 = ps3_ft_now();
        if (ft_s1 - ft_s0 >= ps3_ft_min_ns())
            ps3_ft_span(PS3_FT_SPU_WAIT, (uint32_t)spurs_ft_kidx(), 0x5107u, wid, ft_s0, ft_s1);
    }
    spu_context* ctx = resident->ctx;
    int first = 0;
    /* SPURS_PM_FRESH=1: treat EVERY dispatch as first contact (zeroed LS, image
     * reloaded). A/B against the persistent default: with persistence, the wwsjob
     * PM's first entry attaches + claims exactly ONE ticket then yields, and every
     * RE-entry takes a stale-state path that does NOTHING (LS 0x12A0/0x12C0/0x12F0
     * job-pipeline slots persist; entry only partially re-inits them) -- LBP's
     * 10-ticket loading queue stalls at lanes {1,2} forever. Fresh contexts make
     * each dispatch run the full attach->claim->yield cycle, so repeated kernel
     * dispatches drain the queue one ticket at a time (the persistent-LS claim
     * that motivated the cache predates the r4 word0/word1 layout fix). */
    static SPU_THREAD_LOCAL int s_fresh = -1;
    if (s_fresh < 0) s_fresh = getenv("SPURS_PM_FRESH") ? 1 : 0;
    /* Reset the persistent context when the workload's DATA (queue EA)
     * changes: the game re-points a wid at a NEW job queue once the old one
     * drains, and a real SPURS kernel reloads the policy module fresh on a
     * workload switch. Carrying queue A's LS (header cache 0x14B0, busy
     * masks 0x1270/0x1280, records) into queue B made the PM see stale
     * state and claim nothing (observed: LBP's second loading queue sat at
     * lanes {0,...} while the first drained fine). */
    if (!ctx) {
        ctx = (spu_context*)malloc(sizeof(spu_context));
        if (!ctx) { policy_slot_release(resident); return -1; }
        resident->ctx = ctx;
        first = 1;
        /* SPU_CTX_LOG=1: policy-module contexts are PERSISTENT and there is one
         * per (workload, spu_num) -- a workload with readyCount 8 legitimately
         * runs on several simulated SPUs, which then contend its lock lines
         * exactly as real SPUs would. Counting only the job contexts in
         * spu_lifted_job.h reports "two contexts" for a run that has five, and
         * makes correct multi-SPU execution look like duplicated state. */
        { static SPU_THREAD_LOCAL int s_cl = -1;
          if (s_cl < 0) s_cl = getenv("SPU_CTX_LOG") ? 1 : 0;
          if (s_cl) fprintf(stderr, "[spu-ctx] new POLICY context %p wid=%u spu_num=%u\n",
                            (void*)ctx, wid, spu_num); }
        memset(ctx, 0, sizeof(*ctx));
        spu_context_init(ctx, 0);
    } else if (s_fresh || resident->data != wkl_data ||
               resident->pm != pm_host || resident->size != pm_size ||
               resident->entry != entry || resident->image_id != image_id) {
        if (!s_fresh) {
            static SPU_THREAD_LOCAL int _n = 0;
            if (PS3_LOG_FIRST(_n, 8))
                fprintf(stderr, "[spurs-pm] wid=%u spu=%u workload data changed "
                        "0x%llX -> 0x%llX: fresh PM context\n", wid, spu_num,
                        (unsigned long long)resident->data,
                        (unsigned long long)wkl_data);
        }
        first = 1;
        /* A fresh context drops any parked list, so drop its MFC engine as well:
         * the engine would keep that list's tag incomplete forever. */
        spu_coh_unregister(ctx);
        spu_mfc_release(ctx);
        memset(ctx, 0, sizeof(*ctx));
        spu_context_init(ctx, 0);
    }
    resident->data = wkl_data;
    resident->pm = pm_host; resident->size = pm_size;
    resident->entry = entry; resident->image_id = image_id;
    ctx->image_id    = image_id;
    ctx->policy_mode = 1;

    /* PM image at its load base (first entry only -- LS persists). */
    if (first)
        memcpy(ctx->ls + 0xA00, pm_host, pm_size);

    /* SpursKernelContext at LS 0x100 (fields big-endian, like all guest data). */
    uint8_t* ls = ctx->ls;
#define KBE32(off, v) do { uint32_t _v = (v); ls[(off)] = (uint8_t)(_v >> 24); \
        ls[(off)+1] = (uint8_t)(_v >> 16); ls[(off)+2] = (uint8_t)(_v >> 8);   \
        ls[(off)+3] = (uint8_t)_v; } while (0)
    /* wklCurrentAddr is a 64-BIT pointer (vm::bcptr<void,u64>), so it occupies
     * 0x1D0..0x1D7 and everything after it sits 4 bytes higher than a u32 would
     * put it. Writing it as a u32 shifted the whole tail of the struct:
     * exitToKernelAddr landed on selectWorkloadAddr's slot and selectWorkloadAddr
     * fell off into moduleId, leaving 0x1E4 zero. The wwsjob PM reads both --
     * `lqa $2,0x1e0; bisl $0,$2` at LS 0x3588 calls exitToKernel, and
     * `lqa $7,0x1e0; rotqbyi $6,$7,4; bisl $0,$6` at LS 0x35A0 calls
     * selectWorkload -- so it called our exit intercept for every poll and
     * branched to 0 for every select. It also tests wklCurrentId (0x1DC) against
     * 1, which only reads sensibly at the correct offset. */
    /* tempArea (LS 0x100..0x17F): the real kernel's workload selection GETLLARs the
     * first CellSpurs lock line into it, and modules read it afterwards (GoW:A's
     * WWS jobs test bits around 0x170 -- wklSignal/wklFlagReceiver -- to pick the
     * second workload set's wklInfo; left zero, they computed EA 0 and spun). */
    { extern uint8_t* vm_base;
      if (spurs_ea) memcpy(ls + 0x100, vm_base + spurs_ea, 0x80); }
    KBE32(0x1C0, 0);                    /* spurs EA hi32 (u64 ptr)  */
    KBE32(0x1C4, spurs_ea);             /* spurs EA lo32            */
    KBE32(0x1C8, spu_num);              /* spuNum (virtual SPU / lane row) */
    KBE32(0x1CC, 8);                    /* dmaTagId (kernel's tag)  */
    KBE32(0x1D0, 0);                    /* wklCurrentAddr hi32 (u64 ptr) */
    KBE32(0x1D4, 0xA00);                /* wklCurrentAddr lo32 = PM load base */
    KBE32(0x1D8, wid);                  /* wklCurrentUniqueId       */
    KBE32(0x1DC, wid);                  /* wklCurrentId             */
    KBE32(0x1E0, SPURS_PM_EXIT_TO_KERNEL_LS);
    KBE32(0x1E4, SPURS_PM_SELECT_WORKLOAD_LS);
#undef KBE32

    /* Entry registers per cellSpursModuleEntry. */
    ctx->gpr[0]._u32[0] = SPURS_PM_EXIT_TO_KERNEL_LS;  /* return-to-kernel link */
    ctx->gpr[1]._u32[0] = 0x3FFB0;                     /* stack (PM reloads its own) */
    ctx->gpr[3]._u32[0] = 0x100;                       /* context */
    /* r4 = cellSpursModuleEntry's u64 workload argument in the SPU ABI's
     * preferred doubleword (ABI 2.1.2, fig. 2-5: bytes 0-7, big-endian), so
     * word 0 = high half, word 1 = low half. The wwsjob job-manager PM reads
     * word 1 as the job list header EA and word 0 as an optional AUDIT buffer
     * EA: if non-zero it GETs 256 bytes of "audit header" there and PUTs a
     * 16-byte FFFFFFFF terminator at EA + 0x40 + 16 * spuIndex. The earlier
     * {lo, lo, hi, hi} layout put the list EA in word 0 too, so every PM entry
     * treated the list header as an audit header and wrote FFFF over the
     * list's per-SPU cursor rows (SPU 0-3) and first job slots (SPU 4): the
     * "ticket values exploding" with extra SPUs. Titles pass a 32-bit list
     * EA (high half 0), so audit stays off, as on hardware.
     * GoW:A's wwsjob PM (entry 0x2C30) needs it too: with the low half also
     * in word 0 it wrote its per-SPU status rows (0xFFFFFFFF, 0x10) at header +
     * 0x130 -- inside job list 1 -- so a list of exactly 7 jobs read a garbage
     * 8th entry, counted 22 of 21 and the batch-job wait never finished. */
    ctx->gpr[4]._u32[0] = (uint32_t)(wkl_data >> 32);
    ctx->gpr[4]._u32[1] = (uint32_t)wkl_data;
    ctx->gpr[4]._u32[2] = 0;
    ctx->gpr[4]._u32[3] = 0;
    /* SPURS_PM_ARG_BE=0: the older {lo, lo, hi, hi} layout instead (the
     * big-endian layout above is the default; =1 is accepted and changes
     * nothing). Kept for A/B; the notes that motivated it follow. */
    { static int s_be = -1;
      if (s_be < 0) { const char* e = getenv("SPURS_PM_ARG_BE"); s_be = !(e && *e == '0'); }
      if (!s_be) {
        /* r4 layout is LO-WORD-IN-PREFERRED, not a big-endian u64: verified
         * against the authentic wwsjob PM bytes (entry 0x2BF0 `stqa r4,0x14E0`,
         * attach 0x2720 `brz r4`/`wrch MFC_EAL, r4` -- it treats r4's PREFERRED
         * WORD as the 32-bit queue EA, and `shlqbyi r4,4`'s next word as the aux
         * half). Passing {hi,lo} put 0 in the preferred slot, so the queue attach
         * early-returned on every run: tickets got claimed via the shifted copy
         * but the job-list header never DMA'd in, and the PM wedged forever in
         * its staged-job wait (LBP's post-intro loading freeze). */
        /* ...and the CLAIM path reads the EA from the shlqbyi(r4,4) copy (LS
         * 0x13F0, via func_3038 -> 2418), i.e. r4's SECOND word. Both word 0
         * (attach) and word 1 (claim) must carry the 32-bit EA; the hi half
         * rides in words 2-3. */
        ctx->gpr[4]._u32[0] = (uint32_t)wkl_data;          /* attach reads word0  */
        ctx->gpr[4]._u32[1] = (uint32_t)wkl_data;          /* claim reads word1   */
        ctx->gpr[4]._u32[2] = (uint32_t)(wkl_data >> 32);
        ctx->gpr[4]._u32[3] = (uint32_t)(wkl_data >> 32);
      } }
    ctx->gpr[5]._u32[0] = 0;                           /* poll status */

    g_spurs_pm_polls  = 0;
    g_spurs_pm_exited = 0;
    { LARGE_INTEGER q; QueryPerformanceCounter(&q); g_spurs_pm_t0 = (unsigned long long)q.QuadPart; }

    /* RUN/END tracing: thousands of policy dispatches per second made these
     * unconditional fprintf+fflush pairs a real fraction of movie playback.
     * Default off; SPURS_PM_TRACE=1 restores them (first 64 + every 4096th
     * so an enabled trace can't flood either). */
    static SPU_THREAD_LOCAL int s_trace = -1;
    if (s_trace < 0) s_trace = getenv("SPURS_PM_TRACE") ? 1 : 0;
    static SPU_THREAD_LOCAL unsigned long s_runs = 0;
    unsigned long runno = ++s_runs;
    int log_this = s_trace && (runno <= 64 || (runno & 0xFFF) == 0);
    /* SPURS_WID_STATS=1: per-workload run counts every 2048 runs (which wids
     * the kernel actually schedules; the RUN# log above only samples). */
    { static int s_ws = -1; static unsigned long s_cnt[32];
      if (s_ws < 0) s_ws = getenv("SPURS_WID_STATS") ? 1 : 0;
      if (s_ws) { s_cnt[wid & 31]++;
        if ((runno & 0x7FF) == 0) { fprintf(stderr, "[wid-stats] run#%lu:", runno);
          for (int k = 0; k < 32; k++) if (s_cnt[k]) fprintf(stderr, " w%d=%lu", k, s_cnt[k]);
          fputc(10, stderr); } } }
    if (log_this) {
        fprintf(stderr, "[spurs-pm] RUN#%lu wid=%u data=0x%llX spurs=0x%08X image=%d entry=0xA00 size=%u\n",
                runno, wid, (unsigned long long)wkl_data, spurs_ea, image_id, pm_size);
        fflush(stderr);
    }

    /* SPURS_PM_FLOW=1: capture this run's cross-function transfer trail (drain
     * hook in spu_drain.c) and dump it IF the run performed a PUTLLC (= a queue
     * claim) -- reconstructs the post-claim decision path that ends in a bare
     * exit-to-kernel instead of job staging. Only spu_num 0 is traced (the
     * concurrent lanes would interleave the buffer). */
    extern uint32_t g_pm_flow_buf[]; extern volatile unsigned g_pm_flow_n;
    extern void* volatile g_pm_flow_ctx;
    extern volatile unsigned g_spu_putllc_count;
    extern volatile unsigned g_spu_putllc_sync_hit;
    extern uint32_t g_barrier_sync_watch;
    static SPU_THREAD_LOCAL int s_flow = -1;
    if (s_flow < 0) s_flow = getenv("SPURS_PM_FLOW") ? 1 : 0;
    unsigned putllc_before = g_spu_putllc_sync_hit;
    (void)g_spu_putllc_count;
    /* Trace ONLY the stalled queue's own runs: wkl_data == the sync EA the PPU
     * barrier/ticket probes armed (ticket-pub arms it at first publish, before
     * the PM's first work pass). Idle re-runs of this wid matter as much as the
     * claim run -- the empty-cursor path returns without any PUTLLC. */
    /* SPURS_PM_FLOW_ALL=1: trace EVERY spu_num-0 run of this workload without
     * requiring the PPU-side barrier probe to have armed g_barrier_sync_watch.
     * Needed to answer "does the job's buffer-binding routine (pm 0x3098, the
     * only writer of logicalToBufferNumArray@0x11B0 + jobState@0x12C0) ever
     * execute?" on a run that never reaches the barrier at all -- the RPCS3
     * guest dump shows a working SPU has 0x11B0 bound and runJobNum=0, ours
     * has 0x11B0 == 0 and runJobNum == -1. */
    static SPU_THREAD_LOCAL int s_flow_all = -1;
    if (s_flow_all < 0) s_flow_all = getenv("SPURS_PM_FLOW_ALL") ? 1 : 0;
    int tracing = (s_flow && spu_num == 0 &&
                   (s_flow_all ||
                    (g_barrier_sync_watch &&
                     (uint32_t)wkl_data == g_barrier_sync_watch)));
    if (tracing) {
        void* expected = NULL;
        tracing = __atomic_compare_exchange_n(&g_pm_flow_ctx, &expected, ctx, 0,
                                             __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
    static volatile unsigned long long s_flow_arm_ms;
    extern unsigned long long ps3_ms_now(void);
    if (tracing) {
        g_pm_flow_n = 0;
        s_flow_arm_ms = ps3_ms_now();
        { static SPU_THREAD_LOCAL int _a = 0; if (PS3_LOG_FIRST(_a, 8)) {
            fprintf(stderr, "[pm-flow] ARM wid=%u spu=%u (watch=0x%08X)\n",
                    wid, spu_num, g_barrier_sync_watch); fflush(stderr); } }
        /* Watchdog: the traced run has been observed to HANG the jobmanager
         * kernel thread (the claim run never returns -> no later dispatch can
         * report). A detached thread dumps the partial trail after 5s; the
         * trail's LAST pc names the looping lifted function. */
        { static int s_wd_started = 0;
          if (__atomic_exchange_n(&s_wd_started, 1, __ATOMIC_RELAXED) == 0) {
            extern void spurs_pm_flow_watchdog_start(void);
            spurs_pm_flow_watchdog_start(); } }
    }

    /* SPURS_PM_PIPE=1: trace how the WWS job PIPELINE state evolves across
     * kernel dispatches. Prints the pipeline counters (loadJobState@0x12A0
     * preferred hword, nextLoadJobNum@0x12C0, loadJobNum@0x12D0, runJobNum@
     * 0x12E0, storeJobNum@0x12F0) before and after each PM run, plus the pc
     * spu_run_with_halt halted at. Answers: does re-entry advance the pipeline
     * or reset/stall it? Only wid<=1 (jobmanager) traced, first 4000 runs. */
    static SPU_THREAD_LOCAL int s_pipe = -1;
    if (s_pipe < 0) s_pipe = getenv("SPURS_PM_PIPE") ? 1 : 0;
    /* trace ALL workloads (the jobmanager is wid=12, not <=1); 4000-run cap bounds volume */
    int pipe_this = s_pipe && spu_num == 0;
    uint32_t st_before = 0;
    if (pipe_this) {
        const uint8_t* L = ctx->ls;
        #define HW(o) ((L[(o)+2]<<8)|L[(o)+3])   /* preferred hword (bytes 2-3) */
        st_before = HW(0x12A0);
        (void)st_before;
        #undef HW
    }

    /* SPURS_PM_CLAIMLOG=1: per-run claim-vs-stage summary for BOTH lanes.
     * The loading freeze's signature is a dispatch that CLAIMS a ticket
     * (sync-PUTLLC hit) but never STAGES its job (no loadCommands GET): the
     * claimed ticket is eaten and the batch's barrier can never release
     * (observed: 4-job batch, counter=4, lanes {2,1}). Diffing the sync-PUTLLC
     * and 0xC00-GET counters across each run names that dispatch and shows the
     * LS pipeline state it left behind. */
    static SPU_THREAD_LOCAL int s_claimlog = -1;
    if (s_claimlog < 0) s_claimlog = getenv("SPURS_PM_CLAIMLOG") ? 1 : 0;
    unsigned batch_before = g_wws_batch_gets;
    unsigned claim_before = g_spu_putllc_sync_hit;

    spurs_pm_hang_begin(ctx, wid, spu_num);
    /* SPURS_PM_PROF start marks, only when it is on (a clock read and a 256-byte copy per
     * module run, ~100k runs/s in GoW:A). win32_compat supplies QueryPerformanceCounter. */
    extern SPU_THREAD_LOCAL long long g_spu_interp_ticks;
    extern SPU_THREAD_LOCAL long long g_spu_wrch_ticks, g_spu_rdch_ticks;
    extern SPU_THREAD_LOCAL long long g_spu_rdch_ch_ticks[32];
    static int s_pm_prof_on = -1;
    if (s_pm_prof_on < 0) { const char* e = getenv("SPURS_PM_PROF"); s_pm_prof_on = e && atoi(e) > 0; }
    LARGE_INTEGER prof_t0; prof_t0.QuadPart = 0;
    long long prof_i0 = 0, prof_w0 = 0, prof_r0 = 0, prof_rc0[32];
    unsigned prof_j0 = 0;
    if (s_pm_prof_on) {
        QueryPerformanceCounter(&prof_t0);
        prof_i0 = g_spu_interp_ticks;
        prof_w0 = g_spu_wrch_ticks; prof_r0 = g_spu_rdch_ticks;
        memcpy(prof_rc0, g_spu_rdch_ch_ticks, sizeof prof_rc0);
        prof_j0 = g_spu_job_entries;
    }
    /* PS3_FTRACE: each module run is an SPU busy span, tagged with the image
     * and the workload id (kernel threads never block on a channel, so the
     * channel-wait hook in spu_channels.c records nothing for them). */
    const uint64_t ft_t0 = PS3_FT_ON() ? ps3_ft_now() : 0;
    /* Which workload each kernel thread is running, for the host sampler's
     * PS3_SAMPLE_WIDS filter (ppu_loader.cpp). */
    const int kt = spurs_ft_kidx();
    if (kt < SPURS_KT_MAX) {
#ifdef _WIN32
        g_spurs_kt_tid[kt] = (long)GetCurrentThreadId();
#endif
        g_spurs_kt_wid[kt] = (long)wid;
    }
    const int pm_halted = spu_run_with_halt(entry, ctx);
    if (kt < SPURS_KT_MAX) g_spurs_kt_wid[kt] = -1;
    if (ft_t0) {
        const uint64_t ft_t1 = ps3_ft_now();
        if (ft_t1 - ft_t0 >= ps3_ft_min_ns()) {
            /* who = the host kernel thread, numbered in first-run order (policy
             * contexts all carry spu_id 0, so that would merge every thread). */
            ps3_ft_span(PS3_FT_SPU_BUSY, (uint32_t)spurs_ft_kidx(), (uint32_t)ctx->image_id, (uint32_t)wid, ft_t0, ft_t1);
        }
    }
    /* SPURS_PM_PROF=<s>: every s seconds, the wall time each workload's module
     * runs took, as a share of the window. All of an instance's modules run on
     * one host thread, so the shares add up to how busy that thread is. */
    { static int s_pp = -1; static LARGE_INTEGER s_f, s_w0; static double s_acc[SPURS_PM_MAX_WKL];
      static double s_int[SPURS_PM_MAX_WKL], s_idl[SPURS_PM_MAX_WKL]; static unsigned s_job[SPURS_PM_MAX_WKL];
      static double s_wrc[SPURS_PM_MAX_WKL], s_rdc[SPURS_PM_MAX_WKL];
      static unsigned s_runs[SPURS_PM_MAX_WKL]; static SRWLOCK s_pl = SRWLOCK_INIT;
      static double s_jt[SPURS_PM_MAX_WKL]; static unsigned s_jtn[SPURS_PM_MAX_WKL];
      static unsigned s_jth[SPURS_PM_MAX_WKL][SPU_JOBT_BUCKETS];
      if (s_pp < 0) { const char* e = getenv("SPURS_PM_PROF"); s_pp = e ? atoi(e) : 0;
                      QueryPerformanceFrequency(&s_f); QueryPerformanceCounter(&s_w0); }
      if (s_pp > 0 && wid < SPURS_PM_MAX_WKL) {
          LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
          AcquireSRWLockExclusive(&s_pl);
          s_acc[wid] += (double)(t1.QuadPart - prof_t0.QuadPart) / (double)s_f.QuadPart;
          s_int[wid] += (double)(g_spu_interp_ticks - prof_i0) / (double)s_f.QuadPart;
          s_wrc[wid] += (double)(g_spu_wrch_ticks - prof_w0) / (double)s_f.QuadPart;
          s_rdc[wid] += (double)(g_spu_rdch_ticks - prof_r0) / (double)s_f.QuadPart;
          static double s_rcc[SPURS_PM_MAX_WKL][32];
          for (int ch = 0; ch < 32; ch++)
              s_rcc[wid][ch] += (double)(g_spu_rdch_ch_ticks[ch] - prof_rc0[ch]) / (double)s_f.QuadPart;
          if (g_spurs_pm_polls == 0)   /* the module found no work: a pure poll */
              s_idl[wid] += (double)(t1.QuadPart - prof_t0.QuadPart) / (double)s_f.QuadPart;
          s_runs[wid]++;
          s_job[wid] += g_spu_job_entries - prof_j0;
          /* The job timer's totals belong to this run: take them and zero them. */
          s_jt[wid] += (double)g_spu_jobt_ticks / (double)s_f.QuadPart; g_spu_jobt_ticks = 0;
          s_jtn[wid] += g_spu_jobt_done; g_spu_jobt_done = 0;
          for (int b = 0; b < SPU_JOBT_BUCKETS; b++) { s_jth[wid][b] += g_spu_jobt_hist[b]; g_spu_jobt_hist[b] = 0; }
          const double win = (double)(t1.QuadPart - s_w0.QuadPart) / (double)s_f.QuadPart;
          if (win >= s_pp) {
              char buf[1024]; int p = snprintf(buf, sizeof buf, "[pm-prof] %.1fs:", win);
              double tot = 0, idl = 0;
              double s_accp[SPURS_PM_MAX_WKL];   /* the window's s_acc, kept past the reset below */
              memcpy(s_accp, s_acc, sizeof s_accp);
              unsigned s_jobp[SPURS_PM_MAX_WKL];   /* and its job counts */
              memcpy(s_jobp, s_job, sizeof s_jobp);
              for (unsigned w = 0; w < SPURS_PM_MAX_WKL; w++) {
                  if (!s_runs[w]) continue;
                  tot += s_acc[w]; idl += s_idl[w];
                  /* w<wid>=<share of window>%/<runs>[i<share of that time interpreted>%] */
                  if (p < (int)sizeof buf - 48)
                      p += snprintf(buf + p, sizeof buf - p, " w%u=%.0f%%/%u", w, 100.0 * s_acc[w] / win, s_runs[w]);
                  if (s_int[w] > 0 && p < (int)sizeof buf - 16)
                      p += snprintf(buf + p, sizeof buf - p, "i%.0f%%", 100.0 * s_int[w] / s_acc[w]);
                  if (s_idl[w] > 0 && p < (int)sizeof buf - 16)   /* z = share spent in idle polls */
                      p += snprintf(buf + p, sizeof buf - p, "z%.0f%%", 100.0 * s_idl[w] / s_acc[w]);
                  if (s_job[w] && p < (int)sizeof buf - 16)       /* j = job-code entries */
                      p += snprintf(buf + p, sizeof buf - p, "j%u", s_job[w]);
                  /* m = share in channel writes (MFC commands/DMA), c = in channel reads (waits) */
                  if (s_acc[w] > 0 && (s_wrc[w] > 0 || s_rdc[w] > 0) && p < (int)sizeof buf - 24)
                      p += snprintf(buf + p, sizeof buf - p, "m%.0f%%c%.0f%%",
                                    100.0 * s_wrc[w] / s_acc[w], 100.0 * s_rdc[w] / s_acc[w]);
                  s_acc[w] = 0; s_int[w] = 0; s_idl[w] = 0; s_runs[w] = 0; s_job[w] = 0;
                  s_wrc[w] = 0; s_rdc[w] = 0;
              }
              fprintf(stderr, "%s total=%.0f%% idle=%.0f%%\n", buf, 100.0 * tot / win, 100.0 * idl / win);
              /* [pm-jobt] w<wid>: the job proper (job start to its return to the
               * link) against the whole module run, per job, for workloads that
               * ran jobs. p50/p90 are bucket upper edges (10 us). */
              for (unsigned w = 0; w < SPURS_PM_MAX_WKL; w++) {
                  if (s_jtn[w]) {
                      unsigned q50 = 0, q90 = 0, c = 0;
                      for (int b = 0; b < SPU_JOBT_BUCKETS; b++) {
                          c += s_jth[w][b];
                          if (!q50 && c * 2 >= s_jtn[w]) q50 = 10u * (b + 1);
                          if (!q90 && c * 10 >= s_jtn[w] * 9u) q90 = 10u * (b + 1);
                      }
                      const unsigned jobs = s_jobp[w] ? s_jobp[w] : s_jtn[w];
                      fprintf(stderr, "[pm-jobt] w%u: jobs=%u timed=%u proper=%.1fus/job p50<=%uus p90<=%uus module=%.1fus/job proper/module=%.0f%%\n",
                              w, jobs, s_jtn[w], 1e6 * s_jt[w] / s_jtn[w], q50, q90,
                              1e6 * s_accp[w] / jobs, 100.0 * s_jt[w] / (s_accp[w] > 0 ? s_accp[w] : 1));
                  }
                  s_jt[w] = 0; s_jtn[w] = 0; memset(s_jth[w], 0, sizeof s_jth[w]);
              }
              /* The channel reads (c) by channel, for workloads busy over 5% of the
               * window: [pm-prof-ch] w<wid>: ch<n>=<share of that workload's time>%. */
              for (unsigned w = 0; w < SPURS_PM_MAX_WKL; w++) {
                  double rsum = 0; for (int ch = 0; ch < 32; ch++) rsum += s_rcc[w][ch];
                  if (s_accp[w] > 0.05 * win && rsum > 0) {
                      char cb[400]; int cp = snprintf(cb, sizeof cb, "[pm-prof-ch] w%u:", w);
                      for (int ch = 0; ch < 32; ch++)
                          if (s_rcc[w][ch] >= 0.005 * s_accp[w] && cp < (int)sizeof cb - 20)
                              cp += snprintf(cb + cp, sizeof cb - cp, " ch%d=%.1f%%", ch, 100.0 * s_rcc[w][ch] / s_accp[w]);
                      fprintf(stderr, "%s\n", cb);
                  }
                  for (int ch = 0; ch < 32; ch++) s_rcc[w][ch] = 0;
              }
              s_w0 = t1;
          }
          ReleaseSRWLockExclusive(&s_pl);
      } }
    spurs_pm_hang_end(ctx);
    if (g_spurs_pm_enddump) {
        static int lo = -1, hi = -1; static int left[64];
        if (lo < 0) { const char* e = getenv("SPURS_PM_ENDLS_WID"); lo = 0; hi = -1;
                      if (e) { char* p; lo = (int)strtol(e, &p, 10); hi = (*p == '-') ? (int)strtol(p + 1, 0, 10) : lo; }
                      for (int k = 0; k < 64; k++) left[k] = 3; }
        if ((int)wid >= lo && (int)wid <= hi && wid < 64 && left[wid] > 0) {
            left[wid]--;
            fprintf(stderr, "[pm-end] wid=%u lane=%u pc=0x%05X r3=%08X\n", wid, spu_num,
                    (uint32_t)ctx->pc & SPU_LS_MASK, ctx->gpr[3]._u32[0]);
            spurs_pm_dump_ls(ctx->ls, wid, "pm-end");
            fflush(stderr);
        }
    }
    /* A lane's run should only end through exitToKernel. Anything else strands
     * the jobs it had claimed (GH3's once-per-~10-songs freeze): say how. */
    if (((uint32_t)ctx->pc & SPU_LS_MASK) != SPURS_PM_EXIT_TO_KERNEL_LS) {
        extern unsigned spu_recent_pcs(uint32_t*, unsigned);
        uint32_t r[32]; unsigned n = spu_recent_pcs(r, 32);
        static volatile long s_pe; if (__atomic_add_fetch(&s_pe, 1, __ATOMIC_RELAXED) <= 64) {
            fprintf(stderr, "[pm-end] lane %u ctx=%p halted=%d status=0x%X pc=0x%05X lr=0x%05X r3=0x%08X depth=%u recent:",
                    spu_num, (void*)ctx, pm_halted, ctx->status, (uint32_t)ctx->pc & SPU_LS_MASK,
                    ctx->gpr[0]._u32[0] & SPU_LS_MASK, ctx->gpr[3]._u32[0], ctx->host_depth);
            for (unsigned k = 0; k < n; k++) fprintf(stderr, " %05X", r[k]);
            fprintf(stderr, "\n"); fflush(stderr); } }

    if (s_claimlog) {
        unsigned claims = g_spu_putllc_sync_hit - claim_before;
        unsigned stages = g_wws_batch_gets - batch_before;
        if (claims || stages) {
            static SPU_THREAD_LOCAL unsigned s_cl = 0;
            if (PS3_LOG_FIRST(s_cl, 64)) {
                const uint8_t* L = ctx->ls;
                #define W(o)  ((L[(o)]<<24)|(L[(o)+1]<<16)|(L[(o)+2]<<8)|L[(o)+3])
                fprintf(stderr, "[pm-claim] wid=%u spu=%u first=%d claims=%u stages=%u%s halt_pc=0x%05X "
                        "nextLoad=0x%X loadJob=0x%X runJob=0x%X storeJob=0x%X q13F0=%08X q14E0=%08X\n",
                        wid, spu_num, first, claims, stages,
                        (claims && !stages) ? " <-- CLAIM EATEN" : "",
                        ctx->pc & SPU_LS_MASK,
                        W(0x12C0), W(0x12D0), W(0x12E0), W(0x12F0),
                        W(0x13F0), W(0x14E0));
                fflush(stderr);
                #undef W
            }
        }
    }

    if (pipe_this) {
        /* PER-WID cap: a shared cap gets exhausted by idle wid=0/1 before the
         * jobmanager (wid=12) ever runs in a long/good run. Give each wid its
         * own budget so wid=12's pipeline is always captured. */
        static SPU_THREAD_LOCAL unsigned long s_pn_wid[32] = {0}, s_act_wid[32] = {0};
        unsigned long s_pn = ++s_pn_wid[wid & 31];
        const uint8_t* L = ctx->ls;
        #define HW(o) ((L[(o)+2]<<8)|L[(o)+3])
        #define W(o)  ((L[(o)]<<24)|(L[(o)+1]<<16)|(L[(o)+2]<<8)|L[(o)+3])
        /* Idle runs (state 0, no job numbers) only for the first few; any run that
         * shows pipeline activity is logged up to 3000 per wid, however late. */
        int active = st_before != 0 || HW(0x12A0) != 0 || W(0x12C0) != 0 ||
                     W(0x12D0) != 0xFFFFFFFFu || W(0x12E0) != 0xFFFFFFFFu;
        /* SPURS_PM_PIPE_WID=<n>: also sample every 64th run of that wid (idle or not). */
        static long s_pw = -2; if (s_pw == -2) { const char* e = getenv("SPURS_PM_PIPE_WID"); s_pw = e ? atol(e) : -1; }
        int sampled = (s_pw >= 0 && (long)wid == s_pw && (s_pn % 64) == 0);
        if (s_pn < 8 || sampled || (active && ++s_act_wid[wid & 31] < 3000)) {
            fprintf(stderr, "[pm-pipe] wid=%u run#%lu first=%d halt_pc=0x%05X "
                    "loadState=%u->%u nextLoad=0x%X loadJob=0x%X runJob=0x%X storeJob=0x%X "
                    "lastStore=0x%X\n",
                    wid, s_pn, first, ctx->pc & SPU_LS_MASK,
                    st_before, HW(0x12A0), W(0x12C0), W(0x12D0), W(0x12E0),
                    W(0x12F0), W(0x1300));
            /* WHY-NO-JOB1 probe: AllocateJob (via TryChangeFreeToLoadJob) stores
             * its result at g_WwsJob_jobHeader(0x1420) + g_WwsJob_jobIndex(0x1410).
             * jobHeader low-hword: 0=none(PPU hasn't published)/1=exists/3=nop.
             * Also dump the joblist header in RAM (eaWorkLoad@0x13F0) to see the
             * PPU-published numJobs + claimed jobIndex -- decides whether the PM
             * stops because the header is unpublished (PPU-side) or because our
             * load path bails after claiming. */
            {
                extern uint8_t* vm_base;
                uint32_t eaWL = W(0x13F0);
                fprintf(stderr, "         jobIndex=0x%X jobHeader=0x%08X(cmd=%u) eaWL=0x%08X\n",
                        W(0x1410), W(0x1420), HW(0x1420), eaWL);
                /* Dump the joblist header region eaWL+0x00..0x9F so we can see the
                 * ticket/counter rows AND where the PPU actually wrote the job
                 * headers (ref AllocateJob reads eaWL+0x20; oracle layout puts
                 * jobHeaders at +0x80). If +0x20 is 0 but +0x80 has 0x00xx0001
                 * entries, our AllocateJob reads the WRONG offset. */
                if (vm_base && eaWL) {
                    for (uint32_t o = 0; o < 0xA0; o += 0x10) {
                        const uint8_t* r = vm_base + eaWL + o;
                        #define RW(i) ((r[i]<<24)|(r[i+1]<<16)|(r[i+2]<<8)|r[i+3])
                        fprintf(stderr, "         jl+%02X: %08X %08X %08X %08X\n",
                                o, RW(0), RW(4), RW(8), RW(12));
                        #undef RW
                    }
                }
            }
            fflush(stderr);
            #undef HW
            #undef W
        }
    }

    if (tracing) {
        __atomic_store_n(&g_pm_flow_ctx, NULL, __ATOMIC_RELEASE);
        static SPU_THREAD_LOCAL int s_dumps = 0;
        /* Only dump once the PPU is actually spinning on a job barrier
         * (g_barrier_sync_watch armed by the barrier probe): boot-time idle
         * runs also PUTLLC (queue-check protocol) and would burn the cap. */
        /* Only dump runs that actually had a JOB STAGED. A boot-time idle run
         * has empty loadCommands(0xC00)/bufferSetArray(0xDF0) and its trail says
         * nothing about the Load->Run gate -- with SPURS_PM_FLOW_ALL tracing
         * every run, the dump budget was being spent entirely on idle runs. */
        int has_job = 0;
        for (uint32_t _i = 0xDF0; _i < 0xE10 && !has_job; _i++)
            if (ctx->ls[_i]) has_job = 1;
        for (uint32_t _i = 0xC00; _i < 0xC60 && !has_job; _i++)
            if (ctx->ls[_i]) has_job = 1;
        if (has_job && s_dumps < 6) {
            s_dumps++;
            unsigned n = g_pm_flow_n; if (n > 8192) n = 8192;
            fprintf(stderr, "[pm-flow] dump#%d wid=%u %u transfers (sync-PUTLLC delta %u):",
                    s_dumps, wid, n, g_spu_putllc_sync_hit - putllc_before);
            uint32_t last = 0xFFFFFFFFu; unsigned rep = 0;
            for (unsigned i = 0; i < n; i++) {
                uint32_t pc = g_pm_flow_buf[i] & SPU_LS_MASK;
                if (pc == last) { rep++; continue; }
                if (rep) { fprintf(stderr, "*%u", rep + 1); rep = 0; }
                fprintf(stderr, "%s%05X", (i % 16) ? " " : "\n  ", pc);
                last = pc;
            }
            if (rep) fprintf(stderr, "*%u", rep + 1);
            fprintf(stderr, "\n");
            /* Pipeline LS state right after this traced run: same regions the
             * hung-run watchdog dumps -- shows what the 5 type-1 passes built
             * (records/ring) and why the type-2 loads never dispatch. */
            { static const struct { uint32_t a, len; const char* tag; } R[] = {
                  { 0xC00, 0x60, "cmds" }, { 0xDF0, 0x40, "rec" },
                  { 0xEB0, 0x60, "ring" }, { 0x11B0, 0x20, "mark" },
                  { 0x1270, 0x20, "busy" }, { 0x12A0, 0x10, "st12A0" },
                  { 0x12C0, 0x40, "state" } };
              for (unsigned r = 0; r < sizeof R / sizeof R[0]; r++) {
                  fprintf(stderr, "[pm-ls] %-6s 0x%04X:", R[r].tag, R[r].a);
                  for (uint32_t o = 0; o < R[r].len; o += 4) {
                      if (o && (o & 15) == 0) fprintf(stderr, " |");
                      fprintf(stderr, " %02X%02X%02X%02X",
                              ctx->ls[R[r].a+o], ctx->ls[R[r].a+o+1],
                              ctx->ls[R[r].a+o+2], ctx->ls[R[r].a+o+3]);
                  }
                  fprintf(stderr, "\n");
              } }
            fflush(stderr);
        }
    }

    if (log_this) {
        fprintf(stderr, "[spurs-pm] END wid=%u status=0x%X pc=0x%05X polls=%u exited=%u\n",
                wid, ctx->status, ctx->pc, g_spurs_pm_polls, g_spurs_pm_exited);
        fflush(stderr);
    }

    int result = g_spurs_pm_exited ? 0 : -2;
    policy_slot_release(resident);
    return result;
}

/* ---------------------------------------------------------------------------
 * Real taskset-policy probe (Option 2, A/B vs spurs_pm_build_context).
 *
 * Runs the REAL lifted SPURS taskset policy (tsp_spu_func_00000A00, extracted
 * from libsre vaddr 0x23680) far enough to build the SpursTasksetContext at LS
 * 0x2700, then copies that region out for comparison against the C reimpl. The
 * policy's raw bytes (its data tables at LS 0x27B0-0x2840, read via lqr) come
 * from the embedded array so there is no dependency on libsre's guest load
 * address. The policy will eventually `bi` to the selected task's entry (an
 * address in the task image's LS range, unregistered under image id 100) and
 * halt as branch-to-0 -- by then LS 0x2700 holds the authentic context.
 *
 * image_id 100 is reserved for the policy's own indirect-branch table (the
 * lifted policy registers its 130 funcs there via tsp_spu_recomp_register()
 * under spu_begin_image(100)).  Returns the SPU status at halt.
 * ------------------------------------------------------------------------- */
extern const unsigned char g_taskset_policy_bytes[];
extern const unsigned g_taskset_policy_size;
extern void tsp_spu_func_00000A00(spu_context*);

/* pm-flow hang watchdog: dump the traced run's partial transfer trail + live
 * ctx state 5s after arm if the run hasn't returned (g_pm_flow_ctx still set).
 * Reads of the running ctx are racy by design -- diagnostic only. */
#ifdef _WIN32
#include <windows.h>
static DWORD WINAPI spurs_pm_flow_watchdog(LPVOID p)
{
    (void)p;
    extern uint32_t g_pm_flow_buf[]; extern volatile unsigned g_pm_flow_n;
    extern void* volatile g_pm_flow_ctx;
    Sleep(5000);
    spu_context* ctx = (spu_context*)g_pm_flow_ctx;
    if (!ctx) { fprintf(stderr, "[pm-flow] watchdog: run returned (no hang)\n"); return 0; }
    unsigned n = g_pm_flow_n; if (n > 8192) n = 8192;
    fprintf(stderr, "[pm-flow] HUNG-RUN PARTIAL: %u transfers, live pc=0x%05X "
            "LS12A0=%02X%02X LS12C0=%02X%02X LS12F0=%02X%02X%02X%02X LS1300=%02X%02X:",
            n, ctx->pc & SPU_LS_MASK,
            ctx->ls[0x12A0], ctx->ls[0x12A1],
            ctx->ls[0x12C0], ctx->ls[0x12C1],
            ctx->ls[0x12F0], ctx->ls[0x12F1], ctx->ls[0x12F2], ctx->ls[0x12F3],
            ctx->ls[0x1300], ctx->ls[0x1301]);
    /* Pipeline-state hex dump: batch commands (0xC00), per-slot records (0xDF0),
     * staging ring (0xEB0), FFFF markers (0x11B0), scheduler vars (0x1270-0x1340),
     * queue caches (0x13B0-0x1520). One shot -- everything the state-2 cycle's
     * stage decision reads, so the divergent condition can be hand-simulated. */
    { static const struct { uint32_t a, len; const char* tag; } R[] = {
          { 0xC00, 0x60, "cmds" }, { 0xDF0, 0x40, "rec" }, { 0xEB0, 0x40, "ring" },
          { 0x11B0, 0x20, "mark" }, { 0x1270, 0x40, "sched" }, { 0x12C0, 0x50, "state" },
          { 0x13B0, 0x30, "qvar" }, { 0x14A0, 0x30, "qcache" }, { 0x1440, 0x40, "batchrec" } };
      for (unsigned r = 0; r < sizeof R / sizeof R[0]; r++) {
          fprintf(stderr, "\n[pm-ls] %-8s 0x%04X:", R[r].tag, R[r].a);
          for (uint32_t o = 0; o < R[r].len; o += 4) {
              if (o && (o & 15) == 0) fprintf(stderr, " |");
              fprintf(stderr, " %02X%02X%02X%02X",
                      ctx->ls[R[r].a+o], ctx->ls[R[r].a+o+1],
                      ctx->ls[R[r].a+o+2], ctx->ls[R[r].a+o+3]);
          }
      }
      fprintf(stderr, "\n"); }
    uint32_t last = 0xFFFFFFFFu; unsigned rep = 0; unsigned col = 0;
    for (unsigned i = 0; i < n; i++) {
        uint32_t pc = g_pm_flow_buf[i] & SPU_LS_MASK;
        if (pc == last) { rep++; continue; }
        if (rep) { fprintf(stderr, "*%u", rep + 1); rep = 0; }
        fprintf(stderr, "%s%05X", (col++ % 16) ? " " : "\n  ", pc);
        last = pc;
    }
    if (rep) fprintf(stderr, "*%u", rep + 1);
    fprintf(stderr, "\n"); fflush(stderr);
    return 0;
}
void spurs_pm_flow_watchdog_start(void)
{
    HANDLE h = CreateThread(NULL, 0, spurs_pm_flow_watchdog, NULL, 0, NULL);
    if (h) CloseHandle(h);
}
#else
void spurs_pm_flow_watchdog_start(void) {}
#endif

int spurs_run_taskset_policy_probe(uint32_t taskset_ea, uint32_t taskid,
                                   uint32_t spurs_ea, uint64_t wkl_data,
                                   uint32_t wid, uint8_t* out_2700, uint32_t out_len)
{
    spu_context* ctx = (spu_context*)calloc(1, sizeof(spu_context));
    if (!ctx) return -1;
    spu_context_init(ctx, 0);
    ctx->image_id    = 100;      /* policy's own indirect-branch table */
    ctx->policy_mode = 1;

    /* Policy module image at its load base LS 0xA00 (code + data tables). */
    /* size 0 = this port linked the default stub blob (spurs_policy_blob_weak.c),
     * i.e. it has no lifted taskset policy image -- don't run an empty LS. */
    if (!g_taskset_policy_size ||
        g_taskset_policy_size > SPU_LS_SIZE - 0xA00) { free(ctx); return -1; }
    memcpy(ctx->ls + 0xA00, g_taskset_policy_bytes, g_taskset_policy_size);

    /* SpursKernelContext at LS 0x100 (same layout as spu_run_policy_module). */
    uint8_t* ls = ctx->ls;
#define KBE32(off, v) do { uint32_t _v = (v); ls[(off)] = (uint8_t)(_v >> 24); \
        ls[(off)+1] = (uint8_t)(_v >> 16); ls[(off)+2] = (uint8_t)(_v >> 8);   \
        ls[(off)+3] = (uint8_t)_v; } while (0)
    KBE32(0x1C0, 0);                    KBE32(0x1C4, spurs_ea);
    KBE32(0x1C8, 0);                    KBE32(0x1CC, 8);
    KBE32(0x1D0, 0);                    KBE32(0x1D4, 0xA00);
    KBE32(0x1D8, wid);                  KBE32(0x1DC, wid);
    KBE32(0x1E0, SPURS_PM_EXIT_TO_KERNEL_LS);
    KBE32(0x1E4, SPURS_PM_SELECT_WORKLOAD_LS);
    /* taskset EA also planted at the SpursKernelContext taskset slot the policy
     * reads for its taskset DMA (mirrors the cri-r4 injection at LS 0x27B8). */
#undef KBE32

    ctx->gpr[0]._u32[0] = SPURS_PM_EXIT_TO_KERNEL_LS;
    ctx->gpr[1]._u32[0] = 0x3FFB0;
    ctx->gpr[3]._u32[0] = 0x100;
    ctx->gpr[4]._u32[0] = (uint32_t)(wkl_data >> 32);
    ctx->gpr[4]._u32[1] = (uint32_t)wkl_data;
    ctx->gpr[5]._u32[0] = 0;

    g_spurs_pm_polls = 0; g_spurs_pm_exited = 0;

    fprintf(stderr, "[real-pm] RUN taskset=0x%08X task=%u spurs=0x%08X wkl=0x%llX wid=%u\n",
            taskset_ea, taskid, spurs_ea, (unsigned long long)wkl_data, wid);
    fflush(stderr);

    spu_run_with_halt(tsp_spu_func_00000A00, ctx);

    fprintf(stderr, "[real-pm] END status=0x%X pc=0x%05X polls=%u exited=%u\n",
            ctx->status, ctx->pc, g_spurs_pm_polls, g_spurs_pm_exited);
    fflush(stderr);

    if (out_2700 && out_len) {
        uint32_t n = out_len;
        if ((uint32_t)0x2700 + n > SPU_LS_SIZE) n = SPU_LS_SIZE - 0x2700;
        memcpy(out_2700, ctx->ls + 0x2700, n);
    }
    int st = (int)ctx->status;
    spu_coh_unregister(ctx);
    spu_mfc_release(ctx);
    free(ctx);
    return st;
}
