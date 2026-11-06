/*
 * ps3recomp - cellSpurs HLE implementation
 *
 * Provides the SPURS management API so games can call SPU task/workload
 * functions without crashing.  Full SPU execution requires recompiling
 * SPU programs; this layer provides the scheduling and management APIs.
 *
 * Tasks and workloads are tracked.  If a game provides PPU fallback
 * callbacks, those can be invoked through the task submission path.
 */

#include "cellSpurs.h"
#include "../../runtime/platform/win32_compat.h"
#include "../../runtime/ps3_log.h"
#include "spu_workload.h"   /* SPU image -> lifted-entry dispatch (runtime/spu) */
#include "ps3emu/ps3_ftrace.h"
#include "spurs_taskset.h"  /* REAL BE CellSpursTaskset layout builders (fork Option-B) */
#include "../../runtime/ppu/ppu_memory.h"   /* vm_base (guest mem) */
#include <stdio.h>
#include <stdlib.h>   /* getenv -- an implicit decl returns int, truncating the pointer */
#include <string.h>
#include <stdint.h>
#include "ps3emu/log_first.h"

/* Bridge the real (BE) taskset EA + selected taskId from CreateTask to the image-22
 * SPU dispatch (spu_workload.c), so spurs_pm_build_context can build the leaf's
 * SpursTasksetContext from the real taskset. Set right before dispatch_async (the PPU
 * create path is sequential here). Gated by YDKJ_REAL_TASKSET in the dispatch. */
uint32_t g_ydkj_real_taskset_ea = 0;
uint32_t g_ydkj_real_taskid     = 0;
uint32_t g_ydkj_real_spurs_ea   = 0;   /* real CellSpurs instance EA (for the taskset-policy handoff) */

/* Generic HLE adapter passes GUEST addresses; translate pointer args. CellSpurs
 * is treated opaquely by the game (passed back as a handle), so translating the
 * pointer is enough here. */
#define GUEST_PTR(p, T) ((T)((p) ? (void*)(vm_base + (uint32_t)(uintptr_t)(p)) : (void*)0))

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

/* ---------------------------------------------------------------------------
 * Internal workload tracking
 * -----------------------------------------------------------------------*/

typedef struct {
    int         in_use;
    const void* pm;            /* policy-module image EA (guest) */
    u32         sizePm;
    u64         data;          /* workload data (e.g. joblist EA) */
    u32         spurs_ea;      /* owning CellSpurs instance EA */
    u8          priority[CELL_SPURS_MAX_SPU];
    u32         minContention;
    u32         maxContention;
    u32         readyCount;
    int         started;       /* its SPU program has been dispatched        */
} SpursWorkload;

typedef struct {
    int         in_use;
    u32         id;
    int         active;
    int         completed;
    s32         exitCode;
    void*       entryPoint;
    u64         argA;
} SpursTask;

/* Global workload table (per-SPURS instance in a real system, simplified) */
/* CellSpurs2 (cellSpursInitializeWithAttribute2) has 32 workload slots; the HLE
 * tables cover both sets, AddWorkload only hands out 16..31 on a CellSpurs2. */
#define SPURS_MAXW 32
static SpursWorkload s_workloads[SPURS_MAXW];
static SpursTask     s_tasks[CELL_SPURS_MAX_TASK];
static u32           s_next_task_id = 0;

/* ---------------------------------------------------------------------------
 * Event flag sync side table
 *
 * The CellSpursEventFlag itself is 128 bytes of GAME-owned guest memory in
 * the REAL big-endian kernel layout (see EF_* offsets below) — the SPU side
 * (lifted task-library code) reads/writes it with DMA + atomics, so the PPU
 * HLE must operate on the same guest bytes, never a host struct. This side
 * table (keyed by guest EA) only adds the host mutex + condvar used to block
 * and wake PPU waiters.
 * -----------------------------------------------------------------------*/
#define MAX_EVENT_FLAGS 64

typedef struct {
    uint32_t            ea;      /* guest EA of the 128-byte flag */
#ifdef _WIN32
    CRITICAL_SECTION    cs;
    CONDITION_VARIABLE  cv;
#else
    pthread_mutex_t     mtx;
    pthread_cond_t      cond;
#endif
    int                 initialized;
} EventFlagSync;

static EventFlagSync s_ef_sync[MAX_EVENT_FLAGS];

/* PPU threads inside cellSpursEventFlagWait. The SPU sets a flag with
 * GETLLAR/PUTLLC on the guest struct, which never touched the waiter's
 * condition variable, so every wait ran out its 2 ms poll: FMOD's mixer waits
 * on its SPU task once or twice per 5.3 ms audio block, fell behind, and
 * GH3's songs played ~60% silence (heard as static). */
static volatile long s_ef_ppu_waiters;
static void ef_line_committed(uint32_t line)
{
    if (!s_ef_ppu_waiters) return;
    for (int i = 0; i < MAX_EVENT_FLAGS; i++) {
        EventFlagSync* s = &s_ef_sync[i];
        if (!s->initialized || (s->ea & ~127u) != line) continue;
        /* Take the waiter's lock so the wake cannot fall between its check of
         * the flag and its sleep. Caller holds no lock-line lock. */
#ifdef _WIN32
        EnterCriticalSection(&s->cs); WakeAllConditionVariable(&s->cv); LeaveCriticalSection(&s->cs);
#else
        pthread_mutex_lock(&s->mtx); pthread_cond_broadcast(&s->cond); pthread_mutex_unlock(&s->mtx);
#endif
    }
}

static EventFlagSync* ef_sync_find(uint32_t ea)
{
    for (int i = 0; i < MAX_EVENT_FLAGS; i++) {
        if (s_ef_sync[i].initialized && s_ef_sync[i].ea == ea)
            return &s_ef_sync[i];
    }
    return NULL;
}

static EventFlagSync* ef_sync_alloc(uint32_t ea)
{
    for (int i = 0; i < MAX_EVENT_FLAGS; i++) {
        if (!s_ef_sync[i].initialized) {
            s_ef_sync[i].ea = ea;
            { extern void (*g_spu_line_commit_hook)(uint32_t);
              g_spu_line_commit_hook = ef_line_committed; }
#ifdef _WIN32
            InitializeCriticalSection(&s_ef_sync[i].cs);
            InitializeConditionVariable(&s_ef_sync[i].cv);
#else
            pthread_mutex_init(&s_ef_sync[i].mtx, NULL);
            pthread_cond_init(&s_ef_sync[i].cond, NULL);
#endif
            s_ef_sync[i].initialized = 1;
            return &s_ef_sync[i];
        }
    }
    return NULL;
}

/* Lenient lookup: a flag touched before we saw its Initialize (alternate init
 * paths, e.g. taskset2) still gets a sync slot instead of an error. */
static EventFlagSync* ef_sync_get(uint32_t ea)
{
    EventFlagSync* s = ef_sync_find(ea);
    if (!s) {
        s = ef_sync_alloc(ea);
        if (s)
            fprintf(stderr, "[cellSpurs] event flag 0x%08X used before Initialize "
                            "-- sync slot auto-allocated\n", ea);
    }
    return s;
}

static void ef_sync_free(EventFlagSync* sync)
{
    if (!sync) return;
#ifdef _WIN32
    DeleteCriticalSection(&sync->cs);
    /* CONDITION_VARIABLE has no destroy on Windows */
#else
    pthread_mutex_destroy(&sync->mtx);
    pthread_cond_destroy(&sync->cond);
#endif
    sync->ea = 0;
    sync->initialized = 0;
}

static inline void ef_lock(EventFlagSync* s)
{
#ifdef _WIN32
    EnterCriticalSection(&s->cs);
#else
    pthread_mutex_lock(&s->mtx);
#endif
}

static inline void ef_unlock(EventFlagSync* s)
{
#ifdef _WIN32
    LeaveCriticalSection(&s->cs);
#else
    pthread_mutex_unlock(&s->mtx);
#endif
}

/* Returns 1 if signaled, 0 if it timed out after `ms`. */
static inline int ef_wait_timed(EventFlagSync* s, unsigned ms)
{
#ifdef _WIN32
    return SleepConditionVariableCS(&s->cv, &s->cs, ms) ? 1 : 0;
#else
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += ms / 1000; ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    return pthread_cond_timedwait(&s->cond, &s->mtx, &ts) == 0 ? 1 : 0;
#endif
}

static inline void ef_broadcast(EventFlagSync* s)
{
#ifdef _WIN32
    WakeAllConditionVariable(&s->cv);
#else
    pthread_cond_broadcast(&s->cond);
#endif
}

/* ---------------------------------------------------------------------------
 * REAL CellSpursEventFlag guest layout (128 bytes, big-endian; RPCS3
 * cellSpurs.h is the reference). The SPU task library manipulates this exact
 * layout with GETLLAR/PUTLLC, so every PPU-side access goes through vm_*.
 * -----------------------------------------------------------------------*/
#define EF_EVENTS            0x00u  /* be u16: the event bits */
#define EF_SPU_PENDING_RECV  0x02u  /* be u16: slot bits with met conditions */
#define EF_PPU_WAIT_MASK     0x04u  /* be u16: blocked PPU thread's mask */
#define EF_PPU_WAIT_SLOTMODE 0x06u  /* u8: hi4 = wait slot, lo4 = wait mode */
#define EF_PPU_PENDING_RECV  0x07u  /* u8: 1 when the PPU waiter's cond met */
#define EF_SPU_USED_SLOTS    0x08u  /* be u16: wait slots in use (bit 0x8000>>s) */
#define EF_SPU_WAIT_MODE     0x0Au  /* be u16: per-slot mode (1 = AND) */
#define EF_SPU_PORT          0x0Cu  /* u8 */
#define EF_IS_IWL            0x0Du  /* u8: addr is a wkl instead of taskset */
#define EF_DIRECTION         0x0Eu  /* u8: CELL_SPURS_EVENT_FLAG_* direction */
#define EF_CLEAR_MODE        0x0Fu  /* u8: 0 = AUTO, 1 = MANUAL */
#define EF_SPU_WAIT_MASK_ARR 0x10u  /* be u16 [16]: per-slot wait masks */
#define EF_PENDING_RECV_EVT  0x30u  /* be u16 [16]: events handed to waiters */
#define EF_WAITING_TASK_ID   0x50u  /* u8 [16]: waiting task ids */
#define EF_WAITING_WKL_ID    0x60u  /* u8 [16]: waiting workload ids */
#define EF_ADDR              0x70u  /* be u64: taskset EA (isIwl=0) */
#define EF_EVENT_PORT_ID     0x78u  /* be u32 */
#define EF_EVENT_QUEUE_ID    0x7Cu  /* be u32 */
#define EF_GUEST_SIZE        0x80u

/* Layer-2 hook: wake an SPU task that registered a wait slot and slept via
 * the taskset syscall. Implemented in runtime/spu (task park/wake); the
 * event-flag core only reports WHO must wake. */
extern void spu_taskset_signal_task(uint32_t taskset_ea, uint32_t taskId);

/* Core Set protocol on the guest struct (sync lock held). Mirrors the real
 * kernel/RPCS3 semantics: satisfy registered SPU-task wait slots (slot s uses
 * bit 0x8000>>s in the used/pending/mode words), hand each its received
 * events, auto-clear consumed bits, then signal the tasks. PPU waiters are
 * poll-based here (ef_wait_timed ticks re-check EF_EVENTS), so no PPU
 * wait-slot registration is needed — and deliberately NOT written, so the
 * SPU-side lifted Set code takes its "no PPU waiter" path and simply ORs
 * bits that our poll then observes. */
/* The flag struct is ONE 128-byte line the SPU task library updates with
 * GETLLAR/PUTLLC. A PPU read-modify-write of it must hold the lock-line lock
 * for its whole span, or a PUTLLC landing between the read and the write is
 * silently undone: an SPU wait registered there is lost, nothing ever signals
 * the task, and its PPU partner blocks for good (GH3: FMOD's mixer deadlocked
 * at boot in 1 run of ~4). Raw accesses inside: vm_write16 would take the
 * same (non-recursive) lock. The end notifies reserving SPUs, under the lock
 * as spu_coh_notify_write expects. */
extern void spu_lockline_lock(void);
extern void spu_lockline_unlock(void);
extern int  spu_coh_is_reserved(uint32_t);
extern void spu_coh_notify_write(uint32_t);
static inline u16 ef_r16(uint32_t a) { const uint8_t* p = vm_base + a; return (u16)((p[0] << 8) | p[1]); }
static inline void ef_w16(uint32_t a, u16 v) { uint8_t* p = vm_base + a; p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; }
static inline void ef_line_end(uint32_t ea)
{
    if (spu_coh_is_reserved(ea & ~127u)) spu_coh_notify_write(ea & ~127u);
    spu_lockline_unlock();
}

static void spurs_ef_set_locked(uint32_t ea, u16 bits)
{
    spu_lockline_lock();
    u16 events        = ef_r16(ea + EF_EVENTS);
    u16 used          = (u16)(ef_r16(ea + EF_SPU_USED_SLOTS) &
                              ~ef_r16(ea + EF_SPU_PENDING_RECV));
    u16 waitmode      = ef_r16(ea + EF_SPU_WAIT_MODE);
    u16 eventsToClear = 0;
    u16 pendingRecv   = 0;

    for (int s = 0; s < CELL_SPURS_EVENT_FLAG_MAX_WAIT_SLOTS; s++) {
        u16 bit = (u16)(0x8000u >> s);
        if (!(used & bit)) continue;
        u16 mask = ef_r16(ea + EF_SPU_WAIT_MASK_ARR + 2u * s);
        u16 rel  = (u16)((events | bits) & mask);
        int mode_and = (waitmode & bit) != 0;
        if ((mask & ~rel) == 0 || (!mode_and && rel != 0)) {
            eventsToClear |= rel;
            pendingRecv   |= bit;
            ef_w16(ea + EF_PENDING_RECV_EVT + 2u * s, rel);
        }
    }

    events = (u16)(events | bits);
    if (pendingRecv) {
        ef_w16(ea + EF_SPU_PENDING_RECV,
               (u16)(ef_r16(ea + EF_SPU_PENDING_RECV) | pendingRecv));
        if (vm_read8(ea + EF_CLEAR_MODE) == CELL_SPURS_EVENT_FLAG_CLEAR_AUTO)
            events = (u16)(events & ~eventsToClear);
    }
    ef_w16(ea + EF_EVENTS, events);
    ef_line_end(ea);   /* before signalling: signal_task takes its own locks */

    for (int s = 0; s < CELL_SPURS_EVENT_FLAG_MAX_WAIT_SLOTS; s++) {
        if (pendingRecv & (0x8000u >> s)) {
            uint32_t taskset_ea = (uint32_t)vm_read64(ea + EF_ADDR);
            u32 taskId = vm_read8(ea + EF_WAITING_TASK_ID + s);
            if (ps3_log_verbose())
                fprintf(stderr, "[cellSpurs] EventFlagSet 0x%08X satisfies SPU task "
                        "slot %d (taskset=0x%08X task=%u)\n", ea, s, taskset_ea, taskId);
            spu_taskset_signal_task(taskset_ea, taskId);
        }
    }

    /* SPURS_EF_WAKE_PARKED: a task that parked in WAIT_SIGNAL WITHOUT registering
     * a wait slot in this flag is unreachable by the loop above, which only
     * signals slots present in `used`. YDKJ CRI tasks do exactly that, so a
     * PPU->SPU Set (direction 2) delivered nothing and they slept forever. Same
     * failure class canersaka documented for SPURS queues in Yakuza Dead Souls:
     * the HLE woke only its host condvar, which lifted SPU tasks never wait on.
     * Signalling a task that was not waiting on THIS flag is safe -- signals are
     * latched in guest state and every wait loop re-checks its own predicate. */
    /* Record the bits this Set owes the task waiting on (flag - 0x80),
     * unconditionally. Gating this on EF_DIRECTION made delivery depend on
     * Initialize having already run, and the CRI flags are Set before that --
     * so the post silently never happened. An unclaimed post is harmless: it is
     * only ever consumed by a task that names this exact object in its wait. */
    if (ea > 0x80u) {
        uint32_t obj = ea - 0x80u;
        /* Write NOW *and* record for consume time. The two orderings are both
         * real and neither alone is enough: if the task has not read yet the
         * immediate write is what it sees, and if it is parked (or took the
         * latched-wake path and never parked at all) the WAIT_SIGNAL handler
         * replays it just before returning. Writing the same bits twice is
         * idempotent, so covering both costs nothing and removes the race that
         * made this fire in only 1 of 4 runs. */
        { static int s_od = -1;
          if (s_od < 0) s_od = getenv("SPURS_EF_OBJ_DELIVER") ? 1 : 0;
          if (s_od) {
              vm_write16(obj + EF_EVENTS, bits);
              for (uint32_t s = 0; s < 16; s++)
                  vm_write16(obj + EF_PENDING_RECV_EVT + 2u * s, bits);
          } }
        { extern void spu_ef_bits_post(uint32_t, uint16_t);
          spu_ef_bits_post(obj, bits); }
    }
    if (!pendingRecv) {
        static int s_wp = -1;
        if (s_wp < 0) s_wp = getenv("SPURS_EF_WAKE_PARKED") ? 1 : 0;

        if (s_wp && vm_read8(ea + EF_DIRECTION) == 2) {
            uint32_t taskset_ea = (uint32_t)vm_read64(ea + EF_ADDR);
            /* The task waiting for THIS flag parked on object (flag - 0x80):
             * YDKJ CRI obj 0x006B4500/4780/4A00 <-> flag 0x006B4580/4800/4A80.
             * Fall back to any parked task of the taskset if none matches. */
            extern int spu_taskset_signal_parked_obj(uint32_t, uint32_t);

            int woke = spu_taskset_signal_parked_obj(taskset_ea, ea - 0x80u);
            if (!woke) woke = spu_taskset_signal_parked_obj(taskset_ea, 0);
            if (!woke) {   /* nobody parked yet -- do not lose the wakeup */
                extern void spu_taskset_latch_wake(uint32_t);
                spu_taskset_latch_wake(taskset_ea);
            }
            static int _n = 0;
            if (woke && PS3_LOG_FIRST(_n, 12))
                fprintf(stderr, "[cellSpurs] EventFlagSet 0x%08X woke %d parked "
                                "task(s) on taskset 0x%08X (no wait slot)\n",
                        ea, woke, taskset_ea);
        }
    }
}

/* SPU-side entry (Layer 2): a task's flag Set arriving via the taskset
 * syscall or a runtime bridge. Same protocol, takes the lock itself. */
void spurs_ef_set_from_spu(uint32_t flag_ea, uint16_t bits)
{
    EventFlagSync* sync = ef_sync_get(flag_ea);
    if (!sync) return;
    ef_lock(sync);
    spurs_ef_set_locked(flag_ea, (u16)bits);
    ef_broadcast(sync);
    ef_unlock(sync);
}

/* =========================================================================
 * SPURS core
 *
 * The CellSpurs instance (0x2000 bytes of GAME-owned guest memory) is real
 * shared state: the game's engine pokes it with INLINED atomics (readyCount
 * stores, signal bits) and the policy modules DMA it from the SPU side. So
 * it must hold the REAL big-endian kernel layout — never a host struct.
 * Verified offsets (RPCS3 cellSpurs.h contract):
 *   +0x00 wklReadyCount1[16] (u8/wid)   +0x80 wklState1[16] (u8: 2=runnable)
 *   +0x10 wklIdleSpuCount[16]           +0x90 wklStatus1[16]
 *   +0x20 wklCurrentContention[16]      +0xA0 wklEvent1[16]
 *   +0x40 wklMinContention[16]          +0xB0 wklEnabled (be u32, bit 31-wid)
 *   +0x50 wklMaxContention[16]          +0xBD sysSrvMsgUpdateWorkload (u8)
 *   +0x60 wklFlag (be u64)              +0xB00 wklInfo1[16] (32B each:
 *   +0x70 wklSignal1 (be u16)                  addr u64, arg u64, size u32,
 *   +0x76 nSpus (u8)                           uniqueId u8, prio[8] @+0x18)
 * Our own bookkeeping lives in a host side-table keyed by instance EA.
 * A host "kernel" thread per instance polls readyCount/signal and runs the
 * workload's policy module (spurs_policy.c) — the virtual SPU.
 * =====================================================================*/
enum {
    SPURS_WKL_READY1   = 0x00,
    SPURS_WKL_IDLE2    = 0x10,
    SPURS_WKL_CURCONT  = 0x20,
    SPURS_WKL_MINCONT  = 0x40,
    SPURS_WKL_MAXCONT  = 0x50,
    SPURS_WKL_FLAG     = 0x60,
    SPURS_WKL_SIGNAL1  = 0x70,
    SPURS_NSPUS        = 0x76,
    SPURS_WKL_STATE1   = 0x80,
    SPURS_WKL_ENABLED  = 0xB0,
    SPURS_SYSSRV_MSG   = 0xBD,
    SPURS_WKL_INFO1    = 0xB00,
    SPURS_WKL_INFO_SZ  = 0x20,
    /* CELL_SPURS_SIZE = 4096 (SDK cell/spurs/types.h). The 8192-byte variant
     * is CellSpurs2 (cellSpursInitialize*2* NIDs) which LBP does not use —
     * clearing 0x2000 here overran the game's 4KB heap block and corrupted
     * the allocator (abort in the job pump's first object destruction). */
    SPURS_INST_SIZE    = 0x1000,
};

/* Second workload set (wid 16..31, CellSpurs2 only; RPCS3 cellSpurs.h layout):
 * readyCount2 shares the 0x10 array, state2 is 0xC0, signal2 the be u16 at
 * 0x78, wklInfo2 starts at 0x1000, and current/max contention are packed
 * 4-bit fields (low nibble = wid 0..15, high nibble = wid 16..31). GoW:A
 * registers 18 workloads (EDGE2 and RsxSync are #16/#17). */
static inline u32 spurs_off_ready(u32 wid)  { return wid < 16 ? SPURS_WKL_READY1 + wid : SPURS_WKL_IDLE2 + (wid - 16); }
static inline u32 spurs_off_state(u32 wid)  { return wid < 16 ? SPURS_WKL_STATE1 + wid : 0xC0u + (wid - 16); }
static inline u32 spurs_off_info(u32 wid)   { return wid < 16 ? SPURS_WKL_INFO1 + wid * SPURS_WKL_INFO_SZ
                                                              : 0x1000u + (wid - 16) * SPURS_WKL_INFO_SZ; }
static inline u32 spurs_off_signal(u32 wid) { return wid < 16 ? SPURS_WKL_SIGNAL1 : 0x78u; }
static inline u32 spurs_sig_bit(u32 wid)    { return 0x8000u >> (wid & 15); }
static inline u8  spurs_nib_get(u32 ea, u32 arr, u32 wid)
{ u8 v = *(vm_base + ea + arr + (wid & 15)); return wid < 16 ? (u8)(v & 0x0F) : (u8)(v >> 4); }
static inline void spurs_nib_set(u32 ea, u32 arr, u32 wid, u8 x)
{ u8* q = vm_base + ea + arr + (wid & 15);
  *q = wid < 16 ? (u8)((*q & 0xF0) | (x & 0x0F)) : (u8)((*q & 0x0F) | (u8)(x << 4)); }

#define MAX_SPURS_INST 4
static struct SpursInst {
    u32           ea;          /* 0 = free */
    u32           nspus;
    u32           is2;         /* CellSpurs2 (8 KB, 32 workloads) */
    char          prefix[16];
    volatile long kernel_live; /* poll thread started */
    int           nkern;       /* kernel threads (SPURS_KERN_THREADS) */
    int           kspu[8];     /* SPU played by kernel thread k (SPURS_KERN_SPUS) */
} s_inst[MAX_SPURS_INST];
/* A kernel thread plays one SPU of its instance: it honours each workload's
 * per-SPU priority (0 = never on that SPU). */
struct spurs_kern_arg { struct SpursInst* si; int spu; };

/* LLE libsre mode (set by the port): the real firmware library creates and owns
 * the SPURS instance, so the HLE never sees cellSpursInitialize*. The HLE
 * taskset/task family that stays on HLE still needs the instance to exist --
 * adopt it on first lookup, without the poll "kernel" thread (the real SPURS
 * kernel runs on the SPU threads) and without touching the instance memory. */
int g_spurs_lle = 0;

static struct SpursInst* spurs_inst_find(u32 ea)
{
    for (int i = 0; i < MAX_SPURS_INST; i++)
        if (__atomic_load_n(&s_inst[i].ea, __ATOMIC_ACQUIRE) == ea) return &s_inst[i];
    if (g_spurs_lle && ea) {
        for (int i = 0; i < MAX_SPURS_INST; i++)
            if (!s_inst[i].ea) {
                s_inst[i].ea = ea;
                s_inst[i].nspus = vm_base[ea + 0x76] ? vm_base[ea + 0x76] : 6; /* SPURS_NSPUS */
                s_inst[i].kernel_live = 1;       /* no HLE poll thread */
                fprintf(stderr, "[cellSpurs] LLE: adopted real SPURS instance 0x%08X (nSpus=%u)\n",
                        ea, s_inst[i].nspus);
                return &s_inst[i];
            }
    }
    return NULL;
}

static DWORD WINAPI spurs_kernel_thread(LPVOID p);
static volatile LONG s_wid_busy[SPURS_MAXW];   /* a kernel thread is running this wid's module */
static void spurs_wkl_pm_forget(u32 wid);      /* drop a wid's resolved policy module */
static void spurs_retire_workloads(u32 ea);    /* drop one instance's workload registrations */

/* A workload ID names a workload of ONE instance: the wid APIs reject an ID
 * that another instance registered (upstream v0.15 instance ownership). */
static int spurs_wid_owned(const CellSpurs* spurs, u32 wid)
{
    return wid < SPURS_MAXW && s_workloads[wid].in_use &&
           s_workloads[wid].spurs_ea == (u32)(uintptr_t)spurs;
}

static s32 spurs_initialize_common(u32 spurs_ea, u32 nspus, const char* prefix)
{
    struct SpursInst* si = spurs_inst_find(spurs_ea);
    if (!si) {
        for (int i = 0; i < MAX_SPURS_INST; i++)
            if (!s_inst[i].ea) { si = &s_inst[i]; break; }
    }
    if (!si) return CELL_SPURS_CORE_ERROR_NOMEM;

    si->ea    = spurs_ea;
    si->nspus = (nspus > 0 && nspus <= CELL_SPURS_MAX_SPU) ? nspus : 1;
    memset(si->prefix, 0, sizeof(si->prefix));
    if (prefix) memcpy(si->prefix, prefix, 15);

    /* Real BE instance: zero it, then the few live fields. (The global
     * workload table is NOT wiped here — the title may init several SPURS
     * instances before adding workloads to any of them.) */
    memset(vm_base + spurs_ea, 0, SPURS_INST_SIZE);
    *(vm_base + spurs_ea + SPURS_NSPUS) = (u8)si->nspus;
    vm_write64(spurs_ea + SPURS_WKL_FLAG, 0xFFFFFFFFFFFFFFFFull); /* no receiver */
    /* CellSpursWorkloadFlag {u64 unused0; u32 unused1; u32 flag} at 0x60: the live
     * word is +0xC; wklFlagReceiver (u8) is 0x77. Real init: flag = -1, receiver = 0xFF. */
    vm_write32(spurs_ea + SPURS_WKL_FLAG + 0xC, 0xFFFFFFFFu);
    *(vm_base + spurs_ea + 0x77) = 0xFF;

    if (!si->kernel_live) {
        si->kernel_live = 1;
        /* SPURS_KERN_THREADS=<n>|spus (default 1): host threads running this
         * instance's policy modules; "spus" = the instance's SPU count. One thread
         * serialises work that real SPUs run side by side. */
        int nk = 1;
        { const char* e = getenv("SPURS_KERN_THREADS");
          if (e) nk = !strcmp(e, "spus") ? (int)si->nspus : atoi(e);
          if (nk < 1) nk = 1; if (nk > 6) nk = 6; }
        for (int k = 0; k < 8; k++) si->kspu[k] = k;
        /* SPURS_KERN_SPUS=<a,b,...>: the SPU each kernel thread plays (overrides
         * SPURS_KERN_THREADS). GoW:A "0,1,5": SPU 5 is where BrBDblBufWrkld (prio 1,
         * 0 elsewhere) lives, so one thread owns it as on hardware. */
        { const char* e = getenv("SPURS_KERN_SPUS");
          if (e && *e) {
              int n = 0;
              while (*e && n < 6) {
                  int v = atoi(e);
                  si->kspu[n++] = (v >= 0 && v < CELL_SPURS_MAX_SPU) ? v : 0;
                  while (*e && *e != ',') e++;
                  if (*e == ',') e++;
              }
              if (n) nk = n;
          } }
        si->nkern = nk;
        for (int k = 0; k < nk; k++) {
            static struct spurs_kern_arg ka[MAX_SPURS_INST * 8];
            static int nka;
            struct spurs_kern_arg* a = &ka[nka++ % (MAX_SPURS_INST * 8)];
            a->si = si; a->spu = si->kspu[k];
            CreateThread(NULL, 1u << 20, spurs_kernel_thread, a, 0, NULL);
        }
    }
    printf("[cellSpurs] Initialize \"%s\" ea=0x%08X nSpus=%u (real BE instance + kernel poll)\n",
           si->prefix, spurs_ea, si->nspus);
    return CELL_OK;
}

s32 cellSpursInitialize(CellSpurs* spurs, s32 nSpus, s32 spuPriority,
                        s32 ppuPriority, u8 exitIfNoWork)
{
    (void)spuPriority; (void)ppuPriority; (void)exitIfNoWork;
    if (!spurs)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    return spurs_initialize_common((u32)(uintptr_t)spurs, (u32)nSpus, NULL);
}

s32 cellSpursInitializeWithAttribute(CellSpurs* spurs,
                                     const CellSpursAttribute* attr)
{
    if (!spurs || !attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    u32 spurs_ea = (u32)(uintptr_t)spurs;
    attr = GUEST_PTR(attr, const CellSpursAttribute*);
    return spurs_initialize_common(spurs_ea, attr->nSpus, (const char*)attr->prefix);
}

static void spurs_cancel_attached_queues(u32 spurs_ea);  /* defined with the queue table */

s32 cellSpursFinalize(CellSpurs* spurs)
{
    if (!spurs)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;

    /* After this, no job completion can ever reach the attached queues, so a
     * worker parked on one with an infinite timeout would sleep forever -- and
     * the shutdown that called us then blocks in sys_ppu_thread_join waiting
     * for that worker to notice it should quit. Tokyo Jungle wedges exactly
     * there: "terminate audio thread (6)" then join(6), with tid 6 in
     * event_queue_receive(q=3) and the title never rendering again. Releasing
     * the waiters lets the receive fail, which is the path its loop already
     * handles -- an error return leaves the loop and the thread exits.
     *
     * Done BEFORE the instance lookup: that lookup fails here, and a failed
     * handle is no reason to strand a thread. */
    spurs_cancel_attached_queues((u32)(uintptr_t)spurs);

    struct SpursInst* si = spurs_inst_find((u32)(uintptr_t)spurs);
    if (!si)
        return CELL_SPURS_CORE_ERROR_STAT;
    const u32 ea = si->ea;

    /* A policy module of this instance still running: BUSY, as upstream's
     * scheduler answers while a policy is admitted (the title retries). */
    for (u32 w = 0; w < SPURS_MAXW; w++)
        if (s_workloads[w].in_use && s_workloads[w].spurs_ea == ea && s_wid_busy[w])
            return CELL_SPURS_CORE_ERROR_BUSY;

    printf("[cellSpurs] Finalize(ea=0x%08X)\n", ea);
    __atomic_store_n(&si->ea, 0, __ATOMIC_RELEASE);  /* kernel threads see a dead instance and idle */
    /* Only THIS instance's workloads: wiping the whole table also dropped every
     * other live instance's registrations. Then retire its resident policy
     * contexts (waits for a module run that started before the store above). */
    spurs_retire_workloads(ea);
    spurs_policy_release_instance(ea);
    return CELL_OK;
}

s32 cellSpursAttributeInitialize(CellSpursAttribute* attr, s32 nSpus,
                                 s32 spuPriority, s32 ppuPriority,
                                 u8 exitIfNoWork)
{
    (void)ppuPriority; (void)exitIfNoWork;

    if (!attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;

    /* `attr` is a GUEST address (generic HLE adapter passes r3 raw). */
    attr = GUEST_PTR(attr, CellSpursAttribute*);
    memset(attr, 0, sizeof(CellSpursAttribute));
    attr->nSpus = (nSpus > 0 && nSpus <= CELL_SPURS_MAX_SPU)
                  ? (u32)nSpus : 1;

    for (int i = 0; i < CELL_SPURS_MAX_SPU; i++)
        attr->spuPriority[i] = spuPriority;

    printf("[cellSpurs] AttributeInitialize(nSpus=%d)\n", nSpus);
    return CELL_OK;
}

/* The SDK's cellSpursAttributeInitialize() is an inline wrapper over this
 * internal export (NID 0x95180230), whose arguments are (attr, revision,
 * sdkVersion, nSpus, spuPriority, ppuPriority, exitIfNoWork). Reading them with
 * the public five-argument prototype took the revision (2) as nSpus, the SDK
 * version as spuPriority and the real nSpus as ppuPriority. (GoW:A's
 * "Big" instance has 6 SPUs.) */
s32 _cellSpursAttributeInitialize(CellSpursAttribute* attr, u32 revision,
                                  u32 sdkVersion, s32 nSpus, s32 spuPriority,
                                  s32 ppuPriority, u8 exitIfNoWork)
{
    (void)revision;
    (void)sdkVersion;
    return cellSpursAttributeInitialize(attr, nSpus, spuPriority,
                                        ppuPriority, exitIfNoWork);
}

s32 cellSpursAttributeSetNamePrefix(CellSpursAttribute* attr,
                                    const char* prefix, u32 size)
{
    if (!attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursAttribute*);
    const char* prefix_h = GUEST_PTR(prefix, const char*);

    if (prefix_h && size > 0) {
        u32 copyLen = size < sizeof(attr->prefix) ? size : sizeof(attr->prefix) - 1;
        memcpy(attr->prefix, prefix_h, copyLen);
        attr->prefix[copyLen] = '\0';
        attr->prefixSize = copyLen;
    }

    return CELL_OK;
}

s32 cellSpursAttributeSetSpuThreadGroupType(CellSpursAttribute* attr,
                                            s32 type)
{
    (void)type;
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    return CELL_OK;
}

s32 cellSpursAttributeEnableSpuPrintfIfAvailable(CellSpursAttribute* attr)
{
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    return CELL_OK;
}

s32 cellSpursGetNumSpuThread(const CellSpurs* spurs, u32* nThreads)
{
    if (!spurs || !nThreads)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    struct SpursInst* si = spurs_inst_find((u32)(uintptr_t)spurs);
    u32* nThreads_h = GUEST_PTR(nThreads, u32*);

    if (!si)
        return CELL_SPURS_CORE_ERROR_STAT;

    /* out-param is guest BE */
    vm_write32((u32)(uintptr_t)nThreads, si->nspus);
    (void)nThreads_h;
    return CELL_OK;
}

s32 cellSpursSetMaxContention(CellSpurs* spurs, CellSpursWorkloadId wid,
                              u32 maxContention)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= SPURS_MAXW) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!spurs_wid_owned(spurs, wid)) return CELL_SPURS_CORE_ERROR_SRCH;

    s_workloads[wid].maxContention = maxContention;
    /* Guest-visible too, where AddWorkload put it: the kernel threads read the
     * contention from the instance, so a change applies to later dispatches. */
    spurs_nib_set((u32)(uintptr_t)spurs, SPURS_WKL_MAXCONT, wid,
                  (u8)(maxContention ? (maxContention > 15 ? 15 : maxContention) : 1));
    return CELL_OK;
}

s32 cellSpursSetPriorities(CellSpurs* spurs, CellSpursWorkloadId wid,
                           const u8* priorities)
{
    if (!spurs || !priorities) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    const u8* priorities_h = GUEST_PTR(priorities, const u8*);
    if (wid >= SPURS_MAXW) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!spurs_wid_owned(spurs, wid)) return CELL_SPURS_CORE_ERROR_SRCH;

    memcpy(s_workloads[wid].priority, priorities_h, CELL_SPURS_MAX_SPU);
    /* ...and the published wklInfo priority table, as AddWorkload wrote it. */
    const u32 info = (u32)(uintptr_t)spurs + spurs_off_info(wid);
    for (u32 sn = 0; sn < CELL_SPURS_MAX_SPU; sn++)
        *(vm_base + info + 0x18 + sn) = priorities_h[sn];
    return CELL_OK;
}

/* The lv2 event queues the app attached to SPURS, and the port we handed back.
 * A title may attach more than one (Tokyo Jungle attaches two), so keep them
 * all -- storing a single id let the second attach hide the first. */
#define MAX_SPURS_QUEUES 8
#define SPURS_EVENT_PORT 0u
extern int sys_event_queue_push_by_id(uint32_t queue_id, uint64_t source,
                                      uint64_t data1, uint64_t data2, uint64_t data3);
extern void sys_event_queue_cancel_by_id(uint32_t queue_id);
extern void sys_event_queue_uncancel_by_id(uint32_t queue_id);

static u32 s_spurs_event_queue[MAX_SPURS_QUEUES];
/* WHICH SPURS instance attached each queue. A title with more than one instance
 * -- Tokyo Jungle boots one for its data install and keeps a second for audio --
 * finalises them separately, and cancelling every queue on any Finalize kills
 * the surviving instance's queues too. That reads to the title as "the SPU
 * stopped answering": its sound engine's receives fail with ECANCELED and it
 * tears itself down. Cancel only the queues the instance being finalised owns. */
static u32 s_spurs_event_queue_owner[MAX_SPURS_QUEUES];
static int s_spurs_event_queue_n = 0;
/* Guards the attached-queue list and the port table: attach/detach/finalize run
 * on PPU threads while SPU threads look ports up (upstream v0.15). */
static SRWLOCK s_spurs_queue_lock = SRWLOCK_INIT;
static u32 s_spurs_port_queue[64];

/* Release anyone blocked on the completion queues SPURS attached; see
 * cellSpursFinalize. The ports bound to a cancelled queue are released too. */
static void spurs_cancel_attached_queues(u32 spurs_ea)
{
    AcquireSRWLockExclusive(&s_spurs_queue_lock);
    int keep = 0;
    for (int i = 0; i < s_spurs_event_queue_n; i++) {
        /* owner 0 means "attached before we tracked owners" -- cancel those on
         * any finalize, which is the old behaviour and the safe default. */
        u32 owner = s_spurs_event_queue_owner[i];
        if (spurs_ea == 0 || owner == 0 || owner == spurs_ea) {
            u32 q = s_spurs_event_queue[i];
            for (u32 p = 0; p < 64; p++)
                if (s_spurs_port_queue[p] == q) s_spurs_port_queue[p] = 0;
            sys_event_queue_cancel_by_id(q);
        } else {
            s_spurs_event_queue[keep]       = s_spurs_event_queue[i];
            s_spurs_event_queue_owner[keep] = owner;
            keep++;
        }
    }
    s_spurs_event_queue_n = keep;
    ReleaseSRWLockExclusive(&s_spurs_queue_lock);
}

/* SPU user-event port -> lv2 queue, for SPU code running under SPURS.
 *
 * A SPURS task signals the PPU with send_event(port, ...) exactly as a raw SPU
 * thread does, but there is no lv2 SPU thread group to hold the port binding,
 * so the lv2 delivery path had nowhere to look and dropped the event -- and
 * with it the ack the SPU then blocks on in SPU_RdInMbox. Havok's SPU task
 * stalls on the first physics step that way (port 2), and the PPU step waits
 * for it forever. */
uint32_t spurs_port_queue(uint32_t port)
{
    AcquireSRWLockShared(&s_spurs_queue_lock);
    u32 q = port < 64 ? s_spurs_port_queue[port] : 0;
    ReleaseSRWLockShared(&s_spurs_queue_lock);
    return q;
}

s32 cellSpursAttachLv2EventQueue(CellSpurs* spurs, u32 queue, u8* port,
                                 s32 isDynamic)
{
    if (!spurs || !port) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    AcquireSRWLockExclusive(&s_spurs_queue_lock);
    /* KEEP the queue id. SPURS signals the application through the queue it
     * attaches here, and this discarded it -- so nothing SPURS ever did could
     * wake a thread sitting in sys_event_queue_receive on it. Tokyo Jungle
     * kicks its job chain and then blocks on exactly that receive. */
    if (queue && s_spurs_event_queue_n < MAX_SPURS_QUEUES) {
        /* A previous cellSpursFinalize may have cancelled this queue. Attaching
         * gives it a producer again, so lift that -- otherwise a title that
         * tears its audio down and brings it back finds every receive failing. */
        sys_event_queue_uncancel_by_id(queue);
        int dup = 0;
        for (int i = 0; i < s_spurs_event_queue_n; i++)
            if (s_spurs_event_queue[i] == queue) dup = 1;
        if (!dup) { s_spurs_event_queue_owner[s_spurs_event_queue_n] = (u32)(uintptr_t)spurs;
                    s_spurs_event_queue[s_spurs_event_queue_n++] = queue; }
    }

    u8* port_h = GUEST_PTR(port, u8*);

    /* Static: the caller names the port in *port. Dynamic: pick a free one,
     * 0x10..0x3F first (the low ports are the ones titles ask for by number),
     * and hand it back. This used to write 0 either way, so every attach
     * landed on port 0 and the SPU's real port number routed nowhere. */
    u32 p = *port_h;
    if (isDynamic) {
        p = 64;
        for (u32 i = 0x10; i < 64 && p == 64; i++) if (!s_spurs_port_queue[i]) p = i;
        for (u32 i = 0; i < 0x10 && p == 64; i++) if (!s_spurs_port_queue[i]) p = i;
        if (p == 64) { ReleaseSRWLockExclusive(&s_spurs_queue_lock); return CELL_SPURS_CORE_ERROR_BUSY; }
        *port_h = (u8)p;
    } else if (p >= 64) {
        ReleaseSRWLockExclusive(&s_spurs_queue_lock);
        return CELL_SPURS_CORE_ERROR_INVAL;
    } else if (s_spurs_port_queue[p]) {
        ReleaseSRWLockExclusive(&s_spurs_queue_lock);
        return CELL_SPURS_CORE_ERROR_BUSY;
    }
    s_spurs_port_queue[p] = queue;
    printf("[cellSpurs] AttachLv2EventQueue(queue=%u) port=%u%s\n", queue, p,
           isDynamic ? " (dynamic)" : "");
    ReleaseSRWLockExclusive(&s_spurs_queue_lock);
    return CELL_OK;
}

s32 cellSpursDetachLv2EventQueue(CellSpurs* spurs, u8 port)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    AcquireSRWLockExclusive(&s_spurs_queue_lock);
    if (port < 64) s_spurs_port_queue[port] = 0;
    ReleaseSRWLockExclusive(&s_spurs_queue_lock);
    printf("[cellSpurs] DetachLv2EventQueue(port=%u)\n", port);
    return CELL_OK;
}

/* Validate guest backing before forming a host pointer. A zero VM size is
 * the existing full-address-space convention; mapped native-VA ports can
 * use a null base with a nonzero VM size. */
static int spurs_guest_range_valid(u32 ea, u32 bytes)
{
    extern u32 ppu_vm_size;
    u64 end = (u64)ea + bytes;
    uintptr_t base = (uintptr_t)vm_base;
    return bytes && (vm_base || ppu_vm_size) && end <= 0x100000000ull &&
           (!ppu_vm_size || end <= ppu_vm_size) && base <= UINTPTR_MAX - ea &&
           base + ea <= UINTPTR_MAX - (bytes - 1u);
}

/* =========================================================================
 * Taskset
 * =====================================================================*/

static s32 spurs_create_taskset(CellSpurs* spurs, CellSpursTaskset* taskset,
                                u64 args, const u8* priority, u32 maxContention,
                                u32 taskset_size)
{
    (void)args; (void)priority; (void)maxContention;

    /* Capture the GUEST EAs (raw register values) BEFORE host translation -- the real
     * BE taskset builder writes to guest memory at these EAs. */
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;
    uint32_t spurs_ea   = (uint32_t)(uintptr_t)spurs;

    /* Args arrive as guest effective addresses (ps3_hle_call passes raw guest
     * register values); translate to host before dereferencing. */
    if (!spurs_ea || !taskset_ea) {
        fprintf(stderr, "[cellSpurs] CreateTaskset REJECT null (spurs=0x%08X taskset=0x%08X)\n",
                spurs_ea, taskset_ea);
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    }

    if (!spurs_inst_find(spurs_ea)) {
        fprintf(stderr, "[cellSpurs] CreateTaskset REJECT unregistered spurs=0x%08X (taskset=0x%08X)\n",
                spurs_ea, taskset_ea);
        return CELL_SPURS_CORE_ERROR_STAT;
    }

    if (!spurs_guest_range_valid(taskset_ea, taskset_size))
        return (s32)0x8041090Du; /* host backing safety, before writes */
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    memset(taskset, 0, taskset_size);

    /* Write the REAL big-endian CellSpursTaskset layout (fork Option-B) so the lifted
     * SPU leaf + spurs_pm_build_context read valid data (the native writes above are
     * little-endian = garbage to the SPU). Overwrites 0x00-0x80 with BE fields. */
    /* wid: an HLE taskset is no kernel workload, but SPU-side task APIs read
     * taskset+0x74 and kick that wid when they wake a waiting task (SendSignal's
     * 0x8000>>wid readyCount bump). wid 0 is a live workload in GoW:A (WWS), so
     * hand out ids from the top (31, 30, ...) that AddWorkload fills last. */
    static uint32_t s_ts_wid_next = 31;
    uint32_t ts_wid = s_ts_wid_next > 16 ? s_ts_wid_next-- : 31;
    /* taskset+0x1890 `size` is read by SPU-side task APIs: GoW:A's tasks create
     * follow-up tasks with the SPU cellSpursCreateTask, which requires a
     * CellSpursTaskset2 (size 0x2900) and failed on the host struct's size. */
    /* +0x1898/+0x189C hold the lv2 event-flag ids libsre creates for the taskset;
     * the SPU cellSpursCreateTask rejects a taskset with both zero (0x80410909).
     * No lv2 flag backs them here: nonzero placeholders, unique per taskset. */
    spurs_taskset_init(taskset_ea, spurs_ea, args, ts_wid,
                       taskset_size,
                       /*evf1*/0x7E000000u | (ts_wid << 8) | 1u, /*evf2*/0x7E000000u | (ts_wid << 8) | 2u);
    g_ydkj_real_taskset_ea = taskset_ea;
    { extern void spu_taskset_register(uint32_t); spu_taskset_register(taskset_ea); }

    g_ydkj_real_spurs_ea = spurs_ea;   /* capture for the taskset-policy handoff (LS[0x1C0]) */
    printf("[cellSpurs] CreateTaskset() ea=0x%08X spurs=0x%08X (real BE layout)\n", taskset_ea, spurs_ea);
    return CELL_OK;
}

s32 cellSpursCreateTaskset(CellSpurs* spurs, CellSpursTaskset* taskset,
                           u64 args, const u8* priority, u32 maxContention)
{
    return spurs_create_taskset(spurs, taskset, args, priority, maxContention,
                                CSTS_STRUCT_END);
}

s32 cellSpursCreateTasksetWithAttribute(CellSpurs* spurs,
                                        CellSpursTaskset* taskset,
                                        const CellSpursTasksetAttribute* attr)
{
    u32 attr_ea = (u32)(uintptr_t)attr;
    if (!attr_ea) return (s32)0x80410911u;
    if (attr_ea & 7u) return (s32)0x80410910u;
    if (!spurs_guest_range_valid(attr_ea, sizeof(*attr))) return (s32)0x8041090Du;
    const CellSpursTasksetAttribute* attr_h = GUEST_PTR(attr, const CellSpursTasksetAttribute*);
    u32 size = attr_h->tasksetSize;
    if (size != 0x1900u && size != 0x2900u) return (s32)0x80410902u;
    return spurs_create_taskset(spurs, taskset, 0, NULL, 0, size);
}

/* SPURS v2 API (SDK >= 3.x). The v2 entry points the SDK macros import for
 * cellSpursInitializeWithAttribute / cellSpursCreateTaskset(WithAttribute) when
 * building against the Taskset2 headers. Semantics we model match v1; the
 * attribute/taskset blobs are larger in the guest but only a prefix is read. */
s32 cellSpursInitializeWithAttribute2(CellSpurs* spurs, const CellSpursAttribute* attr)
{
    s32 rc = cellSpursInitializeWithAttribute(spurs, attr);
    if (rc == CELL_OK) {
        struct SpursInst* si = spurs_inst_find((u32)(uintptr_t)spurs);
        if (si) {
            si->is2 = 1;   /* CellSpurs2: 8 KB, second workload set at 0xC0/0x78/0x1000 */
            memset(vm_base + (u32)(uintptr_t)spurs + 0x1000, 0, 0x1000);
            /* flags1 (u8 @0x74) |= SF1_32_WORKLOADS (0x40). Modules read it from the
             * kernel's tempArea copy: GoW:A's WWS job resolving the flag receiver's
             * wklInfo returned EA 0 for wid >= 16 without it and spun on GETLLAR(0). */
            *(vm_base + (u32)(uintptr_t)spurs + 0x74) |= 0x40;
        }
    }
    return rc;
}

s32 cellSpursAttributeEnableSystemWorkload(CellSpursAttribute* attr, const u8* priority,
                                           u32 maxSpu, const u8* isPreem)
{
    (void)priority; (void)isPreem;
    if (!attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    printf("[cellSpurs] AttributeEnableSystemWorkload(maxSpu=%u) (accepted, not modelled)\n", maxSpu);
    return CELL_OK;
}

s32 _cellSpursTasksetAttribute2Initialize(CellSpursTasksetAttribute* attr, u32 revision)
{
    if (!attr)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    /* CellSpursTasksetAttribute2 is 512 bytes in the guest SDK. */
    u32* a = (u32*)GUEST_PTR(attr, void*);
    memset(a, 0, 512);
    a[0] = __builtin_bswap32(revision ? revision : 1);
    printf("[cellSpurs] _TasksetAttribute2Initialize(rev=%u)\n", revision);
    return CELL_OK;
}

s32 cellSpursCreateTaskset2(CellSpurs* spurs, CellSpursTaskset* taskset,
                            const CellSpursTasksetAttribute* attr)
{
    (void)attr;
    return spurs_create_taskset(spurs, taskset, 0, NULL, 0, 0x2900u);  /* sizeof(CellSpursTaskset2) */
}

/* SPURS Task2 API. _cellSpursTaskAttribute2Initialize zeroes the 256-byte guest
 * attribute (revision first, big-endian). cellSpursCreateTask2 is
 *   (taskset, u32* id, eaElf, CellSpursTaskArgument* arg, CellSpursTaskAttribute2* attr)
 * -- r3..r7 (RPCS3 cellSpurs.cpp). The old 3-argument stub read eaElf as the
 * attribute and never registered the task, so the taskset's enabled bitset stayed
 * zero: GoW:A's RsxSync WWS job then signalled one of these tasks with the SPU-side
 * cellSpursSendSignal, got 0x80410905 (no such task) and hit its `stopd` assert.
 * Forward to cellSpursCreateTask, which registers the task in the real BE taskset
 * (enabled+ready bits, TaskInfo) and dispatches a lifted task image if one exists.
 * SPURS_TASK2_STUB=1 restores the accept-only behaviour. */
s32 _cellSpursTaskAttribute2Initialize(u64 attr_ea, u32 revision)
{
    uint32_t a = (uint32_t)attr_ea;
    if (!a) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    memset(vm_base + a, 0, 256);
    vm_write32(a, revision ? revision : 1);
    return CELL_OK;
}

static volatile long long s_ct_stage[6];   /* defined with cellSpursCreateTask */
/* Per-task log lines: the first 16 calls, all of them with SPURS_TASK_LOG=1. */
static int spurs_task_log_on(void)
{
    static int s_all = -1; static volatile LONG s_n;
    if (s_all < 0) s_all = getenv("SPURS_TASK_LOG") ? 1 : 0;
    return s_all || InterlockedIncrement(&s_n) <= 16;
}
s32 cellSpursCreateTask2(u64 taskset_ea, u64 taskId_ea, u64 elf_ea, u64 arg_ea, u64 attr_ea)
{
    static struct { uint32_t ts; uint32_t next; } s_ts[16];
    uint32_t ts = (uint32_t)taskset_ea;
    if (!ts) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (!getenv("SPURS_TASK2_STUB")) {
        /* SPURS_TASK2_PROF=<s>: calls, total and max wall time of this HLE every s seconds. */
        static int s_tp = -1; static LARGE_INTEGER s_f, s_w0; static double s_tot, s_max; static unsigned s_n;
        if (s_tp < 0) { const char* e = getenv("SPURS_TASK2_PROF"); s_tp = e ? atoi(e) : 0;
                        QueryPerformanceFrequency(&s_f); QueryPerformanceCounter(&s_w0); }
        LARGE_INTEGER t0; QueryPerformanceCounter(&t0);
        const uint32_t a = (uint32_t)attr_ea;
        const uint32_t sizeContext = a ? vm_read32(a + 4) : 0;
        const uint32_t eaContext   = a ? (uint32_t)vm_read64(a + 8) : 0;
        const uint32_t lsp_ea      = a ? a + 0x10 : 0;
        s32 rc = cellSpursCreateTask((CellSpursTaskset*)(uintptr_t)ts,
                                     (CellSpursTaskId*)(uintptr_t)(uint32_t)taskId_ea,
                                     (void*)(uintptr_t)(uint32_t)elf_ea,
                                     (void*)(uintptr_t)eaContext, sizeContext,
                                     lsp_ea, (uint32_t)arg_ea);
        if (spurs_task_log_on())
            printf("[cellSpurs] CreateTask2(taskset=0x%08X elf=0x%08X arg=0x%08X ctx=0x%08X size=%u) -> rc=0x%X id=%u\n",
               ts, (uint32_t)elf_ea, (uint32_t)arg_ea, eaContext, sizeContext, (unsigned)rc,
               taskId_ea ? vm_read32((uint32_t)taskId_ea) : 0xFFFFFFFFu);
        if (s_tp > 0) {
            LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
            const double dt = (double)(t1.QuadPart - t0.QuadPart) / (double)s_f.QuadPart;
            s_tot += dt; if (dt > s_max) s_max = dt; s_n++;
            const double win = (double)(t1.QuadPart - s_w0.QuadPart) / (double)s_f.QuadPart;
            if (win >= s_tp) {
                extern volatile long long g_spu_dispatch_fp_ticks, g_spu_dispatch_ct_ticks;
                static long long s_fp0, s_ct0;
                const long long fp = g_spu_dispatch_fp_ticks, ct = g_spu_dispatch_ct_ticks;
                fprintf(stderr, "[task2-prof] %.1fs: %u calls, %.1f%% of wall, avg %.3f ms, max %.3f ms"
                        " (all dispatches: fingerprint %.1f%%, CreateThread %.1f%%)\n",
                        win, s_n, 100.0 * s_tot / win, s_n ? 1000.0 * s_tot / s_n : 0.0, 1000.0 * s_max,
                        100.0 * (double)(fp - s_fp0) / (double)s_f.QuadPart / win,
                        100.0 * (double)(ct - s_ct0) / (double)s_f.QuadPart / win);
                s_fp0 = fp; s_ct0 = ct;
                fprintf(stderr, "[task2-prof]   CreateTask stages %% of wall: elfsize %.1f claim %.1f addtask %.1f logs %.1f dispatch %.1f\n",
                        100.0 * s_ct_stage[1] / (double)s_f.QuadPart / win, 100.0 * s_ct_stage[2] / (double)s_f.QuadPart / win,
                        100.0 * s_ct_stage[3] / (double)s_f.QuadPart / win, 100.0 * s_ct_stage[4] / (double)s_f.QuadPart / win,
                        100.0 * s_ct_stage[5] / (double)s_f.QuadPart / win);
                for (int k = 0; k < 6; k++) s_ct_stage[k] = 0;
                s_tot = 0; s_max = 0; s_n = 0; s_w0 = t1;
            }
        }
        return rc;
    }
    int k = 0;
    while (k < 16 && s_ts[k].ts && s_ts[k].ts != ts) k++;
    if (k == 16) return CELL_SPURS_TASK_ERROR_AGAIN;
    s_ts[k].ts = ts;
    uint32_t id = s_ts[k].next++ & 127u;
    if (taskId_ea) vm_write32((uint32_t)taskId_ea, id);
    static int s_n = 0;
    if (s_n++ < 12 || getenv("SPURS_TASK2_LOG"))
        printf("[cellSpurs] CreateTask2(taskset=0x%08X attr=0x%08X) -> id %u (accepted, not scheduled)\n",
               ts, (uint32_t)attr_ea, id);
    return CELL_OK;
}

s32 cellSpursDestroyTaskset(CellSpursTaskset* taskset)
{
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    if (!taskset)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    printf("[cellSpurs] DestroyTaskset()\n");
    return CELL_OK;
}

s32 cellSpursShutdownTaskset(CellSpursTaskset* taskset)
{
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    if (!taskset)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    printf("[cellSpurs] ShutdownTaskset()\n");
    return CELL_OK;
}

s32 cellSpursJoinTaskset(CellSpursTaskset* taskset)
{
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    if (!taskset)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    printf("[cellSpurs] JoinTaskset()\n");
    /* In a full implementation, wait for all tasks to complete */
    return CELL_OK;
}

s32 cellSpursTasksetAttributeInitialize(CellSpursTasksetAttribute* attr)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTasksetAttribute*);
    memset(attr, 0, sizeof(CellSpursTasksetAttribute));
    attr->revision = 1;
    attr->tasksetSize = CSTS_STRUCT_END;
    return CELL_OK;
}

s32 cellSpursTasksetAttributeSetName(CellSpursTasksetAttribute* attr,
                                      const char* name)
{
    (void)name;
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    return CELL_OK;
}

/* =========================================================================
 * Task
 * =====================================================================*/

/* Real SDK ABI (verified against RPCS3 cellSpurs.cpp:370):
 *   cellSpursCreateTask(taskset, taskId, elf, context, sizeContext,
 *                       CellSpursTaskLsPattern* lsPattern,
 *                       CellSpursTaskArgument*  argument)
 * -- SEVEN args (r3..r9), not six. The old 6-arg form treated r8 as an opaque
 * `attr` and never read r9 (the argument), so the task's 16-byte work-descriptor
 * argument was dropped and TaskInfo.args stayed 0. LBP's audio task DMAs its work
 * from an EA computed out of r3 = that argument (spu_0003 task main @0x17e70:
 * `wrch $ch18, f(r3.word3)`), so a zero argument makes it GET from EA 0 and
 * stall. lsPattern/argument arrive as guest EAs (generic adapter forwards r3..r10). */
/* CreateTask stage timer (SPURS_TASK2_PROF): ticks between consecutive marks,
 * per stage -- 1 elf size, 2 claim, 3 add_task, 4 logs, 5 dispatch. */
static volatile long long s_ct_stage[6];
static _Thread_local long long s_ct_last;
#define CT_STAGE_MARK(k) do { LARGE_INTEGER _q; QueryPerformanceCounter(&_q);     if ((k) > 0) s_ct_stage[(k)] += _q.QuadPart - s_ct_last; s_ct_last = _q.QuadPart; } while (0)
s32 cellSpursCreateTask(CellSpursTaskset* taskset, CellSpursTaskId* taskId,
                        void* elf, void* context, u32 sizeContext,
                        u32 lsPattern_ea, u32 argument_ea)
{
    (void)context; (void)sizeContext;

    /* Read the 16-byte CellSpursTaskArgument + optional LS pattern from guest mem. */
    uint32_t task_arg[4] = {0,0,0,0};
    uint32_t task_lsp[4] = {0,0,0,0};
    if (argument_ea)  for (int _i=0;_i<4;_i++) task_arg[_i] = vm_read32(argument_ea + _i*4);
    if (lsPattern_ea) for (int _i=0;_i<4;_i++) task_lsp[_i] = vm_read32(lsPattern_ea + _i*4);

    /* Capture guest EAs BEFORE host translation (the real BE taskset builder + the
     * SPU DMA use guest EAs). */
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;
    uint32_t elf_ea     = (uint32_t)(uintptr_t)elf;
    uint32_t context_ea = (uint32_t)(uintptr_t)context;

    /* taskId/taskset are guest EAs; translate before deref. elf/context stay
     * guest EAs (handled below — elf is translated for load, context kept EA). */
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    CellSpursTaskId* taskId_h = GUEST_PTR(taskId, CellSpursTaskId*);

    if (!taskset) {
        fprintf(stderr, "[cellSpurs] CreateTask REJECT null taskset (elf=0x%08X)\n", elf_ea);
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    }

    if (!g_ydkj_real_taskset_ea) { /* real-BE init flag (native ->initialized clobbered by BE layout) */
        fprintf(stderr, "[cellSpurs] CreateTask REJECT no-init (taskset=0x%08X elf=0x%08X)\n",
                taskset_ea, elf_ea);
        return CELL_SPURS_TASK_ERROR_STAT;
    }

    /* Find a free task slot IN THIS TASKSET: the first task id whose bit in the
     * taskset's enabled bitset (+0x30, 128 bits, task 0 = MSB) is clear. Task
     * exit clears that bit (spu_taskset_task_exited), so ids recycle the way
     * real SPURS recycles them. The old scan of the global s_tasks[] never
     * freed a slot unless the task was joined; Havok creates a task per step
     * and never joins, so after 128 steps CreateTask failed and the physics
     * step waited forever for a task that was never started.
     * ponytail: s_tasks[] is shared by every taskset and now only records the
     * last task per slot id -- enough for JoinTask2's lookup; key it by
     * (taskset, id) if a title joins tasks in two tasksets at once. */
    /* The search and the enabled bit are one atomic step: a task exiting between
     * a plain read and the later bit set got its enabled bit back with no task
     * behind it, so cellSpursJoinTask2 on that id never returned. */
    u32 free_slot = spurs_taskset_take_slot(taskset_ea, CELL_SPURS_MAX_TASK);
    if (free_slot < CELL_SPURS_MAX_TASK) {
        u32 i = free_slot;
        {
            s_tasks[i].in_use = 1;
            s_tasks[i].id = s_next_task_id++;
            s_tasks[i].active = 1;
            s_tasks[i].completed = 0;
            s_tasks[i].exitCode = 0;
            s_tasks[i].entryPoint = elf;

            /* The guest reads this out-param BIG-ENDIAN, and a native store
             * put it in host order: task 1 came back to the game as 0x01000000,
             * which it then handed to _cellSpursSendSignal. No task by that id
             * exists, the signal was dropped, and the SPU task sat in
             * WAIT_SIGNAL while the PPU waited on the event flag it would have
             * set -- a two-sided deadlock from one missing byte swap.
             *
             * Hand back the SLOT INDEX, not the global counter: `i` is what
             * spurs_taskset_add_task() sets as the taskset's bitset bit and
             * what spu_taskset_signal_task() looks up, so the id the guest
             * signals with has to be the same number. */
            if (taskId) vm_write32((u32)(uintptr_t)taskId, i);
            (void)taskId_h;

            /* Register the task in the REAL BE taskset: writes task_info[slot]
             * (args/elf/context/ls_pattern) + sets enabled+ready bits so the PM's
             * SELECT_TASK picks it. Slot index i = the SPURS taskId (bitset bit).
             * Claim the task first (see spu_taskset_claim; it is the "mark the slot
             * running BEFORE its ready/enabled bits go in" step, and also reports a
             * previous instance still running): once the bits are set, a concurrent
             * task exit may start it. */
            extern int spu_taskset_claim(uint32_t, uint32_t);
            CT_STAGE_MARK(0);
            const uint8_t* host_elf = elf ? GUEST_PTR(elf, const uint8_t*) : NULL;
            const size_t elf_sz = host_elf ? spu_elf_image_size(host_elf, 2u * 1024 * 1024) : 0;
            CT_STAGE_MARK(1);
            const int claimed = elf_sz ? spu_taskset_claim(taskset_ea, i) : 0;
            CT_STAGE_MARK(2);
            if (elf_sz && !claimed)
                fprintf(stderr, "[cellSpurs] CreateTask: taskset 0x%08X slot %u still running -- left to its exit\n",
                        taskset_ea, i);
            { extern void spu_taskset2_set_exit(uint32_t, uint32_t, int, uint32_t);
              spu_taskset2_set_exit(taskset_ea, i, 0, 0); }   /* fresh entry for a recycled id */
            spurs_taskset_add_task(taskset_ea, i, (uint64_t)elf_ea,
                                   (uint64_t)context_ea, task_arg, task_lsp);
            /* Bridge to the image-22 dispatch so build_context uses this taskset+task. */
            g_ydkj_real_taskset_ea = taskset_ea;
            g_ydkj_real_taskid     = i;
            CT_STAGE_MARK(3);

            /* Per-task logs: first 16, then only under SPURS_TASK_LOG=1. Unbuffered
             * stdout shared with every thread made these lines ~20% of GoW:A's main
             * thread at ~200 tasks/s (SPURS_TASK2_PROF). */
            if (spurs_task_log_on())
                printf("[cellSpurs] CreateTask(id=%u, entry=%p, arg=%08X %08X %08X %08X"
                       " ctx=0x%08X lsp=%08X %08X %08X %08X)\n",
                       s_tasks[i].id, elf, task_arg[0], task_arg[1], task_arg[2], task_arg[3],
                       context_ea, task_lsp[0], task_lsp[1], task_lsp[2], task_lsp[3]);

            /* One-shot: dump the memory the task argument points at, to find the
             * pointer that reads back 0 (the task GETs from EA 0 -> some field of
             * its work descriptor is null in our run). Each of the 4 arg words that
             * looks like a valid guest EA gets 64 bytes dumped as BE u32s. */
            if (getenv("SPURS_TASKSET_TRACE")) {
                for (int a = 0; a < 4; a++) {
                    uint32_t p = task_arg[a];
                    if (p < 0x10000 || p >= 0x50000000u) continue;   /* not a plausible EA */
                    fprintf(stderr, "[argdump] arg[%d]=0x%08X ->", a, p);
                    for (int o = 0; o < 64; o += 4)
                        fprintf(stderr, " %08X", vm_read32(p + o));
                    fprintf(stderr, "\n");
                }
                /* Full heap object (arg[2], the audio engine object): the count the
                 * PPU validated is at +0xC8 (16-aligned, <=0x160 = stream count); the
                 * task's per-stream buffer pointers live at +0x124 (=arg[1]). Dump
                 * +0x00..+0x160 so we can see: 0 streams (task should yield) vs N
                 * streams with null data buffers (fill gap). */
                uint32_t ho = task_arg[2];
                if (ho >= 0x10000 && ho < 0x50000000u) {
                    fprintf(stderr, "[heapobj] 0x%08X count@+0xC8=0x%08X\n", ho, vm_read32(ho + 0xC8));
                    for (int o = 0; o < 0x160; o += 16)
                        fprintf(stderr, "  +%03X: %08X %08X %08X %08X\n", o,
                                vm_read32(ho+o), vm_read32(ho+o+4), vm_read32(ho+o+8), vm_read32(ho+o+12));
                }
                /* The descriptor block the SPU task actually DMAs + reads its buffer
                 * pointers from: v10[336..] at v10+1344 = arg[3]-64 (arg[3]=v10+1408).
                 * v10[344]=a1[141], v10[345]=a1[140] (the FMOD DSP buffers). If those
                 * words are 0 here, they are the null source (task GETs from EA 0). */
                uint32_t d = task_arg[3];
                if (d >= 0x10040 && d < 0x50000000u) {
                    uint32_t db = d - 64;   /* 0x0094F6C0 = v10+1344 */
                    fprintf(stderr, "[descblk] v10+1344=0x%08X (a1[141]@+0x20, a1[140]@+0x24):\n", db);
                    for (int o = 0; o < 0x40; o += 16)
                        fprintf(stderr, "  +%02X: %08X %08X %08X %08X\n", o,
                                vm_read32(db+o), vm_read32(db+o+4), vm_read32(db+o+8), vm_read32(db+o+12));
                }
                /* a1 (the FMOD object) = taskId_ea - 628 (sub_48420C passes the taskId
                 * out-param as (_DWORD)a1+628 for task 0). a1[140]/a1[141] (= a1+0x230/
                 * +0x234) are the null DSP-buffer fields. Log the EAs so the next run
                 * can YDKJ_WWATCH=<a1+0x230> to catch who should write it (or prove no
                 * one does). Only for task 0 (offset 628); task 1 uses +688. */
                uint32_t a1 = (uint32_t)(uintptr_t)taskId - 628u;
                if (a1 < 0x50000000u)
                    fprintf(stderr, "[a1obj] a1=0x%08X  a1+0x230(dsp0)=0x%08X val=0x%08X  "
                            "a1+0x234(dsp1)=0x%08X val=0x%08X\n", a1,
                            a1+0x230, vm_read32(a1+0x230), a1+0x234, vm_read32(a1+0x234));
                fflush(stderr);
            }

            /* Run the task's SPU program if a lifted build is registered for it.
             * The registry maps the task ELF (by content fingerprint) to its
             * pre-lifted native entry; dispatch loads the ELF into a local store
             * and runs it with the task arg in r3. INERT until the title
             * registers its lifted SPU set: an unregistered image MISSes and
             * returns 0, preserving the prior "track only" behaviour.
             *
             * NOTE: dispatch is synchronous (runs to completion inline). That
             * suits create+join task patterns; a workload/taskset whose SPU job
             * waits on concurrent PPU-side signals will want the async lv2
             * SPU-thread path instead — wired when a title exercises it. */
            if (claimed) {
                /* elf/context are guest effective addresses; the image pointer is
                 * translated to host memory for fingerprint+load, but context stays
                 * the guest EA (the SPU job's DMA uses guest EAs / r3).
                 * Async: SPURS tasks are persistent workers — running them inline
                 * would block this PPU thread forever (deadlock). Name the task
                 * explicitly: an SPU thread's task exit can rewrite the single-slot
                 * globals between here and the dispatch. Not claimed = a previous
                 * instance in this slot is still running; its exit starts this one. */
                extern int spu_workload_dispatch_claimed_task_async(const uint8_t*, uint32_t, uint32_t,
                                                                    uint32_t, uint32_t);
                CT_STAGE_MARK(4);
                spu_workload_dispatch_claimed_task_async(host_elf, (uint32_t)elf_sz,
                                                         (uint32_t)(uintptr_t)context, taskset_ea, i);
            }
            CT_STAGE_MARK(5);
            return CELL_OK;
        }
    }

    fprintf(stderr, "[cellSpurs] CreateTask: taskset 0x%08X has no free task id (all 128 enabled)\n",
            taskset_ea);
    return CELL_SPURS_TASK_ERROR_NOMEM;
}

/* The SDK's versioned task-attribute initializer. ABI (8 GPR args):
 *   r3=attr r4=revision r5=sdkVersion r6=eaElf r7=eaContext r8=sizeContext
 *   r9=lsPattern r10=argument
 * Stash the task ELF EA + context so cellSpursCreateTaskWithAttribute can
 * dispatch the SPU job. */
s32 _cellSpursTaskAttributeInitialize(CellSpursTaskAttribute* attr, u32 revision,
                                      u32 sdkVersion, u64 eaElf, u64 eaContext,
                                      u32 sizeContext, const void* lsPattern,
                                      const void* argument)
{
    (void)sdkVersion;
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTaskAttribute*);
    memset(attr, 0, sizeof(CellSpursTaskAttribute));
    attr->revision    = revision;
    attr->sizeContext = sizeContext;
    attr->eaContext   = eaContext;
    attr->eaElf       = eaElf;
    /* lsPattern/argument are guest EAs of 16-byte blocks; carry them so
     * CreateTaskWithAttribute writes them into the TaskInfo. The SPU task
     * library refuses blocking waits for a task whose argument is zero or
     * whose lsPattern doesn't cover its stack (0x8041090F). */
    attr->lsPattern_ea = (u32)(uintptr_t)lsPattern;
    attr->argument_ea  = (u32)(uintptr_t)argument;

    /* SPURS_TASKATTR_R8: some callers use a SIX-argument form, passing only
     * r3..r8 -- r9/r10 then hold caller leftovers, not arguments. YDKJ is one:
     * func_00331DA4 sets r3..r8 and nothing else, so our 8-parameter prototype
     * reads lsPattern=r9 (a size, 0x0003D400) and argument=r10 (0x2E, the ASCII
     * terminator its hex-string parser stopped on). r8 is the real 16-byte
     * CellSpursTaskArgument -- proven by its contents matching the quadword the
     * SPU task later receives in r3. A zero/garbage argument makes the SPU task
     * library refuse every blocking wait (0x8041090F), which is exactly how the
     * CRI tasks end up parked in WAIT_SIGNAL forever.
     * CONFIRMED by a second, independently lifted title: Jackbox Party Pack
     * calls the same NID with the same shape -- r3=attr, r4=1, r5=0x00330000,
     * r6=eaElf, r7=sp+0x180, r8=sp+0x170 (two adjacent 16-byte blocks), and
     * r9/r10 never written. Two unrelated callers agreeing settles it, so this
     * is now the default; SPURS_TASKATTR_LEGACY=1 restores the old 8-arg read. */
    { static int s_r8 = -1;
      if (s_r8 < 0) s_r8 = getenv("SPURS_TASKATTR_LEGACY") ? 0 : 1;
      if (s_r8) {
          attr->argument_ea  = (u32)sizeContext;   /* r8 */
          attr->lsPattern_ea = 0;                  /* r9 was a leftover */
          printf("[cellSpurs] TaskAttr R8-form: argument_ea=0x%08X (was 0x%08X)\n",
                 attr->argument_ea, (u32)(uintptr_t)argument);
      } }
    /* SPURS_TASKATTR_DESC: a THIRD caller shape, found in Saints Row 2. Its
     * taskset builder (func_009F56E0) does not pass eaContext/sizeContext/
     * lsPattern as separate arguments at all -- r7 points at a 3-word
     * DESCRIPTOR it fills immediately before the call, and r8 is the argument:
     *
     *   009F58D0  addi r7, r1, 128     ; r7 = sp+0x80  -> the descriptor
     *   009F58D4  addi r8, r1, 176     ; r8 = sp+0xB0  -> CellSpursTaskArgument
     *   009F58E8  stw  r9,  0x80(r1)   ;   [0] context EA
     *   009F58EC  stw  r11, 0x84(r1)   ;   [1] context size, straight out of
     *                                  ;       cellSpursTaskGetContextSaveAreaSize
     *   009F58F0  stw  r25, 0x88(r1)   ;   [2] CellSpursTaskLsPattern*
     *
     * Read the 8-argument way, sizeContext comes out as r8 -- a STACK ADDRESS,
     * not a size (the log shows szctx=267382952 = 0x0FEFF0A8) -- and lsPattern
     * comes out of r9, which this caller never sets, so it is 0. Those are
     * exactly the conditions the SPU task library refuses to run a blocking
     * task under, and the tasks then park in WAIT_SIGNAL forever doing 0 ms of
     * work apiece, which is the symptom this title shows. Off by default: the
     * R8-form above is right for YDKJ and Jackbox and they are unaffected. */
    { static int s_desc = -1;
      if (s_desc < 0) s_desc = getenv("SPURS_TASKATTR_DESC") ? 1 : 0;
      if (s_desc) {
          uint32_t d = (uint32_t)eaContext;          /* r7 */
          attr->eaContext    = vm_read32(d + 0);
          attr->sizeContext  = vm_read32(d + 4);
          attr->lsPattern_ea = vm_read32(d + 8);
          attr->argument_ea  = (u32)sizeContext;     /* r8 */
          printf("[cellSpurs] TaskAttr DESC-form: desc=0x%08X -> ctx=0x%08X size=%u lsp=0x%08X arg=0x%08X\n",
                 d, (u32)attr->eaContext, attr->sizeContext,
                 attr->lsPattern_ea, attr->argument_ea);
      } }
    printf("[cellSpurs] _TaskAttributeInitialize(eaElf=0x%08X ctx=0x%08X szctx=%u lsp=0x%08X arg=0x%08X)\n",
           (u32)eaElf, (u32)eaContext, sizeContext,
           attr->lsPattern_ea, attr->argument_ea);
    return CELL_OK;
}

/* Create a task from a pre-initialized attribute (carries ELF EA + context).
 * Forwards to cellSpursCreateTask, which translates taskset/taskId and runs the
 * SPU image through spu_workload_dispatch. */
s32 cellSpursCreateTaskWithAttribute(CellSpursTaskset* taskset,
                                     CellSpursTaskId* taskId,
                                     CellSpursTaskAttribute* attr)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    CellSpursTaskAttribute* attr_h = GUEST_PTR(attr, CellSpursTaskAttribute*);
    /* Dump the raw attribute: our struct doesn't model lsPattern/argument, and
     * a wait-capable task NEEDS its context size + ls pattern carried through
     * (a no-context task may not block -- SPU task-lib waits then fail with
     * ERROR_STAT). Learn the real field offsets from the bytes. */
    { uint32_t aea = (uint32_t)(uintptr_t)attr;
      static int _n = 0; if (PS3_LOG_FIRST(_n, 6)) {
        fprintf(stderr, "[cellSpurs] CreateTaskWithAttr attr=0x%08X raw:", aea);
        for (int o = 0; o < 0x40; o += 4) fprintf(stderr, " %08X", vm_read32(aea + o));
        fprintf(stderr, "\n"); } }
    /* taskset/taskId forwarded raw (callee translates); elf/context are guest
     * EAs, as are lsPattern/argument (stored by _cellSpursTaskAttributeInitialize;
     * dropping them left the TaskInfo with a zero argument + zero lsPattern and
     * the SPU task library then refuses every blocking wait with 0x8041090F --
     * LBP's binkspu movie-IO task spun forever on that). */
    return cellSpursCreateTask(taskset, taskId,
                               (void*)(uintptr_t)(u32)attr_h->eaElf,
                               (void*)(uintptr_t)(u32)attr_h->eaContext,
                               attr_h->sizeContext,
                               attr_h->lsPattern_ea, attr_h->argument_ea);
}

/* The SDK's versioned taskset-attribute initializer. We forward taskset creation
 * through CreateTaskset (which ignores the attribute), so just zero the struct. */
s32 _cellSpursTasksetAttributeInitialize(CellSpursTasksetAttribute* attr,
                                         u32 revision, u32 sdkVersion, u64 argTaskset,
                                         u64 priority, u32 maxContention)
{
    (void)sdkVersion; (void)argTaskset; (void)priority; (void)maxContention;
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTasksetAttribute*);
    memset(attr, 0, sizeof(CellSpursTasksetAttribute));
    attr->revision = revision ? revision : 1;
    attr->tasksetSize = CSTS_STRUCT_END;
    printf("[cellSpurs] _TasksetAttributeInitialize(rev=%u)\n", revision);
    return CELL_OK;
}

s32 cellSpursJoinTask(CellSpursTaskset* taskset, CellSpursTaskId taskId,
                      s32* exitCode)
{
    (void)taskset;
    s32* exitCode_h = GUEST_PTR(exitCode, s32*);

    printf("[cellSpurs] JoinTask(id=%u)\n", taskId);

    /* Find the task and mark as completed */
    for (u32 i = 0; i < CELL_SPURS_MAX_TASK; i++) {
        if (s_tasks[i].in_use && s_tasks[i].id == taskId) {
            s_tasks[i].completed = 1;
            s_tasks[i].active = 0;
            if (exitCode_h)
                *exitCode_h = s_tasks[i].exitCode;
            s_tasks[i].in_use = 0;
            return CELL_OK;
        }
    }

    return CELL_SPURS_TASK_ERROR_SRCH;
}

s32 cellSpursSendSignal(CellSpursTaskset* taskset, CellSpursTaskId taskId)
{
    /* Capture the guest taskset EA before host translation -- the WAIT_SIGNAL
     * waiter (spu_taskset_wait_signal) keys on the guest EA + taskId. */
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;

    printf("[cellSpurs] SendSignal(taskset=0x%08X id=%u)\n", taskset_ea, taskId);

    /* Deliver the signal for real: set the task's bit in the guest taskset's
     * SIGNALLED bitset and wake its blocked host thread. (Was a documented
     * no-op from the pre-SPU-execution era -- a dropped signal deadlocked any
     * task parked in WAIT_SIGNAL waiting for it.) */
    if (taskset_ea) spu_taskset_signal_task(taskset_ea, taskId);
    return CELL_OK;
}

s32 cellSpursTaskAttributeInitialize(CellSpursTaskAttribute* attr)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTaskAttribute*);
    memset(attr, 0, sizeof(CellSpursTaskAttribute));
    attr->revision = 1;
    return CELL_OK;
}

/* NID: 0x34552FA6. Initialization only; task attachment and completion are
 * separate APIs. Use the firmware's task-domain errors here rather than the
 * legacy module-wide error aliases in cellSpurs.h. */
s32 cellSpursTaskExitCodeInitialize(CellSpursTaskExitCode* exitCode)
{
    u32 ea = (u32)(uintptr_t)exitCode;
    if (!ea) return (s32)0x80410911u;
    if (ea & 0x7Fu) return (s32)0x80410910u;

    /* A zero size denotes the existing fully backed 32-bit VM convention.
     * Native-VA ports may have a null base with a nonzero mapped size. Check
     * before forming a host address, so malformed EAs cannot wrap or write
     * outside a bounded arena. FAULT is a host-backing safety failure. */
    extern u32 ppu_vm_size;
    u64 end = (u64)ea + 0x80u;
    uintptr_t base = (uintptr_t)vm_base;
    if ((!vm_base && !ppu_vm_size) ||
        end > 0x100000000ull ||
        (ppu_vm_size && end > ppu_vm_size) ||
        base > UINTPTR_MAX - ea || base + ea > UINTPTR_MAX - 0x7Fu)
        return (s32)0x8041090Du;

    memset((void*)(base + ea), 0, 0x80u);
    return CELL_OK;
}

/* =========================================================================
 * Workload
 * =====================================================================*/

s32 cellSpursAddWorkload(CellSpurs* spurs, CellSpursWorkloadId* wid,
                         const void* pm, u32 sizePm, u64 data,
                         const u8* priority, u32 minContention,
                         u32 maxContention)
{
    if (!spurs || !wid)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    /* spurs/wid/priority are guest EAs; pm stays a guest EA (it's the SPU
     * program address consumed later by the workload dispatch). */
    uint32_t spurs_ea = (uint32_t)(uintptr_t)spurs;
    struct SpursInst* si = spurs_inst_find(spurs_ea);
    const u8* priority_h = GUEST_PTR(priority, const u8*);

    if (!si)
        return CELL_SPURS_CORE_ERROR_STAT;

    for (u32 i = 0; i < (si->is2 ? SPURS_MAXW : 16u); i++) {
        if (!s_workloads[i].in_use) {
            s_workloads[i].in_use = 1;
            s_workloads[i].pm = pm;
            s_workloads[i].sizePm = sizePm;
            s_workloads[i].data = data;
            s_workloads[i].spurs_ea = spurs_ea;
            s_workloads[i].minContention = minContention;
            s_workloads[i].maxContention = maxContention;
            s_workloads[i].readyCount = 0;

            spurs_wkl_pm_forget(i);   /* a reused wid must resolve its own PM */
            /* ...and must not inherit the previous workload's kicks. */
            spu_lockline_lock();
            vm_base[spurs_ea + spurs_off_ready(i)] = 0;
            ef_w16(spurs_ea + spurs_off_signal(i),
                   (u16)(ef_r16(spurs_ea + spurs_off_signal(i)) & ~spurs_sig_bit(i)));
            ef_line_end(spurs_ea + spurs_off_ready(i));
            spurs_nib_set(spurs_ea, SPURS_WKL_CURCONT, i, 0);

            /* No priority table: every configured lane may run it, published
             * as priority 1 (0 would read as "never on this SPU" to a real
             * kernel in LLE mode). For the kernel threads below this only moves
             * such a workload from the end of a SPU's visiting order (0 sorts
             * last) up among the priority-1 workloads. */
            if (priority_h)
                memcpy(s_workloads[i].priority, priority_h, CELL_SPURS_MAX_SPU);
            else
                memset(s_workloads[i].priority, 1, CELL_SPURS_MAX_SPU);
            fprintf(stderr, "[cellSpurs] wid=%u priority per SPU: %u %u %u %u %u %u %u %u (1 = highest, 0 = never), "
                    "contention min %u max %u\n",
                    i, s_workloads[i].priority[0], s_workloads[i].priority[1], s_workloads[i].priority[2],
                    s_workloads[i].priority[3], s_workloads[i].priority[4], s_workloads[i].priority[5],
                    s_workloads[i].priority[6], s_workloads[i].priority[7],
                    (unsigned)minContention, (unsigned)maxContention);

            /* Publish the workload in the REAL BE instance so the game's
             * inlined kernel protocol (readyCount stores, signal bits, state
             * reads) and the policy module's own instance DMAs see it. */
            u32 info = spurs_ea + spurs_off_info(i);
            vm_write64(info + 0x00, (u64)(uintptr_t)pm);        /* addr */
            vm_write64(info + 0x08, data);                       /* arg  */
            vm_write32(info + 0x10, sizePm);                     /* size */
            vm_write32(info + 0x14, i << 24);                    /* uniqueId */
            for (int b = 0; b < 8; b++)
                *(vm_base + info + 0x18 + b) = priority_h ? priority_h[b] : 1;
            *(vm_base + spurs_ea + spurs_off_state(i)) = 2;    /* runnable */
            if (i < 16)
                *(vm_base + spurs_ea + SPURS_WKL_MINCONT + i) = (u8)(minContention ? minContention : 1);
            spurs_nib_set(spurs_ea, SPURS_WKL_MAXCONT, i, (u8)(maxContention ? maxContention : 1));
            vm_write32(spurs_ea + SPURS_WKL_ENABLED,
                       vm_read32(spurs_ea + SPURS_WKL_ENABLED) | (0x80000000u >> i));
            *(vm_base + spurs_ea + SPURS_SYSSRV_MSG) = 0xFF;

            /* wid out-param is guest BE */
            vm_write32((u32)(uintptr_t)wid, i);
            printf("[cellSpurs] AddWorkload(wid=%u, pm=%p, size=%u)\n",
                   i, pm, sizePm);
            return CELL_OK;
        }
    }

    return CELL_SPURS_CORE_ERROR_NOMEM;
}

/* Real (BE) CellSpursWorkloadAttribute offsets (libspurs layout; the game's
 * inlined SDK code writes the struct directly in guest memory, so it must be
 * read back big-endian at these offsets -- never through a native host struct
 * (see the BIG-ENDIAN WARNING in spurs_taskset.h). */
enum {
    WKATTR_REVISION   = 0x00,   /* be u32 */
    WKATTR_SDKVERSION = 0x04,   /* be u32 */
    WKATTR_PM         = 0x08,   /* be u32: policy-module image EA */
    WKATTR_SIZE       = 0x0C,   /* be u32: policy-module size */
    WKATTR_DATA       = 0x10,   /* be u64: workload data (jobchain/queue EA) */
    WKATTR_PRIORITY   = 0x18,   /* u8[8] */
    WKATTR_MIN_CONT   = 0x20,   /* be u32 */
    WKATTR_MAX_CONT   = 0x24,   /* be u32 */
    WKATTR_NAME_CLASS = 0x28,   /* be u32: char* EA */
    WKATTR_NAME_INST  = 0x2C,   /* be u32: char* EA */
    WKATTR_HOOK       = 0x30,   /* be u32 */
    WKATTR_HOOK_ARG   = 0x34,   /* be u32 */
};

static volatile u32 s_pmwatch_ea = 0, s_pmwatch_sz = 0;
static DWORD WINAPI pm_write_watch(LPVOID unused)
{
    (void)unused;
    u32 ea = s_pmwatch_ea, sz = s_pmwatch_sz;
    u32 last_exe = vm_read32(ea + 0xAF4), last_hot = vm_read32(ea + 0x7E4); /* 0x14f4, 0x11e4 */
    for (int t = 0; t < 6000; t++) {          /* ~60s at 10ms */
        u32 exe = vm_read32(ea + 0xAF4), hot = vm_read32(ea + 0x7E4);
        if ((exe != last_exe && exe != 0) || (hot != last_hot && hot != 0xFFFFFFFFu)) {
            fprintf(stderr, "[pm-watch] PM ASSEMBLED at ~%dms: exe@0x14f4 %08X->%08X hot@0x11e4 %08X->%08X\n",
                    t * 10, last_exe, exe, last_hot, hot);
            FILE* pf = fopen("lbp_spu/pm_wwsjob_complete.bin", "wb");
            if (pf) { fwrite(vm_base + ea, 1, sz, pf); fclose(pf);
                fprintf(stderr, "[pm-watch] wrote lbp_spu/pm_wwsjob_complete.bin (0x%X bytes)\n", sz); }
            return 0;
        }
        Sleep(10);
    }
    fprintf(stderr, "[pm-watch] PM NEVER assembled in 60s (exe@0x14f4=%08X hot@0x11e4=%08X) "
            "-> assembly is HLE'd/external\n", vm_read32(ea + 0xAF4), vm_read32(ea + 0x7E4));
    return 0;
}

s32 cellSpursAddWorkloadWithAttribute(CellSpurs* spurs,
                                       CellSpursWorkloadId* wid,
                                       const CellSpursWorkloadAttribute* attr)
{
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    uint32_t attr_ea = (uint32_t)(uintptr_t)attr;

    /* Decode the REAL BE attribute from guest memory. */
    u32 pm_ea = vm_read32(attr_ea + WKATTR_PM);
    u32 pm_sz = vm_read32(attr_ea + WKATTR_SIZE);
    u64 data  = vm_read64(attr_ea + WKATTR_DATA);
    u32 minc  = vm_read32(attr_ea + WKATTR_MIN_CONT);
    u32 maxc  = vm_read32(attr_ea + WKATTR_MAX_CONT);
    u32 nmcls = vm_read32(attr_ea + WKATTR_NAME_CLASS);
    u32 nmins = vm_read32(attr_ea + WKATTR_NAME_INST);

    {   /* Layout ground truth: dump the raw attr words for the first few calls
         * (if the decode above prints nonsense, these bytes are the arbiter). */
        static int _n = 0;
        if (_n < 3) {
            printf("[cellSpurs] AddWorkloadWA attr=0x%08X raw:", attr_ea);
            for (int o = 0; o < 0x40; o += 4) {
                if ((o & 15) == 0) printf("\n    +%02X:", o);
                printf(" %08X", vm_read32(attr_ea + o));
            }
            printf("\n");
        } else if (_n == 3) {
            printf("[cellSpurs] AddWorkloadWA raw dumps suppressed from here\n");
        }
        printf("[cellSpurs] AddWorkloadWA: pm=0x%08X size=%u data=0x%016llX minC=%u maxC=%u name=%s/%s\n",
               pm_ea, pm_sz, (unsigned long long)data, minc, maxc,
               nmcls ? (const char*)(vm_base + nmcls) : "-",
               nmins ? (const char*)(vm_base + nmins) : "-");
        if (_n == 0 && pm_ea && pm_ea < 0x10000000u) {
            printf("[cellSpurs] PM@0x%08X first 96B:", pm_ea);
            for (int o = 0; o < 96; o += 4) {
                if ((o & 15) == 0) printf("\n    +%02X:", o);
                printf(" %08X", vm_read32(pm_ea + o));
            }
            printf("\n");
        }
        /* PM-COMPLETENESS PROBE (SPURS_PM_DUMP): the wwsjob job-manager PM is
         * ASSEMBLED at runtime -- the embedded ELF (LS 0xA00) has zero HOLES at
         * LS 0xAEE..0x11B0 and the executeStage lives at LS 0x14f4, both filled
         * by SPURS setup. Report whether OUR runtime's PM has that code or the
         * holes, and (once) write the whole image out so we can re-lift it. */
        if (getenv("SPURS_PM_DUMP") && pm_ea && pm_sz >= 0x2200 && pm_sz <= 0x4000) {
            u32 exe = vm_read32(pm_ea + 0xAF4);          /* LS 0x14f4 executeStage */
            int zeros = 0; for (u32 o = 0xEE; o < 0x7B0; o += 4)
                if (vm_read32(pm_ea + o) == 0) zeros += 4;
            fprintf(stderr, "[pm-dump] wid-PM ea=0x%08X sz=0x%X exe@+0xAF4=%08X (want 24F880ED) "
                    "holeZeros=%d/1730 -> %s\n", pm_ea, pm_sz, exe, zeros,
                    (exe == 0x24F880EDu ? "COMPLETE (we assemble it)"
                                        : zeros > 1000 ? "HOLEY (assembly skipped)" : "PARTIAL"));
            static int _dumped = 0;
            if (!_dumped) { _dumped = 1;
                FILE* pf = fopen("lbp_spu/pm_wwsjob_runtime.bin", "wb");
                if (pf) { fwrite(vm_base + pm_ea, 1, pm_sz, pf); fclose(pf);
                    fprintf(stderr, "[pm-dump] wrote lbp_spu/pm_wwsjob_runtime.bin (0x%X bytes)\n", pm_sz); }
                /* WRITE-WATCH: spawn a poller that reports if/when the PM's
                 * placeholder hot-loop (LS 0x11e4, currently 0xFFFFFFFF) and
                 * executeStage (LS 0x14f4, currently 0) get real code written by
                 * PPU code at runtime -> tells us whether our recompiled game
                 * assembles the PM (dump it) or the assembly is HLE'd away. */
                s_pmwatch_ea = pm_ea; s_pmwatch_sz = pm_sz;
                CreateThread(NULL, 1u << 18, pm_write_watch, NULL, 0, NULL);
            }
        }
        _n++;
    }

    /* Forward the BE-decoded values (priority as guest EA of the 8-byte table). */
    return cellSpursAddWorkload(spurs, wid, (const void*)(uintptr_t)pm_ea,
                               pm_sz, data,
                               (const u8*)(uintptr_t)(attr_ea + WKATTR_PRIORITY),
                               minc, maxc);
}

/* The SDK-versioned workload-attribute initializer (the import the game links;
 * NID differs from the non-underscore inline wrapper). Args arrive raw in
 * r3..r10; writes the REAL BE layout so AddWorkloadWithAttribute round-trips. */
s32 _cellSpursWorkloadAttributeInitialize(u64 attr_ea, u32 revision, u32 sdkVersion,
                                          u64 pm_ea, u32 size, u64 data,
                                          u64 prio_ea, u32 minContention)
{
    if (!attr_ea) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + WKATTR_REVISION,   revision);
    vm_write32((u32)attr_ea + WKATTR_SDKVERSION, sdkVersion);
    vm_write32((u32)attr_ea + WKATTR_PM,         (u32)pm_ea);
    vm_write32((u32)attr_ea + WKATTR_SIZE,       size);
    vm_write64((u32)attr_ea + WKATTR_DATA,       data);
    for (int i = 0; i < 8; i++)
        *(vm_base + (u32)attr_ea + WKATTR_PRIORITY + i) =
            prio_ea ? *(vm_base + (u32)prio_ea + i) : 0;
    vm_write32((u32)attr_ea + WKATTR_MIN_CONT, minContention);
    vm_write32((u32)attr_ea + WKATTR_MAX_CONT, 1);   /* 9th arg is beyond the 8-GPR adapter */
    vm_write32((u32)attr_ea + WKATTR_NAME_CLASS, 0);
    vm_write32((u32)attr_ea + WKATTR_NAME_INST,  0);
    vm_write32((u32)attr_ea + WKATTR_HOOK,     0);
    vm_write32((u32)attr_ea + WKATTR_HOOK_ARG, 0);
    printf("[cellSpurs] _WorkloadAttributeInitialize(attr=0x%08X pm=0x%08X size=%u data=0x%llX minC=%u)\n",
           (u32)attr_ea, (u32)pm_ea, size, (unsigned long long)data, minContention);
    return CELL_OK;
}

s32 cellSpursWorkloadAttributeSetName(u64 attr_ea, u64 nameClass_ea, u64 nameInstance_ea)
{
    if (!attr_ea) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + WKATTR_NAME_CLASS, (u32)nameClass_ea);
    vm_write32((u32)attr_ea + WKATTR_NAME_INST,  (u32)nameInstance_ea);
    printf("[cellSpurs] WorkloadAttributeSetName(attr=0x%08X, \"%s\", \"%s\")\n",
           (u32)attr_ea,
           nameClass_ea ? (const char*)(vm_base + (u32)nameClass_ea) : "-",
           nameInstance_ea ? (const char*)(vm_base + (u32)nameInstance_ea) : "-");
    return CELL_OK;
}

s32 cellSpursRemoveWorkload(CellSpurs* spurs, CellSpursWorkloadId wid)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= SPURS_MAXW) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!spurs_wid_owned(spurs, wid)) return CELL_SPURS_CORE_ERROR_SRCH;
    const u32 ea = (u32)(uintptr_t)spurs;

    /* Removing a workload whose module is running (or whose resident policy
     * context is still referenced) is BUSY; otherwise retire the context too,
     * so a later registration under this wid starts fresh. */
    if (s_wid_busy[wid] || spurs_policy_release_workload(ea, wid))
        return CELL_SPURS_CORE_ERROR_BUSY;
    s_workloads[wid].in_use = 0;
    *(vm_base + ea + spurs_off_state(wid)) = 3;          /* removable */
    vm_write32(ea + SPURS_WKL_ENABLED, vm_read32(ea + SPURS_WKL_ENABLED) & ~(0x80000000u >> wid));
    spurs_wkl_pm_forget(wid);
    printf("[cellSpurs] RemoveWorkload(wid=%u)\n", wid);
    return CELL_OK;
}

s32 cellSpursWorkloadAttributeInitialize(CellSpursWorkloadAttribute* attr,
                                         u32 revision, u32 sdkVersion,
                                         const void* pm, u32 sizePm,
                                         u64 data, const u8* priority,
                                         u32 minContention,
                                         u32 maxContention)
{
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursWorkloadAttribute*);
    const u8* priority_h = GUEST_PTR(priority, const u8*);

    memset(attr, 0, sizeof(CellSpursWorkloadAttribute));
    attr->revision = revision;
    attr->sdkVersion = sdkVersion;
    attr->pm = (u64)(uintptr_t)pm;   /* pm kept as guest EA */
    attr->sizePm = sizePm;
    attr->data = data;
    attr->minContention = minContention;
    attr->maxContention = maxContention;

    if (priority_h)
        memcpy(attr->priority, priority_h, CELL_SPURS_MAX_SPU);

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * The SPURS "kernel": one host poll thread per instance (the virtual SPU).
 *
 * The game kicks work by storing a nonzero wklReadyCount1[wid] byte or a
 * wklSignal1 bit into the instance — mostly with INLINED atomics (LBP never
 * calls an API for it beyond cellSpursReadyCountStore). The kernel thread
 * polls those real BE fields, consumes one ready unit (decrement / clear the
 * signal bit, like the real kernel's dispatch), and runs the workload's
 * policy module to completion via spu_run_policy_module.
 * -----------------------------------------------------------------------*/
typedef struct {
    spu_lifted_entry_fn fn;
    int                 image_id;
    int                 resolved;   /* 0=not tried, 1=found, -1=missing */
} WklPm;
static WklPm s_wkl_pm[SPURS_MAXW];

static void spurs_wkl_pm_forget(u32 wid)
{
    if (wid < SPURS_MAXW) memset(&s_wkl_pm[wid], 0, sizeof(s_wkl_pm[wid]));
}

static void spurs_retire_workloads(u32 ea)
{
    for (u32 wid = 0; wid < SPURS_MAXW; wid++)
        if (s_workloads[wid].in_use && s_workloads[wid].spurs_ea == ea) {
            memset(&s_workloads[wid], 0, sizeof(s_workloads[wid]));
            spurs_wkl_pm_forget(wid);
        }
}

static WklPm* spurs_resolve_pm(u32 wid)
{
    WklPm* r = &s_wkl_pm[wid];
    if (r->resolved) return r->resolved > 0 ? r : NULL;
    SpursWorkload* w = &s_workloads[wid];
    uint64_t fp = spu_workload_fingerprint(vm_base + (uint32_t)(uintptr_t)w->pm,
                                           w->sizePm);
    r->fn = spu_workload_find_img(fp, &r->image_id);
    if (!r->fn) {
        /* No lifted image registered for this policy module: interpret it. Set
         * SPURS_PM_INTERP=0 to restore the old "workload will not run" miss. */
        extern void spu_pm_interp_entry(spu_context*);
        const char* ie = getenv("SPURS_PM_INTERP");
        if (!ie || ie[0] != '0') {
            r->fn = spu_pm_interp_entry;
            r->image_id = -1;                 /* pure interpretation */
            printf("[cellSpurs] wid=%u PM not lifted (fp=0x%016llX size=%u): INTERPRETING\n",
                   wid, (unsigned long long)fp, w->sizePm);
            r->resolved = 1;
            return r;
        }
    }
    r->resolved = r->fn ? 1 : -1;
    if (r->fn)
        printf("[cellSpurs] wid=%u PM resolved (fp=0x%016llX image=%d)\n",
               wid, (unsigned long long)fp, r->image_id);
    else {
        printf("[cellSpurs] wid=%u PM NOT LIFTED (fp=0x%016llX size=%u) -- workload will not run\n",
               wid, (unsigned long long)fp, w->sizePm);
        /* SPU_DUMP_MISS=<dir>: write the PM exactly as fingerprinted, like the
         * two miss paths in spu_workload.c. This path refused the workload
         * before either of those ran, so an unlifted PM was never captured. */
        const char* dir = getenv("SPU_DUMP_MISS");
        if (dir && *dir) {
            char path[512];
            snprintf(path, sizeof(path), "%s/spujob_%016llX_%u.bin",
                     dir, (unsigned long long)fp, w->sizePm);
            FILE* f = fopen(path, "wb");
            if (f) {
                fwrite(vm_base + (uint32_t)(uintptr_t)w->pm, 1, w->sizePm, f);
                fclose(f);
                printf("[cellSpurs] wid=%u PM dumped to %s\n", wid, path);
            }
        }
    }
    return r->fn ? r : NULL;
}

/* Thread creation, waiting and Sleep come from runtime/platform/win32_compat.h
 * off Windows (included at the top of this file), which is what lets the three
 * thread bodies below stand as written on every host. SetThreadStackGuarantee
 * has no POSIX counterpart -- it reserves stack for the stack-overflow
 * exception handler, and there is no such handler here -- so it stays behind
 * an _WIN32 guard where it is used.
 *
 * One virtual SPU running a workload's policy module. The WWS job manager
 * runs concurrently across N SPUs (RPCS3: jobmanagerCellSpursKernel0..N): each
 * claims jobs from the shared queue and advances its own lane of the sync
 * barrier. Running the SPUs SEQUENTIALLY deadlocks -- SPU 0 completes its job,
 * then busy-waits at the cross-SPU barrier for lane 1, which the sequential
 * loop can never advance (SPU 1 hasn't run). Concurrency is required, not an
 * optimization: SPU 0's barrier poll observes SPU 1's atomic lane update live. */
struct spurs_pm_worker_arg {
    spu_lifted_entry_fn fn; int image_id;
    const uint8_t* pm; uint32_t pm_size;
    uint64_t arg; uint32_t wid, ea, spu_num;
};
static DWORD WINAPI spurs_pm_worker(LPVOID p)
{
#ifdef _WIN32
    { ULONG g = 256 * 1024; SetThreadStackGuarantee(&g); }
#endif
    struct spurs_pm_worker_arg* a = (struct spurs_pm_worker_arg*)p;
    spu_run_policy_module(a->fn, a->image_id, a->pm, a->pm_size,
                          a->arg, a->wid, a->ea, a->spu_num);
    return 0;
}

/* The SPU this host kernel thread plays (policy modules run on it synchronously). */
static SPU_THREAD_LOCAL int t_kern_spu = -1;

/* SPURS_POLL_SELECT=1: answer cellSpursModulePoll (selectWorkload with isPoll set)
 * with a selection instead of 0. The policy module keeps the SPU while the answer is
 * its own wid; the WWS job manager polls after every job (its decrementer slice is
 * ~70 us, shorter than one EDGE job), so answering 0 ("switch") made every job a
 * whole kernel round trip. Here the caller keeps running while its readyCount is
 * nonzero and no workload this SPU ranks higher is ready and unclaimed; otherwise
 * the higher one is named (the module exits and the kernel pass runs it), and with
 * nothing ready 0x20, the system-service id (exit). */
int spurs_poll_select_on(void)
{
    static int s_on = -1;
    if (s_on < 0) s_on = getenv("SPURS_POLL_SELECT") ? 1 : 0;
    return s_on;
}

/* For the [pm-hang] report: a wid's live readyCount (<< 8) and current contention. */
unsigned spurs_wid_ready_cont(u32 ea, u32 wid)
{
    if (!ea || wid >= SPURS_MAXW) return 0;
    return ((unsigned)*(vm_base + ea + spurs_off_ready(wid)) << 8) | spurs_nib_get(ea, SPURS_WKL_CURCONT, wid);
}

/* A workload this kernel thread's SPU ranks above wid that is ready (or signaled)
 * and unclaimed, else wid. */
static u32 spurs_higher_ready(struct SpursInst* si, u32 ea, u32 wid, int spu)
{
    const u32 own = s_workloads[wid].priority[spu] ? s_workloads[wid].priority[spu] : 16u;
    const u32 enabled = vm_read32(ea + SPURS_WKL_ENABLED);
    const u32 nw = si->is2 ? SPURS_MAXW : 16u;
    for (u32 w = 0; w < nw; w++) {
        if (w == wid || !(enabled & (0x80000000u >> w))) continue;
        if (*(vm_base + ea + spurs_off_state(w)) != 2) continue;
        if (!s_workloads[w].in_use || s_workloads[w].spurs_ea != ea) continue;
        const u32 pr = s_workloads[w].priority[spu];
        if (!pr || pr >= own || s_wid_busy[w]) continue;
        const u32 sig = vm_read32(ea + spurs_off_signal(w)) >> 16;
        if (*(vm_base + ea + spurs_off_ready(w)) || (sig & spurs_sig_bit(w))) return w;
    }
    return wid;
}

u32 spurs_poll_select(u32 ea, u32 wid)
{
    struct SpursInst* si = spurs_inst_find(ea);
    const int spu = t_kern_spu;
    if (!si || wid >= SPURS_MAXW || spu < 0 || spu >= CELL_SPURS_MAX_SPU) return 0x20;
    if (*(vm_base + ea + spurs_off_ready(wid)) == 0) return 0x20;
    return spurs_higher_ready(si, ea, wid, spu);
}

/* SPURS_LANES="<wid>[,<wid>...]=<n>[;...]": let up to n kernel threads run that workload's
 * policy module at once, each as the SPU it plays (spuNum = that SPU), as the real kernel
 * does for a workload whose maxContention allows it. Beyond the registered contention: a
 * title that registered max 1 never had two SPUs in the module on hardware. Default off. */
static int spurs_lanes_for(u32 wid)
{
    static int s_init; static u8 s_n[SPURS_MAXW];
    if (!s_init) {
        for (const char* e = getenv("SPURS_LANES"); e && *e; ) {
            u32 ws[SPURS_MAXW]; int nw = 0;
            while (*e && *e != '=' && *e != ';') {
                int w = atoi(e);
                if (w >= 0 && w < SPURS_MAXW && nw < SPURS_MAXW) ws[nw++] = (u32)w;
                while (*e && *e != ',' && *e != '=' && *e != ';') e++;
                if (*e == ',') e++;
            }
            int n = (*e == '=') ? atoi(++e) : 0;
            if (n > 6) n = 6;
            for (int i = 0; i < nw; i++) s_n[ws[i]] = (u8)(n > 0 ? n : 0);
            while (*e && *e != ';') e++;
            if (*e == ';') e++;
        }
        s_init = 1;
    }
    return wid < SPURS_MAXW ? s_n[wid] : 0;
}

static DWORD WINAPI spurs_kernel_thread(LPVOID p)
{
#ifdef _WIN32
    { ULONG g = 256 * 1024; SetThreadStackGuarantee(&g); }  /* let SO reach the reporter */
#endif
    struct SpursInst* si = ((struct spurs_kern_arg*)p)->si;
    const int my_spu = ((struct spurs_kern_arg*)p)->spu;
    t_kern_spu = my_spu;
    static volatile long s_pm_off = -1;
    if (s_pm_off < 0) s_pm_off = getenv("PS3_NO_SPURS_PM") ? 1 : 0;

    fprintf(stderr, "[spurs-kern] \"%s\" poll thread live: ea=0x%08X pm=%s\n",
            si->prefix, si->ea, s_pm_off ? "DISABLED (PS3_NO_SPURS_PM)" : "enabled");
    fflush(stderr);

    /* Instance change detector (SPURS_KERN_WATCH=1).
     *
     * We assumed the title kicks a workload by poking wklReadyCount/wklSignal
     * with inlined atomics -- but this thread watches exactly those bytes every
     * 1 ms and has never once seen them nonzero, across whole runs. Rather than
     * guess again, shadow the head of the instance and report EVERY byte the
     * title changes. Whatever the real kick is, it has to land in here. */
    static const u32 WATCH_LEN = 0xC0;
    unsigned char shadow[0xC0];
    int shadow_primed = 0;
    int watch = getenv("SPURS_KERN_WATCH") ? 1 : 0;
    int changes_logged = 0;

    /* SPURS_PASS_SLEEP=<ms> (default 1): pause between passes. SPURS_PASS_GREEDY=1
     * skips it right after a pass in which some module found work, so dependent jobs
     * (one workload's output is another's input) do not each wait a timer tick. */
    int pass_sleep = 1, pass_greedy = getenv("SPURS_PASS_GREEDY") ? 1 : 0;
    { const char* e = getenv("SPURS_PASS_SLEEP"); if (e) pass_sleep = atoi(e); }
    /* SPURS_PASS_SLEEP_US=<us> (default 0, off): the pause is <us> on a
     * high-resolution waitable timer instead. Sleep(1) lasts ~1.5-2 ms even at
     * 1 ms timer resolution, and a job the PPU queues just after an empty pass
     * waits that long for a kernel thread (GoW:A: each kernel thread naps ~33%
     * of the time while the main thread spins on its job lists). */
    int pass_sleep_us = 0;
    { const char* e = getenv("SPURS_PASS_SLEEP_US"); if (e) pass_sleep_us = atoi(e); }
#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
    HANDLE pass_timer = pass_sleep_us > 0
        ? CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS) : NULL;
    if (!pass_timer) pass_sleep_us = 0;
#endif
    if (pass_sleep_us > 0) pass_sleep = 1;
    int pass_found_work = 0;
    for (;;) {
        if (!(pass_greedy && pass_found_work) && pass_sleep > 0) {
            /* PS3_FTRACE: the pause between passes, on the thread's SPU row. */
            const uint64_t ft_p0 = PS3_FT_ON() ? ps3_ft_now() : 0;
            if (pass_sleep_us > 0) {
#ifdef _WIN32
                LARGE_INTEGER due; due.QuadPart = -((LONGLONG)pass_sleep_us * 10);
                if (SetWaitableTimer(pass_timer, &due, 0, NULL, NULL, FALSE))
                    WaitForSingleObject(pass_timer, INFINITE);
#else
                usleep((useconds_t)pass_sleep_us);
#endif
            } else
                Sleep((DWORD)pass_sleep);
            if (ft_p0) {
                extern int spurs_ft_kidx(void);
                const uint64_t ft_p1 = ps3_ft_now();
                if (ft_p1 - ft_p0 >= ps3_ft_min_ns())
                    ps3_ft_span(PS3_FT_SPU_WAIT, (uint32_t)spurs_ft_kidx(), 0x51EEu, 0, ft_p0, ft_p1);
            }
        }
        pass_found_work = 0;
        u32 ea = si->ea;
        if (!ea || s_pm_off) continue;

        if (watch) {
            const unsigned char* live = (const unsigned char*)vm_base + ea;
            if (!shadow_primed) { memcpy(shadow, live, WATCH_LEN); shadow_primed = 1; }
            else if (memcmp(shadow, live, WATCH_LEN) != 0) {
                for (u32 o = 0; o < WATCH_LEN; o++) {
                    if (shadow[o] == live[o]) continue;
                    if (changes_logged < 200) {
                        changes_logged++;
                        fprintf(stderr, "[spurs-kern] \"%s\" INSTANCE +0x%02X: %02X -> %02X%s\n",
                                si->prefix, o, shadow[o], live[o],
                                o < 16                    ? "  (wklReadyCount1)" :
                                o >= 0x70 && o < 0x76     ? "  (wklSignal1)"     :
                                o >= 0x80 && o < 0x90     ? "  (wklState1)"      :
                                o >= 0xB0 && o < 0xB4     ? "  (wklEnabled)"     : "");
                    }
                }
                memcpy(shadow, live, WATCH_LEN);
                fflush(stderr);
            }
        }

        u32 enabled = vm_read32(ea + SPURS_WKL_ENABLED);

        /* Kick visibility (can't-miss): log ANY nonzero readyCount/signal state
         * even for wids the dispatch filter below would skip — the game pokes
         * these bytes with inlined atomics and this is our only tap. */
        {
            static int _seen[MAX_SPURS_INST][SPURS_MAXW];
            int slot = (int)(si - s_inst);
            for (u32 w = 0; w < (si->is2 ? SPURS_MAXW : 16u); w++) {
                u32 sig = vm_read32(ea + spurs_off_signal(w)) >> 16;
                u8 rc = *(vm_base + ea + spurs_off_ready(w));
                if ((rc || (sig & spurs_sig_bit(w))) && _seen[slot][w] < 4) {
                    _seen[slot][w]++;
                    fprintf(stderr, "[spurs-kern] \"%s\" POKE wid=%u ready=%u sig=%u enabled=%d state=%u\n",
                            si->prefix, w, rc, (sig & spurs_sig_bit(w)) ? 1 : 0,
                            (enabled >> (31 - w)) & 1,
                            *(vm_base + ea + spurs_off_state(w)));
                }
            }
        }
        if (!enabled) continue;

        /* Visit workloads in this SPU's priority order (1 = highest, 0 = never
         * here -> last), as the real kernel picks the best ready workload: the
         * thread playing SPU 5 serves BrBDblBufWrkld before anything else. */
        const u32 nw = si->is2 ? SPURS_MAXW : 16u;
        u32 order[SPURS_MAXW];
        for (u32 i = 0; i < nw; i++) {
            u32 key = (my_spu < CELL_SPURS_MAX_SPU && s_workloads[i].priority[my_spu])
                          ? s_workloads[i].priority[my_spu] : 16u;
            u32 j = i;
            while (j > 0) {
                u32 pw = order[j - 1];
                u32 pk = (my_spu < CELL_SPURS_MAX_SPU && s_workloads[pw].priority[my_spu])
                             ? s_workloads[pw].priority[my_spu] : 16u;
                if (pk <= key) break;
                order[j] = pw; j--;
            }
            order[j] = i;
        }
        for (u32 oi = 0; oi < nw; oi++) {
            const u32 wid = order[oi];
            if (!(enabled & (0x80000000u >> wid))) continue;
            if (*(vm_base + ea + spurs_off_state(wid)) != 2) continue;
            if (!s_workloads[wid].in_use || s_workloads[wid].spurs_ea != ea) continue;

            /* A SPURS policy module is a PERSISTENT SPU program. The real kernel
             * schedules an ENABLED, runnable workload onto an SPU and the module
             * then polls its OWN job queue in main memory; being enabled is the
             * trigger, not a per-job kick. That is why the title this was written
             * for never wrote wklReadyCount, never set a signal bit, and never
             * called cellSpursReadyCountStore -- on hardware it does not have to
             * (God of War: Ascension does write readyCount: "POKE wid=17 ready=4"). Gating
             * dispatch on a kick meant the module never ran at all, so nothing
             * ever called cellSpursEventFlagSet and the title's loading thread
             * blocked forever.
             *
             * Run one scheduling quantum per enabled workload per pass: the
             * module does its work, exits to the kernel (LS 0x9C0) when it has
             * none, and we re-enter it on the next pass -- which is exactly what
             * the real kernel's dispatch loop does.
             *
             * readyCount/wklSignal are still honoured when a title DOES use them:
             * consume one unit so a kick-driven title paces the same as before. */
            /* Per-SPU priority 0 = this workload never runs on this SPU (GoW:A's
             * BrBDblBufWrkld runs on SPU 5 only). Skip it here as long as another
             * of this instance's kernel threads may run it -- BEFORE touching its
             * readyCount or signal bit: a thread that will not run the workload
             * must not consume its kick. (With SPURS_KERN_SPUS=0,1,5 the SPU 5
             * thread used to take one unit of wid 0's readyCount on every pass and
             * then skip it; that was the only kernel-side difference from the
             * configurations that never corrupted guest state.) */
            if (si->nkern > 1 && my_spu < CELL_SPURS_MAX_SPU && !s_workloads[wid].priority[my_spu]) {
                int elsewhere = 0;
                for (int q = 0; q < si->nkern && q < 8; q++)
                    if (si->kspu[q] < CELL_SPURS_MAX_SPU && s_workloads[wid].priority[si->kspu[q]]) {
                        elsewhere = 1; break; }
                if (elsewhere) continue;
            }
            volatile u8* rdy = vm_base + ea + spurs_off_ready(wid);
            u32 sigoff = spurs_off_signal(wid);                   /* be u16 @0x70 / @0x74 */
            u32 sig = vm_read32(ea + sigoff) >> 16;
            int kicked = (*rdy != 0) || ((sig & spurs_sig_bit(wid)) != 0);
            /* SPURS_KEEP_READY=1: leave readyCount alone (the real kernel never consumes
             * it; the module/job manager lowers it itself). The per-pass decrement was
             * a pacing heuristic for a kick-driven title. */
            { static int s_kr = -1; if (s_kr < 0) s_kr = getenv("SPURS_KEEP_READY") ? 1 : 0;
              if (*rdy && !s_kr) (*rdy)--; }
            if (sig & spurs_sig_bit(wid))
                vm_write32(ea + sigoff, (vm_read32(ea + sigoff) & ~(spurs_sig_bit(wid) << 16)));

            WklPm* r = spurs_resolve_pm(wid);
            if (!r) continue;

            /* Idle backoff: running EVERY enabled workload's module EVERY 1ms
             * pass (x N instance threads) burned ~5 host cores on modules that
             * immediately exit-to-kernel with no work, starving the actual
             * decode/render threads (LBP movie at ~1fps while 500% CPU).
             * A module that keeps finding nothing gets re-run every 2nd, 4th,
             * ... up to 16th pass; an explicit kick (readyCount/signal) resets
             * it to every pass, so kick-driven latency is unchanged. */
            {
                static u8 s_idle[SPURS_MAXW];      /* idle streak (log2 cadence) */
                static u32 s_pass_no;                            /* shared pass counter is fine */
                if (wid == 0) s_pass_no++;
                if (kicked) s_idle[wid] = 0;
                /* SPURS_IDLE_MAX=<n>: cap of the idle cadence (log2, default 4 = every
                 * 16th pass); 0 runs every enabled module every pass. */
                static int s_imax = -1;
                if (s_imax < 0) { const char* e = getenv("SPURS_IDLE_MAX"); s_imax = e ? atoi(e) : 4; }
                u32 cad = 1u << (s_idle[wid] > s_imax ? s_imax : s_idle[wid]);
                if (!kicked && (s_pass_no & (cad - 1)) != 0) continue;
                /* One kernel thread per instance SPU (SPURS_KERN_THREADS): a workload's
                 * module must never run on two of them at once (one persistent PM
                 * context per lane). Claim it; another thread skips it this pass.
                 * A SPURS_LANES workload may be claimed by up to n threads at once,
                 * each running it as the SPU it plays (its own spuNum, so its own
                 * PM context and per-SPU guest state). */
                const int lanes = spurs_lanes_for(wid);
                /* Only as an SPU the workload has a priority on: the title sets up
                 * per-SPU state (GoW:A's EDGE output rings) for those SPUs only. */
                const int lane_ok = lanes > 1 && my_spu >= 0 && my_spu < 6 &&
                                    (!si->nspus || my_spu < si->nspus) &&
                                    s_workloads[wid].priority[my_spu] != 0;
                if (lane_ok) {
                    if (InterlockedIncrement(&s_wid_busy[wid]) > lanes) {
                        InterlockedDecrement(&s_wid_busy[wid]); continue; }
                } else if (lanes > 1) {
                    continue;   /* this thread has no spuNum of its own to run it as */
                } else if (InterlockedCompareExchange(&s_wid_busy[wid], 1, 0) != 0) continue;
                const u32 run_spu = lane_ok ? (u32)my_spu : 0;
                /* SPURS_EDGE_DIAG=<s>: every s seconds, for dispatches of the
                 * wids named by SPURS_EDGE_WIDS (default GoW:A's EDGE0-2,
                 * 15,16,17), how often another of them was busy or had its
                 * readyCount up at that moment: whether they could have run
                 * side by side. */
                { static int s_ed = -1; static u32 s_edm; static volatile LONG s_n, s_other_busy, s_other_ready;
                  static LONG64 s_t0;
                  if (s_ed < 0) {
                      const char* e = getenv("SPURS_EDGE_DIAG"); s_ed = e ? atoi(e) : 0;
                      const char* w = getenv("SPURS_EDGE_WIDS");
                      if (!w) w = "15,16,17";
                      while (*w) { int v = atoi(w); if (v >= 0 && v < 32) s_edm |= 1u << v;
                                   while (*w && *w != ',') w++; if (*w == ',') w++; }
                  }
                  if (s_ed > 0 && (s_edm >> wid & 1u)) {
                      InterlockedIncrement(&s_n);
                      int ob = 0, orr = 0;
                      for (u32 w2 = 0; w2 < 32; w2++) {
                          if (w2 == wid || !(s_edm >> w2 & 1u)) continue;
                          if (s_wid_busy[w2]) ob = 1;
                          if (*(vm_base + ea + spurs_off_ready(w2))) orr = 1;
                      }
                      if (ob) InterlockedIncrement(&s_other_busy);
                      if (orr) InterlockedIncrement(&s_other_ready);
                      LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
                      if (!s_t0) s_t0 = q.QuadPart;
                      else if (q.QuadPart - s_t0 > (LONG64)s_ed * f.QuadPart) {
                          s_t0 = q.QuadPart;
                          fprintf(stderr, "[edge-diag] %ld dispatches: another EDGE wid busy %ld%%, ready %ld%%\n",
                                  s_n, s_n ? 100 * s_other_busy / s_n : 0, s_n ? 100 * s_other_ready / s_n : 0);
                          s_n = s_other_busy = s_other_ready = 0;
                      }
                  } }
                extern SPU_THREAD_LOCAL volatile unsigned g_spurs_pm_polls;
                u32 polls_before = g_spurs_pm_polls;   /* heuristic only */
                (void)polls_before;

            {   static int _n = 0;
                if (_n < 8) { _n++;
                    fprintf(stderr, "[spurs-kern] \"%s\" dispatch wid=%u (enabled, state=2) "
                                    "image=%d ready=%u\n", si->prefix, wid, r->image_id, *rdy);
                    fflush(stderr); } }

            /* SPURS_PM_RERUN_US=<us> (0 = off): after a run that executed jobs, run
             * the same workload again at once (claim kept) until a run finds none
             * or <us> pass. The WWS policy module hands its SPU back after each job
             * (its selectWorkload poll answers "switch"), and paying a whole kernel
             * pass -- every other workload's module, BrBDblBufWrkld's 0.4 ms job
             * included -- between two jobs of one EDGE list made that list's
             * latency, not anyone's compute, the first level's frame time. */
            static long long s_rerun = -1;
            if (s_rerun < 0) { const char* e = getenv("SPURS_PM_RERUN_US");
                               LARGE_INTEGER f; QueryPerformanceFrequency(&f);
                               s_rerun = e ? (long long)f.QuadPart * atoll(e) / 1000000 : 0; }
            /* SPURS_PM_RERUN_WIDS="<wid>[,<wid>...]=<us>[;...]": a window of its own for
             * those workloads. GoW:A's RenJobSet list (wid 11) gates the frame's last
             * job list; with the shared 2 ms window it ran as ~9 visits a frame with
             * ~0.9 ms between them, waiting to be picked again. 10 ms for wids 11,12
             * shortened that chain by 5-6 ms but not the frame: it started later. */
            static volatile LONG s_rrw_init; static long long s_rrw[SPURS_MAXW];
            if (!s_rrw_init) {
                LARGE_INTEGER f; QueryPerformanceFrequency(&f);
                for (const char* e = getenv("SPURS_PM_RERUN_WIDS"); e && *e; ) {
                    u32 ws[SPURS_MAXW]; int nw = 0;
                    while (*e && *e != '=' && *e != ';') {
                        int w = atoi(e);
                        if (w >= 0 && w < SPURS_MAXW && nw < SPURS_MAXW) ws[nw++] = (u32)w;
                        while (*e && *e != ',' && *e != '=' && *e != ';') e++;
                        if (*e == ',') e++;
                    }
                    const long long us = (*e == '=') ? atoll(++e) : 0;
                    for (int i = 0; i < nw; i++) s_rrw[ws[i]] = (long long)f.QuadPart * us / 1000000;
                    while (*e && *e != ';') e++;
                    if (*e == ';') e++;
                }
                s_rrw_init = 1;
            }
            const long long rr_lim = s_rrw[wid] ? s_rrw[wid] : s_rerun;
            LARGE_INTEGER rr_t0; QueryPerformanceCounter(&rr_t0);
            int rr_found = 0;
            for (;;) {
            extern SPU_THREAD_LOCAL unsigned g_spu_job_entries;
            const unsigned jobs_before = g_spu_job_entries;
            /* Live workload arg from the real wklInfo (the game may update it). */
            u64 arg = vm_read64(ea + spurs_off_info(wid) + 8);
            /* Dispatch once per VIRTUAL SPU up to the workload's maxContention:
             * the WWS job manager keys its per-SPU ticket lane off the kernel
             * context's spuNum, and its command lists carry cross-lane BARRIER
             * commands -- with only spu 0 ever dispatched, lane 1 (pre-armed by
             * the PPU for a 2-SPU workload) never advanced and every barrier
             * deadlocked: LBP's post-intro loading froze with 8 ready jobs and
             * the lanes stuck at {1,0,...} against ticket 10+. Sequential
             * per-lane rounds converge where parallel SPUs would. */
            u8 maxcont = spurs_nib_get(ea, SPURS_WKL_MAXCONT, wid);
            if (maxcont < 1) maxcont = 1;
            if (maxcont > 6) maxcont = 6;
            /* SPURS_MAXCONT="<wid>=<n>,...": run those workloads on n virtual
             * SPUs (lanes) at once whatever their registered maxContention --
             * no more than the instance's nSpus (see the sync-row note below). */
            { static int s_mc_init; static u8 s_mc[SPURS_MAXW];
              if (!s_mc_init) {
                  const char* e = getenv("SPURS_MAXCONT");
                  while (e && *e) {
                      int w = atoi(e), n = 0;
                      while (*e && *e != '=' && *e != ',') e++;
                      if (*e == '=') { e++; n = atoi(e); }
                      if (w >= 0 && w < SPURS_MAXW && n > 0) s_mc[w] = (u8)(n > 6 ? 6 : n);
                      while (*e && *e != ',') e++;
                      if (*e == ',') e++;
                  }
                  s_mc_init = 1;
              }
              if (wid < SPURS_MAXW && s_mc[wid]) maxcont = s_mc[wid];
              if (maxcont > si->nspus && si->nspus) maxcont = (u8)si->nspus; }
            /* Do NOT exceed the workload's own contention: per-SPU rows in
             * the WWS sync struct live at +0x40+16*spuNum, and dispatching
             * spuNum >= nSpus made virtual SPU 3 write its bookkeeping row
             * OVER the game's ticket row (row 3) -- observed as ticket values
             * jumping to garbage (141028, 9960...) during the savedata load.
             * The earlier "4 lanes drains the queue" result was partly that
             * scribble. Correct progress comes from PERSISTENT PM contexts
             * (spurs_policy.c), not extra lanes. */
            /* SPURS_FORCE_SPUS=<n>: dispatch every workload for n virtual SPUs
             * regardless of maxContention (A/B: LBP publishes its loading
             * tickets on sync row 3, and the PM's row index = spuNum).
             * 2026-07-23 FINDING: with the completion HLE (LBP_HLE_JOBDONE) in
             * place, forcing 2+ lanes is now COUNTERPRODUCTIVE -- two host
             * threads race on the shared jobIndex atomic in WwsJob_AllocateJob
             * (GETLLAR 0xD0/PUTLLC 0xB4 on the joblist header), making the boot
             * nondeterministic. Single-lane (leave SPURS_FORCE_SPUS unset) is
             * both correct (RPCS3 oracle: jobmanager runs one SPU at a time) and
             * far more stable -- it reaches the furthest boot yet (LBP renders to
             * finish#922: glyphthread+network+camera up). PREFER single-lane. */
            { static int s_fs = -2;
              if (s_fs == -2) { const char* e = getenv("SPURS_FORCE_SPUS");
                s_fs = e ? atoi(e) : -1; }
              if (s_fs > 0) maxcont = (u8)(s_fs > 6 ? 6 : s_fs); }
            /* Lanes: the threads running it now (each a single-SPU run). */
            spurs_nib_set(ea, SPURS_WKL_CURCONT, wid, lane_ok ? (u8)s_wid_busy[wid] : maxcont);
            {
                const uint8_t* pm = (const uint8_t*)vm_base + (uint32_t)(uintptr_t)s_workloads[wid].pm;
                uint32_t sz = s_workloads[wid].sizePm;
                if (maxcont <= 1) {
                    spu_run_policy_module(r->fn, r->image_id, pm, sz, arg, wid, ea, run_spu);
                } else {
                    /* Run the workload's virtual SPUs CONCURRENTLY (see
                     * spurs_pm_worker): each lane advances in parallel so the
                     * cross-SPU barrier resolves. Spawn maxcont-1 workers for
                     * lanes 1..N-1 and run lane 0 on this thread, then join. */
                    HANDLE th[8]; struct spurs_pm_worker_arg wa[8];
                    unsigned nth = 0;
                    for (u32 sn = 1; sn < maxcont && nth < 7; sn++, nth++) {
                        wa[nth].fn = r->fn; wa[nth].image_id = r->image_id;
                        wa[nth].pm = pm; wa[nth].pm_size = sz;
                        wa[nth].arg = arg; wa[nth].wid = wid; wa[nth].ea = ea;
                        wa[nth].spu_num = sn;
                        th[nth] = CreateThread(NULL, 1u << 20, spurs_pm_worker, &wa[nth], 0, NULL);
                    }
                    spu_run_policy_module(r->fn, r->image_id, pm, sz, arg, wid, ea, 0);
                    if (nth) {
                        WaitForMultipleObjects(nth, th, TRUE, INFINITE);
                        for (unsigned k = 0; k < nth; k++) CloseHandle(th[k]);
                    }
                }
            }
            spurs_nib_set(ea, SPURS_WKL_CURCONT, wid, lane_ok && s_wid_busy[wid] > 1 ? (u8)(s_wid_busy[wid] - 1) : 0);
            /* Found work: the module polled the kernel for more (selectWorkload
             * calls > 0) or entered job code. A module that drains its list and
             * exits without a poll did work too; keying on polls alone grew its
             * idle streak and stretched its wake-up cadence. */
            const int did = g_spurs_pm_polls != 0 || g_spu_job_entries != jobs_before;
            rr_found |= did;
            if (!did || rr_lim <= 0 || maxcont > 1) break;
            /* SPURS_RERUN_YIELD=1: end the rerun as soon as a workload this SPU ranks
             * higher is ready and unclaimed, as the real kernel's selection would at
             * the module's next poll. Otherwise a low-priority list (GoW:A's EDGE,
             * priority 8, on up to three lanes) keeps the thread for the whole rerun
             * window while a higher one (HighestPrio6, priority 1-3) waits and the
             * PPU spins on it. Not spurs_poll_select: a module that drains its own
             * list leaves its readyCount 0, and that answers "exit" before looking
             * at the others. Every 60 s it logs how many reruns it ended.
             * Default off: on GoW:A it ends ~10k reruns a minute and changes nothing
             * (HighestPrio6 becomes ready only after the SPU work it depends on). */
            { static int s_ry = -1; static volatile LONG s_ry_n, s_ry_y; static volatile LONG64 s_ry_t;
              if (s_ry < 0) { const char* e = getenv("SPURS_RERUN_YIELD"); s_ry = e ? atoi(e) : 0; }
              if (s_ry && my_spu >= 0 && my_spu < CELL_SPURS_MAX_SPU) {
                  const u32 sel = spurs_higher_ready(si, ea, wid, my_spu);
                  InterlockedIncrement(&s_ry_n);
                  LARGE_INTEGER q; QueryPerformanceCounter(&q);
                  const LONG64 t = s_ry_t;
                  if (!t) InterlockedCompareExchange64(&s_ry_t, q.QuadPart, 0);
                  else { LARGE_INTEGER f; QueryPerformanceFrequency(&f);
                         if (q.QuadPart - t >= 60 * f.QuadPart &&
                             InterlockedCompareExchange64(&s_ry_t, q.QuadPart, t) == t)
                             fprintf(stderr, "[spurs-ry] 60s: rerun checks %ld, yielded %ld\n",
                                     InterlockedExchange(&s_ry_n, 0), InterlockedExchange(&s_ry_y, 0)); }
                  if (sel != wid) { InterlockedIncrement(&s_ry_y); break; } } }
            { LARGE_INTEGER q; QueryPerformanceCounter(&q);
              if (q.QuadPart - rr_t0.QuadPart >= rr_lim) break; }
            }
            if (!rr_found) { if (s_idle[wid] < 8) s_idle[wid]++; }
            else { s_idle[wid] = 0; pass_found_work = 1; }
            InterlockedDecrement(&s_wid_busy[wid]);
            }
        }
    }
}

s32 cellSpursReadyCountStore(CellSpurs* spurs, CellSpursWorkloadId wid,
                             u32 value)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= SPURS_MAXW) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!spurs_wid_owned(spurs, wid)) return CELL_SPURS_CORE_ERROR_SRCH;

    const u32 ea = (u32)(uintptr_t)spurs;
    /* The real store: the instance byte the kernel (poll thread) watches, under
     * the lock-line lock so a reserving SPU sees the line change. */
    spu_lockline_lock();
    s_workloads[wid].readyCount = value;
    vm_base[ea + spurs_off_ready(wid)] = (u8)value;
    ef_line_end(ea + spurs_off_ready(wid));
    {   static int _n = 0;
        if (_n < 32)
            printf("[cellSpurs] ReadyCountStore(wid=%u, value=%u)\n", wid, value);
        else if (_n == 32)
            printf("[cellSpurs] ReadyCountStore further logs suppressed\n");
        _n++;
    }
    return CELL_OK;
}

/* Swap / compare-and-swap on the real instance byte (what the guest's inlined
 * atomics and the kernel threads use), not only the host copy. */
static s32 spurs_ready_exchange(CellSpurs* spurs, CellSpursWorkloadId wid,
                                u32* old, u32 compare, u32 value, int conditional)
{
    if (!spurs || !old) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= SPURS_MAXW) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!spurs_wid_owned(spurs, wid)) return CELL_SPURS_CORE_ERROR_SRCH;
    const u32 at = (u32)(uintptr_t)spurs + spurs_off_ready(wid);
    spu_lockline_lock();
    u8 previous = vm_base[at];
    if (!conditional || previous == compare) {
        vm_base[at] = (u8)value;
        s_workloads[wid].readyCount = value;
        ef_line_end(at);
    } else spu_lockline_unlock();
    vm_write32((u32)(uintptr_t)old, previous);
    return CELL_OK;
}

s32 cellSpursReadyCountSwap(CellSpurs* spurs, CellSpursWorkloadId wid,
                            u32* old, u32 value)
{ return spurs_ready_exchange(spurs, wid, old, 0, value, 0); }

s32 cellSpursReadyCountCompareAndSwap(CellSpurs* spurs,
                                       CellSpursWorkloadId wid,
                                       u32* old, u32 compare, u32 value)
{ return spurs_ready_exchange(spurs, wid, old, compare, value, 1); }

s32 cellSpursWakeUp(CellSpurs* spurs)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;

    /* Start the SPU program of every runnable workload on this instance.
     *
     * cellSpursAddWorkload only REGISTERS the policy module -- it publishes the
     * workload into the BE instance and returns. WakeUp is what the guest
     * expects to actually put it on an SPU, and this was a stub returning
     * CELL_OK, so a title whose work is a WORKLOAD (rather than a taskset task,
     * which cellSpursCreateTask does dispatch) never ran any SPU code at all.
     *
     * Guitar Hero III is one: its job library does AddWorkload(pm, 6720 bytes)
     * -> SetExceptionEventHandler -> ReadyCountStore(8) -> WakeUp, and the
     * decompression jobs it then queues are picked up by that program. With
     * WakeUp inert the jobs were submitted, counted, and never executed or
     * retired -- the PPU blocked forever on a counter nothing could decrement,
     * while the only SPU images running were the two taskset tasks.
     *
     * Dispatch is async for the same reason CreateTask uses it: a workload
     * policy module is a persistent worker, and running it inline would block
     * this PPU thread forever. Guarded by `started` so a second WakeUp (they
     * are common -- the guest calls it whenever it adds work) does not spawn
     * another copy. */
    const uint32_t spurs_ea = (uint32_t)(uintptr_t)spurs;
    for (u32 i = 0; i < SPURS_MAXW; i++) {
        SpursWorkload* w = &s_workloads[i];
        if (!w->in_use || w->started || w->spurs_ea != spurs_ea) continue;
        if (!w->pm || !w->sizePm) continue;
        const uint8_t* host = GUEST_PTR(w->pm, const uint8_t*);
        if (!host) continue;
        w->started = 1;
        fprintf(stderr, "[cellSpurs] WakeUp: starting workload %u pm=0x%08X size=%u\n",
                i, (uint32_t)(uintptr_t)w->pm, w->sizePm);
        fflush(stderr);
        /* One instance. readyCount is how many SPUs the workload MAY run on,
         * and running 2 or 4 changes nothing for the title this was written
         * against -- measured, not assumed -- so the conservative reading
         * stands until a title needs otherwise. */
        spu_workload_dispatch_async(host, w->sizePm, (uint32_t)w->data);
    }
    return CELL_OK;
}

/* =========================================================================
 * Event flags
 * =====================================================================*/

s32 cellSpursEventFlagInitialize(CellSpursTaskset* taskset,
                                 CellSpursEventFlag* eventFlag,
                                 u32 clearMode, u32 direction)
{
    /* All pointers are raw guest EAs; the flag lives in guest memory in the
     * REAL BE layout (the SPU task library DMAs this exact struct). */
    uint32_t eventFlag_ea = (uint32_t)(uintptr_t)eventFlag;
    uint32_t taskset_ea   = (uint32_t)(uintptr_t)taskset;

    if (!eventFlag_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (eventFlag_ea & 0x7F)
        return CELL_SPURS_TASK_ERROR_ALIGN;

    /* Re-initialization: recycle the sync slot. */
    EventFlagSync* old = ef_sync_find(eventFlag_ea);
    if (old) ef_sync_free(old);

    for (uint32_t o = 0; o < EF_GUEST_SIZE; o += 8)
        vm_write64(eventFlag_ea + o, 0);
    vm_write8(eventFlag_ea + EF_DIRECTION,  (uint8_t)direction);
    vm_write8(eventFlag_ea + EF_CLEAR_MODE, (uint8_t)clearMode);
    /* addr = the owning taskset (isIwl=0); SPU-side Set uses it to find whom
     * to signal, and our Set hook passes it to spu_taskset_signal_task. */
    vm_write8(eventFlag_ea + EF_IS_IWL, 0);
    vm_write64(eventFlag_ea + EF_ADDR, (uint64_t)taskset_ea);

    EventFlagSync* sync = ef_sync_alloc(eventFlag_ea);
    if (!sync) {
        printf("[cellSpurs] EventFlagInitialize: no free sync slots!\n");
        return CELL_SPURS_TASK_ERROR_NOMEM;
    }

    printf("[cellSpurs] EventFlagInitialize(clearMode=%u, direction=%u) flagEA=0x%08X taskset=0x%08X\n",
           clearMode, direction, eventFlag_ea, taskset_ea);
    return CELL_OK;
}

s32 cellSpursEventFlagAttachLv2EventQueue(CellSpursEventFlag* eventFlag)
{
    eventFlag = GUEST_PTR(eventFlag, CellSpursEventFlag*);
    if (!eventFlag) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    printf("[cellSpurs] EventFlagAttachLv2EventQueue()\n");
    return CELL_OK;
}

s32 cellSpursEventFlagDetachLv2EventQueue(CellSpursEventFlag* eventFlag)
{
    eventFlag = GUEST_PTR(eventFlag, CellSpursEventFlag*);
    if (!eventFlag) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    printf("[cellSpurs] EventFlagDetachLv2EventQueue()\n");
    return CELL_OK;
}

s32 cellSpursEventFlagSet(CellSpursEventFlag* eventFlag, u16 bits)
{
    uint32_t ea = (uint32_t)(uintptr_t)eventFlag;
    if (!ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    { static unsigned _n=0; if (_n++ < 40 || (_n%1000)==0)   /* unsigned: periodic, wrapping is defined */
        fprintf(stderr, "[cellSpurs] EventFlagSet#%u flagEA=0x%08X bits=0x%04X "
                "events=0x%04X used=0x%04X pend=0x%04X mode=0x%04X\n",
                _n, ea, (unsigned)bits,
                vm_read16(ea + EF_EVENTS), vm_read16(ea + EF_SPU_USED_SLOTS),
                vm_read16(ea + EF_SPU_PENDING_RECV), vm_read16(ea + EF_SPU_WAIT_MODE)); }

    ef_lock(sync);
    spurs_ef_set_locked(ea, bits);
    ef_broadcast(sync);
    ef_unlock(sync);

    return CELL_OK;
}

/* SPURS_EF_STATS=1: per flag, how many PPU waits completed in the last 5 s
 * and their mean / max duration -- one SPU task round trip each. */
static void ef_wait_stats(uint32_t ea, long long t0)
{
    static int on = -1; if (on < 0) on = getenv("SPURS_EF_STATS") ? 1 : 0;
    if (!on) return;
    static uint32_t fea[8]; static unsigned n[8]; static double sum[8], mx[8];
    static long long last; static SRWLOCK lk = SRWLOCK_INIT;
    LARGE_INTEGER now, f; QueryPerformanceCounter(&now); QueryPerformanceFrequency(&f);
    double us = (now.QuadPart - t0) * 1e6 / (double)f.QuadPart;
    AcquireSRWLockExclusive(&lk);
    for (int i = 0; i < 8; i++)
        if (fea[i] == ea || !fea[i]) { fea[i] = ea; n[i]++; sum[i] += us; if (us > mx[i]) mx[i] = us; break; }
    if (!last) last = now.QuadPart;
    if ((now.QuadPart - last) > 5 * f.QuadPart) {
        for (int i = 0; i < 8 && fea[i]; i++) {
            fprintf(stderr, "[ef-stats] flag=0x%08X waits=%.1f/s mean=%.0fus max=%.0fus%c",
                    fea[i], n[i] * (double)f.QuadPart / (now.QuadPart - last),
                    n[i] ? sum[i] / n[i] : 0.0, mx[i], 10);
            n[i] = 0; sum[i] = mx[i] = 0;
        }
        last = now.QuadPart;
    }
    ReleaseSRWLockExclusive(&lk);
}

s32 cellSpursEventFlagWait(CellSpursEventFlag* eventFlag, u16* bits,
                           u32 mode)
{
    uint32_t ea      = (uint32_t)(uintptr_t)eventFlag;
    uint32_t bits_ea = (uint32_t)(uintptr_t)bits;
    if (!ea || !bits_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    u16 pattern = vm_read16(bits_ea);

    ef_lock(sync);

    /* Block until the requested bit pattern is satisfied in the GUEST struct
     * (BE events word at +0x00) — set by an SPU task (lifted code / taskset
     * syscall) or another PPU thread. Poll-based: ef_wait_timed ticks re-read
     * guest memory, so task-side PUTLLC stores are observed without needing
     * the real lv2-event-queue notification path.
     *
     * A wait the SPU never satisfies is a REAL deadlock and must present as
     * one — that is the signal naming the workload that is not executing.
     * SPURS_EF_FORCE=1 restores the old fake for A/B comparison only. */
    static int s_force = -1;
    if (s_force < 0) s_force = getenv("SPURS_EF_FORCE") ? 1 : 0;
    unsigned waits = 0;
    u16 current;
    __atomic_add_fetch(&s_ef_ppu_waiters, 1, __ATOMIC_SEQ_CST);
    LARGE_INTEGER ef_t0; QueryPerformanceCounter(&ef_t0);
    for (;;) {
        current = vm_read16(ea + EF_EVENTS);

        if (mode == CELL_SPURS_EVENT_FLAG_AND) {
            if ((current & pattern) == pattern)
                break;
        } else {
            /* OR mode: any requested bit set */
            if ((current & pattern) != 0)
                break;
        }

        if (!ef_wait_timed(sync, 2)) {
            /* Report the stall once per second, naming what would have to run. */
            if (++waits % 500 == 0) {
                static int _n = 0;
                if (_n < 24) { _n++;
                    fprintf(stderr, "[cellSpurs] EventFlagWait BLOCKED tid=%lu %us on pattern 0x%04X "
                                    "(mode=%s, bits=0x%04X) flagEA=0x%08X -- waiting for an SPU "
                                    "workload to cellSpursEventFlagSet it\n",
                            (unsigned long)GetCurrentThreadId(), waits / 500, pattern,
                            mode == CELL_SPURS_EVENT_FLAG_AND ? "AND" : "OR",
                            current, ea);
                    fflush(stderr);
                }
            }
            if (s_force && waits >= 1000) {
                fprintf(stderr, "[cellSpurs] EventFlagWait: SPURS_EF_FORCE -- faking pattern "
                                "0x%04X (NOT real; A/B only)\n", pattern);
                vm_write16(ea + EF_EVENTS, (u16)(current | pattern));
            }
        }
    }

    __atomic_sub_fetch(&s_ef_ppu_waiters, 1, __ATOMIC_SEQ_CST);
    ef_wait_stats(ea, ef_t0.QuadPart);
    /* Hand back the observed bits; consume the received ones on AUTO clear. */
    vm_write16(bits_ea, current);
    u16 received = (mode == CELL_SPURS_EVENT_FLAG_AND) ? pattern
                                                       : (u16)(current & pattern);
    { static int _n = 0; if (PS3_LOG_FIRST(_n, 40))
        fprintf(stderr, "[cellSpurs] EventFlagWait WAKE tid=%lu flagEA=0x%08X "
                "pattern=0x%04X got=0x%04X (waits=%u)\n",
                (unsigned long)GetCurrentThreadId(), ea, pattern, current, waits); }
    if (vm_read8(ea + EF_CLEAR_MODE) == CELL_SPURS_EVENT_FLAG_CLEAR_AUTO) {
        /* Clear from the CURRENT word: bits an SPU set since `current` was
         * read must survive. */
        spu_lockline_lock();
        ef_w16(ea + EF_EVENTS, (u16)(ef_r16(ea + EF_EVENTS) & ~received));
        ef_line_end(ea);
    }

    ef_unlock(sync);

    return CELL_OK;
}

s32 cellSpursEventFlagTryWait(CellSpursEventFlag* eventFlag, u16* bits,
                              u32 mode)
{
    uint32_t ea      = (uint32_t)(uintptr_t)eventFlag;
    uint32_t bits_ea = (uint32_t)(uintptr_t)bits;
    if (!ea || !bits_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    u16 pattern = vm_read16(bits_ea);

    ef_lock(sync);

    u16 current = vm_read16(ea + EF_EVENTS);

    if (mode == CELL_SPURS_EVENT_FLAG_AND) {
        if ((current & pattern) != pattern) {
            ef_unlock(sync);
            return CELL_SPURS_TASK_ERROR_BUSY;
        }
    } else {
        if ((current & pattern) == 0) {
            ef_unlock(sync);
            return CELL_SPURS_TASK_ERROR_BUSY;
        }
    }

    vm_write16(bits_ea, current);
    u16 received = (mode == CELL_SPURS_EVENT_FLAG_AND) ? pattern
                                                       : (u16)(current & pattern);
    if (vm_read8(ea + EF_CLEAR_MODE) == CELL_SPURS_EVENT_FLAG_CLEAR_AUTO) {
        spu_lockline_lock();
        ef_w16(ea + EF_EVENTS, (u16)(ef_r16(ea + EF_EVENTS) & ~received));
        ef_line_end(ea);
    }

    ef_unlock(sync);
    return CELL_OK;
}

s32 cellSpursEventFlagClear(CellSpursEventFlag* eventFlag, u16 bits)
{
    uint32_t ea = (uint32_t)(uintptr_t)eventFlag;
    if (!ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    ef_lock(sync);
    spu_lockline_lock();
    ef_w16(ea + EF_EVENTS, (u16)(ef_r16(ea + EF_EVENTS) & ~bits));
    ef_line_end(ea);
    ef_unlock(sync);

    return CELL_OK;
}

s32 cellSpursEventFlagGetDirection(CellSpursEventFlag* eventFlag,
                                   u32* direction)
{
    uint32_t ea     = (uint32_t)(uintptr_t)eventFlag;
    uint32_t dir_ea = (uint32_t)(uintptr_t)direction;
    if (!ea || !dir_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    vm_write32(dir_ea, vm_read8(ea + EF_DIRECTION));
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Additional functions needed by Tokyo Jungle (from RPCS3 audit)
 * -----------------------------------------------------------------------*/

/* _cellSpursEventFlagInitialize — internal init with more parameters */
s32 _cellSpursEventFlagInitialize(void* spurs, void* taskset,
                                    CellSpursEventFlag* eventFlag,
                                    u32 clearMode, u32 direction)
{
    (void)spurs; (void)taskset;
    printf("[cellSpurs] _EventFlagInitialize(clearMode=%u, dir=%u)\n",
           clearMode, direction);
    if (!eventFlag) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    /* Forward raw guest pointers; cellSpursEventFlagInitialize translates them
     * (translating here too would double-translate -> out-of-bounds). */
    return cellSpursEventFlagInitialize((CellSpursTaskset*)taskset, eventFlag, clearMode, direction);
}

/* _cellSpursSendSignal — internal signal delivery */
s32 _cellSpursSendSignal(void* taskset, u32 taskId)
{
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;
    printf("[cellSpurs] _SendSignal(taskset=0x%08X id=%u)\n", taskset_ea, taskId);
    /* Real delivery now that SPU tasks execute (comment was stale). */
    if (taskset_ea) spu_taskset_signal_task(taskset_ea, taskId);
    return CELL_OK;
}

/* =========================================================================
 * Job chains (LBP's render path: the game emits SPURS job descriptors and
 * chains them via u64 command words; the jobchain policy module walks the
 * chain on SPU). Facts-first bring-up: decode + log the REAL BE guest
 * structures at the SDK offsets; execution wiring lands once the logged
 * shapes confirm the descriptor formats.
 * =====================================================================*/

/* Real (BE) CellSpursJobChainAttribute offsets. */
enum {
    JCATTR_REVISION   = 0x00,   /* be u32 */
    JCATTR_SDKVERSION = 0x04,   /* be u32 */
    JCATTR_ENTRY      = 0x08,   /* be u32: EA of the first jobchain command word */
    JCATTR_SIZE_DESC  = 0x0C,   /* be u16: sizeJobDescriptor */
    JCATTR_MAX_GRAB   = 0x0E,   /* be u16: maxGrabbedJob */
    JCATTR_PRIORITY   = 0x10,   /* u8[8] */
    JCATTR_MAX_CONT   = 0x18,   /* be u32 */
    JCATTR_AUTO_RDY   = 0x1C,   /* u8 bool: autoReadyCount */
    JCATTR_TAG1       = 0x20,   /* be u32 */
    JCATTR_TAG2       = 0x24,   /* be u32 */
    JCATTR_FIXED_MEM  = 0x28,   /* u8 bool */
    JCATTR_MAX_SIZE_D = 0x2C,   /* be u32 */
    JCATTR_INIT_SPU   = 0x30,   /* be u32 */
    JCATTR_NAME       = 0x34,   /* be u32: char* EA (SetName) */
};

/* Host-side jobchain registry (the CellSpursJobChain guest struct is opaque
 * to the game; we track what we need beside it). */
#define MAX_JOBCHAINS 32
static struct {
    u32 jc_ea;        /* CellSpursJobChain EA (0 = free) */
    u32 entry_ea;     /* first command word EA */
    u16 size_desc;
    u16 max_grab;
    int run_count;
    volatile long running;   /* 1 while a host thread walks this chain */
} s_jobchains[MAX_JOBCHAINS];

static void jc_dump_commands(const char* tag, u32 ea, int max_words)
{
    printf("[cellSpurs] %s chain@0x%08X commands:", tag, ea);
    for (int i = 0; i < max_words; i++) {
        u64 cmd = vm_read64(ea + (u32)i * 8);
        printf("\n    [%2d] 0x%016llX", i, (unsigned long long)cmd);
        if (cmd == 0) { printf(" (halt/empty)"); break; }
    }
    printf("\n");
}

/* SDK ABI (cell/spurs/job_chain.h): the REVISIONS come first -- attr is r5.
 *   _cellSpursJobChainAttributeInitialize(jmRevision, sdkRevision, attr,
 *       jobChainEntry, sizeJobDescriptor, maxGrabbedJob, priorityTable,
 *       maxContention, [stack: autoRequestSpuCount, tag1, tag2,
 *       isFixedMemAlloc, maxSizeJobDescriptor, initialRequestSpuCount]) */
s32 _cellSpursJobChainAttributeInitialize(u32 jmRevision, u32 sdkRevision, u64 attr_ea,
                                          u64 entry_ea, u32 sizeJobDescriptor,
                                          u32 maxGrabbedJob, u64 prio_ea, u32 maxContention)
{
    if (!attr_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + JCATTR_REVISION,   jmRevision);
    vm_write32((u32)attr_ea + JCATTR_SDKVERSION, sdkRevision);
    vm_write32((u32)attr_ea + JCATTR_ENTRY,      (u32)entry_ea);
    vm_write32((u32)attr_ea + JCATTR_SIZE_DESC,
               ((sizeJobDescriptor & 0xFFFFu) << 16) | (maxGrabbedJob & 0xFFFFu));
    for (int i = 0; i < 8; i++)
        *(vm_base + (u32)attr_ea + JCATTR_PRIORITY + i) =
            prio_ea ? *(vm_base + (u32)prio_ea + i) : 0;
    vm_write32((u32)attr_ea + JCATTR_MAX_CONT, maxContention);
    /* args 9+ (autoReadyCount, tag1, tag2, isFixedMemAlloc, maxSizeJobDescriptor,
     * initSpuCount) are on the guest stack, beyond the 8-GPR HLE adapter -- defaults. */
    vm_write32((u32)attr_ea + JCATTR_AUTO_RDY,   0);
    vm_write32((u32)attr_ea + JCATTR_TAG1,       0);
    vm_write32((u32)attr_ea + JCATTR_TAG2,       0);
    vm_write32((u32)attr_ea + JCATTR_FIXED_MEM,  0);
    vm_write32((u32)attr_ea + JCATTR_MAX_SIZE_D, 0);
    vm_write32((u32)attr_ea + JCATTR_INIT_SPU,   0);
    vm_write32((u32)attr_ea + JCATTR_NAME,       0);
    printf("[cellSpurs] _JobChainAttributeInitialize(attr=0x%08X entry=0x%08X sizeDesc=%u maxGrab=%u maxCont=%u)\n",
           (u32)attr_ea, (u32)entry_ea, sizeJobDescriptor, maxGrabbedJob, maxContention);
    return CELL_OK;
}

s32 cellSpursJobChainAttributeSetName(u64 attr_ea, u64 name_ea)
{
    if (!attr_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + JCATTR_NAME, (u32)name_ea);
    printf("[cellSpurs] JobChainAttributeSetName(attr=0x%08X, \"%s\")\n",
           (u32)attr_ea, name_ea ? (const char*)(vm_base + (u32)name_ea) : "-");
    return CELL_OK;
}

/* Register (or re-register) a job chain handle. Shared by both Create forms. */
static s32 jc_register(u32 jc_ea, u32 entry_ea, u16 size_desc, u16 max_grab,
                       const char* who)
{
    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (!s_jobchains[i].jc_ea || s_jobchains[i].jc_ea == jc_ea) {
            s_jobchains[i].jc_ea     = jc_ea;
            s_jobchains[i].entry_ea  = entry_ea;
            s_jobchains[i].size_desc = size_desc;
            s_jobchains[i].max_grab  = max_grab;
            s_jobchains[i].run_count = 0;
            return CELL_OK;
        }
    }
    printf("[cellSpurs] %s: registry full\n", who);
    return CELL_OK;
}

s32 cellSpursCreateJobChainWithAttribute(u64 spurs_ea, u64 jc_ea, u64 attr_ea)
{
    if (!spurs_ea || !jc_ea || !attr_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    u32 entry   = vm_read32((u32)attr_ea + JCATTR_ENTRY);
    u32 sd_mg   = vm_read32((u32)attr_ea + JCATTR_SIZE_DESC);
    u32 name_ea = vm_read32((u32)attr_ea + JCATTR_NAME);

    {   /* Layout ground truth (same rationale as AddWorkloadWA). */
        static int _n = 0;
        if (_n < 3) {
            printf("[cellSpurs] CreateJobChainWA jc=0x%08X attr=0x%08X raw:", (u32)jc_ea, (u32)attr_ea);
            for (int o = 0; o < 0x40; o += 4) {
                if ((o & 15) == 0) printf("\n    +%02X:", o);
                printf(" %08X", vm_read32((u32)attr_ea + o));
            }
            printf("\n");
            if (entry && entry < 0x10000000u) jc_dump_commands("CreateJobChainWA", entry, 16);
        } else if (_n == 3) {
            printf("[cellSpurs] CreateJobChainWA raw dumps suppressed from here\n");
        }
        _n++;
    }
    printf("[cellSpurs] CreateJobChainWithAttribute(jc=0x%08X entry=0x%08X sizeDesc=%u maxGrab=%u name=\"%s\")\n",
           (u32)jc_ea, entry, sd_mg >> 16, sd_mg & 0xFFFFu,
           name_ea ? (const char*)(vm_base + name_ea) : "-");

    return jc_register((u32)jc_ea, entry, (u16)(sd_mg >> 16), (u16)(sd_mg & 0xFFFFu),
                       "CreateJobChainWithAttribute");
}

/* cellSpursCreateJobChain -- the no-attribute form, and the one the WWS job
 * manager's own wrapper calls. Same registration, with the fields passed
 * directly instead of read back out of a CellSpursJobChainAttribute:
 *
 *   cellSpursCreateJobChain(CellSpurs*, CellSpursJobChain*, const u64* entry,
 *                           u16 sizeJobDescriptor, u16 maxGrabbedJob,
 *                           const u8 priorityTable[8], u32 maxContention,
 *                           bool autoRequestSpuCount, u32 tag1, u32 tag2)
 *
 * tag1/tag2 are guest-stack args, past the 8-GPR HLE adapter; nothing here
 * needs them. Without this entry point the chain is never registered at all,
 * so the later kick finds "UNKNOWN chain (no Create seen)" and returns
 * CELL_OK having walked nothing -- and the title waits forever on SPU work
 * that was never started. Gran Turismo 5 Prologue parks its whole boot there. */
s32 cellSpursCreateJobChain(u64 spurs_ea, u64 jc_ea, u64 entry_ea,
                            u32 sizeJobDescriptor, u32 maxGrabbedJob,
                            u64 prio_ea, u32 maxContention,
                            u32 autoRequestSpuCount)
{
    (void)spurs_ea; (void)prio_ea;
    if (!jc_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 3)) {
        printf("[cellSpurs] CreateJobChain(jc=0x%08X entry=0x%08X sizeDesc=%u maxGrab=%u "
               "maxCont=%u autoReq=%u)\n",
               (u32)jc_ea, (u32)entry_ea, sizeJobDescriptor, maxGrabbedJob,
               maxContention, autoRequestSpuCount);
        if (entry_ea && (u32)entry_ea < 0x10000000u)
            jc_dump_commands("CreateJobChain", (u32)entry_ea, 16);
    }
    return jc_register((u32)jc_ea, (u32)entry_ea, (u16)sizeJobDescriptor,
                       (u16)maxGrabbedJob, "CreateJobChain");
}

/* ---------------------------------------------------------------------------
 * Job-chain execution
 *
 * A job chain is a u64 command stream (SDK cell/spurs/job_commands.h): the low
 * 3 bits select the opcode, and a nonzero word whose low 3 bits are 0 IS a job
 * pointer. The real jobchain policy module walks this on an SPU, fetching each
 * CellSpursJobHeader and running its binary.
 *
 * LBP's draw pipeline is built on this -- the jobs emit the GCM commands. With
 * the chain unimplemented the FIFO starves, the RSX `ref` fence stops advancing
 * and the game spins on it forever (the boot hang: ref frozen at 0x2B3 while
 * the chain sat un-run).
 *
 * We walk the stream on a host thread (one per chain; the real thing is async
 * on SPUs, so RunJobChain must not block the PPU) and push each job binary
 * through the same fingerprint -> lifted-SPU dispatch that already runs this
 * title's workload images.
 * -----------------------------------------------------------------------*/
enum {                              /* CellSpursJobHeader (48 B) */
    JH_EA_BINARY     = 0x00,        /* be u64: job binary EA (low 3 bits = flags) */
    JH_SIZE_BINARY   = 0x08,        /* be u16: binary size >> 4                   */
    JH_JOB_TYPE      = 0x2C,        /* u8                                         */
    JH_SIZE          = 0x30,
};

static void jc_run_one_job(u32 job_ea, int idx, u32 size_desc)
{
    u64 ea_bin_raw = vm_read64(job_ea + JH_EA_BINARY);
    u32 ea_bin     = (u32)(ea_bin_raw & ~7ull);          /* low 3 bits = flags */
    u32 size_bin   = ((vm_read32(job_ea + JH_SIZE_BINARY) >> 16) & 0xFFFFu) << 4;
    u8  job_type   = *(vm_base + job_ea + JH_JOB_TYPE);

    { static int _n = 0;
      if (_n < 8) {
          printf("[cellSpurs]   job[%d] @0x%08X eaBinary=0x%08X size=%u type=0x%02X hdr:",
                 idx, job_ea, ea_bin, size_bin, job_type);
          for (int o = 0; o < JH_SIZE; o += 4) {
              if ((o & 15) == 0) printf("\n      +%02X:", o);
              printf(" %08X", vm_read32(job_ea + o));
          }
          printf("\n");
      }
      _n++; }

    if (!ea_bin || !size_bin || ea_bin >= 0x10000000u) {
        static int _b = 0;
        if (PS3_LOG_FIRST(_b, 4))
            printf("[cellSpurs]   job[%d]: implausible binary (ea=0x%08X size=%u) -- skipped\n",
                   idx, ea_bin, size_bin);
        return;
    }
    /* A jobchain job is NOT a SPURS task: Sony's jm2 stages the whole working
     * set in local store and enters the job's CRT with r3 = CellSpursJobContext2*
     * and r4 = the job descriptor, both LS pointers. Dispatching it on the task
     * ABI (arg EA in r3) left it reading a zeroed context and parking with no
     * DMA traffic at all. spu_workload_dispatch_job reproduces jm2's staging.
     * Synchronous by design: a job runs to completion, and the chain's own
     * NEXT/CALL/RET commands are what order them. */
    spu_workload_dispatch_job(vm_base + ea_bin, size_bin, job_ea, size_desc);
}

/* Walk one chain's command stream. Bounded: a malformed or self-looping stream
 * must not spin a host thread forever. */
/* ---------------------------------------------------------------------------
 * Job guards
 *
 * A job chain can begin with a GUARD command (op 7, ext 7|(1<<3)), whose EA is
 * the CellSpursJobGuard. The chain BLOCKS there until the guard's notify count
 * reaches zero; the PPU releases it with cellSpursJobGuardNotify once the job's
 * parameters are ready. With autoReset the count is restored afterwards, so a
 * looping chain waits again on the next pass.
 *
 * These were unimplemented, and the walker fell through the GUARD command into
 * the default (ignore) case. Tokyo Jungle's "soc-job" chain is
 *
 *     [0] GUARD 0x02932C00      [1] JOB 0x02932D00
 *     [2] SYNC                  [3] NEXT -> [0]        (a loop)
 *
 * so the job ran immediately, on the first pass, against a parameter block the
 * game had not filled in yet -- 0x02937580 read back as all zeroes -- and the
 * SPU went on to compute a wild DMA address from it. Honouring the guard is
 * what makes "run this job when I say so" mean anything.
 * -----------------------------------------------------------------------*/

#define MAX_JOBGUARDS 32
static struct {
    u32 ea;          /* guest CellSpursJobGuard address (128-byte aligned) */
    u32 count;       /* notifications still outstanding                    */
    u32 reset;       /* value to restore when autoReset is set             */
    u32 pending;     /* notifications received, not yet consumed        */
    u32 auto_reset;
    int active;
} s_jobguards[MAX_JOBGUARDS];

static int jg_find(u32 ea)
{
    for (int i = 0; i < MAX_JOBGUARDS; i++)
        if (s_jobguards[i].active && s_jobguards[i].ea == ea) return i;
    return -1;
}

/* cellSpursJobGuardInitialize(CellSpursJobChain* jobChain,
 *                             CellSpursJobGuard* jobGuard,
 *                             u32 notifyCount, u8 requestSpuCount, u8 autoReset)
 * Argument order confirmed against the guest: r3 is the chain (0x02932A80,
 * the handle CreateJobChainWithAttribute registered) and r4 is the guard
 * (0x02932C00, the EA the chain's GUARD command carries). */
s32 cellSpursJobGuardInitialize(u64 jc_ea, u64 guard_ea, u32 notify_count,
                                u32 request_spu_count, u32 auto_reset)
{
    (void)jc_ea; (void)request_spu_count;
    u32 ea = (u32)guard_ea;
    if (!ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    int i = jg_find(ea);
    if (i < 0)
        for (i = 0; i < MAX_JOBGUARDS; i++) if (!s_jobguards[i].active) break;
    if (i >= MAX_JOBGUARDS) return CELL_SPURS_TASK_ERROR_AGAIN;

    s_jobguards[i].ea         = ea;
    s_jobguards[i].count      = notify_count;
    s_jobguards[i].reset      = notify_count;
    s_jobguards[i].pending    = 0;
    s_jobguards[i].auto_reset = auto_reset;
    s_jobguards[i].active     = 1;

    /* Mirror the count into guest memory. The SPU side reads the guard on
     * hardware; keeping the two consistent costs nothing and makes a guest-side
     * peek meaningful. */
    vm_write32(ea, notify_count);

    printf("[cellSpurs] JobGuardInitialize(guard=0x%08X chain=0x%08X notify=%u "
           "autoReset=%u)\n", ea, (u32)jc_ea, notify_count, auto_reset);
    return CELL_OK;
}

/* cellSpursJobGuardNotify(CellSpursJobGuard* jobGuard) -- one argument. */
s32 cellSpursJobGuardNotify(u64 guard_ea)
{
    u32 ea = (u32)guard_ea;
    int i = jg_find(ea);
    if (i < 0) {
        printf("[cellSpurs] JobGuardNotify(0x%08X) -- unknown guard\n", ea);
        return CELL_SPURS_TASK_ERROR_INVAL;
    }
    /* Count notifications as CREDITS rather than decrementing to a floor of
     * zero. The guard used to reload its count at the moment the walker
     * OBSERVED zero, which threw away any notify that arrived while the job
     * was still running -- the title notified 6 times and the chain completed
     * 2 laps, so the thread waiting on the other 4 completions never woke. */
    s_jobguards[i].pending++;
    { u32 remaining = (s_jobguards[i].pending >= s_jobguards[i].reset)
                    ? 0u : s_jobguards[i].reset - s_jobguards[i].pending;
      s_jobguards[i].count = remaining;
      vm_write32(ea, remaining); }
    { static int _n = 0;
      if (PS3_LOG_FIRST(_n, 8))
          printf("[cellSpurs] JobGuardNotify(0x%08X) -> %u remaining\n",
                 ea, s_jobguards[i].count); }
    return CELL_OK;
}

/* cellSpursJobGuardReset(CellSpursJobGuard*) -- restore the initial count. */
s32 cellSpursJobGuardReset(u64 guard_ea)
{
    u32 ea = (u32)guard_ea;
    int i = jg_find(ea);
    if (i < 0) return CELL_SPURS_TASK_ERROR_INVAL;
    s_jobguards[i].count = s_jobguards[i].reset;
    vm_write32(ea, s_jobguards[i].count);
    return CELL_OK;
}

/* Wait for a guard to be released. Returns 1 if it opened, 0 on timeout (the
 * chain then gives up this pass rather than spinning the walker thread
 * forever). An unknown guard opens immediately -- a chain guarded by something
 * we never saw initialised must not deadlock the walker. */
static int jg_wait(u32 ea)
{
    int i = jg_find(ea);
    if (i < 0) return 1;
    for (int spin = 0; spin < 20000; spin++) {     /* ~20 s at 1 ms */
        if (s_jobguards[i].pending >= s_jobguards[i].reset) {
            /* Consume exactly one release worth of credits, so notifies that
             * arrived during the previous lap still count. */
            s_jobguards[i].pending -= s_jobguards[i].reset;
            if (!s_jobguards[i].auto_reset) s_jobguards[i].reset = 0;
            s_jobguards[i].count = s_jobguards[i].reset;
            vm_write32(ea, s_jobguards[i].count);
            return 1;
        }
#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }
    printf("[cellSpurs] guard 0x%08X still closed after 20 s -- chain gives up\n", ea);
    return 0;
}

/* Tell the application a lap of chain work is done.
 *
 * Timing is the whole point here. The obvious place is after the walk
 * FINISHES, but a guarded chain never finishes: it loops back to its GUARD and
 * blocks for the next notify. Tokyo Jungle notifies the guard, then waits on
 * this queue, and would only notify again after the event arrives -- so
 * signalling at walker exit deadlocks both sides. Signal when the WORK
 * completes: at END, and on reaching a guard with jobs run this lap. */
/* __declspec(thread) for MSVC only: MinGW GCC ignores it, which would make
 * these process-wide. spurs_job.c and cellSpurs.c must agree. */
#if defined(_MSC_VER)
#  define SPURS_JOB_TLS __declspec(thread)
#else
#  define SPURS_JOB_TLS __thread
#endif
extern SPURS_JOB_TLS uint32_t g_spurs_job_mbox, g_spurs_job_mbox_intr, g_spurs_job_cmd;
extern SPURS_JOB_TLS int g_spurs_job_mbox_valid;

/* SPURS_EVENT_D3=<n>: probe. The value a job query returns to the PPU is read
 * from the completion event's data3 field (the guest stores r7 at sp+0xB8 and
 * reads the count back from sp+0xBC). This lets that be confirmed by observing
 * the returned count change, without having to guess the real value first. */
static int d3_probe_valid = 0;
static u64 spurs_event_data3_probe(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPURS_EVENT_D3");
                 d3_probe_valid = e ? 1 : 0; v = e ? atoi(e) : 0; }
    return (u64)(unsigned)v;
}

static void jc_signal_done(u32 jc_ea, int job_index)
{
    /* Carry the job's mailbox answer in the completion event. On hardware the
     * SPU's outbound/interrupt mailbox is what the SPURS event delivers; a
     * synthetic payload means the caller reads back zero for whatever it
     * asked. Keep the chain EA in data1 for anything that used it.
     *
     * data2 is WHICH job finished: the PPU side of a job chain counts on it.
     * GT5P's audio service does
     *     do { JobGuardNotify; receive(q); i = data2; run_callback(i); }
     *     while (i + 1 < queued);
     * so a data2 pinned at 0 never ends that loop -- with the chain's lwmutex
     * held, which blocks the main thread's teardown behind it. A chain-level
     * signal (job_index < 0) sends 0. */
    u64 d2 = (u64)(unsigned)(job_index < 0 ? 0 : job_index);
    u64 mbox = 0, d3 = 0;
    (void)spurs_event_data3_probe();   /* prime d3_probe_valid */
    /* A PPU-side query submits a job and reads the reply out of the completion
     * event's data3 (the guest stores r7 at sp+0xB8 and reads the value back
     * from sp+0xBC, the low half). Tokyo Jungle asks each DSP module for its
     * input and output bus count this way -- commands 0x105 and 0x106 -- and
     * the SPU answers both in its outbound mailbox. We computed that mailbox
     * and then dropped it, pushing 0, so every module registered with a bus
     * count of 0 and the bus setup rejected the first bus it was asked for
     * ("sgxbus.c: out of range bus"): the check errors when count < requested,
     * and 0 < 1.
     *
     * Only the two count queries are forwarded. Neighbouring commands
     * (0x102/0x103/0x104) reply with LS ADDRESSES rather than counts, and
     * func_00201D38 stores each reply into module+0x1C/+0x20/+0x24. The
     * mailbox never goes in data2, which is the job index above: a mailbox
     * value there would end the consumer's wait early. */
    enum { SGX_Q_IN_BUSES = 0x105, SGX_Q_OUT_BUSES = 0x106 };
    const int answers_a_count = (g_spurs_job_cmd == SGX_Q_IN_BUSES ||
                                 g_spurs_job_cmd == SGX_Q_OUT_BUSES);
    if (g_spurs_job_mbox_valid) {
        mbox = g_spurs_job_mbox;
        d3 = g_spurs_job_mbox_intr; (void)d3;
        g_spurs_job_mbox_valid = 0;
    }
    for (int i = 0; i < s_spurs_event_queue_n; i++) {
        int rc = sys_event_queue_push_by_id(s_spurs_event_queue[i],
                                            SPURS_EVENT_PORT, jc_ea, d2,
                                            d3_probe_valid
                                                ? spurs_event_data3_probe()
                                                : (answers_a_count ? mbox : 0));
        static int n = 0;
        if (PS3_LOG_FIRST(n, 8))
            printf("[cellSpurs] chain 0x%08X job %d done -> event queue %u "
                   "(data2=%llu rc=%d)\n",
                   jc_ea, job_index, s_spurs_event_queue[i],
                   (unsigned long long)d2, rc);
    }
}

static void jc_execute(u32 entry_ea, u32 jc_ea, u32 size_desc)
{
    u32 pc = entry_ea, ret_pc = 0;
    int jobs = 0;
    /* Bound IDLE steps, not total steps. A SPURS job chain is frequently a
     * SERVICE LOOP -- guard, job, sync, next, repeat -- and is meant to run for
     * the lifetime of the subsystem. A flat cap on total commands therefore
     * kills a perfectly healthy chain: Tokyo Jungle's audio chain hit it after
     * 1332 jobs and its sound pipeline simply stopped. What actually indicates
     * a malformed chain is spinning through commands WITHOUT running a job, so
     * count that instead and reset it whenever real work happens. */
    int idle = 0;
    for (;; idle++) {
        if (idle > 4096) break;
        u64 cmd = vm_read64(pc);
        u32 op  = (u32)(cmd & 7);
        u32 ext = (u32)(cmd & 127);

        if (cmd != 0 && op == 0) {                    /* JOB */
            { extern u32 g_spurs_job_ls_handle; g_spurs_job_ls_handle = jc_ea; }
            int idx = jobs++;
            jc_run_one_job((u32)(cmd & ~7ull), idx, size_desc);
            idle = 0;                    /* real work: the chain is healthy */
            /* Signal per JOB, not per lap. The application waits once for each
             * unit of work it queued -- this title notified the guard 7 times
             * and sat in event_queue_receive 30 times -- so batching the
             * completion into one event per pass leaves it waiting for
             * completions that, from its point of view, never arrive. */
            jc_signal_done(jc_ea, idx);
            pc += 8; continue;
        }
        if (op == 1) { pc = (u32)(cmd & ~7ull); continue; }   /* RESET_PC */
        if (op == 3) { pc = (u32)(cmd & ~7ull); continue; }   /* NEXT     */
        if (op == 4) { ret_pc = pc + 8; pc = (u32)(cmd & ~7ull); continue; }  /* CALL */
        if (op == 7) {
            if (ext == (7 | (15 << 3))) {                     /* END */
                printf("[cellSpurs] chain 0x%08X: END after %d job(s)\n", jc_ea, jobs);
                if (jobs) jc_signal_done(jc_ea, jobs - 1);
                return;
            }
            if (ext == (7 | (14 << 3))) {                     /* RET */
                if (!ret_pc) return;
                pc = ret_pc; ret_pc = 0; continue;
            }
            if (ext == (7 | (0 << 3))) {                      /* ABORT */
                printf("[cellSpurs] chain 0x%08X: ABORT\n", jc_ea);
                return;
            }
        }
        if (op == 7 && ext == (7 | (1 << 3))) {           /* GUARD */
            u32 g_ea = (u32)(cmd & ~127ull);
            if (!jg_wait(g_ea)) return;                  /* still closed */
            pc += 8; continue;
        }
        /* NOP(0) / SYNC+LWSYNC(2, one virtual SPU: nothing to wait for) /
         * FLUSH(5) / JOBLIST(6, nested arrays TODO) / SET_LABEL */
        pc += 8;
    }
    printf("[cellSpurs] chain 0x%08X: 4096 commands with no job run after %d "
           "job(s) -- malformed?\n",
           jc_ea, jobs);
}

static DWORD WINAPI jc_thread(LPVOID p)
{
    int slot = (int)(intptr_t)p;
    /* TIMING PROBE (SPURS_JC_DELAY=ms): the real jm2 chain walker is async and
     * picks up jobs as the PPU appends them + fills their descriptors. Our walk
     * is one-shot; if it reads descriptors before the PPU populates the I/O
     * (n_dma=0, empty ioBuffer), deferring the walk should let real I/O appear.
     * Confirms timing-vs-never before committing to the async rewrite. */
    { const char* d = getenv("SPURS_JC_DELAY");
      if (d && *d) Sleep((unsigned)atoi(d)); }
    jc_execute(s_jobchains[slot].entry_ea, s_jobchains[slot].jc_ea,
               s_jobchains[slot].size_desc);
    s_jobchains[slot].running = 0;
    /* Tell the application the chain is done. Without this the walk finishes
     * in silence and a thread waiting on the attached queue never runs again. */
    jc_signal_done(s_jobchains[slot].jc_ea, -1);
    return 0;
}

/* cellSpursRunJobChain -- start job chain execution (async, like the real one). */
/* cellSpursRunJobChain(const CellSpursJobChain* jobChain) -- ONE argument.
 *
 * This was declared (spurs, jobChain), so the ABI adapter fed it r3=spurs,
 * r4=<whatever>, and the chain it looked up was r4 -- never the handle
 * CreateJobChainWithAttribute registered. Every run logged
 * "UNKNOWN chain (no Create seen)" and returned CELL_OK without walking
 * anything, so the title sat in event_queue_receive waiting for SPU work that
 * was never started. Tokyo Jungle blocks its whole boot there.
 *
 * Verified against the guest: Create takes the chain in r4 (r3 is the CellSpurs)
 * and passes 0x02932A80; Run then arrives with r3=0x02932A80. Join and Shutdown
 * are the same one-argument shape. */
static s32 jc_start(u64 jc_ea, const char* who)
{
    static int s_off = -1;
    if (s_off < 0) s_off = getenv("PS3_NO_JOBCHAIN") ? 1 : 0;

    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (s_jobchains[i].jc_ea != (u32)jc_ea) continue;
        s_jobchains[i].run_count++;
        if (s_jobchains[i].run_count <= 3) {
            printf("[cellSpurs] %s(jc=0x%08X) run#%d entry=0x%08X\n",
                   who, (u32)jc_ea, s_jobchains[i].run_count, s_jobchains[i].entry_ea);
            jc_dump_commands(who, s_jobchains[i].entry_ea, 16);
        }
        if (s_off || !s_jobchains[i].entry_ea) return CELL_OK;
        /* Coalesce: a chain already being walked must not start twice. */
        if (_InterlockedCompareExchange(&s_jobchains[i].running, 1, 0) == 0) {
            HANDLE th = CreateThread(NULL, 1u << 20, jc_thread, (LPVOID)(intptr_t)i, 0, NULL);
            if (th) CloseHandle(th);
            else s_jobchains[i].running = 0;
        }
        return CELL_OK;
    }
    printf("[cellSpurs] %s(jc=0x%08X) -- UNKNOWN chain (no Create seen)\n", who, (u32)jc_ea);
    return CELL_OK;
}

s32 cellSpursRunJobChain(u64 jc_ea)
{
    return jc_start(jc_ea, "RunJobChain");
}

/* cellSpursShutdownJobChain(CellSpursJobChain*) -- the other half of the pair.
 *
 * A title that starts a chain with Run/Kick ends it with Shutdown, then blocks
 * in Join until the chain has actually stopped. Leaving Shutdown unimplemented
 * is not harmless: the import logs UNIMPLEMENTED and returns whatever was in
 * r3, so the caller reads a garbage result for "did the shutdown take?" and a
 * title that checks it can decide the subsystem failed and tear itself down.
 *
 * Clearing `running` is what makes the pair honest: the host walker stops
 * grabbing new work, and the Join that follows has something true to observe.
 * Found in Tokyo Jungle, whose sound engine shuts its chain down during init
 * and then reports "failed to recv data from SPU" when the handshake does not
 * complete. */
s32 cellSpursShutdownJobChain(u64 jc_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] ShutdownJobChain(jc=0x%08X)\n", (u32)jc_ea);
    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (s_jobchains[i].jc_ea == (u32)jc_ea) {
            s_jobchains[i].running = 0;   /* stop the walker; Join can now succeed */
            return CELL_OK;
        }
    }
    /* Not one of ours: still CELL_OK -- the chain is, trivially, not running. */
    return CELL_OK;
}

/* cellSpursJoinJobChain(const CellSpursJobChain*) -- one argument, as above. */
s32 cellSpursJoinJobChain(u64 jc_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] JoinJobChain(jc=0x%08X)\n", (u32)jc_ea);
    /* Run/Kick walk the chain on a host thread, so Join has to wait for it.
     * Returning at once told the title its jobs were done while they were
     * still running. The Simpsons Arcade Game runs and joins a chain every
     * frame, and those jobs DMA the fragment programs and vertices of draws
     * already queued in the FIFO: the RSX drain then reached the draws first
     * (black frames, glyph quads built from half-written vertices), and the
     * next frame's Run was coalesced away while this walker was still busy,
     * so that frame's data was never written at all. */
    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (s_jobchains[i].jc_ea != (u32)jc_ea) continue;
        while (s_jobchains[i].running) {
#ifdef _WIN32
            Sleep(0);
#else
            usleep(0);
#endif
        }
    }
    return CELL_OK;
}

s32 cellSpursJobChainGetError(u64 jc_ea, u64 cause_out_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] JobChainGetError(jc=0x%08X)\n", (u32)jc_ea);
    if (cause_out_ea) vm_write32((u32)cause_out_ea, 0);
    return CELL_OK;
}

/* cellSpursKickJobChain(CellSpursJobChain* jobChain, u8 numReadyCount)
 *
 * Two arguments, jobChain in r3 -- the same one-handle shape as Run/Join, not
 * (spurs, jobChain). It is the *other* way a title starts a chain: Run for one
 * that was created ready to go, Kick to hand the SPUs another `numReadyCount`
 * units of an already-created chain. Titles built on the WWS job manager use
 * Kick, so leaving this a no-op meant their chains were created and then never
 * walked -- which looks exactly like a hang with no error anywhere. */
s32 cellSpursKickJobChain(u64 jc_ea, u32 numReadyCount)
{
    (void)numReadyCount;
    return jc_start(jc_ea, "KickJobChain");
}

/* =========================================================================
 * SPURS queues (PPU<->SPU bounded FIFO; LBP's audio instance uses these).
 * Log-only bring-up: capture the shapes before wiring real state.
 * =====================================================================*/

/* SDK ABI (cell/spurs/queue.h):
 *   _cellSpursQueueInitialize(CellSpurs*, CellSpursTaskset*, CellSpursQueue*,
 *       const void* buffer, u32 size, u32 depth, CellSpursQueueDirection) */
s32 _cellSpursQueueInitialize(u64 spurs_ea, u64 taskset_ea, u64 queue_ea,
                              u64 buffer_ea, u32 size, u32 depth, u32 direction)
{
    (void)spurs_ea;
    if (!queue_ea || !buffer_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (!size || !depth)         return CELL_SPURS_TASK_ERROR_INVAL;

    /* Same 128-byte big-endian line as the LF variant below, and the title's
     * own call says so: it passes q=0x032B2980 with buffer=0x032B2A00, exactly
     * 0x80 apart, just as its LFQueue passes 0x4059FD00/0x4059FD80. The
     * consumer is task 5 (image 1), whose CellSpursTaskArgument points into
     * this same 0x032B2xxx block and which reads the line with GETLLAR/PUTLLC,
     * so leaving it uninitialised handed recompiled SPU code a zero-depth
     * queue to reason about. */
    uint32_t q = (uint32_t)queue_ea;
    for (uint32_t o = 0; o < 128; o += 4) vm_write32(q + o, 0);
    /* W3 at +0x0C is the ring modulus the consumer reduces indices by: the
     * empty test is (W1-W0) mod 2*W3 and the buffer slot is index mod W3
     * (derived at 0x12914..0x129B8). Leaving it zero degenerated both and made
     * the queue look permanently empty no matter what a producer wrote. */
    vm_write32(q + 0x0C, depth);
    vm_write32(q + 0x10, size);
    vm_write32(q + 0x14, depth);
    vm_write64(q + 0x18, (u64)(uint32_t)buffer_ea);
    /* NOT direction at +0x24 and NOT init at +0x2C. Both sit inside the
     * 16-byte group at +0x20..+0x2F that the consumer owns and shifts wholesale
     * (shlqbyi <group>,1 at 0x12A0C). The trace shows the two values we used to
     * write there marching through it one byte per dequeue --
     *   +0x24: 00000002 -> 00000200 -> 00020000 -> 02000000
     *   +0x2C: 00000001 -> 00000100 -> 00010000 -> 01000000
     * -- i.e. we were feeding garbage into the SPU's own state every cycle.
     * Whatever holds direction/init, it is not these offsets. */
    vm_write64(q + 0x70, (u64)(uint32_t)taskset_ea);
    memset(vm_base + (uint32_t)buffer_ea, 0, (size_t)size * depth);

    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8))
        printf("[cellSpurs] _QueueInitialize(taskset=0x%08X q=0x%08X buf=0x%08X size=%u depth=%u dir=%u) -> BE line written\n",
               (u32)taskset_ea, (u32)queue_ea, (u32)buffer_ea, size, depth, direction);
    return CELL_OK;
}

s32 cellSpursQueueClear(u64 queue_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] QueueClear(q=0x%08X)\n", (u32)queue_ea);
    return CELL_OK;
}

/* SDK ABI: cellSpursQueueSize(CellSpursQueue*, unsigned int* size): the number of entries in the queue.
 *
 * Without a handler the import was unresolved: it returned CELL_OK without writing *size, so a title that waits for a
 * queue to drain (QueueSize, then a short sleep while the size is above zero) read stale stack and waited out its whole
 * retry budget every frame. The occupancy is the ring's: W0 (+0x00) the pop index, W1 (+0x04) the push index, both
 * normalised (~v marks an operation in progress) and running modulo 2*W3 (+0x0C, the depth), the same model
 * cellSpursQueuePushBody uses. A queue with no depth was never initialised. */
s32 cellSpursQueueSize(u64 queue_ea, u64 size_ea)
{
    if (!queue_ea || !size_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (queue_ea & 0x7F)       return CELL_SPURS_TASK_ERROR_ALIGN;

    uint32_t q   = (uint32_t)queue_ea;
    int32_t  w0  = (int32_t)vm_read32(q + 0x00);
    int32_t  w1  = (int32_t)vm_read32(q + 0x04);
    int32_t  mod = (int32_t)vm_read32(q + 0x0C);
    if (mod <= 0) return CELL_SPURS_TASK_ERROR_INVAL;

    int32_t  pop  = (w0 < 0) ? ~w0 : w0;
    int32_t  push = (w1 < 0) ? ~w1 : w1;
    uint32_t n    = (uint32_t)((pop > push) ? push - pop + 2 * mod : push - pop);
    vm_write32((uint32_t)size_ea, n);

    static int _n = 0;
    if (_n < 8 || (n && _n < 64))
        printf("[cellSpurs] QueueSize(q=0x%08X) -> %u (pop=%d push=%d mod=%d)\n", q, n, pop, push, mod);
    _n++;
    return CELL_OK;
}

/* SDK ABI: cellSpursQueuePushBody(CellSpursQueue*, const void* buffer, bool isBlocking) */
s32 cellSpursQueuePushBody(u64 queue_ea, u64 data_ea, u32 isBlocking)
{
    /* SPURS_QUEUE_PUSH=1 -- EXPERIMENT, off by default.
     *
     * A deliberately self-revealing probe rather than a finished push. The
     * waiter half of the protocol is known (see PROGRESS.md phase 19): the
     * 16-byte group at queue+0x20 is a ring of blocked task ids, byte[0] the
     * count and byte[13] the cursor. The DATA half is not: which ring slot an
     * element belongs in has never been observed, because nothing has ever
     * pushed and so the consumer has never taken its non-empty path.
     *
     * So push into a CANDIDATE slot (a counter kept in the push1 half of the
     * line at +0x0C), wake the registered waiter, and let the consumer's own
     * DMA say where it actually looks. Its read address names the correct slot
     * whether or not the guess was right, which turns one run into the answer.
     * Everything is logged for that reason. */
    static int s_exp = -1;
    if (s_exp < 0) s_exp = getenv("SPURS_QUEUE_PUSH") ? 1 : 0;

    static int _n = 0;
    if (!s_exp) {
        if (_n < 8)
            printf("[cellSpurs] QueuePushBody(q=0x%08X data=0x%08X blocking=%u)\n",
                   (u32)queue_ea, (u32)data_ea, isBlocking);
        else if (_n == 8)
            printf("[cellSpurs] QueuePushBody further logs suppressed\n");
        _n++;
        return CELL_OK;
    }

    if (!queue_ea || !data_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    uint32_t q   = (uint32_t)queue_ea;
    uint32_t sz  = vm_read32(q + 0x10);
    uint32_t dep = vm_read32(q + 0x14);
    uint32_t buf = (uint32_t)vm_read64(q + 0x18);
    uint32_t tsp = (uint32_t)vm_read64(q + 0x70);      /* eaSignal: taskset */
    if (!sz || !dep || !buf) return CELL_SPURS_TASK_ERROR_INVAL;

    /* Derived ring model (see PROGRESS.md phase 22). Working the dataflow of
     * the consumer's decision at 0x12914..0x129D0 backwards:
     *
     *   r11 = normalise(W0)        W0 = queue+0x00
     *   r2  = normalise(W1)        W1 = queue+0x04
     *   r60 = W3                   W3 = queue+0x0C
     *     where normalise(v) is  cgti/nor/selb  ==  (v >= 0 ? v : ~v)
     *
     *   r81 = r2 - r11
     *   r79 = (r2 + r60) - (r11 - r60)  =  r2 - r11 + 2*r60
     *   r68 = (r11 > r2) ? r79 : r81    =  (r2 - r11) mod 2*r60
     *   ceqi r77, r68, 0                -> the EMPTY test
     *
     * and, separately at 0x129A8..0x129B8,
     *
     *   r14 = (r60 > r11) ? r11 : r11 - r60   =  index mod r60
     *
     * which is the buffer slot. So this is an ordinary ring: W0 is the pop
     * index, W1 the push index, indices run modulo 2*W3, and the slot is the
     * index modulo W3 -- the standard scheme that keeps "full" distinguishable
     * from "empty". W3 is therefore the depth.
     *
     * And that is the bug: the initialiser left queue+0x0C at ZERO, so r60 = 0
     * and the whole computation degenerates -- occupancy is (W1-W0) mod 0 and
     * the slot is index mod 0. With both indices at 0 the empty test is
     * trivially true, which is precisely why the consumer parked every time and
     * never once read the element buffer, whatever the producer wrote. */
    uint8_t* qp = vm_base + q;

    int32_t  sync  = (int32_t)vm_read32(q + 0x00);          /* W0, pop  */
    int32_t  tail  = (int32_t)vm_read32(q + 0x04);          /* W1, push */
    int32_t  mod   = (int32_t)vm_read32(q + 0x0C);          /* W3       */
    if (mod <= 0) mod = (int32_t)dep;                       /* pre-fix queues */

    int32_t  nsync = (sync < 0) ? ~sync : sync;             /* normalise */
    int32_t  cur   = (tail < 0) ? ~tail : tail;
    uint32_t slot  = (uint32_t)(cur % mod);

    memcpy(vm_base + buf + (size_t)slot * sz, vm_base + (uint32_t)data_ea, sz);

    int32_t  nxt   = (cur + 1) % (2 * mod);
    int32_t  ntail = nxt;
    vm_write32(q + 0x04, (uint32_t)ntail);

    /* Waiter ring at +0x20: byte[0] = count, byte[1..12] = task ids. */
    uint32_t waiters = qp[0x20];
    int woke = -1;
    if (waiters > 0 && waiters <= 3 && tsp) {
        /* The entry area is m_bs[1..3] ONLY -- bytes 0x21..0x23. Shifting the
         * whole 16-byte group (as a first cut of this did) walks straight over
         * direction at +0x24 and init at +0x2C and corrupts the header; the
         * trace caught it as direction turning from 00000002 into 00020000. */
        woke = qp[0x21];
        qp[0x21] = qp[0x22];
        qp[0x22] = qp[0x23];
        qp[0x23] = 0;
        qp[0x20] = (uint8_t)(waiters - 1);
        extern void spu_taskset_signal_task(uint32_t taskset_ea, uint32_t taskId);
        spu_taskset_signal_task(tsp, (uint32_t)woke);
    }

    if (PS3_LOG_FIRST(_n, 16))
        printf("[cellSpurs] QueuePush#%d q=0x%08X slot=%u (cur=%u->%u) buf=0x%08X+0x%X "
               "size=%u depth=%u waiters=%u woke=%d sync=%d/%d tail=%d->%d taskset=0x%08X\n",
               _n, q, slot, cur, nxt, buf, slot * sz, sz, dep, waiters, woke, sync, nsync, tail, ntail, tsp);
    return CELL_OK;
}

/* =========================================================================
 * Misc SPURS surface the title links (logged CELL_OK stubs).
 * =====================================================================*/

s32 cellSpursRequestIdleSpu(u64 spurs_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 4)) printf("[cellSpurs] RequestIdleSpu(spurs=0x%08X)\n", (u32)spurs_ea);
    return CELL_OK;
}

s32 cellSpursGetInfo(u64 spurs_ea, u64 info_ea)
{
    struct SpursInst* si = spurs_inst_find((u32)spurs_ea);
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 4)) printf("[cellSpurs] GetInfo(spurs=0x%08X info=0x%08X)\n", (u32)spurs_ea, (u32)info_ea);
    if (info_ea) {
        vm_write32((u32)info_ea, si ? si->nspus : 1);   /* nSpus */
        for (int o = 4; o < 0x28; o += 4) vm_write32((u32)info_ea + o, 0);
    }
    return CELL_OK;
}

s32 cellSpursSetExceptionEventHandler(u64 spurs_ea, u64 handler_ea, u64 arg_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 4))
        printf("[cellSpurs] SetExceptionEventHandler(spurs=0x%08X handler=0x%08X)\n",
               (u32)spurs_ea, (u32)handler_ea);
    return CELL_OK;
}

#ifndef CELL_SPURS_POLICY_MODULE_ERROR_INVAL
#define CELL_SPURS_POLICY_MODULE_ERROR_INVAL        0x80410802u
#define CELL_SPURS_POLICY_MODULE_ERROR_PERM         0x80410809u
#define CELL_SPURS_POLICY_MODULE_ERROR_BUSY         0x8041080Au
#define CELL_SPURS_POLICY_MODULE_ERROR_ALIGN        0x80410810u
#define CELL_SPURS_POLICY_MODULE_ERROR_NULL_POINTER 0x80410811u
#endif

/* cellSpursGetWorkloadFlag (NID 0xC765B995): *flag = &spurs->wklFlag. GoW:A hands the
 * address to a WWS job that waits on the flag's lock line; unresolved, the out-param
 * stayed 0 and the job spun on GETLLAR of EA 0 forever. */
s32 cellSpursGetWorkloadFlag(u64 spurs_ea, u64 flag_pp)
{
    if (!spurs_ea || !flag_pp) return CELL_SPURS_POLICY_MODULE_ERROR_NULL_POINTER;
    if (spurs_ea & 0x7F) return CELL_SPURS_POLICY_MODULE_ERROR_ALIGN;
    vm_write32((u32)flag_pp, (u32)spurs_ea + SPURS_WKL_FLAG);
    return CELL_OK;
}

/* _cellSpursWorkloadFlagReceiver (NID 0xA73BF47E): name (or release) the workload
 * that receives the flag. Mirrors RPCS3: set requires no current receiver, clear
 * requires the caller to be it; either way the flag word resets to -1. */
s32 _cellSpursWorkloadFlagReceiver(u64 spurs_ea, u32 wid, u32 is_set)
{
    if (!spurs_ea) return CELL_SPURS_POLICY_MODULE_ERROR_NULL_POINTER;
    if (spurs_ea & 0x7F) return CELL_SPURS_POLICY_MODULE_ERROR_ALIGN;
    if (wid >= SPURS_MAXW || !s_workloads[wid].in_use) return CELL_SPURS_POLICY_MODULE_ERROR_INVAL;
    u8* recv = vm_base + (u32)spurs_ea + 0x77;
    if (is_set) { if (*recv != 0xFF) return CELL_SPURS_POLICY_MODULE_ERROR_BUSY; }
    else        { if (*recv != (u8)wid) return CELL_SPURS_POLICY_MODULE_ERROR_PERM; }
    vm_write32((u32)spurs_ea + SPURS_WKL_FLAG + 0xC, 0xFFFFFFFFu);
    *recv = is_set ? (u8)wid : 0xFF;
    printf("[cellSpurs] WorkloadFlagReceiver(wid=%u, set=%u)\n", wid, is_set);
    return CELL_OK;
}

s32 cellSpursGetWorkloadInfo(u64 spurs_ea, u32 wid, u64 info_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] GetWorkloadInfo(wid=%u info=0x%08X)\n", wid, (u32)info_ea);

    if (!spurs_ea || !info_ea)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= SPURS_MAXW || !s_workloads[wid].in_use)
        return CELL_SPURS_CORE_ERROR_INVAL;

    /* CellSpursWorkloadInfo is a GUEST out-buffer (cell/spurs/workload_types.h),
     * big-endian, 32-bit pointers. This used to return CELL_OK writing NOTHING,
     * so the guest read whatever stale bytes sat at info_ea as the descriptor.
     * Fill the header from our workload mirror + the live BE instance counters;
     * zero the pointer/name/hook fields we do not track. Byte fields are single
     * bytes (endian-neutral); multi-byte fields go through vm_write* (BE). */
    const SpursWorkload* w = &s_workloads[wid];
    u32 base = (u32)info_ea;
    for (u32 o = 0; o < 0x30; o += 4) vm_write32(base + o, 0);   /* header clean */

    vm_write64(base + 0x00, w->data);                            /* data        */
    for (int b = 0; b < 8 && b < CELL_SPURS_MAX_SPU; b++)
        *(vm_base + base + 0x08 + b) = w->priority[b];           /* priority[8] */
    vm_write32(base + 0x10, (u32)(uintptr_t)w->pm);              /* policyModule (32-bit EA) */
    vm_write32(base + 0x14, w->sizePm);                          /* sizePolicyModule */
    /* nameClass/nameInstance (0x18/0x1C): not tracked -> left 0 by the zero above. */

    u32 se = (u32)spurs_ea;
    *(vm_base + base + 0x20) = spurs_nib_get(se, SPURS_WKL_CURCONT, wid);   /* contention   */
    *(vm_base + base + 0x21) = (u8)w->minContention;                       /* minContention */
    *(vm_base + base + 0x22) = (u8)w->maxContention;                       /* maxContention */
    *(vm_base + base + 0x23) = *(vm_base + se + spurs_off_ready(wid));     /* readyCount   */
    *(vm_base + base + 0x24) = wid < 16 ? *(vm_base + se + SPURS_WKL_IDLE2 + wid) : 0; /* idleSpuRequest */
    u32 sig = vm_read32(se + spurs_off_signal(wid)) >> 16;
    *(vm_base + base + 0x25) = (sig & spurs_sig_bit(wid)) ? 1 : 0;         /* hasSignal    */
    /* SPURS_WINFO_LOG=1: what the title polls (GoW:A waits for readyCount==0 &&
     * contention==0 to call a workload idle). Logs value changes per wid. */
    { static int s_wl = -1; if (s_wl < 0) s_wl = getenv("SPURS_WINFO_LOG") ? 1 : 0;
      static u32 s_last[SPURS_MAXW]; static int s_n = 0;
      if (s_wl && wid < SPURS_MAXW) {
          u32 v = (u32)*(vm_base + base + 0x20) << 16 | (u32)*(vm_base + base + 0x23) << 8 | 1u;
          if (v != s_last[wid] && s_n++ < 400) {
              s_last[wid] = v;
              fprintf(stderr, "[winfo] wid=%u contention=%u readyCount=%u\n", wid,
                      *(vm_base + base + 0x20), *(vm_base + base + 0x23));
          }
      } }
    return CELL_OK;
}

s32 cellSpursShutdownWorkload(u64 spurs_ea, u32 wid)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] ShutdownWorkload(wid=%u)\n", wid);
    return CELL_OK;
}

s32 cellSpursWaitForWorkloadShutdown(u64 spurs_ea, u32 wid)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] WaitForWorkloadShutdown(wid=%u)\n", wid);
    return CELL_OK;
}

/* =========================================================================
 * Task LS patterns and context save area sizing.
 *
 * Saints Row 2 builds every taskset through this sequence (func_009F56E0):
 *
 *   _cellSpursTasksetAttributeInitialize(attr, ...)
 *   cellSpursTasksetAttributeSetName(attr, name)
 *   cellSpursTasksetAttributeSetTasksetSize(attr, size)
 *   cellSpursCreateTasksetWithAttribute(spurs, taskset, attr)
 *   ...
 *   _cellSpursQueueInitialize(...) / cellSpursQueueAttachLv2EventQueue(q)
 *   cellSpursTaskGetReadOnlyAreaPattern(&ro,   elf)
 *   cellSpursTaskGetContextSaveAreaSize (&size, &ls)   ls = default & ~ro
 *   _cellSpursTaskAttributeInitialize(attr2, .., size, &ls, ..)
 *
 * The ABI of the last two is not guesswork -- it is readable straight off the
 * call site. At 0x009F5878 the caller sets r3 = sp+0x70 and r4 = r25, where
 * r25 = sp+0x90 (set at 0x009F5744) is exactly the 16-byte pattern the two
 * `andc`+`std` pairs at 0x009F5888..0x009F5894 had just built. After the call
 * it does `lwz r11, 0x70(r1)` -- reading a u32 back out of r3 -- checks it
 * against 0x2D400 and bails if it does not fit, then stores the size and the
 * pattern pointer side by side at sp+0x84/sp+0x88 for the task attribute. So:
 *
 *   s32 cellSpursTaskGetContextSaveAreaSize(u32* size_out,
 *                                           const CellSpursTaskLsPattern* ls);
 *
 * A task's context save area holds the SPU register file plus every 2 KB local
 * store block the task may dirty; the read-only blocks (its code/rodata, which
 * can simply be reloaded from the ELF) are masked out by the caller first.
 * Local store is 256 KB and the pattern is 128 bits, so one bit == 2 KB.
 * =====================================================================*/

#define SPURS_LS_BLOCK      2048u    /* 256 KB local store / 128 pattern bits */
#define SPURS_CTX_HDR       2048u    /* 128 SPU registers x 16 bytes          */
#define SPURS_CTX_ALIGN     128u     /* CELL_SPURS_TASK_CONTEXT_SAVE_AREA_ALIGN */

/* cellSpursTaskGetReadOnlyAreaPattern(CellSpursTaskLsPattern* pattern,
 *                                     const void* elf)
 *
 * Mark the local-store blocks covered by the image's NON-writable PT_LOAD
 * segments. The caller clears these out of its default pattern, so anything we
 * fail to report simply ends up saved as well -- costing context space, never
 * correctness. Bit numbering is PPC-style (bit 0 = MSB of the first word), and
 * the size path below only ever popcounts the result, so a bit-order mistake
 * here cannot change the computed size.
 */
s32 cellSpursTaskGetReadOnlyAreaPattern(u64 pattern_ea, u64 elf_ea)
{
    if (!pattern_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    u64 word[2] = { 0, 0 };

    if (elf_ea) {
        const uint8_t* img = vm_base + (uint32_t)elf_ea;
        if (img[0] == 0x7F && img[1] == 'E' && img[2] == 'L' && img[3] == 'F' &&
            img[4] == 1 /*ELFCLASS32*/ && img[5] == 2 /*ELFDATA2MSB*/) {
            uint32_t e_phoff = ((uint32_t)img[0x1C] << 24) | ((uint32_t)img[0x1D] << 16) |
                               ((uint32_t)img[0x1E] << 8)  |  (uint32_t)img[0x1F];
            uint16_t e_phentsize = (uint16_t)((img[0x2A] << 8) | img[0x2B]);
            uint16_t e_phnum     = (uint16_t)((img[0x2C] << 8) | img[0x2D]);
            if (e_phentsize < 0x20) e_phentsize = 0x20;

            for (uint16_t i = 0; i < e_phnum; i++) {
                const uint8_t* ph = img + e_phoff + (size_t)i * e_phentsize;
                #define BE32(p) (((uint32_t)(p)[0] << 24) | ((uint32_t)(p)[1] << 16) | \
                                 ((uint32_t)(p)[2] << 8)  |  (uint32_t)(p)[3])
                uint32_t p_type  = BE32(ph + 0x00);
                uint32_t p_vaddr = BE32(ph + 0x08);
                uint32_t p_memsz = BE32(ph + 0x14);
                uint32_t p_flags = BE32(ph + 0x18);
                #undef BE32
                if (p_type != 1 /*PT_LOAD*/ || !p_memsz) continue;
                if (p_flags & 2 /*PF_W*/)                continue;   /* writable: must be saved */
                if ((uint64_t)p_vaddr + p_memsz > SPU_LS_SIZE) continue;

                uint32_t first = p_vaddr / SPURS_LS_BLOCK;
                uint32_t last  = (p_vaddr + p_memsz - 1) / SPURS_LS_BLOCK;
                for (uint32_t b = first; b <= last && b < 128; b++)
                    word[b >> 6] |= (u64)1 << (63 - (b & 63));
            }
        }
    }

    vm_write64((uint32_t)pattern_ea,     word[0]);
    vm_write64((uint32_t)pattern_ea + 8, word[1]);

    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8))
        printf("[cellSpurs] TaskGetReadOnlyAreaPattern(elf=0x%08X) -> %016llX %016llX\n",
               (u32)elf_ea, (unsigned long long)word[0], (unsigned long long)word[1]);
    return CELL_OK;
}

/* cellSpursTaskGetContextSaveAreaSize(u32* size_out,
 *                                     const CellSpursTaskLsPattern* lsPattern) */
s32 cellSpursTaskGetContextSaveAreaSize(u64 size_out_ea, u64 pattern_ea)
{
    if (!size_out_ea || !pattern_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    u64 hi = vm_read64((uint32_t)pattern_ea);
    u64 lo = vm_read64((uint32_t)pattern_ea + 8);

    uint32_t blocks = 0;
    for (int i = 0; i < 64; i++) {
        if (hi & ((u64)1 << i)) blocks++;
        if (lo & ((u64)1 << i)) blocks++;
    }

    uint32_t size = SPURS_CTX_HDR + blocks * SPURS_LS_BLOCK;
    size = (size + (SPURS_CTX_ALIGN - 1)) & ~(SPURS_CTX_ALIGN - 1);

    vm_write32((uint32_t)size_out_ea, size);

    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8))
        printf("[cellSpurs] TaskGetContextSaveAreaSize(ls=%016llX %016llX, %u blocks) -> %u (0x%X)\n",
               (unsigned long long)hi, (unsigned long long)lo, blocks, size, size);
    return CELL_OK;
}

/* cellSpursBarrierInitialize(CellSpursTaskset* taskset, CellSpursBarrier* barrier,
 *                            u32 total)
 *
 * A barrier is 128 bytes, 128-aligned, opaque in the SDK: the PPU only
 * initialises it, and the tasks of `taskset` meet at it from the SPU through
 * the barrier code linked into their own images. That code reads the count
 * of tasks still to arrive as a u32 at +0x04 and the owning taskset's EA at
 * +0x34, the layout reverse engineering of the SPU side has found; the rest
 * starts zeroed. Up to 128 tasks, one per taskset slot.
 */
s32 cellSpursBarrierInitialize(u64 taskset_ea, u64 barrier_ea, u32 total)
{
    if (!taskset_ea || !barrier_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if ((taskset_ea & 127u) || (barrier_ea & 127u)) return CELL_SPURS_TASK_ERROR_ALIGN;
    if (!total || total > 128u) return CELL_SPURS_TASK_ERROR_INVAL;

    for (uint32_t o = 0; o < 128u; o += 8u) vm_write64((uint32_t)barrier_ea + o, 0);
    vm_write32((uint32_t)barrier_ea + 0x04, total);
    vm_write32((uint32_t)barrier_ea + 0x34, (uint32_t)taskset_ea);

    printf("[cellSpurs] BarrierInitialize(taskset=0x%08X, barrier=0x%08X, total=%u)\n",
           (u32)taskset_ea, (u32)barrier_ea, total);
    return CELL_OK;
}

/* cellSpursGetTasksetInfo(const CellSpursTaskset* taskset, CellSpursTasksetInfo* info)
 *
 * CellSpursTasksetInfo is 128 per-task entries of 0x48 bytes, then the
 * taskset's own fields: argument (u64, +0x2400), workload id (+0x2408), last
 * scheduled task (+0x240C), name, exception handler and its argument
 * (+0x2410..+0x241B), and the taskset's size (+0x241C). The taskset-level
 * fields are filled from the taskset itself; the per-task entries are left
 * zeroed, which reads as "no task", since nothing here tracks a task's state
 * the way the SPU kernel does. Twisted Metal's post-FX init reads the
 * workload id back from it.
 */
#define TSI_ARGUMENT     0x2400u
#define TSI_WORKLOAD     0x2408u
#define TSI_LAST_TASK    0x240Cu
#define TSI_SIZE         0x241Cu
#define TSI_BYTES        0x2420u

s32 cellSpursGetTasksetInfo(u64 taskset_ea, u64 info_ea)
{
    if (!taskset_ea || !info_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (taskset_ea & 127u) return CELL_SPURS_TASK_ERROR_ALIGN;

    const uint32_t ts = (uint32_t)taskset_ea, info = (uint32_t)info_ea;
    for (uint32_t o = 0; o < TSI_BYTES; o += 4u) vm_write32(info + o, 0);
    vm_write64(info + TSI_ARGUMENT, vm_read64(ts + CSTS_ARGS));
    vm_write32(info + TSI_WORKLOAD, vm_read32(ts + CSTS_WID));
    vm_write32(info + TSI_LAST_TASK, 0);
    vm_write32(info + TSI_SIZE, vm_read32(ts + CSTS_SIZE_FIELD));
    return CELL_OK;
}

/* NID 0x8F122EF8: the SDK supports the base and extended taskset sizes.
 * Attributes retain the existing host layout; creation writes guest BE fields. */
s32 cellSpursTasksetAttributeSetTasksetSize(CellSpursTasksetAttribute* attr, u32 size)
{
    u32 ea = (u32)(uintptr_t)attr;
    if (!ea) return (s32)0x80410911u;
    if (ea & 7u) return (s32)0x80410910u;
    if (size != 0x1900u && size != 0x2900u) return (s32)0x80410902u;
    if (!spurs_guest_range_valid(ea, sizeof(*attr))) return (s32)0x8041090Du;
    attr = GUEST_PTR(attr, CellSpursTasksetAttribute*);
    attr->tasksetSize = size;
    return CELL_OK;
}

/* cellSpursQueueAttachLv2EventQueue(CellSpursQueue*)
 * The queue's completion signalling runs through our own event-flag path, so
 * there is no lv2 queue to bind; accept and log. */
s32 cellSpursQueueAttachLv2EventQueue(u64 queue_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] QueueAttachLv2EventQueue(q=0x%08X)\n", (u32)queue_ea);
    return CELL_OK;
}

/* _cellSpursLFQueueInitialize(void* pTasksetOrSpurs, CellSpursLFQueue* pQueue,
 *      const void* buffer, u32 size, u32 depth, u32 direction)
 *
 * This one CANNOT be a stub, because the consumer is recompiled SPU code that
 * reads the queue out of main memory itself. Disassembling image 1 around the
 * task's WAIT_SIGNAL call site (LS 0x12B80) shows the real protocol: it builds
 * a 128-byte line at LS 0x80 and commits it with
 *
 *   wrch MFC_LSA,0x80 / MFC_EAH / MFC_EAL / MFC_Size,128 / MFC_Cmd,0xB4 (PUTLLC)
 *   rdch MFC_RdAtomicStat ; brnz -> retry
 *
 * i.e. a GETLLAR/PUTLLC lock-line atomic on the queue's own cache line. So the
 * bytes in guest memory ARE the interface, and they have to be the real
 * big-endian CellSyncLFQueue: one 128-byte, 128-aligned line.
 *
 *   0x00 pop1     0x10 size     0x18 buffer(u64)  0x24 direction  0x2C init
 *   0x08 push1    0x14 depth    0x20 bs[4]        0x28 v1         0x70 eaSignal
 *
 * The title's own call corroborates the size: it passes q=0x4059FD00 with
 * buffer=0x4059FD80, exactly 128 bytes later.
 *
 * NOTE: libs/sync/cellSync.c has a CellSyncLFQueue too, but that one is a
 * HOST-native struct (atomic_uint, a 64-bit host buffer pointer). It is fine
 * for a queue both of whose ends are HLE, and completely wrong here -- writing
 * it into guest memory would hand the SPU a host pointer where a 32-bit big-
 * endian EA belongs. Hence a separate, guest-accurate initializer rather than
 * delegating to it.
 *
 * What is set here is only what the arguments determine outright: size, depth,
 * buffer, direction, and the init flag, over a zeroed line (the documented
 * empty state). The bs[]/v1 slot state machine is left zero -- see PROGRESS.md;
 * it is not guessed at. */
s32 _cellSpursLFQueueInitialize(u64 owner_ea, u64 queue_ea, u64 buffer_ea,
                                u32 size, u32 depth, u32 direction)
{
    if (!queue_ea || !buffer_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (!size || !depth)         return CELL_SPURS_TASK_ERROR_INVAL;

    uint32_t q = (uint32_t)queue_ea;
    for (uint32_t o = 0; o < 128; o += 4) vm_write32(q + o, 0);

    /* W3 at +0x0C is the ring modulus the consumer reduces indices by: the
     * empty test is (W1-W0) mod 2*W3 and the buffer slot is index mod W3
     * (derived at 0x12914..0x129B8). Leaving it zero degenerated both and made
     * the queue look permanently empty no matter what a producer wrote. */
    vm_write32(q + 0x0C, depth);
    vm_write32(q + 0x10, size);
    vm_write32(q + 0x14, depth);
    vm_write64(q + 0x18, (u64)(uint32_t)buffer_ea);   /* bcptr<void,u64> */
    /* NOT direction at +0x24 and NOT init at +0x2C. Both sit inside the
     * 16-byte group at +0x20..+0x2F that the consumer owns and shifts wholesale
     * (shlqbyi <group>,1 at 0x12A0C). The trace shows the two values we used to
     * write there marching through it one byte per dequeue --
     *   +0x24: 00000002 -> 00000200 -> 00020000 -> 02000000
     *   +0x2C: 00000001 -> 00000100 -> 00010000 -> 01000000
     * -- i.e. we were feeding garbage into the SPU's own state every cycle.
     * Whatever holds direction/init, it is not these offsets. */                          /* init: constructed */
    vm_write64(q + 0x70, (u64)(uint32_t)owner_ea);    /* eaSignal <- taskset/spurs */

    memset(vm_base + (uint32_t)buffer_ea, 0, (size_t)size * depth);

    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8))
        printf("[cellSpurs] _LFQueueInitialize(owner=0x%08X q=0x%08X buf=0x%08X size=%u depth=%u dir=%u) -> BE line written\n",
               (u32)owner_ea, (u32)queue_ea, (u32)buffer_ea, size, depth, direction);
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * _cellSpursLFQueuePushBody(queue_ea, elem_ea, isBlocking)
 *
 * PPU-side push onto a SPURS lock-free queue. The consumer here is the SCE
 * "Edge Zlib" decompressor: a SPURS task that pops 32-byte
 * EdgeZlibInflateQElement records and inflates one block each. We have no
 * SPURS taskset dispatch for it, so the push is serviced synchronously on the
 * calling PPU thread with a host inflate (the PPU-side fallback strategy the
 * runtime already documents for SPU jobs). Element layout (big-endian):
 *
 *   +0x00 eaCompressed            +0x10 eaNumElementsToDecompress (u32 counter)
 *   +0x04 compressedSize          +0x14 eaEventFlag (CellSpursEventFlag*, or 0)
 *   +0x08 eaUncompressed          +0x18 eventFlagBits (u16)
 *   +0x0C uncompressedSize
 *
 * Anything that does not decode as such an element is only logged (the old
 * behaviour for an unregistered import: report success, drop the data).
 * -------------------------------------------------------------------------*/
extern int stbi_zlib_decode_buffer(char* obuffer, int olen, const char* ibuffer, int ilen);
extern int stbi_zlib_decode_noheader_buffer(char* obuffer, int olen, const char* ibuffer, int ilen);
extern uint32_t ppu_vm_size;        /* guest address-space size (ppu_loader.cpp) */

s32 _cellSpursLFQueuePushBody(u64 queue_ea, u64 elem_ea, u32 isBlocking)
{
    uint32_t q = (uint32_t)queue_ea, e = (uint32_t)elem_ea;
    if (!q || !e) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    (void)isBlocking;

    uint32_t esize = vm_read32(q + 0x10);
    uint32_t w[8];
    for (int i = 0; i < 8; i++) w[i] = vm_read32(e + 4u * (uint32_t)i);
    uint16_t bits = (uint16_t)(w[6] >> 16);   /* eventFlagBits: u16 at +0x18 */

    static int s_log = 0;
    int log_this = (s_log < 24) || getenv("SPURS_LFQ_LOG");
    if (log_this) {
        s_log++;
        uint32_t c0 = (w[0] > 0x10000 && (uint64_t)w[0] + 4 <= (ppu_vm_size ? (uint64_t)ppu_vm_size : 0xFFFF0000ull)) ? vm_read32(w[0]) : 0;
        printf("[cellSpurs] LFQueuePush q=0x%08X esize=%u elem=0x%08X: %08X %08X %08X %08X %08X %08X %08X %08X  (src[0]=%08X)\n",
               q, esize, e, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], c0);
        for (int i = 0; i < 8; i++) {
            if (w[i] > 0x10000 && (uint64_t)w[i] + 16 <= (ppu_vm_size ? (uint64_t)ppu_vm_size : 0xFFFF0000ull)) {
                const uint8_t* b = vm_base + (w[i] & ~1u);
                char ln[160];
                snprintf(ln, sizeof ln, "[cellSpurs]    elem.w%d -> %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n", i,
                         b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
                fputs(ln, stdout);
            }
        }
    }

    /* Observed element (BCUS98232): {eaIn, eaOut, compSize, outCapacity,
     * &counter|1, eventFlag, bits<<16, 0}. eaIn is RAW deflate (first byte 0x95
     * = final dynamic block), not zlib-wrapped. */
    uint32_t csz = w[2], usz = w[3];
    uint64_t vmsz = ppu_vm_size ? (uint64_t)ppu_vm_size : 0xFFFF0000ull;   /* 0 = whole 4 GB space backed */
    int sane = (esize == 32) && w[0] > 0x10000 && w[1] > 0x10000 &&
               csz > 0 && usz > 0 && csz <= (1u << 20) && usz <= (1u << 20) &&
               (uint64_t)w[0] + csz <= vmsz && (uint64_t)w[1] + usz <= vmsz;
    if (!sane) {
        static int s_bad = 0;
        if (s_bad++ < 8)
            printf("[cellSpurs] LFQueuePush: unrecognized element (esize=%u) -- dropped\n", esize);
        return CELL_OK;
    }

    const uint8_t* src = vm_base + w[0];
    uint8_t* dst = vm_base + w[1];
    int ok = 0;
    int n_out = 0;
    {
        /* Raw deflate is the norm here; accept a zlib-wrapped stream too. */
        /* The caller reads the compressed block into the TAIL of the output
         * buffer and inflates in place (eaIn + compSize == eaOut + outCap in
         * the observed element), so decode from a private copy of the input. */
        uint8_t* tmp = (uint8_t*)malloc(csz);
        if (!tmp) return CELL_OK;
        memcpy(tmp, src, csz);
        src = tmp;
        int hdr = ((src[0] & 0x0F) == 8) && ((((unsigned)src[0] << 8) | src[1]) % 31u == 0);
        /* w6 = eventBits<<16 | skip: the element asks for outCap bytes starting `skip` bytes into
         * the decompressed block (FIOS streams a compressed PSARC entry this way; GoW:A reads its .dcb
         * files in 0x40/0x10 slices). Ignoring skip hands back the block prefix every time. */
        uint32_t skip = w[6] & 0xFFFFu;
        n_out = 0;
        if (skip == 0)
            n_out = hdr ? stbi_zlib_decode_buffer((char*)dst, (int)usz, (const char*)src, (int)csz)
                        : stbi_zlib_decode_noheader_buffer((char*)dst, (int)usz, (const char*)src, (int)csz);
        if (n_out <= 0 && usz < (1u << 17)) {
            /* Partial request (a slice of a larger block): inflate the whole block privately and
             * hand back [skip, skip+outCap). One-entry cache: slices of one block arrive in a row. */
#if defined(_MSC_VER)
#  define ZSLICE_TLS __declspec(thread)
#else
#  define ZSLICE_TLS _Thread_local
#endif
            /* per thread: several guest threads push elements concurrently */
            static ZSLICE_TLS char* s_blk = NULL; static ZSLICE_TLS int s_blk_n = 0;
            static ZSLICE_TLS uint32_t s_key_ea = 0, s_key_csz = 0; static ZSLICE_TLS uint8_t s_key_head[16];
            int hit = s_blk && s_key_ea == w[0] && s_key_csz == csz &&
                      !memcmp(s_key_head, src, csz < 16 ? csz : 16);
            if (!hit) {
                const int cap2 = 1 << 17;
                if (!s_blk) s_blk = (char*)malloc((size_t)cap2);
                s_blk_n = s_blk ? (hdr ? stbi_zlib_decode_buffer(s_blk, cap2, (const char*)src, (int)csz)
                                       : stbi_zlib_decode_noheader_buffer(s_blk, cap2, (const char*)src, (int)csz)) : 0;
                s_key_ea = w[0]; s_key_csz = csz; memcpy(s_key_head, src, csz < 16 ? csz : 16);
                if (s_blk_n <= 0) s_key_csz = 0;
            }
            if (s_blk_n > (int)skip) {
                int avail = s_blk_n - (int)skip;
                n_out = (avail < (int)usz) ? avail : (int)usz;
                memcpy(dst, s_blk + skip, (size_t)n_out);
            }
        }
        ok = (n_out > 0 && n_out <= (int)usz);
        if (!ok) {
            static int s_mm = 0;
            if (s_mm++ < 16)
                printf("[cellSpurs] LFQueuePush: inflate failed (%d) csz=%u cap=%u hdr=%d src=%02X%02X%02X%02X %02X%02X%02X%02X ... tail=%02X%02X%02X%02X -- element dropped\n",
                       n_out, csz, usz, hdr, src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
                       src[csz - 4], src[csz - 3], src[csz - 2], src[csz - 1]);
        }
        free(tmp);
    }
    if (!ok) return CELL_OK;

    /* Completion: the decompressor decrements the caller's pending-element
     * counter, then (when it reaches zero) raises the event flag. */
    uint32_t ctr = w[4] & ~3u;     /* low bits are tag flags (observed 0x...CAD) */
    uint32_t left = 1;
    if (ctr > 0x10000 && (uint64_t)ctr + 4 <= vmsz) {
        uint32_t* p = (uint32_t*)(vm_base + ctr);
        uint32_t old = __atomic_load_n(p, __ATOMIC_SEQ_CST), nv;
        do {
            uint32_t cur = __builtin_bswap32(old);
            left = cur ? cur - 1u : 0u;
            nv = __builtin_bswap32(left);
        } while (!__atomic_compare_exchange_n(p, &old, nv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
        ppu_resv_break_store(ctr);
    }
    if (left == 0 && w[5] > 0x10000 && bits)
        cellSpursEventFlagSet((CellSpursEventFlag*)(uintptr_t)w[5], bits);

    static int s_done = 0;
    s_done++;
    static int s_zlog = -1;
    if (s_zlog < 0) { const char* e = getenv("SPURS_ZLOG"); s_zlog = (e && *e && *e != '0') ? 1 : 0; }
    if (s_zlog)   /* SPURS_ZLOG=1: every element, raw words included */
        printf("[ZLIB] #%d in=%08X out=%08X csz=%u cap=%u w4=%08X w5=%08X w6=%08X w7=%08X -> %d\n",
               s_done, w[0], w[1], csz, usz, w[4], w[5], w[6], w[7], n_out);
    else if (s_done <= 6 || (s_done % 2048) == 0)
        printf("[cellSpurs] LFQueuePush: inflated %u -> %d bytes (#%d, counter now %u)\n",
               csz, n_out, s_done, left);
    return CELL_OK;
}

s32 cellSpursLFQueueAttachLv2EventQueue(u64 queue_ea)
{
    static int _n = 0;
    if (PS3_LOG_FIRST(_n, 8)) printf("[cellSpurs] LFQueueAttachLv2EventQueue(q=0x%08X)\n", (u32)queue_ea);
    return CELL_OK;
}
