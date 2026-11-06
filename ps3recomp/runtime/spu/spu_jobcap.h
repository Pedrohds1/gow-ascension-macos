/*
 * SPU job capture (SPU_JOB_CAPTURE, spu_channels.c) and replay
 * (SPU_JOB_REPLAY, spu_job_replay.c): the file layout both sides share and the
 * hooks the MFC, channel and dispatch paths call. The format itself is
 * documented in the header comment of the "SPU job capture" block in
 * spu_channels.c; docs/EDGE_NATIVE.md has the design.
 */
#ifndef SPU_JOBCAP_H
#define SPU_JOBCAP_H

#include "spu_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPU_JOBCAP_VERSION 2

enum { JC_GET = 1, JC_PUT = 2, JC_GETLLAR = 3, JC_PUTLLC = 4, JC_PUTLLUC = 5,
       JC_RDCH = 6, JC_POLL = 7, JC_PMCALL = 8, JC_IRQ = 9, JC_RCHCNT = 10,
       JC_END = 15 };
enum { JC_END_RETURN = 1, JC_END_HALT = 2, JC_END_NEXTJOB = 3 };

/* Context state at job start (version 2). These are the fields lifted code or
 * the drain loop read directly instead of through a recorded channel read:
 * without int_enable/event_mask/event_status a replayed context never takes
 * the interrupts the WWS policy module's completion tail depends on. All
 * 32-bit, so the block has no padding. */
typedef struct {
    uint32_t int_enable, event_status, event_mask, srr0;
    uint32_t decrementer, status, stop_code, host_depth;
    uint32_t mfc_lsa, mfc_eah, mfc_eal, mfc_size, mfc_tag;
    uint32_t mfc_tag_mask, mfc_tag_status;
    uint32_t mfc_tag_completed, mfc_queue_count;   /* the context's MFC engine */
    uint32_t resv_ea, resv_valid, atomic_stat;
    uint32_t list_stall_mask, list_stall_stat;
    uint32_t irq_saved, irq_resume_pc;
    int32_t  module_img_a00;
    uint32_t steps;
    uint32_t list_stall_elem_lsa[32], list_stall_remaining[32];
    uint32_t list_stall_dest_lsa[32], list_stall_cmd[32];
    uint32_t list_stall_ea_base[32][2];            /* {lo, hi} */
    uint8_t  resv_line[128];
    uint32_t irq_gpr[128][4];                      /* as ctx->irq_gpr stores them */
} spu_jobcap_state;

typedef struct {
    char     magic[8];
    uint32_t version, header_size, ls_size, gpr_count, wid;
    int32_t  image_id, resident_ovl, resident_task;
    uint32_t spu_id;
    int32_t  policy_mode;
    uint32_t entry_pc, link, seq, ms, code_slots;
    struct { uint32_t lsa, size, source_ea; int32_t image_id; } code[SPU_RESIDENT_CODE_SLOTS];
    spu_jobcap_state st;                           /* version 2 */
} spu_jobcap_header;

/* One event record; `n` payload bytes follow it. */
typedef struct {
    uint32_t type, n, pc, a, b, c, d, index;
} spu_jobcap_event;

/* Capture side (spu_channels.c). g_spu_jobcap_on > 0 only while the env is set. */
extern int g_spu_jobcap_on_v;
void spu_jobcap_xfer(spu_context* spu, uint32_t cmd, uint64_t ea, uint32_t lsa,
                     uint32_t size, const void* bytes);
/* Nonzero while a capture or a replay counts trampoline hops (spu_drain.c
 * spu_hop calls spu_job_hop) and wants each interrupt taken (spu_job_irq). */
extern int g_spu_jobhop_on_v;
void spu_job_hop(spu_context* ctx);
void spu_job_irq(spu_context* ctx);
/* spu_drain_call returning at its return_pc (the host-frame return of a
 * lifted call): a captured job's return to its link ends the capture. */
void spu_job_drain_return(spu_context* ctx, uint32_t return_pc);
/* Nonzero when something wants spu_job_drain_return called (capture, replay,
 * or the SPURS_PM_PROF job timer). */
extern int g_spu_jobret_on_v;
/* SPURS_PM_PROF job timer: the job proper's wall time per host thread (QPC
 * ticks) and a histogram in 10 us buckets, the last one open-ended.
 * spurs_policy.c consumes and zeroes both after each module run. */
#define SPU_JOBT_BUCKETS 101
extern SPU_THREAD_LOCAL long long g_spu_jobt_ticks;
extern SPU_THREAD_LOCAL unsigned g_spu_jobt_done;
extern SPU_THREAD_LOCAL unsigned g_spu_jobt_hist[SPU_JOBT_BUCKETS];
/* The context's MFC engine tag state, read (set = 0) or written (set = 1). */
void spu_jobcap_mfc_state(spu_context* ctx, uint32_t* tag_completed,
                          uint32_t* queue_count, int set);

/* Replay side (spu_job_replay.c). g_spu_jobrep_on is 1 only inside
 * spu_job_replay_run; every hook below is a no-op for a context that is not
 * the replay session's. Hooks that return int return nonzero when they served
 * the operation. */
extern int g_spu_jobrep_on_v;
/* The hot paths test the flags through these names. A build without
 * SPU_JOBLAB (CMake PS3RECOMP_SPU_JOBLAB, off by default) makes them constant
 * 0, so capture, replay and the job timer cost nothing there: the run-time
 * checks measured -1.1% level fps (swapped A/B, env unset). The _v variables
 * stay so the cold code still builds. */
#ifndef SPU_JOBLAB
#define SPU_JOBLAB 0
#endif
#if SPU_JOBLAB
#define g_spu_jobcap_on g_spu_jobcap_on_v
#define g_spu_jobhop_on g_spu_jobhop_on_v
#define g_spu_jobret_on g_spu_jobret_on_v
#define g_spu_jobrep_on g_spu_jobrep_on_v
#else
#define g_spu_jobcap_on 0
#define g_spu_jobhop_on 0
#define g_spu_jobret_on 0
#define g_spu_jobrep_on 0
#endif
int  spu_jobrep_xfer(spu_context* ctx, uint32_t cmd, uint64_t ea, uint32_t lsa, uint32_t size);
int  spu_jobrep_atomic(spu_context* ctx, uint32_t cmd, uint32_t ea, uint32_t lsa);
int  spu_jobrep_rdch(spu_context* ctx, uint32_t channel, u128* out);
int  spu_jobrep_rchcnt(spu_context* ctx, uint32_t channel, uint32_t* out);
void spu_jobrep_pmcall(spu_context* ctx);
int  spu_jobrep_poll(spu_context* ctx);
void spu_jobrep_jobstart(spu_context* ctx);
void spu_jobrep_halt(spu_context* ctx);
void spu_jobrep_hop(spu_context* ctx);
void spu_jobrep_irq(spu_context* ctx);
/* Replays every job_*.bin in `dir`; returns the process exit code (0 = every
 * job identical). */
int  spu_job_replay_run(const char* dir);

#ifdef __cplusplus
}
#endif

#endif /* SPU_JOBCAP_H */
