/*
 * SPU job replay: SPU_JOB_REPLAY=<dir> (docs/EDGE_NATIVE.md, "Replay").
 *
 * Re-runs SPU_JOB_CAPTURE files (format: spu_channels.c, "SPU job capture";
 * structs in spu_jobcap.h) on the lifted code, in isolation: no PPU, no GPU,
 * no other SPU. The port calls spu_job_replay_run() once the SPU images,
 * overlays and fragments are registered (static constructors) and exits with
 * its return value.
 *
 * One job: a fresh context gets the captured LS, GPRs, dispatch fields and
 * state block and enters at the snapshot pc through the same bracket the WWS
 * policy module enters a job with (`bisle`: host_depth++, dispatch,
 * spu_drain_call(link)). It runs under spu_run_with_halt, as on a kernel
 * thread, until it ends where its capture closed: the return to its link, a
 * halt, or the next job start. Everything the job asks
 * of the outside is served from the transcript, in order:
 *   GET -> the recorded bytes into LS; GETLLAR -> the recorded line;
 *   PUTLLC -> the recorded verdict; RDCH / RCHCNT -> the recorded value;
 *   POLL -> the recorded r3 and LS 0x100..0x17F; IRQ -> taken at the
 *   recorded hop (event bits a peer raised are re-raised there).
 * A request that is not the next recorded event, or differs from it in type,
 * cmd, ea, lsa, size, channel or target, is a divergence: the job stops there
 * and the report names the event index, both sides and the pc.
 * Outputs are compared, not performed: every PUT / PUTLLC / PUTLLUC payload
 * against the transcript, then the GPRs and the whole LS against the END
 * record. The first differing byte is reported.
 *
 *   SPU_JOB_REPLAY=<dir|file>      replay every job_*.bin in <dir> (or one file)
 *   SPU_JOB_REPLAY_REPEAT=<n>      run each job n times, report the median time
 *
 * Exit code 0 when every job is identical.
 */
#include "spu_context.h"
#include "spu_jobcap.h"
#include "spu_coherency.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../platform/win32_compat.h"

#if defined(_MSC_VER)
#  define JR_TLS __declspec(thread)
#else
#  define JR_TLS __thread
#endif

int g_spu_jobrep_on_v = 0;

typedef void (*jr_fn)(spu_context*);
extern int    spu_run_with_halt(void (*)(spu_context*), spu_context*);
extern jr_fn  spu_lookup(uint32_t addr, int image_id);
extern void   spu_halt(spu_context*);
extern void   spu_mfc_release(spu_context*);
extern void   spu_overlay_note_get(spu_context*, uint32_t ea, const uint8_t* ls, uint32_t size);
extern unsigned spu_recent_pcs(uint32_t* out, unsigned max);
extern void   spu_fn_probe_job(const char* name, int run);   /* spu_fn_probe.c */

typedef struct {
    spu_context*             ctx;
    const uint8_t*           buf;
    size_t                   len;
    const spu_jobcap_header* h;
    size_t                   off;      /* next unconsumed event record */
    const spu_jobcap_event*  nx;       /* the record at off, NULL past the end */
    uint32_t                 hops;     /* trampoline hops since the last consumed event */
    unsigned long long       hop_total;
    long long                deadline; /* QPC: a run past it is a divergence (no progress) */
    int                      entered, ended, end_kind, diverged;
    uint32_t                 halt_pc;
    unsigned                 outputs, out_bad, notes;
    char                     why[768];
    char                     out_why[320];
} jr_session;

static JR_TLS jr_session* t_jr;

static const char* jr_type_name(uint32_t t)
{
    switch (t) {
    case JC_GET: return "GET"; case JC_PUT: return "PUT"; case JC_GETLLAR: return "GETLLAR";
    case JC_PUTLLC: return "PUTLLC"; case JC_PUTLLUC: return "PUTLLUC"; case JC_RDCH: return "RDCH";
    case JC_POLL: return "POLL"; case JC_PMCALL: return "PMCALL"; case JC_IRQ: return "IRQ";
    case JC_RCHCNT: return "RCHCNT"; case JC_END: return "END"; default: return "?";
    }
}

/* One recorded event, in the same terms the replay's request is printed in. */
static void jr_describe(const spu_jobcap_event* e, char* out, size_t n)
{
    switch (e->type) {
    case JC_GET: case JC_PUT:
        snprintf(out, n, "%s cmd=0x%X ea=0x%08X lsa=0x%05X size=%u", jr_type_name(e->type), e->a, e->b, e->c, e->d);
        break;
    case JC_GETLLAR: case JC_PUTLLC: case JC_PUTLLUC:
        snprintf(out, n, "%s cmd=0x%X ea=0x%08X lsa=0x%05X", jr_type_name(e->type), e->a, e->b, e->c);
        break;
    case JC_RDCH:    snprintf(out, n, "RDCH ch=%u", e->a); break;
    case JC_RCHCNT:  snprintf(out, n, "RCHCNT ch=%u", e->a); break;
    case JC_POLL:    snprintf(out, n, "POLL"); break;
    case JC_PMCALL:  snprintf(out, n, "PMCALL target=0x%05X link=0x%05X", e->a, e->b); break;
    case JC_IRQ:     snprintf(out, n, "IRQ srr0=0x%05X status=0x%X mask=0x%X after %u hops", e->a, e->b, e->c, e->d); break;
    case JC_END:     snprintf(out, n, "END reason=%u pc=0x%05X", e->a, e->b); break;
    default:         snprintf(out, n, "type %u", e->type); break;
    }
}

static const spu_jobcap_event* jr_at(const jr_session* s, size_t off)
{
    if (off + sizeof(spu_jobcap_event) > s->len) return NULL;
    const spu_jobcap_event* e = (const spu_jobcap_event*)(s->buf + off);
    if (off + sizeof *e + e->n > s->len) return NULL;
    return e;
}

static void jr_consume(jr_session* s)
{
    s->off += sizeof(spu_jobcap_event) + s->nx->n;
    s->nx = jr_at(s, s->off);
    s->hops = 0;
}

/* Stops the job: the report keeps the first reason. spu_halt longjmps back to
 * spu_run_with_halt; it only returns if no run is armed, so callers bail out
 * on a NULL/0 result. */
static void jr_diverge(jr_session* s, const char* fmt, ...)
{
    if (!s->diverged) {
        s->diverged = 1;
        va_list ap; va_start(ap, fmt);
        int p = vsnprintf(s->why, sizeof s->why, fmt, ap);
        va_end(ap);
        if (p < 0) p = 0;
        uint32_t r[8]; const unsigned k = spu_recent_pcs(r, 8);
        p += snprintf(s->why + p, p < (int)sizeof s->why ? sizeof s->why - p : 0,
                      " (pc=0x%05X lr=0x%05X; recent hops:", (unsigned)(s->ctx->pc & SPU_LS_MASK),
                      (unsigned)(s->ctx->gpr[0]._u32[0] & SPU_LS_MASK));
        for (unsigned i = 0; i < k && p < (int)sizeof s->why; i++)
            p += snprintf(s->why + p, sizeof s->why - p, " %05X", r[i]);
        if (p < (int)sizeof s->why) snprintf(s->why + p, sizeof s->why - p, ")");
    }
    s->ctx->status = SPU_STATUS_STOPPED_BY_HALT;
    spu_halt(s->ctx);
}

/* The next recorded event, which must be of `type`; `got` says what the
 * replay asked for. NULL = divergence (already reported). */
static const spu_jobcap_event* jr_expect(jr_session* s, uint32_t type, const char* got)
{
    const spu_jobcap_event* e = s->nx;
    if (!e) { jr_diverge(s, "transcript ends (truncated file), replay issued %s", got); return NULL; }
    if (e->type != type) {
        char want[160]; jr_describe(e, want, sizeof want);
        jr_diverge(s, "event #%u: recorded %s, replay issued %s", e->index, want, got);
        return NULL;
    }
    return e;
}

/* A field of the request that differs from the recorded event. */
static int jr_mismatch(jr_session* s, const spu_jobcap_event* e, const char* got)
{
    char want[160]; jr_describe(e, want, sizeof want);
    jr_diverge(s, "event #%u: recorded %s, replay issued %s", e->index, want, got);
    return 1;
}

/* An output the job produced (PUT payload, offered atomic line) against the
 * recorded one. Not a divergence: what the job reads next comes from the
 * transcript either way, so it runs on and the END compare still reports. */
static void jr_cmp_output(jr_session* s, const spu_jobcap_event* e, const uint8_t* got, uint32_t size)
{
    s->outputs++;
    const uint8_t* want = (const uint8_t*)(e + 1);
    for (uint32_t i = 0; i < size; i++) {
        const uint8_t g = got ? got[i] : 0;   /* NULL: SDCRZ writes zeros */
        if (g == want[i]) continue;
        if (!s->out_bad++)
            snprintf(s->out_why, sizeof s->out_why,
                     "event #%u %s ea=0x%08X size=%u: first difference at +0x%X (ea 0x%08X): "
                     "recorded %02X, replay %02X", e->index, jr_type_name(e->type), e->b, size,
                     i, e->b + i, want[i], g);
        return;
    }
}

static void jr_note(jr_session* s, const spu_jobcap_event* e, uint32_t ch, uint32_t live, uint32_t rec)
{
    if (s->notes++ < 3)
        fprintf(stderr, "[replay]   note: event #%u ch %u: live context value 0x%X, recorded 0x%X "
                "(a context field the state block does not restore)\n", e->index, ch, live, rec);
}

int spu_jobrep_xfer(spu_context* ctx, uint32_t cmd, uint64_t ea, uint32_t lsa, uint32_t size)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return 0;
    const int sdcrz = (cmd & 0xFFFFu) == MFC_SDCRZ_CMD;
    const int put = sdcrz || !(cmd & 0x40u);
    char got[128];
    snprintf(got, sizeof got, "%s cmd=0x%X ea=0x%08X lsa=0x%05X size=%u", put ? "PUT" : "GET",
             cmd, (uint32_t)ea, lsa, size);
    const spu_jobcap_event* e = jr_expect(s, put ? JC_PUT : JC_GET, got);
    if (!e) return 1;
    if (e->a != cmd || e->b != (uint32_t)ea || e->c != lsa || e->d != size || e->n != size)
        return jr_mismatch(s, e, got);
    if (put) {
        jr_cmp_output(s, e, sdcrz ? NULL : ctx->ls + lsa, size);
        jr_consume(s);
    } else {
        memcpy(ctx->ls + lsa, e + 1, size);
        jr_consume(s);
        /* As the live GET path does: a payload that is registered code makes
         * its overlay or EDGE fragment resident. */
        spu_overlay_note_get(ctx, (uint32_t)ea, ctx->ls + lsa, size);
    }
    return 1;
}

int spu_jobrep_atomic(spu_context* ctx, uint32_t cmd, uint32_t ea, uint32_t lsa)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return 0;
    const uint32_t type = cmd == MFC_GETLLAR_CMD ? JC_GETLLAR : cmd == MFC_PUTLLC_CMD ? JC_PUTLLC : JC_PUTLLUC;
    char got[128];
    snprintf(got, sizeof got, "%s cmd=0x%X ea=0x%08X lsa=0x%05X", jr_type_name(type), cmd, ea, lsa);
    const spu_jobcap_event* e = jr_expect(s, type, got);
    if (!e) return 1;
    if (e->a != cmd || e->b != ea || e->c != lsa || e->n != 128) return jr_mismatch(s, e, got);
    uint8_t* ls = ctx->ls + lsa;
    if (type == JC_GETLLAR) {
        memcpy(ls, e + 1, 128);
        memcpy(ctx->resv_line, e + 1, 128);
        ctx->resv_ea = ea; ctx->resv_valid = (int)((ea & ~127u) | 1u); ctx->atomic_stat = 4;
        ctx->dbg_getllar++;
    } else {
        jr_cmp_output(s, e, ls, 128);
        ctx->atomic_stat = type == JC_PUTLLC ? (e->d ? 0u : 1u) : 2u;
        ctx->resv_valid = 0;
    }
    jr_consume(s);
    return 1;
}

int spu_jobrep_rdch(spu_context* ctx, uint32_t channel, u128* out)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return 0;
    char got[48]; snprintf(got, sizeof got, "RDCH ch=%u", channel);
    const spu_jobcap_event* e = jr_expect(s, JC_RDCH, got);
    if (!e) { memset(out, 0, sizeof *out); return 1; }
    if (e->a != channel || e->n != 16) { jr_mismatch(s, e, got); memset(out, 0, sizeof *out); return 1; }
    memcpy(out, e + 1, 16);
    const uint32_t rec = out->_u32[0];
    /* Channels whose value is pure context state: the replay's own value must
     * agree, or the state block missed a field. */
    int pure = 1; uint32_t live = 0;
    switch (channel) {
    case SPU_RdMachStat:      live = ctx->int_enable & 1u;  break;
    case SPU_RdEventMask:     live = ctx->event_mask;       break;
    case SPU_RdSRR0:          live = ctx->srr0;             break;
    case MFC_RdTagMask:       live = ctx->mfc_tag_mask;     break;
    case MFC_RdTagStat:       live = ctx->mfc_tag_status;   break;
    case MFC_RdAtomicStat:    live = ctx->atomic_stat;      break;
    case MFC_RdListStallStat: live = ctx->list_stall_stat;  break;
    default: pure = 0; break;
    }
    if (pure && live != rec) jr_note(s, e, channel, live, rec);
    /* The read's own side effects. RdEventStat returns event_status whole, so
     * the recorded value IS the context's event status at that read (bits a
     * peer raised included). */
    if (channel == MFC_RdListStallStat) ctx->list_stall_stat = 0;   /* clear on read */
    if (channel == SPU_RdEventStat) ctx->event_status = rec;
    jr_consume(s);
    return 1;
}

int spu_jobrep_rchcnt(spu_context* ctx, uint32_t channel, uint32_t* out)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return 0;
    char got[48]; snprintf(got, sizeof got, "RCHCNT ch=%u", channel);
    const spu_jobcap_event* e = jr_expect(s, JC_RCHCNT, got);
    if (!e) { *out = 0; return 1; }
    if (e->a != channel) { jr_mismatch(s, e, got); *out = 0; return 1; }
    *out = e->b;
    if (channel == MFC_RdListStallStat && (ctx->list_stall_stat ? 1u : 0u) != e->b)
        jr_note(s, e, channel, ctx->list_stall_stat ? 1u : 0u, e->b);
    if (channel == SPU_RdEventStat) ctx->event_status = e->c;   /* incl. an Lr the poll raised */
    jr_consume(s);
    return 1;
}

void spu_jobrep_pmcall(spu_context* ctx)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return;
    const uint32_t pc = ctx->pc & SPU_LS_MASK, link = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
    if (pc == s->h->link) {
        /* The capture closed here (END reason 1, the job's return to its link). */
        if (s->nx && s->nx->type == JC_END && s->nx->a == JC_END_RETURN) {
            s->ended = 1; s->end_kind = JC_END_RETURN; s->halt_pc = pc;
            ctx->status = SPU_STATUS_STOPPED_BY_HALT;
            spu_halt(ctx);
            return;
        }
    }
    char got[64]; snprintf(got, sizeof got, "PMCALL target=0x%05X link=0x%05X", pc, link);
    const spu_jobcap_event* e = jr_expect(s, JC_PMCALL, got);
    if (!e) return;
    if (e->a != pc || e->b != link) { jr_mismatch(s, e, got); return; }
    jr_consume(s);
}

int spu_jobrep_poll(spu_context* ctx)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return 0;
    const spu_jobcap_event* e = jr_expect(s, JC_POLL, "POLL");
    if (!e) return 1;
    if (e->n != 0x80) return jr_mismatch(s, e, "POLL");
    ctx->gpr[3] = spu_make_preferred_u32(e->a);
    memcpy(ctx->ls + 0x100, e + 1, 0x80);
    jr_consume(s);
    return 1;
}

/* A policy-module job start (where g_spu_job_entries counts one). The first
 * is the captured job's own entry; a second means the capture closed there. */
void spu_jobrep_jobstart(spu_context* ctx)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return;
    if (!s->entered) { s->entered = 1; return; }
    if (s->nx && s->nx->type == JC_END && s->nx->a == JC_END_NEXTJOB) {
        s->ended = 1; s->end_kind = JC_END_NEXTJOB; s->halt_pc = ctx->pc & SPU_LS_MASK;
        ctx->status = SPU_STATUS_STOPPED_BY_HALT;
        spu_halt(ctx);
        return;
    }
    char got[64]; snprintf(got, sizeof got, "a second job start at 0x%05X", (unsigned)(ctx->pc & SPU_LS_MASK));
    jr_expect(s, JC_END, got);
    jr_diverge(s, "replay started another job at 0x%05X", (unsigned)(ctx->pc & SPU_LS_MASK));
}

void spu_jobrep_halt(spu_context* ctx)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return;
    if (!s->ended) { s->ended = 1; s->end_kind = JC_END_HALT; s->halt_pc = ctx->pc & SPU_LS_MASK; }
}

void spu_jobrep_hop(spu_context* ctx)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return;
    s->hops++;
    /* An interrupt the capture took this many hops after the previous event:
     * raise the event bits it saw (an Lr from a peer arrives at an arbitrary
     * moment; bits the job raised itself are already set). */
    const spu_jobcap_event* e = s->nx;
    if (e && e->type == JC_IRQ && s->hops == e->d) SPU_EVENT_SET(ctx, e->b);
    if ((++s->hop_total & 0xFFFFu) == 0) {
        LARGE_INTEGER q; QueryPerformanceCounter(&q);
        if (q.QuadPart > s->deadline) {
            char want[160] = "the end of the transcript";
            if (e) jr_describe(e, want, sizeof want);
            jr_diverge(s, "no progress: %llu hops without reaching event #%u (%s)",
                       s->hop_total, e ? e->index : 0u, want);
        }
    }
}

void spu_jobrep_irq(spu_context* ctx)
{
    jr_session* s = t_jr;
    if (!s || s->ctx != ctx) return;
    char got[128];
    snprintf(got, sizeof got, "IRQ srr0=0x%05X status=0x%X mask=0x%X after %u hops",
             (unsigned)(ctx->srr0 & SPU_LS_MASK), ctx->event_status, ctx->event_mask, s->hops);
    const spu_jobcap_event* e = jr_expect(s, JC_IRQ, got);
    if (!e) return;
    if (e->a != (ctx->srr0 & SPU_LS_MASK) || e->d != s->hops) { jr_mismatch(s, e, got); return; }
    jr_consume(s);
}

/* The PM's `bisle` into a job (gowpm0 at LS 0x3364): one lifted frame deep,
 * drained up to the link -- the shape the captured job ran in (interrupts are
 * on from the state block). The drain returning at the link is the job's
 * return, where the capture closed (END reason 1). If the transcript goes on,
 * the PM's continuation at the link runs as the direct transfer the lifted
 * bracket makes. */
static void jr_entry(spu_context* ctx)
{
    jr_session* s = t_jr;
    const uint32_t link = s->h->link;
    const int32_t si = (int32_t)ctx->image_id;
    ctx->host_depth++;
    spu_indirect_branch(ctx);
    spu_drain_call(ctx, link);
    if (s->nx && s->nx->type == JC_END && s->nx->a == JC_END_RETURN) {
        s->ended = 1; s->end_kind = JC_END_RETURN; s->halt_pc = ctx->pc & SPU_LS_MASK;
        ctx->host_depth--;
        return;
    }
    ctx->host_depth--;
    spu_img_restore(ctx, si);
    ctx->pc = link;
    jr_fn fn = spu_lookup(link, ctx->image_id);
    g_spu_trampoline_fn = fn ? fn : spu_indirect_branch;
}

static void jr_load(spu_context* ctx, const spu_jobcap_header* h, const uint8_t* buf)
{
    spu_context_init(ctx, h->spu_id);
    ctx->image_id = h->image_id;
    ctx->policy_mode = h->policy_mode;
    ctx->resident_ovl = h->resident_ovl;
    ctx->resident_task = h->resident_task;
    for (unsigned k = 0; k < SPU_RESIDENT_CODE_SLOTS && k < h->code_slots; ++k) {
        ctx->resident_code[k].lsa = h->code[k].lsa;
        ctx->resident_code[k].size = h->code[k].size;
        ctx->resident_code[k].source_ea = h->code[k].source_ea;
        ctx->resident_code[k].image_id = h->code[k].image_id;
    }
    ctx->resident_code_mask = 0;
    for (unsigned k = 0; k < SPU_RESIDENT_CODE_SLOTS; ++k)
        if (ctx->resident_code[k].image_id) ctx->resident_code_mask |= 1u << k;
    const spu_jobcap_state* st = &h->st;
    ctx->int_enable = st->int_enable; ctx->event_status = st->event_status;
    ctx->event_mask = st->event_mask; ctx->srr0 = st->srr0;
    ctx->decrementer = st->decrementer; ctx->status = st->status; ctx->stop_code = st->stop_code;
    ctx->mfc_lsa = st->mfc_lsa; ctx->mfc_eah = st->mfc_eah; ctx->mfc_eal = st->mfc_eal;
    ctx->mfc_size = st->mfc_size; ctx->mfc_tag = st->mfc_tag;
    ctx->mfc_tag_mask = st->mfc_tag_mask; ctx->mfc_tag_status = st->mfc_tag_status;
    ctx->resv_ea = st->resv_ea; ctx->resv_valid = (int)st->resv_valid; ctx->atomic_stat = st->atomic_stat;
    ctx->list_stall_mask = st->list_stall_mask; ctx->list_stall_stat = st->list_stall_stat;
    ctx->irq_resume_pc = st->irq_resume_pc;   /* irq_saved: spu_run_with_halt forgets it anyway */
    ctx->module_img_a00 = st->module_img_a00;
    for (unsigned t = 0; t < 32; ++t) {
        ctx->list_stall_elem_lsa[t] = st->list_stall_elem_lsa[t];
        ctx->list_stall_remaining[t] = st->list_stall_remaining[t];
        ctx->list_stall_dest_lsa[t] = st->list_stall_dest_lsa[t];
        ctx->list_stall_cmd[t] = st->list_stall_cmd[t];
        ctx->list_stall_ea_base[t] = (uint64_t)st->list_stall_ea_base[t][1] << 32 | st->list_stall_ea_base[t][0];
    }
    memcpy(ctx->resv_line, st->resv_line, sizeof ctx->resv_line);
    memcpy(ctx->irq_gpr, st->irq_gpr, sizeof ctx->irq_gpr);
    memcpy(ctx->ls, buf + h->header_size, SPU_LS_SIZE);
    memcpy(ctx->gpr, buf + h->header_size + h->ls_size, sizeof ctx->gpr);
    ctx->pc = h->entry_pc;
    uint32_t tc = st->mfc_tag_completed, qc = st->mfc_queue_count;
    spu_jobcap_mfc_state(ctx, &tc, &qc, 1);
}

/* One replay of one job. Returns 1 when identical; `why` gets the first
 * difference otherwise. */
static int jr_run_once(spu_context* ctx, const uint8_t* buf, size_t len, double* ms,
                       char* why, size_t whyn, unsigned* notes)
{
    static jr_session s;   /* one session at a time, off the stack */
    memset(&s, 0, sizeof s);
    const spu_jobcap_header* h = (const spu_jobcap_header*)buf;
    s.ctx = ctx; s.buf = buf; s.len = len; s.h = h;
    s.off = (size_t)h->header_size + h->ls_size + 16u * h->gpr_count;
    s.nx = jr_at(&s, s.off);
    jr_load(ctx, h, buf);
    LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    s.deadline = t0.QuadPart + 10 * f.QuadPart;
    t_jr = &s;
    spu_run_with_halt(jr_entry, ctx);
    QueryPerformanceCounter(&t1);
    t_jr = NULL;
    *ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart;
    *notes += s.notes;

    int ok = 0;
    const spu_jobcap_event* e = s.nx;
    if (s.diverged) {
        snprintf(why, whyn, "diverged: %s", s.why);
    } else if (!s.ended) {
        snprintf(why, whyn, "the run returned without halting (pc=0x%05X)", (unsigned)(ctx->pc & SPU_LS_MASK));
    } else if (!e || e->type != JC_END) {
        char want[160] = "nothing (truncated)";
        if (e) jr_describe(e, want, sizeof want);
        snprintf(why, whyn, "the run ended at pc=0x%05X with events left: next is #%u %s",
                 s.halt_pc, e ? e->index : 0u, want);
    } else if (e->a != (uint32_t)s.end_kind || e->b != s.halt_pc) {
        snprintf(why, whyn, "END differs: recorded reason %u at pc=0x%05X, replay reason %d at pc=0x%05X",
                 e->a, e->b, s.end_kind, s.halt_pc);
    } else if (e->n != sizeof ctx->gpr + SPU_LS_SIZE) {
        snprintf(why, whyn, "END record payload is %u bytes", e->n);
    } else {
        const uint8_t* want = (const uint8_t*)(e + 1);
        const uint8_t* g = (const uint8_t*)ctx->gpr;
        size_t i = 0;
        while (i < sizeof ctx->gpr && g[i] == want[i]) i++;
        if (i < sizeof ctx->gpr) {
            const unsigned r = (unsigned)(i / 16), w = (unsigned)(i % 16 / 4);
            snprintf(why, whyn, "final r%u word %u differs: recorded %08X, replay %08X",
                     r, w, ((const uint32_t*)want)[i / 4], ((const uint32_t*)g)[i / 4]);
        } else {
            want += sizeof ctx->gpr;
            i = 0;
            while (i < SPU_LS_SIZE && ctx->ls[i] == want[i]) i++;
            if (i < SPU_LS_SIZE)
                snprintf(why, whyn, "final LS differs first at 0x%05X: recorded %02X, replay %02X",
                         (unsigned)i, want[i], ctx->ls[i]);
            else if (s.out_bad)
                snprintf(why, whyn, "%u of %u outputs differ; %s", s.out_bad, s.outputs, s.out_why);
            else
                ok = 1;
        }
    }
    if (ok && s.out_bad == 0 && e && e->d != (uint32_t)ctx->resident_ovl)
        fprintf(stderr, "[replay]   note: resident_ovl %d at the end, recorded %u\n", ctx->resident_ovl, e->d);
    spu_coh_unregister(ctx);
    spu_mfc_release(ctx);
    return ok;
}

#ifdef _WIN32
typedef struct { char path[600]; char name[80]; unsigned wid, seq; } jr_file;

static int jr_cmp_file(const void* a, const void* b)
{
    const jr_file* x = (const jr_file*)a; const jr_file* y = (const jr_file*)b;
    return x->seq != y->seq ? (x->seq < y->seq ? -1 : 1) : strcmp(x->name, y->name);
}

static int jr_cmp_double(const void* a, const void* b)
{
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

typedef struct { const char* dir; int rc; } jr_args;

static DWORD WINAPI jr_thread(LPVOID p)
{
    jr_args* args = (jr_args*)p;
    const char* dir = args->dir;
    long repeat = 1;
    { const char* r = getenv("SPU_JOB_REPLAY_REPEAT"); if (r && *r) repeat = atol(r); if (repeat < 1) repeat = 1; }

    /* <dir>/job_*.bin, or one file. */
    jr_file* files = NULL; size_t nf = 0, cap = 0;
    const DWORD attr = GetFileAttributesA(dir);
    char pattern[600];
    const int single = attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
    if (single) snprintf(pattern, sizeof pattern, "%s", dir);
    else snprintf(pattern, sizeof pattern, "%s/job_*.bin", dir);
    WIN32_FIND_DATAA fd;
    HANDLE hf = FindFirstFileA(pattern, &fd);
    if (hf != INVALID_HANDLE_VALUE) {
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            if (nf == cap) { cap = cap ? cap * 2 : 256; files = (jr_file*)realloc(files, cap * sizeof *files); }
            jr_file* jf = &files[nf++];
            memset(jf, 0, sizeof *jf);
            snprintf(jf->name, sizeof jf->name, "%s", fd.cFileName);
            if (single) snprintf(jf->path, sizeof jf->path, "%s", dir);
            else snprintf(jf->path, sizeof jf->path, "%s/%s", dir, fd.cFileName);
            sscanf(fd.cFileName, "job_%u_%u", &jf->wid, &jf->seq);
        } while (FindNextFileA(hf, &fd));
        FindClose(hf);
    }
    if (!nf) {
        fprintf(stderr, "[replay] no job_*.bin captures in %s\n", dir);
        args->rc = 2;
        return 0;
    }
    qsort(files, nf, sizeof *files, jr_cmp_file);
    fprintf(stderr, "[replay] %zu captures in %s, %ld run(s) each\n", nf, dir, repeat);

    spu_context* ctx = (spu_context*)_aligned_malloc(sizeof(spu_context), 64);
    double* job_ms = (double*)calloc(nf, sizeof(double));
    double* run_ms = (double*)calloc((size_t)repeat, sizeof(double));
    size_t identical = 0, timed = 0;
    double sum_min = 0;
    unsigned notes_total = 0;
    for (size_t k = 0; k < nf; ++k) {
        FILE* f = fopen(files[k].path, "rb");
        if (!f) { fprintf(stderr, "[replay] %s FAIL cannot open\n", files[k].name); continue; }
        fseek(f, 0, SEEK_END); const long len = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t* buf = len > 0 ? (uint8_t*)malloc((size_t)len) : NULL;
        const size_t got = buf ? fread(buf, 1, (size_t)len, f) : 0;
        fclose(f);
        const spu_jobcap_header* h = (const spu_jobcap_header*)buf;
        const char* bad = NULL;
        if (!buf || got != (size_t)len || (size_t)len < sizeof *h) bad = "short file";
        else if (memcmp(h->magic, "SPUJCAP", 8)) bad = "not a job capture";
        else if (h->version != SPU_JOBCAP_VERSION)
            bad = "capture version 1 has no context state block (re-capture with this build)";
        else if (h->header_size < sizeof *h || h->ls_size != SPU_LS_SIZE || h->gpr_count != 128 ||
                 (size_t)len < (size_t)h->header_size + h->ls_size + 16u * h->gpr_count)
            bad = "header does not match this build";
        if (bad) {
            fprintf(stderr, "[replay] %s FAIL %s\n", files[k].name, bad);
            free(buf);
            continue;
        }
        int ok = 1; char why[1024] = "";
        unsigned notes = 0;
        for (long r = 0; r < repeat; ++r) {
            char w[1024] = "";
            spu_fn_probe_job(files[k].name, (int)r);
            const int one = jr_run_once(ctx, buf, (size_t)len, &run_ms[r], w, sizeof w, &notes);
            if (!one && ok) {
                ok = 0;
                if (repeat > 1) snprintf(why, sizeof why, "run %ld: %s", r + 1, w);
                else snprintf(why, sizeof why, "%s", w);
            }
        }
        qsort(run_ms, (size_t)repeat, sizeof *run_ms, jr_cmp_double);
        const double med = run_ms[repeat / 2];
        job_ms[timed++] = med;
        sum_min += run_ms[0];
        notes_total += notes;
        if (ok) identical++;
        if (repeat > 1)
            fprintf(stderr, "[replay] %s %s %.3f ms median of %ld (min %.3f max %.3f) entry=0x%05X ovl=%d%s%s\n",
                    files[k].name, ok ? "PASS" : "FAIL", med, repeat, run_ms[0], run_ms[repeat - 1],
                    h->entry_pc, h->resident_ovl, ok ? "" : "  ", why);
        else
            fprintf(stderr, "[replay] %s %s %.3f ms entry=0x%05X ovl=%d%s%s\n", files[k].name,
                    ok ? "PASS" : "FAIL", med, h->entry_pc, h->resident_ovl, ok ? "" : "  ", why);
        free(buf);
    }
    spu_fn_probe_job(NULL, 0);   /* flush the last job's [fnpt] */
    if (timed) {
        qsort(job_ms, timed, sizeof *job_ms, jr_cmp_double);
        double sum = 0; for (size_t k = 0; k < timed; ++k) sum += job_ms[k];
        fprintf(stderr, "[replay] time per job%s: median %.3f ms, min %.3f, max %.3f, mean %.3f (%zu jobs)\n",
                repeat > 1 ? " (each the median of its runs)" : "", job_ms[timed / 2], job_ms[0],
                job_ms[timed - 1], sum / (double)timed, timed);
        /* The A/B figure: each job's fastest run, summed. Noise only adds time,
         * so the minimum moves least when something else takes the core. */
        if (repeat > 1)
            fprintf(stderr, "[replay] sum of per-job minimum: %.3f ms\n", sum_min);
    }
    if (notes_total)
        fprintf(stderr, "[replay] %u live-state notes (see above)\n", notes_total);
    fprintf(stderr, "[replay] %zu/%zu jobs identical\n", identical, nf);
    fflush(stderr);
    _aligned_free(ctx);
    free(job_ms); free(run_ms); free(files);
    args->rc = identical == nf ? 0 : 1;
    return 0;
}

int spu_job_replay_run(const char* dir)
{
#if !SPU_JOBLAB
    fprintf(stderr, "[replay] SPU_JOB_REPLAY=%s: this build has no job-lab hooks (configure with -DPS3RECOMP_SPU_JOBLAB=ON)\n", dir);
    return 1;
#endif
    /* Nothing else runs: no capture of the replay itself, and every hook
     * answers for the one session context. */
    g_spu_jobcap_on_v = 0;
    g_spu_jobrep_on_v = 1;
    g_spu_jobhop_on_v = 1; g_spu_jobret_on_v = 1;
    jr_args args = { dir, 1 };
    /* A kernel thread's depth of lifted frames needs more than the main
     * thread's 1 MB. */
    HANDLE th = CreateThread(NULL, 256u << 20, jr_thread, &args, STACK_SIZE_PARAM_IS_A_RESERVATION | CREATE_SUSPENDED, NULL);
    if (!th) { fprintf(stderr, "[replay] cannot start the replay thread\n"); return 1; }
    /* Timing runs: one core away from core 0 and high priority, so another
     * process landing on the thread moves the numbers less. */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(th, (DWORD_PTR)1 << 2);
    SetThreadPriority(th, THREAD_PRIORITY_HIGHEST);
    ResumeThread(th);
    WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
    g_spu_jobrep_on_v = 0;
    g_spu_jobhop_on_v = 0;
    return args.rc;
}
#else
int spu_job_replay_run(const char* dir)
{
    fprintf(stderr, "[replay] SPU_JOB_REPLAY=%s: the runner is Windows-only for now\n", dir);
    return 1;
}
#endif
