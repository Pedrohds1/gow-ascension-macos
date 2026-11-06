/*
 * SPU function boundary probe (job lab, used with SPU_JOB_REPLAY).
 *
 * A lab lift wraps one lifted function (the port's analysis/spu_fnprobe_lift.py):
 *     probe(ctx, fn, 0); fn_impl(ctx); probe(ctx, fn, 1);
 * and may route that function's (and its callees') LS reads through
 * spu_fn_probe_read. During the first run of each replayed job, every call of
 * the function prints one line:
 *     [fnp] <fn> <job> #<call> <us> us  r3..r10=<preferred words>  r1=<sp>
 *           wr=<LS ranges written>  rd=<LS ranges read>  gpr=<registers changed>
 * Ranges are 16-byte lines, merged; "rd" only when reads are routed. With
 * SPU_JOB_REPLAY_REPEAT > 1 the function is also timed in every run, and each
 * job ends with [fnpt] <fn> <job> <min us of call 0> <call 1> ...: the per-call
 * minimum, the figure to compare. A --blocks lift also prints, per call of run
 * 0, [fnpb] <fn> <job> #<call> <block pc>:<count> ... (blocks entered).
 *
 *   SPU_FN_PROBE=<hex fn>           the function the lift wrapped (e.g. E550)
 *   SPU_FN_PROBE_SAVE=<dir>[,<n>]   also write the first n calls (default 64) as
 *                                   fnp_<fn>_<job>_<call>.bin: "SPUFNP1\0", u32 fn,
 *                                   u32 0, entry GPRs (2 KB), entry LS, exit GPRs,
 *                                   exit LS. The gate for a native replacement.
 *
 * Outside a replay, or without SPU_FN_PROBE, the probe returns at once. The
 * timing (QPC from after the entry snapshot to before the exit diff) includes
 * the read routing when the lift routes reads.
 */
#include "spu_context.h"
#include "spu_jobcap.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../platform/win32_compat.h"

#ifdef _WIN32
static int s_fnp_on = -1;          /* -1 env not read yet */
static uint32_t s_fnp_fn;
static char s_fnp_dir[260];
static long s_fnp_save_max = 64, s_fnp_saved;
static const char* s_fnp_job = "";
static int s_fnp_run;               /* replay run index of the current job; only run 0 is probed */
static unsigned s_fnp_call;
static int s_fnp_open;
static u128 s_fnp_gpr[128];
static uint8_t s_fnp_ls[SPU_LS_SIZE];
static uint8_t s_fnp_rd[SPU_LS_SIZE / 16];
static int s_fnp_reads;
static LARGE_INTEGER s_fnp_t0;
static double s_fnp_min[64];
static unsigned s_fnp_ncalls;
static uint32_t* s_fnp_blk; static const uint32_t* s_fnp_blk_pc; static unsigned s_fnp_nblk;

static void fnp_init(void)
{
    const char* e = getenv("SPU_FN_PROBE");
    s_fnp_on = 0;
    if (!e || !*e) return;
    s_fnp_fn = (uint32_t)strtoul(e, NULL, 16);
    const char* d = getenv("SPU_FN_PROBE_SAVE");
    if (d && *d) {
        snprintf(s_fnp_dir, sizeof s_fnp_dir, "%s", d);
        char* c = strchr(s_fnp_dir, ',');
        if (c) { *c = 0; s_fnp_save_max = atol(c + 1); }
        CreateDirectoryA(s_fnp_dir, NULL);
    }
    s_fnp_on = 1;
    fprintf(stderr, "[fnp] probing func 0x%05X%s%s\n", s_fnp_fn, s_fnp_dir[0] ? ", saving to " : "", s_fnp_dir);
}

/* The replay names the job and run before each run (spu_job_replay.c). */
/* The replay names the job and run before each run (spu_job_replay.c), and
 * NULL once at the end. A new job first flushes the last one's minimums. */
void spu_fn_probe_job(const char* name, int run)
{
    if (s_fnp_on > 0 && (run == 0 || !name) && s_fnp_ncalls) {
        char b[1024]; size_t p = 0;
        for (unsigned k = 0; k < s_fnp_ncalls && k < 64 && p + 16 < sizeof b; k++)
            p += (size_t)snprintf(b + p, sizeof b - p, " %.2f", s_fnp_min[k]);
        fprintf(stderr, "[fnpt] %05X %s%s\n", s_fnp_fn, s_fnp_job, b);
        s_fnp_ncalls = 0;
    }
    if (!name) return;
    s_fnp_job = name; s_fnp_run = run; s_fnp_call = 0; s_fnp_open = 0;
}

/* A --blocks lab lift hands over its counter table on every call. */
void spu_fn_probe_blocks(uint32_t* cnt, const uint32_t* pc, unsigned n)
{
    s_fnp_blk = cnt; s_fnp_blk_pc = pc; s_fnp_nblk = n;
}

static void fnp_time(double us)
{
    if (s_fnp_call < 64) {
        if (s_fnp_run == 0 || us < s_fnp_min[s_fnp_call]) s_fnp_min[s_fnp_call] = us;
        if (s_fnp_call + 1 > s_fnp_ncalls) s_fnp_ncalls = s_fnp_call + 1;
    }
}

static int fnp_ranges(char* out, size_t n, const uint8_t* mark)
{
    size_t p = 0; int runs = 0;
    for (uint32_t i = 0; i < SPU_LS_SIZE / 16; ) {
        if (!mark[i]) { i++; continue; }
        uint32_t j = i; while (j < SPU_LS_SIZE / 16 && mark[j]) j++;
        if (p + 24 < n) p += (size_t)snprintf(out + p, n - p, "%s%05X+%X", runs ? "," : "", i * 16, (j - i) * 16);
        runs++; i = j;
    }
    if (!runs && n) snprintf(out, n, "-");
    return runs;
}

void spu_fn_probe(spu_context* ctx, uint32_t fn, int phase)
{
    if (s_fnp_on < 0) fnp_init();
    if (!s_fnp_on || !g_spu_jobrep_on || fn != s_fnp_fn) return;
    if (s_fnp_run != 0) {           /* timing only */
        if (phase == 0) { if (!s_fnp_open++) QueryPerformanceCounter(&s_fnp_t0); return; }
        if (!s_fnp_open || --s_fnp_open) return;
        LARGE_INTEGER t1, f; QueryPerformanceCounter(&t1); QueryPerformanceFrequency(&f);
        fnp_time((double)(t1.QuadPart - s_fnp_t0.QuadPart) * 1e6 / (double)f.QuadPart);
        s_fnp_call++;
        return;
    }
    if (phase == 0) {
        if (s_fnp_open++) return;   /* nested: only the outermost call */
        if (s_fnp_blk) memset(s_fnp_blk, 0, s_fnp_nblk * sizeof *s_fnp_blk);
        memcpy(s_fnp_gpr, ctx->gpr, sizeof s_fnp_gpr);
        memcpy(s_fnp_ls, ctx->ls, SPU_LS_SIZE);
        memset(s_fnp_rd, 0, sizeof s_fnp_rd);
        s_fnp_reads = 0;
        QueryPerformanceCounter(&s_fnp_t0);
        return;
    }
    if (!s_fnp_open || --s_fnp_open) return;
    LARGE_INTEGER t1, f; QueryPerformanceCounter(&t1); QueryPerformanceFrequency(&f);
    const double us = (double)(t1.QuadPart - s_fnp_t0.QuadPart) * 1e6 / (double)f.QuadPart;
    fnp_time(us);
    if (s_fnp_blk) {
        static char bs[8192]; size_t bp = 0;
        for (unsigned k = 0; k < s_fnp_nblk && bp + 24 < sizeof bs; k++)
            if (s_fnp_blk[k]) bp += (size_t)snprintf(bs + bp, sizeof bs - bp, " %05X:%u", s_fnp_blk_pc[k], s_fnp_blk[k]);
        fprintf(stderr, "[fnpb] %05X %s #%u%s\n", fn, s_fnp_job, s_fnp_call, bp ? bs : " -");
    }
    static uint8_t wr[SPU_LS_SIZE / 16];
    for (uint32_t i = 0; i < SPU_LS_SIZE / 16; i++)
        wr[i] = memcmp(ctx->ls + i * 16, s_fnp_ls + i * 16, 16) != 0;
    static char ws[4096], rs[4096];
    fnp_ranges(ws, sizeof ws, wr);
    if (s_fnp_reads) fnp_ranges(rs, sizeof rs, s_fnp_rd); else snprintf(rs, sizeof rs, "(not routed)");
    char gs[600]; size_t gp = 0;
    for (int r = 0; r < 128; r++)
        if (memcmp(&ctx->gpr[r], &s_fnp_gpr[r], 16) && gp + 6 < sizeof gs)
            gp += (size_t)snprintf(gs + gp, sizeof gs - gp, "%sr%d", gp ? "," : "", r);
    if (!gp) snprintf(gs, sizeof gs, "-");
    char args[200]; size_t ap = 0;
    for (int r = 3; r <= 10; r++)
        ap += (size_t)snprintf(args + ap, sizeof args - ap, "%s%08X", r > 3 ? " " : "", s_fnp_gpr[r]._u32[0]);
    fprintf(stderr, "[fnp] %05X %s #%u %.2f us  r3..r10=%s  r1=%05X  wr=%s  rd=%s  gpr=%s\n",
            fn, s_fnp_job, s_fnp_call, us, args, s_fnp_gpr[1]._u32[0] & SPU_LS_MASK, ws, rs, gs);
    if (s_fnp_dir[0] && s_fnp_saved < s_fnp_save_max) {
        char path[400];
        snprintf(path, sizeof path, "%s/fnp_%05X_%.*s_%u.bin", s_fnp_dir, fn,
                 (int)(strlen(s_fnp_job) > 4 ? strlen(s_fnp_job) - 4 : strlen(s_fnp_job)), s_fnp_job, s_fnp_call);
        FILE* o = fopen(path, "wb");
        if (o) {
            const uint32_t hd[4] = { 0x46555053u, 0x0031504Eu, fn, 0 };   /* "SPUFNP1\0" little-endian */
            fwrite(hd, 4, 4, o);
            fwrite(s_fnp_gpr, 1, sizeof s_fnp_gpr, o); fwrite(s_fnp_ls, 1, SPU_LS_SIZE, o);
            fwrite(ctx->gpr, 1, sizeof s_fnp_gpr, o); fwrite(ctx->ls, 1, SPU_LS_SIZE, o);
            fclose(o); s_fnp_saved++;
        }
    }
    s_fnp_call++;
}

/* LS read routed by the lab lift: records the line while a probe window is open. */
u128 spu_fn_probe_read(const spu_context* ctx, uint32_t lsa)
{
    if (s_fnp_open) { s_fnp_rd[(lsa & SPU_LS_MASK) >> 4] = 1; s_fnp_reads = 1; }
    return spu_ls_read128(ctx, lsa);
}
#else
void spu_fn_probe_job(const char* name, int run) { (void)name; (void)run; }
void spu_fn_probe_blocks(uint32_t* cnt, const uint32_t* pc, unsigned n) { (void)cnt; (void)pc; (void)n; }
void spu_fn_probe(spu_context* ctx, uint32_t fn, int phase) { (void)ctx; (void)fn; (void)phase; }
u128 spu_fn_probe_read(const spu_context* ctx, uint32_t lsa) { return spu_ls_read128(ctx, lsa); }
#endif
