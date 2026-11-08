/*
 * ps3recomp - platform-neutral RSX draw engine (see rsx_draw_engine.h)
 *
 * The orchestration carried across from libs/video/rsx_live_draw.c, the
 * NV4097 -> D3D12 engine a title has shipped on. Line numbers in the comments
 * below are that file's, so the two can be read side by side; the file itself
 * is vendored and stays exactly as it is.
 *
 * What is deliberately left out, because it is not renderer behaviour a title
 * needs: the shader disk cache, movie mode and its compositor, the a010
 * probe, and every YZ_PERF_PROFILE block.
 */
#include "rsx_draw_engine.h"

#include "rsx_fp_decompiler.h"
#include "rsx_primitives.h"
#include "rsx_restart_cuts.h"
#include "rsx_vertex_formats.h"
#include "rsx_vp_decompiler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "ps3emu/log_first.h"

/* Guest memory and the RSX offset resolvers, declared the way every backend
 * that reads guest data declares them (rsx_vertex_fetch.h). */
extern u8* vm_base;
u32 cellGcmResolveLocated(int local, u32 offset);
/* cellGcmSys.c: the display buffers registered before the engine came up. */
void cellGcm_replay_display_buffers_to_engine(void);
/* Non-zero only when the offset's page is in the IO table, so a caller can
 * tell IO-mapped main memory from "assume VRAM". */
u32 cellGcmResolveIO(u32 offset);
/* The guest VM's OOB guard: the size a host backed, or 0 for "the whole 32-bit
 * space is backed and no check is needed" (runtime/memory/vm.h). */
extern u32 ppu_vm_size;

/* ------------------------------------------------------------------------- */

#define ENG_MAX_SURFACES   64
#define ENG_MAX_ZDEPTHS    64
#define ENG_MAX_TEXTURES   RSX_DRAW_ENGINE_TEXTURE_CACHE
#define ENG_MAX_VTEXTURES  64
#define ENG_MAX_PIPELINES  8192
#define ENG_PIPELINE_INDEX 16384u           /* a power of two, twice ENG_MAX_PIPELINES */
#define ENG_MAX_BATCHES    256
#define ENG_VERT_STRIDE    (RSX_DSP_NUM_VERTEX_ATTR * 16u)
#define ENG_INVALID        0xFFFFFFFFu
/* Guest textures larger than this are a misdecoded register, not an image;
 * rsx_live_draw.c rejects the same 4096 bound (2033). */
#define ENG_MAX_TEX_DIM    4096u
/* A surface declaration beyond this is a guest pointer read as clip
 * dimensions. rsx_live_draw.c's surface-guard (2625) preserves the existing
 * target rather than destroying a live one. */
#define ENG_MAX_SURFACE_DIM 8192u

#define ENG_VP_CB_BYTES    ((RSX_DSP_NUM_CONSTANTS + 2) * 16u)
#define ENG_FP_MAX_BYTES   0x10000u

#define M_ALPHA_TEST_ENABLE   0x0304
#define M_ALPHA_FUNC          0x0308
#define M_ALPHA_REF           0x030C
#define M_BLEND_ENABLE        0x0310
#define M_BLEND_SFACTOR       0x0314
#define M_BLEND_DFACTOR       0x0318
#define M_BLEND_EQUATION      0x0320
#define M_COLOR_MASK          0x0324
#define M_STENCIL_TEST_ENABLE 0x0328
#define M_STENCIL_MASK        0x032C
#define M_STENCIL_FUNC        0x0330
#define M_STENCIL_FUNC_REF    0x0334
#define M_STENCIL_FUNC_MASK   0x0338
#define M_STENCIL_OP_FAIL     0x033C
#define M_STENCIL_OP_ZFAIL    0x0340
#define M_STENCIL_OP_ZPASS    0x0344
#define M_TWO_SIDED_STENCIL   0x0348
#define M_BACK_STENCIL_FUNC   0x0350
#define M_BACK_STENCIL_OP_FAIL  0x035C
#define M_BACK_STENCIL_OP_ZFAIL 0x0360
#define M_BACK_STENCIL_OP_ZPASS 0x0364
#define M_DEPTH_BOUNDS_ENABLE 0x0380
#define M_DEPTH_BOUNDS_MIN    0x0384
#define M_DEPTH_BOUNDS_MAX    0x0388
#define M_SCISSOR_HORIZONTAL  0x08C0
#define M_SCISSOR_VERTICAL    0x08C4
#define M_DEPTH_FUNC          0x0A6C
#define M_DEPTH_WRITE         0x0A70
#define M_DEPTH_TEST_ENABLE   0x0A74
#define M_ZSTENCIL_CLEAR      0x1D8C

/* ---- module state -------------------------------------------------------- */

typedef struct {
    u32 location, offset;
    u32 w, h;
    u32 pitch;              /* guest bytes per row, from SET_SURFACE_PITCH */
    rsx_be_format fmt;
    u32 handle;
    u32 feedback;          /* last immutable own-target copy, engine-owned */
    u64 color_serial;       /* g.write_serial of the last colour write, or 0 */
    int guest_dirty;        /* rendered since guest memory last got a copy   */
    struct {                /* read as Y16_X16_FLOAT, filled on the GPU (eng_color_rg16f) */
        u32 tex, format, remap, w, h;
        u64 serial;         /* color_serial it was filled at */
    } rg16[2];
} eng_surface;

typedef struct {
    u32 location, offset;
    u32 w, h;
    u32 handle;
    u32 snapshot;          /* sampleable copy, or 0                        */
    int cleared;
    int had_write;
    int snapshot_valid;
    u64 depth_serial;      /* g.write_serial of the last depth write or clear */
    int guest_dirty;       /* written since guest memory last got a copy     */
    u32 zfmt, pitch;       /* SURFACE_FORMAT depth field and zeta pitch then */
    u32 used_w, used_h;    /* the clip that write went through              */
    struct {               /* read as A8R8G8B8, filled on the GPU (eng_zdepth_argb8) */
        u32 tex, format, remap, w, h;
        u64 serial;        /* depth_serial it was filled from               */
    } argb[2];
} eng_zdepth;

typedef struct {
    u32 location, offset, format, width, height, pitch, remap, cubemap;
    u32 handle;
    u64 content_hash;
    u64 sparse_hash;            /* eng_texture_sparse_hash, the per-frame check */
    u32 last_full_frame;        /* g.frames of the last full content hash */
    u32 last_hash_frame;
    u64 last_use_serial;
    u32 depth;                  /* slices of a 3D texture, else 1 */
    u32 levels;                 /* SET_TEXTURE_FORMAT mipmap count it was built with */
    int rewritten;              /* guest bytes rewritten by the engine itself */
    u32 full_shift;             /* RSX_TEXHASH_ADAPT: full-hash period = FULL_EVERY << this */
} eng_texture;

typedef struct {
    u64 key;
    u32 handle;
    u8 fixed;
    u16 tex_used;           /* units the pixel program samples */
    u64 fp_hash;            /* eng_fp_debug_hash of the guest fragment program */
    u64 text_hash;          /* the two programs' HLSL, for the pipeline catalog */
    u8 prov;                /* built at boot from the catalog, not yet checked against the decompiler */
    u8 topo;                /* topologies the catalog has recorded for it */
} eng_pipeline;

typedef struct { u32 first, count; } eng_batch;

typedef struct {
    u32 location, offset, pitch, width, height;
    int valid;
} eng_display_buffer;

static struct {
    const rsx_draw_backend* be;
    int ready;
    int default_on;
    u32 width, height;

    rsx_dispatch rsx;

    eng_surface surfaces[ENG_MAX_SURFACES];
    u32 n_surfaces;
    eng_zdepth zdepths[ENG_MAX_ZDEPTHS];
    u32 n_zdepths;
    eng_texture textures[ENG_MAX_TEXTURES];
    u32 n_textures;
    eng_pipeline pipelines[ENG_MAX_PIPELINES];
    u32 n_pipelines;
    u16 pipeline_index[ENG_PIPELINE_INDEX];   /* key hash -> pipelines[] slot + 1, 0 empty */
    eng_display_buffer display_buffers[8];

    u32 frames;
    u64 texture_use_serial;
    u32 guest_draws;
    u32 last_guest_draws;
    u32 last_present_surface;
    u32 last_flip_buffer;
    u64 write_serial;       /* orders colour and depth writes to one address */

    /* staging: decoded texture levels, the constant blocks, the index list */
    u8* tex_staging;
    u32 tex_staging_cap;
    u8* vp_cb;
    u8* fp_cb;
    u32 fp_cb_cap;
    u32* indices;
    u32 index_cap;

    rsx_fp_constant_block fp_constants;
} g;

/* The decompilers' HLSL. The VP decompiler builds bodies up to 192 KB. */
static char s_vs_hlsl[256 * 1024];
static char s_ps_hlsl[256 * 1024];

/* ENG_PHASE_STATS (see sink_end): host microseconds, and the texture-path
 * buckets it splits the "tex" phase into. */
static int s_stats_s = -1;
static int s_stats_meth = -1;   /* ENG_PHASE_STATS_METHODS=1: also time every method (two clock reads each) */
/* CPU time of the calling thread, in microseconds: the walker's own cost,
 * which other processes' load does not inflate the way wall time is. */
static unsigned long long eng_thread_cpu_us(void)
{
#ifdef _WIN32
    FILETIME c, e, k, u;
    if (!GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) return 0;
    return ((((unsigned long long)k.dwHighDateTime << 32) | k.dwLowDateTime) +
            (((unsigned long long)u.dwHighDateTime << 32) | u.dwLowDateTime)) / 10ull;
#else
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (unsigned long long)ts.tv_sec * 1000000ull + (unsigned long long)ts.tv_nsec / 1000ull;
#endif
}
static unsigned long long eng_now_us(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (unsigned long long)ts.tv_sec * 1000000ull + (unsigned long long)ts.tv_nsec / 1000ull;
}
enum { ST_ZSNAP, ST_Z2G, ST_CSNAP, ST_S2G, ST_HASH, ST_REUP, ST_MISS, ST_FIDX, ST_FREMAP, ST_FPREP, ST_FMANY,
       ST_PGET, ST_PFILL, ST_PRESEED, ST_HSPARSE, ST_HFULL, ST_N };
static const char* const s_st_sub_name[ST_N] = { "zsnap", "z2guest", "csnap", "s2guest", "hash", "reupload", "miss",
                                                  "f.index", "f.remap", "f.prep", "f.fetch", "p.get",
                                                  "p.fill", "p.reseed", "h.sparse", "h.full" };
/* ...and what a pipeline-lookup cache could skip: how often a draw's inputs
 * equal the previous draw's (fp memo slot, 0x08E4, the fragment texture
 * registers, 0x1EA0, the final key) and how many fragment programs a window
 * sees. */
static struct {
    u32 memo_last, memo_other, memo_miss, same_key, same_fp, same_ftex, same_vp, lookups, creates;
    u32 nfp, fp_seen[16384];
    u32 fill_samestruct, full_small, full_big, l0_hit, l0_checked, l0_differ;
    u32 mg_draws, mg_state, mg_state_fp, mg_all, mg_list, mg_quiet;
    u32 rp_state, rp_tex, rp_checked, rp_differ;   /* RSX_DRAW_REPEAT */
    u32 rp_regmiss, rp_constmiss, tp_z, tp_own, tp_surf, tp_depthfmt, tp_plain;
    u64 full_bytes;
    u32 prev_fp, prev_vp, prev_ftex[16 * 7 + 16];
    u64 prev_key;
} s_pc;
static u32 s_pc_frames0;
/* Engine entry points other than a draw (clears, flips, guest reads/writes of
 * surfaces, blit-shadow drops, display buffers, flushes, presents) seen so far. */
static u32 s_nd_events;
static u32 s_rp_hist[0x2000u / 4u];          /* ENG_PHASE_STATS: registers that broke a repeat */
static int s_tex_plain;                     /* RSX_DRAW_REPEAT: sink_bind_textures took only the plain cache path */
static unsigned long long s_st_sub_us[ST_N];
static u32 s_st_sub_n[ST_N];
#define ENG_ST_BEGIN() const unsigned long long st_t0_ = s_stats_s > 0 ? eng_now_us() : 0
#define ENG_ST_END(k) do { if (s_stats_s > 0) { s_st_sub_us[k] += eng_now_us() - st_t0_; s_st_sub_n[k]++; } } while (0)

/* The draw accumulator: one BEGIN_END group's batches, references and cuts. */
static struct {
    eng_batch arr[ENG_MAX_BATCHES];
    u32 n_arr;
    eng_batch idx[ENG_MAX_BATCHES];
    u32 n_idx;
    u32 n_packets;

    rsx_vertex_ref* refs;
    u32 n_refs, cap_refs, n_source_refs;
    int refs_remapped;
    rsx_vertex_remap ref_remap;

    u32* cuts;
    u32 n_cuts, cap_cuts;

    u8* verts;
    u64 verts_cap;
    u32 n_verts;

    /* Canonical ATTR0..15 float4 snapshots for register-based vertices.
     * Bounded separately from compact GPU staging; never fetch stale arrays. */
    float (*immediate)[RSX_DSP_NUM_VERTEX_ATTR][4];
    u32 n_immediate, cap_immediate;

    rsx_vertex_layout_plan layout;
    rsx_vertex_fetch_plan fetch_plan;
    int fetch_ok;
    int gpu_vtx;            /* RSX_VTX_GPU: vertices left to the backend (s_vjob) */

    /* INLINE_ARRAY stream for this group, when the vertices came down the
     * FIFO rather than from a vertex array. */
    const u8* inl;
    u32 inl_bytes;
    u32 inline_dropped_at_begin;
} dc;

/* ---- guest memory -------------------------------------------------------- */

/* rsx_dsp_* locations are RSX_LOCATION_LOCAL = 0 / _MAIN = 1;
 * cellGcmResolveLocated's argument is the opposite sense (1 = local), and
 * getting that backwards resolves every VRAM object through the IO table. */
static rsx_vertex_guest_ptr_fn s_guest_reader;
static void* s_guest_user;

void rsx_draw_engine_set_guest_memory(rsx_vertex_guest_ptr_fn reader, void* user)
{
    s_guest_reader = reader;
    s_guest_user = user;
}

static const u8* eng_guest_ptr(void* user, u32 location, u32 offset,
                               u32 min_bytes)
{
    (void)user;
    if (!min_bytes || min_bytes > (256u << 20)) return NULL;
    if (s_guest_reader) return s_guest_reader(s_guest_user, location, offset, min_bytes);
    if (!vm_base) return NULL;
    const u32 ea = cellGcmResolveLocated(location == RSX_LOCATION_LOCAL, offset);
    if (!ea) return NULL;
    if (ppu_vm_size && (u64)ea + min_bytes > ppu_vm_size) return NULL;
    return vm_base + ea;
}

static u64 eng_fnv1a(const void* data, u32 n, u64 hash)
{
    const u8* p = (const u8*)data;
    for (u32 i = 0; i < n; i++) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
}

/* ---- render state -------------------------------------------------------- */

void rsx_draw_engine_decode_render_state(const rsx_dispatch* rsx,
                                         rsx_be_render_state* rs)
{
    memset(rs, 0, sizeof(*rs));
    rs->alpha_test_enable = rsx_dsp_reg(rsx, M_ALPHA_TEST_ENABLE) & 1;
    rs->alpha_func   = rsx_dsp_reg(rsx, M_ALPHA_FUNC);
    rs->alpha_ref_raw = rsx_dsp_reg(rsx, M_ALPHA_REF);
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(rsx, &sf);
    rs->alpha_ref_format = sf.color_format;
    rs->rt_fp16 = sf.color_format == RSX_SURFACE_FMT_F_W16Z16Y16X16;
    rs->blend_enable = rsx_dsp_reg(rsx, M_BLEND_ENABLE) & 1;
    const u32 sfac = rsx_dsp_reg(rsx, M_BLEND_SFACTOR);
    const u32 dfac = rsx_dsp_reg(rsx, M_BLEND_DFACTOR);
    const u32 eq   = rsx_dsp_reg(rsx, M_BLEND_EQUATION);
    rs->sf_rgb = sfac & 0xFFFF; rs->sf_a = sfac >> 16;
    rs->df_rgb = dfac & 0xFFFF; rs->df_a = dfac >> 16;
    rs->eq_rgb = eq & 0xFFFF;   rs->eq_a = eq >> 16;
    rs->depth_test  = rsx_dsp_reg(rsx, M_DEPTH_TEST_ENABLE) & 1;
    rs->depth_write = rsx_dsp_reg(rsx, M_DEPTH_WRITE) & 1;
    rs->depth_func  = rsx_dsp_reg(rsx, M_DEPTH_FUNC);
    rs->cull_enable = rsx_dsp_reg(rsx, 0x183C) & 1;
    rs->cull_face   = rsx_dsp_reg(rsx, 0x1830);
    rs->front_face  = rsx_dsp_reg(rsx, 0x1834);
    /* The RAW register: a game-written 0 is a real "write no colour channel"
     * (a depth-prime pass), and rsx_dispatch_init seeds the nv40 reset value
     * so never-written reads as all-on. */
    rs->color_mask  = rsx_dsp_reg(rsx, M_COLOR_MASK);
    rs->color_mask_mrt = rsx_dsp_reg(rsx, 0x0370) & 0xFFF0u;
    rs->blend_mrt      = rsx_dsp_reg(rsx, 0x036C) & 0xEu;
    rs->stencil_enable    = rsx_dsp_reg(rsx, M_STENCIL_TEST_ENABLE) & 1;
    rs->stencil_two_sided = rsx_dsp_reg(rsx, M_TWO_SIDED_STENCIL) & 1;
    rs->s_func       = rsx_dsp_reg(rsx, M_STENCIL_FUNC);
    rs->s_func_mask  = rsx_dsp_reg(rsx, M_STENCIL_FUNC_MASK) & 0xFF;
    rs->s_write_mask = rsx_dsp_reg(rsx, M_STENCIL_MASK) & 0xFF;
    rs->s_fail       = rsx_dsp_reg(rsx, M_STENCIL_OP_FAIL);
    rs->s_zfail      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZFAIL);
    rs->s_zpass      = rsx_dsp_reg(rsx, M_STENCIL_OP_ZPASS);
    rs->bs_func      = rsx_dsp_reg(rsx, M_BACK_STENCIL_FUNC);
    rs->bs_fail      = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_FAIL);
    rs->bs_zfail     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZFAIL);
    rs->bs_zpass     = rsx_dsp_reg(rsx, M_BACK_STENCIL_OP_ZPASS);
    rs->depth_bounds_test = rsx_dsp_reg(rsx, M_DEPTH_BOUNDS_ENABLE) & 1;
}

/* State a disabled stage leaves behind is not identity. The registers keep
 * whatever the last user wrote: GoW:A leaves stale blend factors (249 source
 * alpha values seen) and compare functions behind disabled stages, and keying
 * on them made ~8,200 pipelines out of ~900 real ones -- the table filled
 * after ~13 minutes of play and new draws dropped. A field is zeroed here only
 * where every backend ignores it while its stage is off; RSX_PIPE_KEY_RAW=1
 * keeps the old keying (diagnostic). */
static void eng_rs_canonical(rsx_be_render_state* r)
{
    static int raw = -1;
    if (raw < 0) { const char* e = getenv("RSX_PIPE_KEY_RAW"); raw = e && *e == '1'; }
    if (raw) return;
    if (!r->alpha_test_enable) r->alpha_func = 0;
    if (!r->blend_enable && !r->blend_mrt)
        r->sf_rgb = r->df_rgb = r->sf_a = r->df_a = r->eq_rgb = r->eq_a = 0;
    if (!r->depth_test) r->depth_func = r->depth_write = 0;   /* no test, no write (both backends) */
    if (!r->cull_enable) r->cull_face = 0;
    if (!r->stencil_enable) {
        r->stencil_two_sided = 0;
        r->s_func = r->s_func_mask = r->s_write_mask = r->s_fail = r->s_zfail = r->s_zpass = 0;
    }
    if (!r->stencil_two_sided) r->bs_func = r->bs_fail = r->bs_zfail = r->bs_zpass = 0;
}

/* The alpha REFERENCE is deliberately absent: it lives in the fragment
 * constant block, so a title animating a fade must not compile a new
 * pipeline per frame. Enable and compare mode still select a variant.
 * Every field is listed by name so struct padding can never be identity
 * (rsx_live_draw.c:3151-3185). */
#define ENG_RS_FIELDS(X) \
    X(alpha_test_enable) X(alpha_func) X(blend_enable) X(sf_rgb) X(df_rgb) X(sf_a) X(df_a) \
    X(eq_rgb) X(eq_a) X(depth_test) X(depth_write) X(depth_func) X(cull_enable) X(cull_face) \
    X(front_face) X(color_mask) X(color_mask_mrt) X(blend_mrt) X(rt_fp16) X(stencil_enable) \
    X(stencil_two_sided) X(s_func) X(s_func_mask) X(s_write_mask) X(s_fail) X(s_zfail) \
    X(s_zpass) X(bs_func) X(bs_fail) X(bs_zfail) X(bs_zpass) X(depth_bounds_test)

/* The bytes the key takes from a render state, in hashing order. */
static u32 eng_rs_key_bytes(const rsx_be_render_state* rs, u8* out)
{
    u32 n = 0;
#define ENG_RS_PUT(name) memcpy(out + n, &rs->name, sizeof(rs->name)); n += (u32)sizeof(rs->name);
    ENG_RS_FIELDS(ENG_RS_PUT)
#undef ENG_RS_PUT
    return n;
}

u64 rsx_draw_engine_hash_render_state(const rsx_be_render_state* rs_in, u64 hash)
{
    rsx_be_render_state canon = *rs_in;
    eng_rs_canonical(&canon);
    u8 bytes[sizeof(rsx_be_render_state)];
    return eng_fnv1a(bytes, eng_rs_key_bytes(&canon, bytes), hash);
}

/* A catalog key made before the render state was canonical: FNV-1a steps are
 * invertible (the prime is odd), so peel the raw state's bytes off the end of
 * the old key and hash the canonical state back on. */
static u64 eng_rekey_canonical(u64 old_key, const rsx_be_render_state* rs)
{
    u8 bytes[sizeof(rsx_be_render_state)];
    u32 n = eng_rs_key_bytes(rs, bytes);
    const u64 inv = 0xCE965057AFF6957Bull;   /* 1099511628211^-1 mod 2^64 */
    u64 h = old_key;
    while (n) { h *= inv; h ^= bytes[--n]; }
    return rsx_draw_engine_hash_render_state(rs, h);
}

/* ---- topology expansion -------------------------------------------------- */

/* Every expansion is bounded by the restart cuts, so a strip that the guest
 * broke with the sentinel index gets no connecting triangle across the break.
 * Missing that decode is what upstream's aa7fb63 records as "the exploded
 * spiky mesh that occluded the scene". */
u32 rsx_draw_engine_topology_index_count(u32 primitive, u32 source_refs,
                                         const u32* cuts, u32 cut_count)
{
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES:
        return source_refs - source_refs % 3u;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON: {
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 3) total += (count - 2) * 3;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUAD_STRIP: {
        /* Two vertices per quad after the first pair, so a segment of n
         * carries (n - 2) / 2 quads and an odd trailing vertex is dropped. */
        u32 total = 0;
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            (void)begin;
            if (count >= 4) total += ((count - 2u) / 2u) * 6u;
        }
        return total;
    }
    case RSX_PRIMITIVE_QUADS:
        return (source_refs / 4u) * 6u;
    default:
        return 0;
    }
}

static u32 eng_topology_vertex(const u32* occurrence_to_unique, u32 occurrence)
{
    return occurrence_to_unique ? occurrence_to_unique[occurrence] : occurrence;
}

void rsx_draw_engine_write_topology_indices(u32 primitive, u32 source_refs,
                                            const u32* cuts, u32 cut_count,
                                            const u32* occurrence_to_unique,
                                            u32* indices)
{
    u32 write = 0;
    const u32 segments = cut_count + 1;
    switch (primitive) {
    case RSX_PRIMITIVE_TRIANGLES: {
        const u32 count = source_refs - source_refs % 3u;
        for (u32 o = 0; o < count; o++)
            indices[write++] = eng_topology_vertex(occurrence_to_unique, o);
        break;
    }
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            /* Odd triangles swap their first two vertices, so the whole strip
             * keeps one winding. */
            for (u32 i = 0; i + 2 < count; i++) {
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 1u - (i & 1u));
                indices[write++] = eng_topology_vertex(
                    occurrence_to_unique, begin + i + 2u);
            }
        }
        break;
    /* A polygon is a convex fan around its own first vertex, which is what
     * both of this tree's other renderers make of one. */
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 3) continue;
            for (u32 i = 1; i + 1 < count; i++) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + i + 1u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUAD_STRIP:
        for (u32 s = 0; s < segments; s++) {
            u32 begin, count;
            rsx_restart_segment_bounds(cuts, cut_count, source_refs, s,
                                       &begin, &count);
            if (count < 4) continue;
            /* Every quad is split along the diagonal between the two pairs,
             * so each one keeps the strip's winding without alternating the
             * way a triangle strip has to; a cut simply starts the pairing
             * again from the segment's own first vertex. This is the
             * expansion the vtable path's emit_vertices performs. */
            for (u32 q = 0; q + 3 < count; q += 2) {
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 1u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 3u);
                indices[write++] = eng_topology_vertex(occurrence_to_unique, begin + q + 2u);
            }
        }
        break;
    case RSX_PRIMITIVE_QUADS:
        for (u32 q = 0; q < source_refs / 4u; q++) {
            const u32 base = q * 4u;
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 1u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 2u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base + 3u);
            indices[write++] = eng_topology_vertex(occurrence_to_unique, base);
        }
        break;
    default:
        break;
    }
}

/* Does this primitive become an indexed triangle list, and is the index buffer
 * needed at all? rsx_vertex_topology_plan answers that for the reference
 * engine, whose own expansion stops at quads. Quad strips and polygons are
 * this engine's, so their two cases are decided here rather than in the
 * shared helper, which the vendored rsx_live_draw.c also calls. */
static int eng_topology_rebuild(u32 primitive, int refs_remapped, int* indexed)
{
    if (primitive == RSX_PRIMITIVE_QUAD_STRIP ||
        primitive == RSX_PRIMITIVE_POLYGON) {
        (void)refs_remapped;
        *indexed = 1;
        return 1;
    }
    return rsx_vertex_topology_plan(primitive, refs_remapped, indexed);
}

/* ---- surfaces ------------------------------------------------------------ */

static rsx_be_format eng_surface_format(u32 color_format)
{
    switch (color_format & 0x1Fu) {
    case RSX_SURFACE_FMT_F_W16Z16Y16X16: return RSX_BE_FMT_R16G16B16A16F;
    case 0x0C: return RSX_BE_FMT_R32G32B32A32F;
    case 0x0D: return RSX_BE_FMT_R32F;
    default:   return RSX_BE_FMT_R8G8B8A8;
    }
}

/* Decode the guest's own bytes behind a colour surface into host R,G,B,A rows
 * so a freshly created target starts with what the title CPU-initialised it
 * to. Returns the staging pointer, or NULL when the bytes are unreadable or
 * the format is not one the decoder handles. */
static const void* eng_surface_seed(u32 location, u32 offset, u32 w, u32 h,
                                    rsx_be_format fmt, u32* out_row_bytes)
{
    *out_row_bytes = 0;
    if (fmt != RSX_BE_FMT_R8G8B8A8) return NULL;
    /* Only a surface in IO-mapped MAIN memory is seeded. RSX local memory is
     * reserved and not backed by this tree's guest VM -- runtime/memory/vm.h
     * maps the whole space PROT_NONE and commits main memory and the stacks,
     * leaving a runner to commit what its own title needs -- so reading a VRAM
     * surface's backing would fault long before it could help. Both halves are
     * checked: the DMA context says which space the offset is in, and the IO
     * table says the page is really mapped. */
    if (location != RSX_LOCATION_MAIN ||
        (!s_guest_reader && !cellGcmResolveIO(offset))) return NULL;
    rsx_tex_layout tl;
    /* A8R8G8B8 with the LN bit: a surface is linear, never swizzled. */
    rsx_texture_layout(0x85u | 0x20u, w, h, &tl);
    if (!tl.face_bytes || !tl.dst_row_bytes) return NULL;
    const u8* src = eng_guest_ptr(NULL, location, offset, tl.face_bytes);
    if (!src) return NULL;
    const u32 bytes = tl.dst_row_bytes * tl.rows;
    if (g.tex_staging_cap < bytes) {
        u8* n = (u8*)realloc(g.tex_staging, bytes);
        if (!n) return NULL;
        g.tex_staging = n;
        g.tex_staging_cap = bytes;
    }
    rsx_texture_decode(g.tex_staging, tl.dst_row_bytes, src, w, h, &tl,
                       rsx_texture_argb_is_rgba());
    *out_row_bytes = tl.dst_row_bytes;
    return g.tex_staging;
}

/* The colour target for this (location, offset), created or reallocated when
 * the size or format moved. Returns a slot index, or ENG_INVALID.
 * rsx_live_draw.c's surface_get (2605-2690).
 *
 * A smaller clip on the same rows is the same memory, not a new surface: with
 * the same format and pitch, RSX draws into the top-left of what is already
 * there. God of War: Ascension renders its scene into a 1280x720 target, then
 * runs a 640x360 pass at the same address and pitch and copies the full
 * 1280x720 out for its SPU anti-aliasing; recreating the target at 640x360
 * lost the scene and left that quarter of the frame black. */
static u32 eng_surface_get(u32 location, u32 offset, u32 want_w, u32 want_h,
                          rsx_be_format want_fmt, u32 pitch)
{
    if (!want_w) want_w = g.width;
    if (!want_h) want_h = g.height;

    u32 slot = ENG_MAX_SURFACES;
    for (u32 i = 0; i < g.n_surfaces; i++)
        if (g.surfaces[i].location == location && g.surfaces[i].offset == offset) {
            const eng_surface* s = &g.surfaces[i];
            if (s->fmt == want_fmt &&
                ((s->w == want_w && s->h == want_h) ||
                 (pitch && s->pitch == pitch && s->w >= want_w && s->h >= want_h)))
                return i;
            slot = i;
            break;
        }

    /* Never destroy a usable render target because one malformed command
     * decoded a guest pointer as clip dimensions. */
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM) {
        static u32 logs = 0;
        if (PS3_LOG_FIRST(logs, 8))
            fprintf(stderr, "[rsx engine] rejected implausible surface 0x%X %ux%u;"
                            " keeping the %s target\n", offset, want_w, want_h,
                    slot < ENG_MAX_SURFACES ? "existing" : "absent");
        return slot < ENG_MAX_SURFACES ? slot : ENG_INVALID;
    }
    if (slot == ENG_MAX_SURFACES) {
        if (g.n_surfaces >= ENG_MAX_SURFACES) {
            static u32 full_logs = 0;
            if (PS3_LOG_FIRST(full_logs, 8))
                fprintf(stderr, "[eng-tables] SURFACE TABLE FULL (%u): no slot for 0x%X %ux%u frame %llu -- its draws are dropped\n",
                        (unsigned)ENG_MAX_SURFACES, offset, want_w, want_h, (unsigned long long)g.frames);
            return ENG_INVALID;
        }
        slot = g.n_surfaces;
    }

    u32 seed_row = 0;
    const void* seed = eng_surface_seed(location, offset, want_w, want_h,
                                        want_fmt, &seed_row);
    const u32 handle = g.be->color_target_create(g.be->user, want_fmt,
                                                 want_w, want_h, seed, seed_row);
    if (!handle)
        return (slot < g.n_surfaces && g.surfaces[slot].handle) ? slot : ENG_INVALID;

    eng_surface* s = &g.surfaces[slot];
    if (s->feedback) { g.be->texture_release(g.be->user, s->feedback); s->feedback = 0; }
    for (u32 i = 0; i < 2; i++) if (s->rg16[i].tex) g.be->texture_release(g.be->user, s->rg16[i].tex);
    memset(s->rg16, 0, sizeof s->rg16);
    if (s->handle) g.be->color_target_release(g.be->user, s->handle);
    s->location = location; s->offset = offset;
    s->w = want_w; s->h = want_h; s->fmt = want_fmt;
    s->handle = handle;
    s->color_serial = 0;
    s->guest_dirty = 0;
    if (slot == g.n_surfaces) g.n_surfaces++;
    { static u32 logs = 0; if (PS3_LOG_FIRST(logs, 16))
        fprintf(stderr, "[rsx engine] surface %u:0x%08X %ux%u fmt %d%s\n",
                location, offset, want_w, want_h, (int)want_fmt,
                seed ? " (seeded from guest memory)" : ""); }
    return slot;
}

/* The colour targets SET_SURFACE_COLOR_TARGET names, as slots into
 * g.surfaces, with A first. One for an ordinary draw; an MRT set adds B, C
 * and D, each created at target A's size and format, which is the rule the
 * vtable path's target_for binds by. Resolution stops at the first member
 * that cannot be created, as that path stops at the first gap. Returns how
 * many were resolved, or 0. */
static u32 eng_current_target_set(u32 slots[RSX_BE_MAX_COLOR_TARGETS])
{
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(&g.rsx, &sf);
    /* SET_SURFACE_COLOR_TARGET 2 selects B alone; every other value starts at
     * A, and the MRT selectors add B, C and D on top of it. */
    const u32 sel = (sf.color_target == CELL_GCM_SURFACE_TARGET_1) ? 1u : 0u;
    const rsx_be_format fmt = eng_surface_format(sf.color_format);
    const u32 first = eng_surface_get(sf.color_location[sel], sf.color_offset[sel],
                                      sf.clip_w, sf.clip_h, fmt, sf.color_pitch[sel]);
    if (first == ENG_INVALID) return 0;
    g.surfaces[first].pitch = sf.color_pitch[sel];
    slots[0] = first;
    u32 n = 1;

    static const u32 mrt_from[3] = {
        CELL_GCM_SURFACE_TARGET_MRT1, CELL_GCM_SURFACE_TARGET_MRT2,
        CELL_GCM_SURFACE_TARGET_MRT3
    };
    for (u32 i = 0; i < 3; i++) {
        if (sf.color_target < mrt_from[i]) break;
        /* At A's size, which may be larger than the clip (see above): the
         * framebuffer takes A's extent, and every attachment must cover it. */
        const u32 slot = eng_surface_get(sf.color_location[i + 1],
                                         sf.color_offset[i + 1],
                                         g.surfaces[first].w, g.surfaces[first].h, fmt,
                                         sf.color_pitch[i + 1]);
        if (slot == ENG_INVALID) break;
        g.surfaces[slot].pitch = sf.color_pitch[i + 1];
        /* A set naming one buffer twice would attach the same target twice,
         * which no host API allows. */
        int seen = 0;
        for (u32 k = 0; k < n; k++) if (slots[k] == slot) seen = 1;
        if (seen) break;
        slots[n++] = slot;
    }
    return n;
}

/* ---- coherence with guest memory around 2D transfers ------------------- */

/* Bytes per pixel of a surface a 2D transfer can touch: A8R8G8B8, and the
 * FP16 RGBA target an HDR pass renders into (Twisted Metal copies its bloom
 * out of one for the SPU post-FX). 0 for any other format. */
static u32 eng_transfer_bpp(rsx_be_format fmt)
{
    switch (fmt) {
    case RSX_BE_FMT_R8G8B8A8:      return 4u;
    case RSX_BE_FMT_R16G16B16A16F: return 8u;
    default:                       return 0u;
    }
}

/* The surface holding guest bytes [offset, offset + pitch * h), with the byte
 * position of `offset` inside its row (*xb) and its row (*y). Only a surface
 * laid out with the transfer's own pitch qualifies: another pitch puts the
 * rows elsewhere. The CLOSEST base at or below the address wins: surfaces'
 * guest ranges overlap, and an FP16 target right after an RGBA8 one must not
 * resolve into the RGBA8 one's rows (rsx_live_draw.c's resolve takes the same
 * rule, and its comment records copying out of a stale surface at 0x0 before
 * it did). */
static u32 eng_surface_at(u32 location, u32 offset, u32 pitch, u32* xb, u32* y)
{
    u32 best = ENG_INVALID;
    for (u32 i = 0; i < g.n_surfaces; i++) {
        const eng_surface* s = &g.surfaces[i];
        if (!s->handle || s->location != location || offset < s->offset) continue;
        const u32 bpp = eng_transfer_bpp(s->fmt);
        if (!bpp || s->pitch != pitch) continue;
        const u32 d = offset - s->offset;
        if (d / pitch >= s->h || d % pitch >= s->w * bpp) continue;
        if (best != ENG_INVALID && g.surfaces[best].offset > s->offset) continue;
        best = i;
        *xb = d % pitch;
        *y = d / pitch;
    }
    return best;
}

static u8* s_coherence_rows;
static u32 s_coherence_cap;

/* RSX_ENGINE_COHERENCE_LOG=<n>: log the first n transfers the engine syncs,
 * with how many non-zero pixels moved -- the question when a composite comes
 * out black is whether the GPU surface had anything in it. */
static int eng_coherence_log(void)
{
    static int left = -1;
    if (left < 0) { const char* e = getenv("RSX_ENGINE_COHERENCE_LOG"); left = e ? atoi(e) : 0; }
    return left > 0 ? left-- : 0;
}

/* Pixels of `bpp` bytes whose colour (the first three quarters: R,G,B) is
 * not all zero. */
static u32 eng_nonzero_pixels(const u8* rows, u32 n, u32 bpp)
{
    u32 k = 0;
    for (u32 i = 0; i < n; i++, rows += bpp) {
        u8 any = 0;
        for (u32 b = 0; b < bpp * 3u / 4u; b++) any |= rows[b];
        k += any != 0;
    }
    return k;
}

static u8* eng_coherence_buffer(u32 bytes)
{
    if (s_coherence_cap < bytes) {
        u8* n = (u8*)realloc(s_coherence_rows, bytes);
        if (!n) return NULL;
        s_coherence_rows = n;
        s_coherence_cap = bytes;
    }
    return s_coherence_rows;
}

static void pd_note_xfer(const char* what, u32 location, u32 offset, u32 w, u32 h,
                         u32 surface, u32 nonzero);
static int pd_xfer_wanted(void);

/* RSX_GUEST_READ_FAST=0: rsx_draw_engine_guest_read as it was (the pass-dump
 * pixel count always computed, rows converted in place then copied), for A/B.
 * Read once; rsx_draw_engine_set_walker_fast overrides it for tests. */
static int s_guest_read_fast = -1;
static int eng_guest_read_fast(void)
{
    if (s_guest_read_fast < 0) {
        const char* e = getenv("RSX_GUEST_READ_FAST");
        s_guest_read_fast = (e && e[0] == '0') ? 0 : 1;
    }
    return s_guest_read_fast;
}

/* One row of 32-bit pixels between the backend's R,G,B,A and the guest's
 * A8R8G8B8 bytes (or straight through with TEX_RGBA). Copied a byte at a
 * time, with the TEX_RGBA test inside the loop, these two conversions were
 * about a quarter of the FIFO thread's time while Twisted Metal booted; at
 * the -O2 the runtime builds with, GCC vectorises neither form. Each pixel is
 * one 32-bit rotate instead: bytes s0..s3 load as s0 | s1<<8 | s2<<16 | s3<<24
 * on a little-endian host (ps3emu/endian.h), so rotating left by 8 gives
 * s3,s0,s1,s2 (to the guest) and right by 8 gives s1,s2,s3,s0 (from it). */
static void eng_row_convert(u8* dst, const u8* src, u32 w, int rgba, int to_guest)
{
    if (rgba) { memcpy(dst, src, (size_t)w * 4u); return; }
    for (u32 c = 0; c < w; c++) {
        u32 v;
        memcpy(&v, src + (size_t)c * 4u, 4);
        v = to_guest ? (v << 8) | (v >> 24) : (v >> 8) | (v << 24);
        memcpy(dst + (size_t)c * 4u, &v, 4);
    }
}

void rsx_draw_engine_guest_read(u32 location, u32 offset, u32 pitch, u32 w, u32 h)
{
    s_nd_events++;
    if (!g.ready || !g.be->readback || !pitch || !w || !h) return;
    u32 xb = 0, y = 0;
    const u32 slot = eng_surface_at(location, offset, pitch, &xb, &y);
    if (slot == ENG_INVALID) {
        if (eng_coherence_log())
            fprintf(stderr, "[rsx engine] guest_read %u:0x%08X %ux%u: no surface\n",
                    location, offset, w, h);
        pd_note_xfer("read", location, offset, w, h, ENG_INVALID, 0);
        return;
    }
    const eng_surface* s = &g.surfaces[slot];
    const u32 bpp = eng_transfer_bpp(s->fmt);
    /* The transfer counts 32-bit pixels, so w * 4 bytes per row; an FP16
     * surface pixel is two of them. Read back every surface pixel those bytes
     * touch, then store exactly the bytes. */
    u32 bytes = w * 4u;
    if (xb + bytes > s->w * bpp) bytes = s->w * bpp - xb;
    if (y + h > s->h) h = s->h - y;
    const u32 x = xb / bpp, skip = xb % bpp;
    const u32 n = (skip + bytes + bpp - 1u) / bpp;
    u8* rows = eng_coherence_buffer(n * bpp * h);
    if (!rows) return;
    g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
    g.be->readback(g.be->user, s->handle, x, y, n, h, rows, n * bpp);
    if (eng_coherence_log())
        fprintf(stderr, "[rsx engine] guest_read %u:0x%08X %ux%u <- surface 0x%08X (%u bpp) at %u,%u:"
                        " %u non-zero pixels\n", location, offset, w, h, s->offset, bpp, x, y,
                eng_nonzero_pixels(rows, n * h, bpp));
    /* The non-zero count is only for the pass dump's index: scanning every
     * byte of a full-screen readback for a line nobody writes was a tenth of
     * the walker. RSX_GUEST_READ_FAST=0 scans unconditionally again (A/B). */
    const int fast = eng_guest_read_fast();
    if (!fast || pd_xfer_wanted())
        pd_note_xfer("read", location, offset, w, h, s->offset, eng_nonzero_pixels(rows, n * h, bpp));
    const int rgba = rsx_texture_argb_is_rgba();
    for (u32 r = 0; r < h; r++) {
        u8* dst = (u8*)eng_guest_ptr(NULL, location, offset + r * pitch, bytes);
        if (!dst) return;
        u8* src = rows + (size_t)r * n * bpp;
        if (fast && bpp == 4u && !rgba && !skip && bytes == n * 4u) {
            /* Whole pixels: convert straight into guest memory, one pass
             * instead of converting in place and then copying. */
            eng_row_convert(dst, src, n, 0, 1);
            continue;
        }
        if (bpp == 8u) {
            /* R,G,B,A halves, each stored big-endian: the layout
             * rsx_texture_decode reads W16_Z16_Y16_X16_FLOAT from. */
            for (u32 c = 0; c < n * 4u; c++) {
                const u8 t = src[2 * c]; src[2 * c] = src[2 * c + 1]; src[2 * c + 1] = t;
            }
        } else if (!rgba) {
            eng_row_convert(src, src, n, 0, 1);   /* in place: one pixel per load/store */
        }
        memcpy(dst, src + skip, bytes);
    }
}

void rsx_draw_engine_guest_wrote(u32 location, u32 offset, u32 pitch, u32 w, u32 h)
{
    s_nd_events++;
    if (!g.ready || !g.be->color_target_write || !pitch || !w || !h) return;
    u32 x = 0, y = 0;
    const u32 slot = eng_surface_at(location, offset, pitch, &x, &y);
    /* color_target_write takes R,G,B,A bytes: an A8R8G8B8 surface only. */
    if (slot == ENG_INVALID || g.surfaces[slot].fmt != RSX_BE_FMT_R8G8B8A8) {
        if (eng_coherence_log())
            fprintf(stderr, "[rsx engine] guest_wrote %u:0x%08X %ux%u: no A8R8G8B8 surface\n",
                    location, offset, w, h);
        pd_note_xfer("wrote", location, offset, w, h, ENG_INVALID, 0);
        return;
    }
    x /= 4u;
    const eng_surface* s = &g.surfaces[slot];
    if (x + w > s->w) w = s->w - x;
    if (y + h > s->h) h = s->h - y;
    u8* rows = eng_coherence_buffer(w * 4u * h);
    if (!rows) return;
    const int rgba = rsx_texture_argb_is_rgba();
    for (u32 r = 0; r < h; r++) {
        const u8* src = eng_guest_ptr(NULL, location, offset + r * pitch, w * 4u);
        if (!src) return;
        eng_row_convert(rows + (size_t)r * w * 4u, src, w, rgba, 0);
    }
    if (eng_coherence_log())
        fprintf(stderr, "[rsx engine] guest_wrote %u:0x%08X %ux%u -> surface 0x%08X at %u,%u:"
                        " %u non-zero pixels\n", location, offset, w, h, s->offset, x, y,
                eng_nonzero_pixels(rows, w * h, 4u));
    g.be->color_target_write(g.be->user, s->handle, x, y, w, h, rows, w * 4u);
}

static u32 eng_current_surface(void)
{
    u32 slots[RSX_BE_MAX_COLOR_TARGETS];
    return eng_current_target_set(slots) ? slots[0] : ENG_INVALID;
}

/* ---- per-zeta depth ------------------------------------------------------ */

/* One depth target per guest zeta address. Sharing a single resource across a
 * title's shadow, scene and post passes cross-contaminates later depth tests;
 * upstream's ef3271b records that as "the black player-character mass". */
static u32 eng_zdepth_get(u32 location, u32 offset, u32 rt_w, u32 rt_h)
{
    /* The attachment must cover the whole canvas: an early pass declaring a
     * smaller clip than the live viewport otherwise gets a target the host
     * API rejects when a later, larger pass binds it. */
    u32 want_w = rt_w > g.width ? rt_w : g.width;
    u32 want_h = rt_h > g.height ? rt_h : g.height;
    if (want_w > ENG_MAX_SURFACE_DIM || want_h > ENG_MAX_SURFACE_DIM)
        return ENG_INVALID;

    u32 slot = ENG_MAX_ZDEPTHS;
    for (u32 i = 0; i < g.n_zdepths; i++) {
        eng_zdepth* z = &g.zdepths[i];
        if (z->location == location && z->offset == offset) {
            if (z->w >= want_w && z->h >= want_h) return i;
            slot = i;
            if (want_w < z->w) want_w = z->w;
            if (want_h < z->h) want_h = z->h;
            break;
        }
    }
    if (slot == ENG_MAX_ZDEPTHS) {
        if (g.n_zdepths >= ENG_MAX_ZDEPTHS) {
            static u32 full_logs = 0;
            if (PS3_LOG_FIRST(full_logs, 8))
                fprintf(stderr, "[eng-tables] ZETA TABLE FULL (%u): no slot for 0x%X %ux%u frame %llu -- its draws are dropped\n",
                        (unsigned)ENG_MAX_ZDEPTHS, offset, want_w, want_h, (unsigned long long)g.frames);
            return ENG_INVALID;
        }
        slot = g.n_zdepths;
    }

    const u32 handle = g.be->depth_target_create(g.be->user, want_w, want_h);
    if (!handle)
        return (slot < g.n_zdepths && g.zdepths[slot].handle) ? slot : ENG_INVALID;

    eng_zdepth* z = &g.zdepths[slot];
    if (z->handle) g.be->depth_target_release(g.be->user, z->handle);
    for (u32 i = 0; i < 2; i++)
        if (z->argb[i].tex) g.be->texture_release(g.be->user, z->argb[i].tex);
    memset(z->argb, 0, sizeof z->argb);
    z->location = location; z->offset = offset;
    z->w = want_w; z->h = want_h;
    z->handle = handle;
    z->snapshot = 0;
    z->cleared = 0;
    z->had_write = 0;
    z->snapshot_valid = 0;
    z->depth_serial = 0;
    z->guest_dirty = 0;
    if (slot == g.n_zdepths) g.n_zdepths++;
    return slot;
}

/* A depth target read back as a texture, but only once the pass has executed
 * a depth-WRITING draw: a write-enable bit alone does not prove the pass
 * produced a usable depth map, and a clear-only zeta falls through to guest
 * memory instead (rsx_live_draw.c:6016-6018, 6265-6269). */
static u32 eng_zdepth_snapshot(u32 slot)
{
    eng_zdepth* z = &g.zdepths[slot];
    /* Why a snapshot was refused, once per reason: such a texture goes through
     * guest memory instead, which costs a GPU wait and a re-upload. */
    static int s_why[3];
    if (!z->handle || !z->had_write) {
        if (!s_why[0]++) fprintf(stderr, "[eng-zsnap] refused: zeta 0x%08X %ux%u handle %u had_write %u\n",
                                 z->offset, z->w, z->h, z->handle, (u32)z->had_write);
        return 0;
    }
    if (z->snapshot_valid && z->snapshot) return z->snapshot;
    if (!g.be->depth_snapshot) return 0;
    const u32 tex = g.be->depth_snapshot(g.be->user, z->handle, z->w, z->h);
    if (!tex) {
        if (!s_why[1]++) fprintf(stderr, "[eng-zsnap] backend failed: zeta 0x%08X %ux%u\n",
                                 z->offset, z->w, z->h);
        return 0;
    }
    z->snapshot = tex;
    z->snapshot_valid = 1;
    return tex;
}

/* ---- textures ------------------------------------------------------------ */

static rsx_be_format eng_texfmt(rsx_texfmt f)
{
    switch (f) {
    case RSX_TEXFMT_R8:              return RSX_BE_FMT_R8;
    case RSX_TEXFMT_R8G8:            return RSX_BE_FMT_R8G8;
    case RSX_TEXFMT_R8G8B8A8:        return RSX_BE_FMT_R8G8B8A8;
    case RSX_TEXFMT_BC1:             return RSX_BE_FMT_BC1;
    case RSX_TEXFMT_BC2:             return RSX_BE_FMT_BC2;
    case RSX_TEXFMT_BC3:             return RSX_BE_FMT_BC3;
    case RSX_TEXFMT_R16:             return RSX_BE_FMT_R16;
    case RSX_TEXFMT_R16G16:          return RSX_BE_FMT_R16G16;
    case RSX_TEXFMT_R16G16F:         return RSX_BE_FMT_R16G16F;
    case RSX_TEXFMT_R16G16B16A16F:   return RSX_BE_FMT_R16G16B16A16F;
    case RSX_TEXFMT_R32F:            return RSX_BE_FMT_R32F;
    case RSX_TEXFMT_R32G32B32A32F:   return RSX_BE_FMT_R32G32B32A32F;
    default:                         return RSX_BE_FMT_R8;
    }
}

/* How many guest bytes a texture occupies: the whole mip chain, times six for
 * a cube map. That is what the content hash has to cover, or a title
 * animating one face or one lower level keeps its stale upload. */
static u32 eng_texture_span(u32 fmt, u32 w, u32 h, u32 levels, u32 pitch, int cube)
{
    if (cube) return rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) * 6u;
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 n = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!n) return 0;
    return lv[n - 1].offset + lv[n - 1].tl.face_bytes;
}

/* A 3D texture: each level is `depth` slices (halving with the level), one
 * face_bytes each -- consecutive for a linear image, one Morton block for a
 * swizzled one. */
static u32 eng_texture_span_3d(u32 fmt, u32 w, u32 h, u32 depth, u32 levels, u32 pitch)
{
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 n = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!n) return 0;
    u32 total = 0, d = depth ? depth : 1u;
    for (u32 m = 0; m < n; m++) { total += lv[m].tl.face_bytes * d; if (d > 1) d >>= 1; }
    return total;
}

/* Slice z of a swizzled 3D level, de-interleaved into row-major rows of
 * w * bpp bytes, so the 2D decode can convert it as a linear image. */
static void eng_deswizzle_3d_slice(u8* dst, const u8* src, u32 w, u32 h, u32 d, u32 z, u32 bpp)
{
    u32 lw = 0, lh = 0, ld = 0;
    while ((1u << lw) < w) lw++;
    while ((1u << lh) < h) lh++;
    while ((1u << ld) < d) ld++;
    for (u32 y = 0; y < h; y++)
        for (u32 x = 0; x < w; x++)
            memcpy(dst + ((size_t)y * w + x) * bpp,
                   src + (size_t)rsx_swizzle_offset_3d(x, y, z, lw, lh, ld) * bpp, bpp);
}

/* One hash per cached texture per presented frame. Word-at-a-time FNV over
 * the whole span is deliberately cheap: this is a mutation detector, not a
 * content id (rsx_live_draw.c:2002-2026). */
/* RSX_TEXHASH_LANES=0: the single-chain hash below instead of the four-lane
 * one (A/B). Read once, so a run compares one function with itself. */
static int s_texhash_lanes = -1;
static int eng_texhash_lanes(void)
{
    if (s_texhash_lanes < 0) {
        const char* e = getenv("RSX_TEXHASH_LANES");
        s_texhash_lanes = (e && e[0] == '0') ? 0 : 1;
    }
    return s_texhash_lanes;
}

static u64 eng_texture_content_hash(u32 location, u32 offset, u32 span,
                                    int* readable)
{
    const u8* src = span ? eng_guest_ptr(NULL, location, offset, span) : NULL;
    if (!src) { *readable = 0; return 0; }
    u64 hash = 1469598103934665603ull;
    u32 i = 0;
    if (eng_texhash_lanes()) {
        /* The same FNV step on four independent chains, 32 bytes a round:
         * one chain waits on its multiply every 8 bytes, which made this a
         * compute loop well short of memory speed. Each step is a bijection
         * of its lane, and so is folding the lanes together, so any one
         * changed word still always changes the result -- the guarantee the
         * single chain gave. */
        u64 h1 = hash ^ 0x9E3779B97F4A7C15ull, h2 = hash ^ 0xC2B2AE3D27D4EB4Full,
            h3 = hash ^ 0x165667B19E3779F9ull;
        for (; i + 32 <= span; i += 32) {
            u64 w0, w1, w2, w3;
            memcpy(&w0, src + i, 8);
            memcpy(&w1, src + i + 8, 8);
            memcpy(&w2, src + i + 16, 8);
            memcpy(&w3, src + i + 24, 8);
            hash = (hash ^ w0) * 1099511628211ull;
            h1 = (h1 ^ w1) * 1099511628211ull;
            h2 = (h2 ^ w2) * 1099511628211ull;
            h3 = (h3 ^ w3) * 1099511628211ull;
        }
        hash = (hash ^ h1) * 1099511628211ull;
        hash = (hash ^ h2) * 1099511628211ull;
        hash = (hash ^ h3) * 1099511628211ull;
    }
    for (; i + 8 <= span; i += 8) {
        u64 word;
        memcpy(&word, src + i, sizeof(word));
        hash ^= word;
        hash *= 1099511628211ull;
    }
    for (; i < span; i++) { hash ^= src[i]; hash *= 1099511628211ull; }
    *readable = 1;
    return hash;
}

/* The cheap per-frame change check: 64 bytes out of every 2 KB, plus the
 * last 64. A rewrite of the whole texture (a video frame, a UI redraw, a
 * streamed replacement) changes it at once; a small write between the
 * sampled blocks is left to the full hash, which every texture still gets
 * every ENG_TEXHASH_FULL_EVERY frames. */
#define ENG_TEXHASH_STRIDE      2048u
#define ENG_TEXHASH_FULL_EVERY  8u          /* a power of two */
#define ENG_TEXHASH_SMALL       16384u      /* at or below: full every frame */
static u64 eng_texture_sparse_hash(u32 location, u32 offset, u32 span,
                                   int* readable)
{
    const u8* src = span ? eng_guest_ptr(NULL, location, offset, span) : NULL;
    if (!src) { *readable = 0; return 0; }
    u64 hash = 1469598103934665603ull;
    if (eng_texhash_lanes()) {
        /* Four chains over each sampled 64-byte block (two words each), the
         * next blocks prefetched: the single chain waited on a multiply every
         * word and on each block's miss in turn. Bijective per lane and in the
         * fold, like the full hash, so a changed sampled word still always
         * changes the result. */
        u64 h1 = hash ^ 0x9E3779B97F4A7C15ull, h2 = hash ^ 0xC2B2AE3D27D4EB4Full,
            h3 = hash ^ 0x165667B19E3779F9ull;
        for (u32 i = 0; i + 64 <= span; i += ENG_TEXHASH_STRIDE) {
            if (i + 4u * ENG_TEXHASH_STRIDE + 64u <= span) __builtin_prefetch(src + i + 4u * ENG_TEXHASH_STRIDE);
            u64 w[8];
            memcpy(w, src + i, sizeof w);
            hash = (hash ^ w[0]) * 1099511628211ull; h1 = (h1 ^ w[1]) * 1099511628211ull;
            h2 = (h2 ^ w[2]) * 1099511628211ull;     h3 = (h3 ^ w[3]) * 1099511628211ull;
            hash = (hash ^ w[4]) * 1099511628211ull; h1 = (h1 ^ w[5]) * 1099511628211ull;
            h2 = (h2 ^ w[6]) * 1099511628211ull;     h3 = (h3 ^ w[7]) * 1099511628211ull;
        }
        hash = (hash ^ h1) * 1099511628211ull;
        hash = (hash ^ h2) * 1099511628211ull;
        hash = (hash ^ h3) * 1099511628211ull;
    } else
    for (u32 i = 0; i + 64 <= span; i += ENG_TEXHASH_STRIDE)
        for (u32 k = 0; k < 64; k += 8) {
            u64 word;
            memcpy(&word, src + i + k, sizeof(word));
            hash = (hash ^ word) * 1099511628211ull;
        }
    for (u32 i = span > 64 ? span - 64 : 0; i < span; i++)
        hash = (hash ^ src[i]) * 1099511628211ull;
    *readable = 1;
    return hash;
}

/* Whether a cached texture gets the full hash at this use: always when it
 * is small or under PS3RECOMP_ENG_TEXHASH_FULL=1 (the old behaviour, read
 * at engine init), otherwise once ENG_TEXHASH_FULL_EVERY frames have passed
 * since its last one. New entries start staggered by slot, so the full
 * hashes of a scene's textures spread over the frames instead of landing on
 * the same one. */
static int s_texhash_full_always;
/* RSX_TEXHASH_ADAPT=<n> (0 = off, the fixed period): a big texture whose full
 * hash came out unchanged doubles its period, up to FULL_EVERY << n; any
 * change found sets it back to FULL_EVERY. Most of a scene's textures never
 * change, and their full hashes were ~1 GB/s of the walker's reads (GoW:A,
 * Megera). The sparse check still runs every frame, so a texture rewritten
 * as a whole is still caught at once; only a small write between its sampled
 * blocks can wait the longer period. */
static int s_texhash_adapt = -1;
static u32 eng_texhash_adapt(void)
{
    if (s_texhash_adapt < 0) { const char* e = getenv("RSX_TEXHASH_ADAPT"); s_texhash_adapt = e ? atoi(e) : 0;
                               if (s_texhash_adapt < 0) s_texhash_adapt = 0;
                               if (s_texhash_adapt > 5) s_texhash_adapt = 5; }
    return (u32)s_texhash_adapt;
}
static int eng_texture_full_rehash(u32 span, u32 last_full_frame, u32 shift)
{
    return s_texhash_full_always || span <= ENG_TEXHASH_SMALL ||
           (u32)(g.frames - last_full_frame) >= (ENG_TEXHASH_FULL_EVERY << shift);
}

static int eng_staging_reserve(u32 bytes)
{
    if (g.tex_staging_cap >= bytes) return 1;
    u8* n = (u8*)realloc(g.tex_staging, bytes);
    if (!n) return 0;
    g.tex_staging = n;
    g.tex_staging_cap = bytes;
    return 1;
}

/* Decode a guest texture out of guest memory and hand every face and level to
 * the backend. Returns the backend handle, or 0. `into`, when nonzero, is an
 * existing 2D texture of exactly this key to rewrite instead of creating one
 * (rsx_draw_backend.texture_upload_ordered). */
/* RSX_TEX3D_INPLACE=0: a changed volume gets a new texture and the old one is
 * released, as before. On, it is rewritten in place like a 2D one -- the uploads
 * are ordered behind the draws that sampled the old texels the same way -- and
 * GoW:A's colour-grading LUT, which changes nearly every frame, stops costing a
 * create and a release (each a wait for the GPU) per change. */
static int eng_tex3d_inplace(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("RSX_TEX3D_INPLACE"); s = !(e && e[0] == '0'); }
    return s;
}

static u32 eng_texture_upload(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                              u32 levels, u32 pitch, int cube, u32 remap, u32 depth,
                              u32 into)
{
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 nlv = rsx_texture_mip_chain(fmt, w, h, levels, pitch, lv);
    if (!nlv || !lv[0].tl.face_bytes) return 0;
    /* A volume: every slice of every level decoded and uploaded on its own.
     * inFAMOUS 2's occlusion volumes read an 8x8x8 cone LUT; uploaded as its
     * first slice and sampled in 2D it darkened whole boxes round every prop. */
    if (depth > 1 && !cube && g.be->texture_create_3d && g.be->texture_upload_3d &&
        !lv[0].tl.compressed && lv[0].tl.bytes_per_texel) {
        const u32 span = eng_texture_span_3d(fmt, w, h, depth, levels, pitch);
        const u8* src = eng_guest_ptr(NULL, location, offset, span);
        if (!src) return 0;
        /* `into`: rewrite an existing volume of this key in place (the cache's
         * reupload, see eng_tex3d_inplace) instead of a new one per change. */
        const u32 handle = into ? into
            : g.be->texture_create_3d(g.be->user, eng_texfmt(lv[0].tl.fmt), w, h, depth, nlv, remap, fmt);
        if (!handle) return 0;
        const u8* level_src = src;
        u32 d = depth;
        for (u32 m = 0; m < nlv; m++) {
            const rsx_tex_layout* tl = &lv[m].tl;
            const u32 lin_bytes = tl->swizzled ? tl->row_bytes * tl->rows : 0u;
            if (!eng_staging_reserve(tl->dst_row_bytes * tl->rows + lin_bytes)) {
                if (!into) g.be->texture_release(g.be->user, handle);
                return 0;
            }
            for (u32 z = 0; z < d; z++) {
                if (tl->swizzled) {
                    u8* lin = g.tex_staging + tl->dst_row_bytes * tl->rows;
                    eng_deswizzle_3d_slice(lin, level_src, lv[m].w, lv[m].h, d, z, tl->bytes_per_texel);
                    rsx_tex_layout ltl = *tl; ltl.swizzled = 0;
                    rsx_texture_decode(g.tex_staging, tl->dst_row_bytes, lin, lv[m].w, lv[m].h,
                                       &ltl, rsx_texture_argb_is_rgba());
                } else {
                    rsx_texture_decode(g.tex_staging, tl->dst_row_bytes,
                                       level_src + (size_t)z * tl->face_bytes, lv[m].w, lv[m].h,
                                       tl, rsx_texture_argb_is_rgba());
                }
                g.be->texture_upload_3d(g.be->user, handle, z, m, lv[m].w, lv[m].h,
                                        g.tex_staging, tl->dst_row_bytes);
            }
            level_src += (size_t)tl->face_bytes * d;
            if (d > 1) d >>= 1;
        }
        return handle;
    }
    const u32 faces = cube ? 6u : 1u;
    const u32 face_stride = cube
        ? rsx_texture_cube_face_stride(fmt, w, h, levels, pitch) : 0u;
    const u32 span = eng_texture_span(fmt, w, h, levels, pitch, cube);
    const u8* src = eng_guest_ptr(NULL, location, offset, span);
    if (!src) return 0;

    const u32 handle = into ? into
        : g.be->texture_create(g.be->user, eng_texfmt(lv[0].tl.fmt), w, h, nlv, faces, remap, fmt);
    if (!handle) return 0;

    for (u32 f = 0; f < faces; f++) {
        const u8* face = src + (size_t)f * face_stride;
        for (u32 m = 0; m < nlv; m++) {
            const rsx_tex_layout* tl = &lv[m].tl;
            if (!eng_staging_reserve(tl->dst_row_bytes * tl->rows)) {
                if (!into) g.be->texture_release(g.be->user, handle);
                return 0;
            }
            rsx_texture_decode(g.tex_staging, tl->dst_row_bytes,
                               face + lv[m].offset, lv[m].w, lv[m].h, tl,
                               rsx_texture_argb_is_rgba());
            g.be->texture_upload(g.be->user, handle, f, m, lv[m].w, lv[m].h,
                                 g.tex_staging, tl->dst_row_bytes, tl->rows);
        }
    }
    return handle;
}

/* The cache slot for a texture unit's bytes, keyed on where they are and what
 * the registers say they are. An already-cached entry is re-hashed at most
 * once per presented frame and re-decoded when the guest changed it, which is
 * what makes an animated UI or a video texture update at all. A miss with the
 * cache full evicts the least recently used entry rather than returning
 * nothing: returning white "made the recovered orphanage render as flat
 * green/black geometry" (rsx_live_draw.c:2326-2334). */
static u32 eng_texture_slot(u32 location, u32 offset, u32 fmt, u32 w, u32 h,
                            u32 levels, u32 pitch, int cube, u32 remap, u32 depth)
{
    if (!w || !h || w > ENG_MAX_TEX_DIM || h > ENG_MAX_TEX_DIM) return 0;
    if (!depth) depth = 1;
    const u32 span = depth > 1 ? eng_texture_span_3d(fmt, w, h, depth, levels, pitch)
                               : eng_texture_span(fmt, w, h, levels, pitch, cube);
    if (!span) return 0;

    /* A scene binds the same few hundred textures every frame, and a linear
     * scan of the whole cache per bind averaged several hundred entries.
     * Remember where each key was last found; a remembered slot is used only
     * if its entry still matches, otherwise the scan runs as before. Keys are
     * unique in the cache, so the result is the same either way. */
    static u16 s_slot_memo[4096];
    const u32 memo = (offset >> 4 ^ offset >> 16 ^ fmt * 0x9E37u ^ w * 31u ^ h ^
                      location << 11) & 4095u;
#define ENG_TEXTURE_KEY_MATCHES(e) ((e)->location == location && (e)->offset == offset && \
        (e)->format == fmt && (e)->width == w && (e)->height == h && \
        (e)->pitch == pitch && (e)->remap == remap && \
        (e)->cubemap == (u32)(cube != 0) && (e)->depth == depth)
    eng_texture* hit = NULL;
    if (s_slot_memo[memo] < g.n_textures && ENG_TEXTURE_KEY_MATCHES(&g.textures[s_slot_memo[memo]])) {
        hit = &g.textures[s_slot_memo[memo]];
    } else {
        for (u32 i = 0; i < g.n_textures; i++)
            if (ENG_TEXTURE_KEY_MATCHES(&g.textures[i])) {
                hit = &g.textures[i];
                s_slot_memo[memo] = (u16)i;
                break;
            }
    }
#undef ENG_TEXTURE_KEY_MATCHES
    if (hit) {
        eng_texture* e = hit;
        if (e->handle && e->last_hash_frame != g.frames) {
            /* Every cached texture used to be hashed in full every frame;
             * in a big scene that is hundreds of MB a second for the few
             * that change. Now the sparse check every frame, and the full
             * hash when it differs, when this entry's turn has come, for a
             * small texture, or after a write the engine saw. */
            int readable = 0;
            u64 hash = e->content_hash;
            ENG_ST_BEGIN();
            unsigned long long st_h0_ = s_stats_s > 0 ? eng_now_us() : 0;
            const u64 sparse = eng_texture_sparse_hash(location, offset, span,
                                                       &readable);
            if (s_stats_s > 0) { const unsigned long long n_ = eng_now_us(); s_st_sub_us[ST_HSPARSE] += n_ - st_h0_; s_st_sub_n[ST_HSPARSE]++; st_h0_ = n_; }
            /* The engine just stored a surface or depth target over these
             * bytes: they are new, so upload without hashing them all. The
             * stored hash becomes unknown and the next ordinary check
             * re-uploads once rather than miss a change. */
            if (e->rewritten && readable) {
                hash = ~e->content_hash;
                e->rewritten = 0;
            } else if (readable && (sparse != e->sparse_hash ||
                             eng_texture_full_rehash(span, e->last_full_frame, e->full_shift))) {
                hash = eng_texture_content_hash(location, offset, span, &readable);
                e->last_full_frame = g.frames;
                if (readable) {
                    if (hash != e->content_hash || sparse != e->sparse_hash) e->full_shift = 0;
                    else if (e->full_shift < eng_texhash_adapt()) e->full_shift++;
                }
                if (s_stats_s > 0) {
                    s_st_sub_us[ST_HFULL] += eng_now_us() - st_h0_; s_st_sub_n[ST_HFULL]++;
                    if (span <= ENG_TEXHASH_SMALL) s_pc.full_small++; else s_pc.full_big++;
                    s_pc.full_bytes += span;
                }
            }
            if (readable) e->sparse_hash = sparse;
            e->last_hash_frame = g.frames;
            ENG_ST_END(ST_HASH);
            if (readable && hash != e->content_hash) {
                ENG_ST_BEGIN();
                /* In place when the backend orders the upload after the draws
                 * that sampled the old bytes; a volume keeps create-and-swap. */
                /* The key does not hold the mip count: a different one needs a
                 * new image, as the levels to upload would not all exist. */
                const u32 into = (g.be->texture_upload_ordered && e->levels == levels &&
                                  (depth <= 1 || (eng_tex3d_inplace() && e->depth == depth)))
                                     ? e->handle : 0u;
                const u32 fresh = eng_texture_upload(location, offset, fmt, w, h,
                                                     levels, pitch, cube, remap, depth, into);
                if (fresh && !into) {
                    g.be->texture_release(g.be->user, e->handle);
                    e->handle = fresh;
                    e->levels = levels;
                }
                if (fresh) e->content_hash = hash;
                ENG_ST_END(ST_REUP);
            }
        }
        e->last_use_serial = ++g.texture_use_serial;
        return e->handle;
    }

    ENG_ST_BEGIN();
    u32 index;
    u32 evicted = 0;
    /* RSX_ENG_TEXCAP=<n>: smaller live limit, to stress the eviction path. */
    static u32 s_texcap;
    if (!s_texcap) {
        const char* e = getenv("RSX_ENG_TEXCAP");
        const u32 v = e ? (u32)atoi(e) : 0;
        s_texcap = (v >= 16 && v < ENG_MAX_TEXTURES) ? v : ENG_MAX_TEXTURES;
    }
    if (g.n_textures < s_texcap) {
        index = g.n_textures++;
    } else {
        index = 0;
        for (u32 i = 1; i < g.n_textures; i++)
            if (g.textures[i].last_use_serial < g.textures[index].last_use_serial)
                index = i;
        evicted = g.textures[index].handle;
    }

    eng_texture e;
    memset(&e, 0, sizeof(e));
    e.location = location; e.offset = offset; e.format = fmt;
    e.width = w; e.height = h; e.pitch = pitch;
    e.remap = remap; e.cubemap = (u32)(cube != 0); e.depth = depth; e.levels = levels;
    e.last_hash_frame = g.frames;
    e.last_full_frame = g.frames - (index & (ENG_TEXHASH_FULL_EVERY - 1u));
    e.last_use_serial = ++g.texture_use_serial;
    { int readable = 0;
      e.content_hash = eng_texture_content_hash(location, offset, span, &readable);
      e.sparse_hash = eng_texture_sparse_hash(location, offset, span, &readable); }
    e.handle = eng_texture_upload(location, offset, fmt, w, h, levels, pitch,
                                  cube, remap, depth, 0u);
    if (e.handle && evicted) g.be->texture_release(g.be->user, evicted);
    if (e.handle || !evicted) g.textures[index] = e;
    ENG_ST_END(ST_MISS);
    return e.handle;
}

/* ---- surfaces read through guest memory --------------------------------- */

/* A texture may name a surface's bytes in a format the surface was not
 * rendered in: a deferred renderer reads its D24S8 depth buffer as A8R8G8B8
 * to rebuild positions, samples its Z16 shadow map as DEPTH16, and reads back
 * as X32_FLOAT a float it packed into an RGBA8 target. On RSX that is only
 * memory. Here the surface is stored into guest memory in its own layout --
 * the bytes RSX would have left there -- and the texture is then decoded from
 * guest memory like any other, so every format the decoder knows reads them
 * the way RSX would. A surface is stored at most once per write to it. */

/* Re-hash, on their next use rather than next frame, the cached textures that
 * start inside [offset, offset + bytes): their bytes changed within this one. */
static void eng_textures_touch(u32 location, u32 offset, u32 bytes)
{
    for (u32 i = 0; i < g.n_textures; i++) {
        eng_texture* e = &g.textures[i];
        if (e->location == location && e->offset >= offset && e->offset - offset < bytes) {
            e->last_hash_frame = g.frames - 1u;
            e->last_full_frame = g.frames - ENG_TEXHASH_FULL_EVERY;   /* hash it all */
            e->full_shift = 0;
            e->rewritten = 1;
        }
    }
}

/* Formats that read an RGBA8 surface's 32-bit pixels as something else: the
 * host image cannot be sampled as them, the guest bytes can. */
static int eng_format_reinterprets_rgba8(u32 base_fmt)
{
    switch (base_fmt) {
    case RSX_TEX_FMT_DEPTH24_D8:
    case 0x91u:     /* DEPTH24_D8_FLOAT */
    case 0x92u:     /* DEPTH16          */
    case 0x93u:     /* DEPTH16_FLOAT    */
    case 0x95u:     /* Y16_X16          */
    case 0x9Cu:     /* X32_FLOAT        */
    case 0x9Fu:     /* Y16_X16_FLOAT    */
        return 1;
    default:
        return 0;
    }
}

/* Store a colour surface into guest memory, A,R,G,B per pixel at its own
 * pitch. Returns 0 when it cannot, and the caller samples as before. */
static int eng_surface_to_guest(u32 slot)
{
    eng_surface* s = &g.surfaces[slot];
    if (!s->guest_dirty) return 1;
    if (s->fmt != RSX_BE_FMT_R8G8B8A8 || s->pitch < s->w * 4u || !g.be->readback)
        return 0;
    rsx_draw_engine_guest_read(s->location, s->offset, s->pitch, s->w, s->h);
    s->guest_dirty = 0;
    eng_textures_touch(s->location, s->offset, s->pitch * s->h);
    return 1;
}

static float* s_depth_rows;
static u32 s_depth_cap;

/* Store a depth target into guest memory as its zeta format lays it out:
 * Z16 as one big-endian 16-bit value, Z24S8 as a big-endian word with the
 * depth in the top 24 bits. The stencil byte is written as 0: the host depth
 * target keeps no stencil to copy. Returns 0 when it cannot. */
static u8* s_stencil_rows;
static u32 s_stencil_cap;

static int eng_zdepth_to_guest(u32 slot)
{
    eng_zdepth* z = &g.zdepths[slot];
    if (!z->guest_dirty) return 1;
    if (!g.be->depth_readback || !z->handle) return 0;
    const u32 bpp = (z->zfmt == 1u) ? 2u : 4u;      /* SURFACE_FORMAT Z16 = 1 */
    const u32 w = z->used_w < z->w ? z->used_w : z->w;
    const u32 h = z->used_h < z->h ? z->used_h : z->h;
    if (!w || !h || z->pitch < w * bpp) return 0;
    if (s_depth_cap < w * h) {
        float* n = (float*)realloc(s_depth_rows, (size_t)w * h * sizeof(float));
        if (!n) return 0;
        s_depth_rows = n;
        s_depth_cap = w * h;
    }
    g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
    if (g.be->depth_readback(g.be->user, z->handle, 0, 0, w, h, s_depth_rows,
                             w * (u32)sizeof(float)) != 0)
        return 0;
    /* The stencil byte of a D24S8 word, when the backend can read it back. */
    const u8* sten = NULL;
    if (bpp == 4u && g.be->stencil_readback) {
        if (s_stencil_cap < w * h) {
            u8* n = (u8*)realloc(s_stencil_rows, (size_t)w * h);
            if (n) { s_stencil_rows = n; s_stencil_cap = w * h; }
        }
        if (s_stencil_cap >= w * h &&
            g.be->stencil_readback(g.be->user, z->handle, 0, 0, w, h, s_stencil_rows, w) == 0)
            sten = s_stencil_rows;
    }
    for (u32 r = 0; r < h; r++) {
        u8* dst = (u8*)eng_guest_ptr(NULL, z->location, z->offset + r * z->pitch, w * bpp);
        if (!dst) return 0;
        const float* src = s_depth_rows + (size_t)r * w;
        const u8* srow = sten ? sten + (size_t)r * w : NULL;
        for (u32 c = 0; c < w; c++) {
            float d = src[c];
            if (!(d > 0.0f)) d = 0.0f;
            if (d > 1.0f) d = 1.0f;
            if (bpp == 2u) {
                const u32 v = (u32)(d * 65535.0f + 0.5f);
                dst[0] = (u8)(v >> 8); dst[1] = (u8)v;
            } else {
                /* Float32 rounding at this magnitude can turn depth 1 into
                 * 0x1000000. Quantize in double before storing the 24-bit value. */
                const u32 v = (u32)((double)d * 16777215.0 + 0.5);
                dst[0] = (u8)(v >> 16); dst[1] = (u8)(v >> 8); dst[2] = (u8)v;
                dst[3] = srow ? srow[c] : 0;
            }
            dst += bpp;
        }
    }
    z->guest_dirty = 0;
    eng_textures_touch(z->location, z->offset, z->pitch * h);
    if (eng_coherence_log())
        fprintf(stderr, "[rsx engine] depth 0x%08X (%s, pitch %u) %ux%u stored to guest memory\n",
                z->offset, bpp == 2u ? "Z16" : "Z24S8", z->pitch, w, h);
    return 1;
}

/* A depth write or clear through the current zeta, recorded with what the
 * guest-memory copy needs to lay it out. */
static void eng_zdepth_wrote(u32 slot, const rsx_dsp_surface* sf)
{
    eng_zdepth* z = &g.zdepths[slot];
    z->depth_serial = ++g.write_serial;
    z->guest_dirty = 1;
    z->zfmt = sf->depth_format;
    z->pitch = sf->zeta_pitch;
    z->used_w = sf->clip_w ? sf->clip_w : g.width;
    z->used_h = sf->clip_h ? sf->clip_h : g.height;
}

static void eng_surfaces_wrote(const u32* targets, u32 n_targets)
{
    const u64 serial = ++g.write_serial;
    for (u32 i = 0; i < n_targets; i++) {
        g.surfaces[targets[i]].color_serial = serial;
        g.surfaces[targets[i]].guest_dirty = 1;
    }
}

/* SET_TEXTURE_FILTER's min field is [18:16] (1 NEAREST, 2 LINEAR, then 3..6,
 * the four combinations of a nearest/linear minification with a
 * nearest/linear mip filter) and mag [26:24]; SET_TEXTURE_CONTROL0 carries
 * max LOD at [18:7] and min LOD at [30:19], both 4.8 fixed point. RSX's LOD
 * bias, SET_TEXTURE_FILTER [12:0], is not applied: the reference engine
 * leaves it at zero too. */
static void eng_decode_sampler(u32 filter, u32 wrap, u32 control0,
                               rsx_be_sampler_desc* out, u32 border_color, u32 format)
{
    const u32 fb = format & 0x9Fu;          /* DEPTH24_D8, DEPTH24_D8_FLOAT, DEPTH16, DEPTH16_FLOAT */
    out->border_color = (fb >= 0x90u && fb <= 0x93u) ? border_color : 0u;
    const u32 minf = (filter >> 16) & 7u;
    const u32 magf = (filter >> 24) & 7u;
    out->min_linear  = (u8)(minf == 2 || minf == 4 || minf == 6);
    out->mag_linear  = (u8)(magf == 2);
    out->mip_present = (u8)(minf >= 3);
    out->mip_linear  = (u8)(minf == 5 || minf == 6);
    out->wrap_s = (u8)(wrap & 0xFu);
    out->wrap_t = (u8)((wrap >> 8) & 0xFu);
    out->wrap_r = (u8)((wrap >> 16) & 0xFu);
    out->min_lod = (float)((control0 >> 19) & 0xFFFu) / 256.0f;
    out->max_lod = out->mip_present ? (float)((control0 >> 7) & 0xFFFu) / 256.0f
                                    : 0.0f;
    if (out->max_lod < out->min_lod) out->max_lod = out->min_lod;
}

/* ---- pipelines ----------------------------------------------------------- */

static u32 eng_vtex_mask(void)
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_VERTEX_TEXTURES; u++) {
        rsx_dsp_vertex_texture vt;
        rsx_dsp_get_vertex_texture(&g.rsx, u, &vt);
        if (vt.enabled && vt.width && vt.height) mask |= 1u << u;
    }
    return mask;
}

/* Units addressed in texels (RSX_TEX_FMT_UNNORM), with each one's size in
 * dim. No host sampling mode is unnormalised for these, so the fragment
 * program divides the coordinates instead, and the divisor makes the sizes
 * pipeline identity (rsx_live_draw.c keys them the same way). */
static u32 eng_unnorm_units(u32 dim[RSX_DSP_NUM_TEXTURES][2])
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        dim[u][0] = dim[u][1] = 0;
        if (!t.enabled || !(t.format & RSX_TEX_FMT_UNNORM) || !t.width || !t.height)
            continue;
        mask |= 1u << u;
        dim[u][0] = t.width;
        dim[u][1] = t.height;
    }
    return mask;
}

/* Units bound to a 3D texture (SET_TEXTURE_FORMAT dimension 3 with a
 * depth), declared Texture3D and sampled with three coordinates. Only when
 * the backend can create one: otherwise the first slice stays a 2D texture
 * and the program samples it in 2D, as before. */
static u32 eng_volume_mask(void)
{
    if (!g.be->texture_create_3d || !g.be->texture_upload_3d) return 0;
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        if (t.enabled && !t.cubemap && t.dimension == 3 && t.depth > 1 && t.width && t.height)
            mask |= 1u << u;
    }
    return mask;
}

static u32 eng_cube_mask(void)
{
    u32 mask = 0;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        /* A cube face is square by construction, so a non-square image
         * cannot be one; the fragment program is compiled against this mask
         * and a texturecube slot filled with a 2D texture is a validation
         * failure, not a wrong pixel. */
        if (t.enabled && t.cubemap && t.width && t.width == t.height)
            mask |= 1u << u;
    }
    return mask;
}

/* Are both of the guest's own programs resident? One answer, used by the
 * layout and by the pipeline, because they must not disagree: the built-in
 * program declares all sixteen inputs, so narrowing the layout for it would
 * leave the pipeline's vertex descriptor short of what the shader reads. */
static int eng_guest_programs(const u8** out_vp, u32* out_vp_instrs,
                              const u8** out_fp, u32* out_fp_size)
{
    const u8* vp_uc = NULL;
    u32 vp_instrs = 0;
    const u32 start = rsx_dsp_vp_start(&g.rsx);
    if (start < RSX_DSP_VP_INSTR) {
        vp_uc = (const u8*)(g.rsx.vp + start * 4);
        vp_instrs = rsx_vp_program_size_instrs(
            vp_uc, (RSX_DSP_VP_INSTR - start) * 16u);
    }

    /* A zero SET_SHADER_PROGRAM is "none", not "the program at offset 0": the
     * register carries the location in its low two bits, so a real program can
     * never read back as zero. */
    const u8* fp_uc = NULL;
    u32 fp_size = 0;
    if (rsx_dsp_reg(&g.rsx, 0x08E4)) {
        u32 fp_loc = 0;
        const u32 fp_off = rsx_dsp_fragment_program(&g.rsx, &fp_loc);
        fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, 16);
        fp_size = fp_uc ? rsx_fp_program_size(fp_uc, ENG_FP_MAX_BYTES) : 0;
        if (fp_size) fp_uc = eng_guest_ptr(NULL, fp_loc, fp_off, fp_size);
        if (!fp_uc) fp_size = 0;
    }

    if (out_vp) *out_vp = vp_uc;
    if (out_vp_instrs) *out_vp_instrs = vp_instrs;
    if (out_fp) *out_fp = fp_uc;
    if (out_fp_size) *out_fp_size = fp_size;
    return vp_instrs && fp_size;
}

/* The layout the active vertex program actually reads. An uncertain analysis
 * falls back to all sixteen registers (rsx_live_draw.c:3625-3643). */
static void eng_vertex_layout(rsx_vertex_layout_plan* layout)
{
    u32 mask = 0xFFFFu;
    const u8* uc = NULL;
    u32 instrs = 0;
    if (eng_guest_programs(&uc, &instrs, NULL, NULL)) {
        rsx_vp_input_analysis analysis = { 0xFFFFu, 0 };
        if (rsx_vp_analyze_inputs(uc, instrs * 16u, &analysis) == (int)instrs &&
            analysis.exact && analysis.input_mask)
            mask = analysis.input_mask;
    }
    rsx_vertex_layout_plan_init(layout, mask);
}

/* What a draw runs before a title has loaded programs of its own.
 *
 * The reference engine has no fallback and drops such a draw, because a title
 * always has both programs resident by the time it draws anything. A general
 * toolkit does not get to assume that: the host harness's fixed-function modes
 * are exactly this case, and so is the first frame of a title that clears and
 * flips before its first SET_TRANSFORM_PROGRAM. It is written in the same HLSL
 * shape the decompilers emit -- ATTRn inputs, the COLOR0/TEXCOORD varyings, the
 * VPConst block and the viewport epilogue -- so it goes through the one
 * pipeline_create the backend already implements rather than needing a second
 * entry point for a built-in shader.
 *
 * Vertex constant rows 0..3 are the transform when the guest programmed them
 * and identity when it did not: an all-zero matrix would collapse every vertex
 * onto the origin. */
static const char* const kEngFixedVS =
"struct VSInput {\n"
"    float4 a0:ATTR0;  float4 a1:ATTR1;  float4 a2:ATTR2;  float4 a3:ATTR3;\n"
"    float4 a4:ATTR4;  float4 a5:ATTR5;  float4 a6:ATTR6;  float4 a7:ATTR7;\n"
"    float4 a8:ATTR8;  float4 a9:ATTR9;  float4 a10:ATTR10; float4 a11:ATTR11;\n"
"    float4 a12:ATTR12; float4 a13:ATTR13; float4 a14:ATTR14; float4 a15:ATTR15;\n"
"};\n"
"struct VSOutput {\n"
"    float4 pos:SV_Position; float4 col0:COLOR0; float4 col1:COLOR1;\n"
"    float4 fog:FOG;\n"
"    float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;\n"
"    float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7;\n"
"    float4 t8:TEXCOORD8; float4 t9:TEXCOORD9;\n"
"};\n"
"cbuffer VPConst : register(b0) {\n"
"    float4 vp_c[512];\n"
"    float4 vp_posscale;\n"
"    float4 vp_posoffset;\n"
"};\n"
"VSOutput main(VSInput input) {\n"
"    float4 _p = float4(dot(vp_c[0], input.a0), dot(vp_c[1], input.a0),\n"
"                       dot(vp_c[2], input.a0), dot(vp_c[3], input.a0));\n"
"    float _m = dot(abs(vp_c[0]), 1.0) + dot(abs(vp_c[1]), 1.0)\n"
"             + dot(abs(vp_c[2]), 1.0) + dot(abs(vp_c[3]), 1.0);\n"
"    if (_m == 0.0) _p = input.a0;\n"
"    VSOutput Out;\n"
"    Out.pos = float4(_p.xyz * vp_posscale.xyz + _p.w * vp_posoffset.xyz, _p.w);\n"
"    Out.col0 = input.a3; Out.col1 = float4(0,0,0,1); Out.fog = (float4)0;\n"
"    Out.t0 = input.a8; Out.t1 = (float4)0; Out.t2 = (float4)0; Out.t3 = (float4)0;\n"
"    Out.t4 = (float4)0; Out.t5 = (float4)0; Out.t6 = (float4)0; Out.t7 = (float4)0;\n"
"    Out.t8 = (float4)0; Out.t9 = (float4)0;\n"
"    return Out;\n"
"}\n";

static const char* const kEngFixedPS =
"struct PSInput {\n"
"    float4 position : SV_POSITION; float4 col0 : COLOR0; float4 col1 : COLOR1;\n"
"    float4 fog : FOG;\n"
"    float4 tc0:TEXCOORD0; float4 tc1:TEXCOORD1; float4 tc2:TEXCOORD2; float4 tc3:TEXCOORD3;\n"
"    float4 tc4:TEXCOORD4; float4 tc5:TEXCOORD5; float4 tc6:TEXCOORD6; float4 tc7:TEXCOORD7;\n"
"};\n"
"float4 main(PSInput input) : SV_TARGET { return input.col0; }\n";

/* The texture units a translated pixel program samples. Every sampling call
 * the decompiler emits names its unit's sampler, rsx_samp[N], whatever the
 * texture's own spelling (rsx_tex[N] or, with a cube unit, rsx_texN); the
 * declaration's rsx_samp[16] is out of range and so not counted. */
static u32 eng_sampled_units(const char* ps)
{
    u32 mask = 0;
    for (const char* q = ps; (q = strstr(q, "rsx_samp[")) != NULL; q += 9) {
        const unsigned long u = strtoul(q + 9, NULL, 10);
        if (u < RSX_BE_MAX_TEXTURES) mask |= 1u << u;
    }
    return mask;
}

/* The pipeline for this draw, built once per distinct key. A negative result
 * is cached too (handle 0), so a program pair that will not translate is not
 * retried on every draw of every frame. `out_tex_used` gets the units the
 * pixel program samples: all of them for the built-in pair, which keeps
 * every enabled unit bound as before. */
/* Up to 16 comma-separated hex hashes from an environment variable. */
static u32 eng_parse_hash_list(const char* name, u64 out[16])
{
    const char* e = getenv(name);
    u32 n = 0;
    while (e && *e && n < 16) {
        char* end;
        const u64 v = strtoull(e, &end, 16);
        if (end == e) break;
        if (v) out[n++] = v;
        if (*end != ',') break;
        e = end + 1;
    }
    return n;
}

/* The per-draw fragment-program debug knobs; nonzero drops the draw.
 *
 * RSX_ENGINE_SKIP_FP=<hex>[,<hex>...] drops every draw through those
 * fragment programs: which pass paints a wrong smear, halo or black band is
 * found by removing passes one at a time.
 *
 * RSX_ENGINE_REGDUMP_FP=<hex>[,<hex>...] prints the non-zero words of the
 * whole RSX register file at the first draw through each of those programs,
 * so the state of a broken pass can be diffed register by register against a
 * working one -- which state it sets that the engine does not honour. */
static int eng_debug_fp(u64 fp_hash, u32 pipeline)
{
    static int init;
    static u64 skip[16], dump[16];
    static u32 n_skip, n_dump, dumped;
    if (!init) {
        init = 1;
        n_skip = eng_parse_hash_list("RSX_ENGINE_SKIP_FP", skip);
        n_dump = eng_parse_hash_list("RSX_ENGINE_REGDUMP_FP", dump);
    }
    for (u32 i = 0; i < n_dump; i++) {
        if (dump[i] != fp_hash || (dumped & (1u << i))) continue;
        dumped |= 1u << i;
        fprintf(stderr, "[rsx regdump] fp %016llx pipe %u frame %u begin\n",
                (unsigned long long)fp_hash, pipeline, g.frames);
        for (u32 r = 0; r < RSX_DSP_NUM_REGS; r++)
            if (g.rsx.regs[r])
                fprintf(stderr, "[rsx regdump] 0x%04X = 0x%08X\n", r * 4u, g.rsx.regs[r]);
        fprintf(stderr, "[rsx regdump] fp %016llx end\n", (unsigned long long)fp_hash);
    }
    for (u32 i = 0; i < n_skip; i++)
        if (skip[i] == fp_hash) return 1;
    return 0;
}

/* A guest fragment program's name for the debug knobs: its structural hash
 * (inline constants excluded) seeded with the FNV offset basis, the value
 * RSX_FP_DEBUG matches in the decompiler. Unlike a backend pipeline handle it
 * does not depend on the order programs are first drawn in, so a program
 * found in one run is named the same in the next. 0 for the built-in one. */
static u64 eng_fp_debug_hash(const u8* fp_uc, u32 fp_size)
{
    return fp_uc && fp_size
        ? rsx_fp_structural_hash(fp_uc, fp_size, 1469598103934665603ull) : 0;
}

/* The vertex program's share of the pipeline key, eng_fnv1a over its bytes
 * from the FNV offset basis. A byte-at-a-time FNV over up to 8.5 KB was most
 * of eng_pipeline_get on every draw, for the handful of programs a frame
 * uses. Remembered by CONTENT: an entry answers only when its stored copy of
 * the bytes is equal to the program's bytes now (memcmp), so a re-upload, a
 * patch of one word or a different start can never reuse a stale hash, and
 * the result is always exactly what eng_fnv1a would return.
 * RSX_VP_HASH_CACHE=0 hashes every draw again (A/B). */
#define ENG_VP_HASH_MEMO 16u
static struct {
    u32 bytes;                              /* 0 = empty */
    u64 hash;
    u8 copy[RSX_DSP_VP_INSTR * 16u];
} s_vp_memo[ENG_VP_HASH_MEMO];
static u32 s_vp_memo_next;
static int s_vp_hash_cache = -1;

static u64 eng_vp_hash(const u8* vp_uc, u32 bytes)
{
    const u64 basis = 1469598103934665603ull;
    if (s_vp_hash_cache < 0) {
        const char* e = getenv("RSX_VP_HASH_CACHE");
        s_vp_hash_cache = (e && e[0] == '0') ? 0 : 1;
    }
    if (!s_vp_hash_cache || !vp_uc || !bytes || bytes > sizeof s_vp_memo[0].copy)
        return eng_fnv1a(vp_uc, bytes, basis);
    for (u32 i = 0; i < ENG_VP_HASH_MEMO; i++)
        if (s_vp_memo[i].bytes == bytes && !memcmp(s_vp_memo[i].copy, vp_uc, bytes))
            return s_vp_memo[i].hash;
    const u64 hash = eng_fnv1a(vp_uc, bytes, basis);
    const u32 slot = s_vp_memo_next++ % ENG_VP_HASH_MEMO;
    memcpy(s_vp_memo[slot].copy, vp_uc, bytes);
    s_vp_memo[slot].bytes = bytes;
    s_vp_memo[slot].hash = hash;
    return hash;
}

void rsx_draw_engine_set_walker_fast(int fast)
{
    s_guest_read_fast = fast;
    s_vp_hash_cache = fast;
    s_texhash_lanes = fast;
    memset(s_vp_memo, 0, sizeof s_vp_memo);
    s_vp_memo_next = 0;
}

/* RSX_PIPE_FAST=0 turns off both shortcuts below (A/B). Each gives exactly
 * the result of the code it skips. */
static int s_pipe_fast = -1;
static int eng_pipe_fast(void)
{
    if (s_pipe_fast < 0) { const char* e = getenv("RSX_PIPE_FAST"); s_pipe_fast = (e && e[0] == '0') ? 0 : 1; }
    return s_pipe_fast;
}

/* What eng_pipeline_get reads off the fragment program on every draw -- the
 * sampled units, the units gamma and sign expansion could apply to, and the
 * structural hash -- remembered per program, by CONTENT like eng_vp_hash:
 * an entry answers only when its stored copy equals the program's bytes now.
 * ~2000 draws a frame walked a handful of programs four times each. */
#define ENG_FP_MEMO       16u
#define ENG_FP_MEMO_BYTES 4096u
static struct {
    u32 bytes;                              /* 0 = empty */
    u32 tex, gamma, expand;
    u64 seed, hash;                         /* rsx_fp_structural_hash(seed) */
    u64 hash0;                              /* stats: the seed-0 hash at fill */
    u32 serial;                             /* bumped at every fill (RSX_PIPE_CACHE) */
    u8 copy[ENG_FP_MEMO_BYTES];
} s_fp_memo[ENG_FP_MEMO];
static u32 s_fp_memo_next, s_fp_memo_last, s_fp_memo_serial;

static int eng_fp_memo_find(const u8* fp_uc, u32 fp_size)
{
    if (!eng_pipe_fast() || !fp_uc || !fp_size || fp_size > ENG_FP_MEMO_BYTES) return -1;
    const u32 last = s_fp_memo_last;
    if (s_fp_memo[last].bytes == fp_size && !memcmp(s_fp_memo[last].copy, fp_uc, fp_size)) {
        s_pc.memo_last++;
        return (int)last;
    }
    for (u32 i = 0; i < ENG_FP_MEMO; i++)
        if (i != last && s_fp_memo[i].bytes == fp_size && !memcmp(s_fp_memo[i].copy, fp_uc, fp_size)) {
            s_fp_memo_last = i;
            s_pc.memo_other++;
            return (int)i;
        }
    s_pc.memo_miss++;
    ENG_ST_BEGIN();
    const u32 slot = s_fp_memo_next++ % ENG_FP_MEMO;
    memcpy(s_fp_memo[slot].copy, fp_uc, fp_size);
    s_fp_memo[slot].bytes = fp_size;
    rsx_fp_texture_unit_sets(fp_uc, fp_size, &s_fp_memo[slot].tex,
                             &s_fp_memo[slot].gamma, &s_fp_memo[slot].expand);
    s_fp_memo[slot].seed = 0;
    s_fp_memo[slot].hash = rsx_fp_structural_hash(fp_uc, fp_size, 0);
    s_fp_memo[slot].hash0 = s_fp_memo[slot].hash;
    s_fp_memo[slot].serial = ++s_fp_memo_serial;
    for (u32 i = 0; i < ENG_FP_MEMO; i++)
        if (i != slot && s_fp_memo[i].bytes == fp_size && s_fp_memo[i].hash0 == s_fp_memo[slot].hash0) {
            s_pc.fill_samestruct++;
            break;
        }
    s_fp_memo_last = slot;
    ENG_ST_END(ST_PFILL);
    return (int)slot;
}

/* RSX_PIPE_CACHE=1: the previous draw's answer, when every input the lookup
 * below reads equals that draw's. ~84% of draws repeat the previous key; such
 * a draw still walks both programs and matches the fp memo (so the fragment
 * program's bytes, constants included, are compared in full), but skips the
 * unit decodes, the constant collection and the key. The inputs: the program
 * registers, the fragment and vertex texture unit words the masks and shadow
 * descriptors are built from, the vertex program's bytes, the fp memo slot
 * (its fill serial, so a refilled slot never matches), the layout, the
 * targets and the decoded render state (less ALPHA_REF, which the key leaves
 * out too). g.fp_constants still holds that draw's constants: only this
 * function writes it, and any run of the full path drops the cache until it
 * reaches a lookup or a create. =2 runs the full path on every hit and
 * compares. */
typedef struct {
    u32 vp_start, vp_instrs, fp_size, memo, memo_serial;
    u32 r08e4, ctrl, r1fec, r0208;
    u32 ftex[RSX_DSP_NUM_TEXTURES * 7u];
    u32 vtex[RSX_DSP_NUM_VERTEX_TEXTURES * 8u];
    u32 lmask, lstride, rt_fmt, rt_count;
    rsx_be_render_state rs;
} eng_pl_sig;
typedef struct { u32 handle, tex_used; int fixed; u64 fp_hash; u32 count; unsigned long long nf; } eng_pl_was;
static struct {
    int valid;
    eng_pl_sig sig;
    u8 vp[RSX_DSP_VP_INSTR * 16u];
    u32 handle, tex_used;
    int fixed;
    u64 fp_hash;
    unsigned long long nonfinite;           /* what the collection added to the counter */
} s_pl;
static int s_pl_mode = -1;
static int s_pl_last_hit;                   /* the last eng_pipeline_get matched the previous draw's inputs */
static int eng_pipe_cache(void)
{
    if (s_pl_mode < 0) { const char* e = getenv("RSX_PIPE_CACHE"); s_pl_mode = e ? atoi(e) : 0; }
    return s_pl_mode;
}

static void eng_pl_signature(eng_pl_sig* s, const rsx_vertex_layout_plan* layout,
                             const rsx_be_render_state* rs, rsx_be_format rt_fmt, u32 rt_count,
                             u32 vp_instrs, u32 fp_size, int memo)
{
    memset(s, 0, sizeof *s);
    s->vp_start = rsx_dsp_vp_start(&g.rsx);
    s->vp_instrs = vp_instrs;
    s->fp_size = fp_size;
    s->memo = (u32)memo;
    s->memo_serial = s_fp_memo[memo].serial;
    s->r08e4 = rsx_dsp_reg(&g.rsx, 0x08E4u);
    s->ctrl = rsx_dsp_shader_control(&g.rsx);
    s->r1fec = rsx_dsp_reg(&g.rsx, 0x1FECu);
    s->r0208 = rsx_dsp_reg(&g.rsx, 0x0208u);
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        for (u32 k = 0; k < 6u; k++)                /* +0x04 FORMAT .. +0x18 IMAGE_RECT */
            s->ftex[u * 7u + k] = rsx_dsp_reg(&g.rsx, 0x1A04u + u * 0x20u + k * 4u);
        s->ftex[u * 7u + 6u] = rsx_dsp_reg(&g.rsx, 0x1840u + u * 4u);
    }
    for (u32 u = 0; u < RSX_DSP_NUM_VERTEX_TEXTURES; u++)
        for (u32 k = 0; k < 8u; k++)
            s->vtex[u * 8u + k] = rsx_dsp_reg(&g.rsx, 0x0900u + u * 0x20u + k * 4u);
    s->lmask = layout->mask;
    s->lstride = layout->stride;
    s->rt_fmt = (u32)rt_fmt;
    s->rt_count = rt_count;
    memcpy(&s->rs, rs, sizeof *rs);           /* padding too: decode zeroed it */
    s->rs.alpha_ref_raw = 0;
}

/* The full path reached a lookup or a create: remember its answer. With a
 * check armed (=2), first compare it with what the cache would have given. */
static void eng_pl_keep(const eng_pl_sig* sig, const u8* vp_uc, u32 vp_instrs, u32 handle,
                        u32 tex_used, int fixed, u64 fp_hash, unsigned long long nf,
                        const eng_pl_was* w, const rsx_fp_constant_block* was_consts)
{
    if (w) {
        const u32 was_count = w->count;
        const unsigned long long was_nf = w->nf;
        const int bad = w->handle != handle || w->tex_used != tex_used || w->fixed != fixed ||
                        w->fp_hash != fp_hash || was_count != g.fp_constants.count || was_nf != nf ||
                        memcmp(was_consts->values, g.fp_constants.values, (size_t)was_count * 16u);
        s_pc.l0_checked++;
        if (bad && ++s_pc.l0_differ <= 32)
            fprintf(stderr, "[pipe-cache-check] hit differs: handle %u/%u tex %x/%x fp %016llx/%016llx "
                            "consts %u/%u nonfinite %llu/%llu 0x08E4 %08X\n",
                    w->handle, handle, w->tex_used, tex_used, (unsigned long long)w->fp_hash,
                    (unsigned long long)fp_hash, was_count, g.fp_constants.count, was_nf, nf, sig->r08e4);
    }
    memcpy(&s_pl.sig, sig, sizeof *sig);
    memcpy(s_pl.vp, vp_uc, (size_t)vp_instrs * 16u);
    s_pl.handle = handle;
    s_pl.tex_used = tex_used;
    s_pl.fixed = fixed;
    s_pl.fp_hash = fp_hash;
    s_pl.nonfinite = nf;
    s_pl.valid = 1;
}

/* The depth bounds test in the pixel program, for a backend whose device has
 * none (rsx_draw_backend.depth_bounds_native; Metal, so MoltenVK). GoW:A bounds
 * its deferred lights to their depth range and its sky passes to the far plane
 * with it; ignored, every light lit its whole scissor rectangle and the sky
 * passes ran over the geometry. Per pipeline handle: the unit its program reads
 * the zeta snapshot from, plus one (0: none). RSX_DBT_EMULATE=0 turns it off. */
static u8 s_pipe_dbt[ENG_MAX_PIPELINES + 1];
/* Pipelines whose PSConstants end in fp_shs[16] (rsx_fp_apply_shadow_scale). */
static u8 s_pipe_shs[ENG_MAX_PIPELINES + 1];
/* RSX_SHADOW_REGION=0: shadow maps sampled from a depth snapshot read the whole zeta, as
 * before (a cascade drawn into part of a shared target is then read at the wrong scale). */
static int eng_shadow_region_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("RSX_SHADOW_REGION"); on = !(e && e[0] == '0'); }
    return on;
}
static int eng_dbt_emulated(void)
{
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("RSX_DBT_EMULATE");
        v = !(e && *e == '0') && g.be->depth_bounds_native && !g.be->depth_bounds_native(g.be->user);
        if (v) fprintf(stderr, "[rsx engine] depth bounds test run in the pixel program (the device has none)\n");
    }
    return v;
}

/* ---- pipeline catalog (RSX_PIPE_CATALOG=<file>) ------------------------------
 * A pipeline was built the first time a draw needed it: decompile, HLSL ->
 * SPIR-V, and -- the expensive part on the first draw with each topology --
 * the host compile (MoltenVK's MSL translation plus the Metal compiler), all
 * on the thread that walks the FIFO. Every new material or effect stalled the
 * frame it first appeared in. The catalog records each pipeline the engine
 * builds (its two programs' HLSL, render state, layout, target format and
 * count) and the topologies it was drawn with; the next boot builds all of
 * them, variants included, at the first draw (the boot logos), so play finds
 * them ready.
 * A pipeline built from the catalog is provisional: the first draw that looks
 * its key up still runs the decompiler (cheap next to the host compile) and
 * compares the HLSL with the recorded text; a decompiler that has changed
 * since gets a fresh build, so a stale catalog costs a stall, never a wrong
 * shader. The file is rewritten compacted at each boot and appended to while
 * playing; a torn last record (a crash) is dropped. */
#define ENG_CAT_MAGIC   0x31544350u   /* "PCT1" */
#define ENG_CAT_VERSION 2u   /* 2: keys from canonical render state */
#define ENG_CAT_REC_P   0x50495045u
#define ENG_CAT_REC_T   0x4F504F54u
typedef struct { u32 magic, version, rs_size, layout_size; } eng_cat_hdr;
typedef struct {
    u32 tag, fixed;
    u64 key, text_hash, fp_hash;
    rsx_be_render_state rs;
    rsx_vertex_layout_plan layout;
    u32 stride, rt_fmt, rt_count, topo, vs_len, ps_len;
} eng_cat_p;
typedef struct { u32 tag, topo; u64 key; } eng_cat_t;
static FILE* s_cat_f;
static int s_cat_state = -1;           /* -1 not yet, 0 off, 1 on */
static u32 s_pipe_entry[ENG_MAX_PIPELINES + 1];   /* backend handle -> pipelines[] index + 1 */
static u32 s_cat_midgame_logs;
static struct { u32 pn, vn; double pms, vms, pmax, vmax; } s_cat_mid;   /* builds after boot */

static u64 eng_cat_text_hash(const char* vs, const char* ps)
{
    u64 h = eng_fnv1a(vs, (u32)strlen(vs), 1469598103934665603ull);
    return eng_fnv1a(ps, (u32)strlen(ps), h ^ 0x9E3779B97F4A7C15ull);
}

static void eng_cat_write_p(FILE* f, const eng_pipeline* p, const rsx_be_render_state* rs,
                            const rsx_vertex_layout_plan* layout, u32 stride, u32 rt_fmt,
                            u32 rt_count, const char* vs, const char* ps)
{
    if (!f) return;
    eng_cat_p r;
    memset(&r, 0, sizeof r);
    r.tag = ENG_CAT_REC_P; r.fixed = p->fixed;
    r.key = p->key; r.text_hash = p->text_hash; r.fp_hash = p->fp_hash;
    r.rs = *rs; r.layout = *layout;
    r.stride = stride; r.rt_fmt = rt_fmt; r.rt_count = rt_count; r.topo = p->topo;
    r.vs_len = (u32)strlen(vs); r.ps_len = (u32)strlen(ps);
    fwrite(&r, sizeof r, 1, f);
    fwrite(vs, 1, r.vs_len, f);
    fwrite(ps, 1, r.ps_len, f);
    fflush(f);
}

static double eng_cat_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void eng_cat_insert(u32 idx)
{
    const u64 key = g.pipelines[idx].key;
    u32 ix = (u32)(key ^ key >> 29 ^ key >> 47) & (ENG_PIPELINE_INDEX - 1u);
    while (g.pipeline_index[ix]) ix = (ix + 1u) & (ENG_PIPELINE_INDEX - 1u);
    g.pipeline_index[ix] = (u16)(idx + 1u);
}

/* Once, at the first pipeline lookup: build what the file lists, rewrite it
 * compacted, keep it open for appending. */
static void eng_cat_init(void)
{
    if (s_cat_state >= 0) return;
    s_cat_state = 0;
    const char* path = getenv("RSX_PIPE_CATALOG");
    if (!path || !*path || !eng_pipe_fast()) return;
    s_cat_state = 1;
    const double t0 = eng_cat_ms();
    u8* buf = NULL; long size = 0;
    FILE* f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
        if (size > 0 && (buf = (u8*)malloc((size_t)size)) != NULL &&
            fread(buf, 1, (size_t)size, f) != (size_t)size) { free(buf); buf = NULL; }
        fclose(f);
    }
    /* Parse: the last P record per key wins, T records OR into its topologies. */
    enum { CAT_MAX = ENG_MAX_PIPELINES };
    static const eng_cat_p* recs[CAT_MAX];
    static u32 topo[CAT_MAX], last[CAT_MAX], order[CAT_MAX];
    static u32 tab[ENG_PIPELINE_INDEX];   /* key hash -> recs index + 1 */
    u32 ordinal = 0;
    memset(tab, 0, sizeof tab);
    u32 n = 0, torn = 0;
    const eng_cat_hdr want = { ENG_CAT_MAGIC, ENG_CAT_VERSION,
                               (u32)sizeof(rsx_be_render_state), (u32)sizeof(rsx_vertex_layout_plan) };
    /* Version 1 had the same records with keys over the raw render state:
     * rekey them (eng_rekey_canonical); T records follow their P, so a small
     * old -> new map carries the topologies over. */
    eng_cat_hdr v1 = want; v1.version = 1u;
    const int from_v1 = buf && size >= (long)sizeof v1 && !memcmp(buf, &v1, sizeof v1);
    static u64 remap_old[2 * ENG_PIPELINE_INDEX], remap_new[2 * ENG_PIPELINE_INDEX];
    if (from_v1) {
        memset(remap_old, 0, sizeof remap_old);
        fprintf(stderr, "[pipe-catalog] %s: version 1, rekeying over the canonical render state\n", path);
    }
    if (buf && size >= (long)sizeof want && (from_v1 || !memcmp(buf, &want, sizeof want))) {
        long off = (long)sizeof want;
        while (off + 4 <= size) {
            u32 tag; memcpy(&tag, buf + off, 4);
            u64 key = 0; u32 tp = 0; const eng_cat_p* p = NULL;
            if (tag == ENG_CAT_REC_P && off + (long)sizeof(eng_cat_p) <= size) {
                p = (const eng_cat_p*)(buf + off);
                const long end = off + (long)sizeof(eng_cat_p) + (long)p->vs_len + (long)p->ps_len;
                if (end > size || p->vs_len > (1u << 20) || p->ps_len > (1u << 20)) { torn = 1; break; }
                if (from_v1) {
                    const u64 nk = eng_rekey_canonical(p->key, &p->rs);
                    u32 m = (u32)(p->key ^ p->key >> 31) & (2u * ENG_PIPELINE_INDEX - 1u);
                    while (remap_old[m] && remap_old[m] != p->key) m = (m + 1u) & (2u * ENG_PIPELINE_INDEX - 1u);
                    remap_old[m] = p->key; remap_new[m] = nk;
                    ((eng_cat_p*)p)->key = nk;     /* buf is ours; the rewrite below keeps it */
                }
                key = p->key; tp = p->topo; off = end;
            } else if (tag == ENG_CAT_REC_T && off + (long)sizeof(eng_cat_t) <= size) {
                eng_cat_t t; memcpy(&t, buf + off, sizeof t);
                key = t.key; tp = t.topo; off += (long)sizeof t;
                if (from_v1) {
                    u32 m = (u32)(key ^ key >> 31) & (2u * ENG_PIPELINE_INDEX - 1u);
                    while (remap_old[m] && remap_old[m] != key) m = (m + 1u) & (2u * ENG_PIPELINE_INDEX - 1u);
                    if (!remap_old[m]) { ordinal++; continue; }   /* its P was lost */
                    key = remap_new[m];
                }
            } else { torn = 1; break; }
            ordinal++;
            u32 ix = (u32)(key ^ key >> 29 ^ key >> 47) & (ENG_PIPELINE_INDEX - 1u);
            while (tab[ix] && recs[tab[ix] - 1u]->key != key) ix = (ix + 1u) & (ENG_PIPELINE_INDEX - 1u);
            if (!tab[ix]) {
                if (!p || n == CAT_MAX) continue;   /* a T before its P, or full */
                recs[n] = p; topo[n] = tp; last[n] = ordinal; tab[ix] = ++n;
            } else {
                if (p) recs[tab[ix] - 1u] = p;
                topo[tab[ix] - 1u] |= tp;
                last[tab[ix] - 1u] = ordinal;
            }
        }
    } else if (buf) {
        fprintf(stderr, "[pipe-catalog] %s: another format, starting over\n", path);
    }
    /* Least recently seen first (a record is touched each session its
     * pipeline is used); build only the newest `cap`, so the tables keep room
     * for what play brings (RSX_PIPE_CATALOG_MAX, default 6144 of 8192). */
    for (u32 i = 0; i < n; i++) order[i] = i;
    for (u32 i = 1; i < n; i++) {           /* insertion sort: the file is nearly in order */
        const u32 v = order[i]; u32 j = i;
        while (j > 0 && last[order[j - 1]] > last[v]) { order[j] = order[j - 1]; j--; }
        order[j] = v;
    }
    u32 cap = 6144u;
    { const char* e = getenv("RSX_PIPE_CATALOG_MAX"); if (e && *e) cap = (u32)atoi(e); }
    if (cap > ENG_MAX_PIPELINES - 1024u) cap = ENG_MAX_PIPELINES - 1024u;
    const u32 first = n > cap ? n - cap : 0;
    u32 built = 0, variants = 0;
    for (u32 k = first; k < n && g.n_pipelines < ENG_MAX_PIPELINES; k++) {
        const u32 i = order[k];
        const eng_cat_p* r = recs[i];
        char* vs = (char*)malloc(r->vs_len + 1u);
        char* ps = (char*)malloc(r->ps_len + 1u);
        if (!vs || !ps) { free(vs); free(ps); break; }
        memcpy(vs, (const u8*)(r + 1), r->vs_len); vs[r->vs_len] = 0;
        memcpy(ps, (const u8*)(r + 1) + r->vs_len, r->ps_len); ps[r->ps_len] = 0;
        const u32 h = g.be->pipeline_create(g.be->user, vs, ps, &r->rs, &r->layout, r->stride,
                                            (rsx_be_format)r->rt_fmt, r->rt_count);
        if (h) {
            if (g.be->pipeline_prewarm && topo[i]) g.be->pipeline_prewarm(g.be->user, h, topo[i]);
            if (g.be->pipeline_variants) variants += (u32)__builtin_popcount(g.be->pipeline_variants(g.be->user, h));
            eng_pipeline* e = &g.pipelines[g.n_pipelines];
            memset(e, 0, sizeof *e);
            e->key = r->key; e->handle = h; e->fixed = (u8)r->fixed;
            e->tex_used = (u16)(r->fixed ? 0xFFFFu : eng_sampled_units(ps));
            e->fp_hash = r->fp_hash; e->text_hash = r->text_hash;
            e->prov = 1; e->topo = (u8)topo[i];
            if (h <= ENG_MAX_PIPELINES) s_pipe_entry[h] = g.n_pipelines + 1u;
            eng_cat_insert(g.n_pipelines);
            g.n_pipelines++;
            built++;
        }
        free(vs); free(ps);
    }
    /* Rewrite compacted: one P per pipeline, its topologies folded in. */
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* w = fopen(tmp, "wb");
    if (w) {
        fwrite(&want, sizeof want, 1, w);
        /* the oldest drop off once the file holds twice what a boot builds */
        for (u32 k = n > 2u * cap ? n - 2u * cap : 0; k < n; k++) {
            const u32 i = order[k];
            eng_cat_p r = *recs[i];
            r.topo = topo[i];
            fwrite(&r, sizeof r, 1, w);
            fwrite((const u8*)(recs[i] + 1), 1, (size_t)r.vs_len + r.ps_len, w);
        }
        if (fclose(w) == 0) rename(tmp, path); else remove(tmp);
    }
    free(buf);
    s_cat_f = fopen(path, "ab");
    if (s_cat_f && ftell(s_cat_f) == 0) fwrite(&want, sizeof want, 1, s_cat_f);
    fprintf(stderr, "[pipe-catalog] %u of %u pipelines built at boot (%u topology variants) in %.0f ms%s\n",
            built, n, variants, eng_cat_ms() - t0, torn ? "; torn tail dropped" : "");
    (void)last;
}

/* After a draw: topologies the backend built for this pipeline since the
 * catalog last heard. */
static void eng_cat_note_draw(u32 handle, double ms)
{
    if (!s_cat_f || !handle || handle > ENG_MAX_PIPELINES || !s_pipe_entry[handle] || !g.be->pipeline_variants)
        return;
    eng_pipeline* p = &g.pipelines[s_pipe_entry[handle] - 1u];
    const u32 m = g.be->pipeline_variants(g.be->user, handle);
    if (!(m & ~(u32)p->topo)) return;
    p->topo |= (u8)m;
    const eng_cat_t t = { ENG_CAT_REC_T, p->topo, p->key };
    fwrite(&t, sizeof t, 1, s_cat_f);
    fflush(s_cat_f);
    s_cat_mid.vn++; s_cat_mid.vms += ms; if (ms > s_cat_mid.vmax) s_cat_mid.vmax = ms;
    if (PS3_LOG_FIRST(s_cat_midgame_logs, 16))
        fprintf(stderr, "[pipe-catalog] new variant mid-game: pipeline %016llx topologies 0x%X, draw took %.1f ms\n",
                (unsigned long long)p->key, m, ms);
    if (s_cat_mid.vn % 50u == 0)
        fprintf(stderr, "[pipe-catalog] mid-game so far: %u pipelines created (%.0f ms, max %.1f), "
                        "%u variants built (%.0f ms, max %.1f)\n",
                s_cat_mid.pn, s_cat_mid.pms, s_cat_mid.pmax, s_cat_mid.vn, s_cat_mid.vms, s_cat_mid.vmax);
}

static u32 eng_pipeline_get(const rsx_vertex_layout_plan* layout,
                            const rsx_be_render_state* rs,
                            rsx_be_format rt_fmt, u32 rt_count, int* out_fixed,
                            u32* out_tex_used, u64* out_fp_hash)
{
    *out_fixed = 1;
    *out_tex_used = 0;
    *out_fp_hash = 0;
    if (s_cat_state < 0) eng_cat_init();
    const u8* vp_uc = NULL;
    const u8* fp_uc = NULL;
    u32 vp_instrs = 0, fp_size = 0;
    /* No resident program pair means the built-in one; see kEngFixedVS. */
    const int fixed = !eng_guest_programs(&vp_uc, &vp_instrs, &fp_uc, &fp_size);

    const int pl_mode = eng_pipe_cache();
    int memo_early = -1, pl_armed = 0;
    s_pl_last_hit = 0;
    eng_pl_sig pl_sig;
    eng_pl_was pl_was;
    static rsx_fp_constant_block pl_consts;
    if (pl_mode && !fixed) {
        memo_early = eng_fp_memo_find(fp_uc, fp_size);
        if (memo_early >= 0) {
            eng_pl_signature(&pl_sig, layout, rs, rt_fmt, rt_count, vp_instrs, fp_size, memo_early);
            if (s_pl.valid && !memcmp(&pl_sig, &s_pl.sig, sizeof pl_sig) &&
                !memcmp(s_pl.vp, vp_uc, (size_t)vp_instrs * 16u)) {
                s_pc.l0_hit++;
                s_pl_last_hit = 1;
                if (pl_mode != 2) {
                    g_rsx_fp_nonfinite_constants += s_pl.nonfinite;
                    *out_fixed = s_pl.fixed;
                    *out_tex_used = s_pl.tex_used;
                    *out_fp_hash = s_pl.fp_hash;
                    return s_pl.handle;
                }
                pl_armed = 1;
                pl_was.handle = s_pl.handle; pl_was.tex_used = s_pl.tex_used;
                pl_was.fixed = s_pl.fixed; pl_was.fp_hash = s_pl.fp_hash;
                pl_was.count = g.fp_constants.count; pl_was.nf = s_pl.nonfinite;
                memcpy(pl_consts.values, g.fp_constants.values, (size_t)g.fp_constants.count * 16u);
            }
        }
    }
    s_pl.valid = 0;

    const u32 fp_ctrl  = rsx_dsp_shader_control(&g.rsx);
    const u32 srgb_output = !fixed && rsx_fp_srgb_output_enabled(
        rsx_dsp_reg(&g.rsx, 0x1FECu), fp_ctrl,
        rsx_dsp_reg(&g.rsx, 0x0208u) & 0x1Fu);
    const u32 cube_mask = eng_cube_mask();
    const u32 vol_mask = eng_volume_mask();
    const u32 vtex_mask = eng_vtex_mask();
    rsx_fp_shadow_desc shadow[RSX_DSP_NUM_TEXTURES] = {{0}};
    u32 shadow_mask = 0;
    const int memo = fixed ? -1 : memo_early >= 0 ? memo_early : eng_fp_memo_find(fp_uc, fp_size);
    const u32 sampled_units = fixed ? 0 : memo >= 0 ? s_fp_memo[memo].tex
                                                    : rsx_fp_texture_units(fp_uc, fp_size);
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++) {
        if (!(sampled_units & (1u << u))) continue;
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        rsx_fp_shadow_desc_from_texture(&shadow[u], t.format, t.dimension,
            t.cubemap, t.mipmaps, t.filter, t.wrap, t.control0);
        if (shadow[u].enabled) shadow_mask |= 1u << u;
    }
    u8 gamma_masks[RSX_DSP_NUM_TEXTURES] = {0};
    if (!fixed) for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; ++u) {
        rsx_dsp_texture t; rsx_dsp_get_texture(&g.rsx, u, &t);
        if (t.enabled) gamma_masks[u] = rsx_texture_gamma_mask(t.format, t.remap, t.filter, t.wrap);
    }
    u32 gamma_units = 0;
    if (memo >= 0) {
        for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; ++u)
            if (gamma_masks[u]) gamma_units |= 1u << u;
        gamma_units &= s_fp_memo[memo].gamma;
    } else if (!fixed) {
        gamma_units = rsx_fp_texture_gamma_units(fp_uc, fp_size, gamma_masks);
    }
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; ++u)
        if (!(gamma_units & (1u << u))) gamma_masks[u] = 0;
    /* Texture sign expansion: SET_TEXTURE_ADDRESS's UNSIGNED_REMAP_BIASED
     * asks that every fetch come back re-centred to -1..1. Same rule as
     * gamma: only units the program samples count. */
    u8 expand_masks[RSX_DSP_NUM_TEXTURES] = {0};
    if (!fixed) for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; ++u) {
        rsx_dsp_texture t; rsx_dsp_get_texture(&g.rsx, u, &t);
        if (t.enabled) expand_masks[u] = rsx_texture_expand_mask(t.format, t.remap, t.filter, t.wrap);
    }
    u32 expand_units = 0;
    if (memo >= 0) {
        for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; ++u)
            if (expand_masks[u]) expand_units |= 1u << u;
        expand_units &= s_fp_memo[memo].expand;
    } else if (!fixed) {
        expand_units = rsx_fp_texture_expand_units(fp_uc, fp_size, expand_masks);
    }
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; ++u)
        if (!(expand_units & (1u << u))) expand_masks[u] = 0;

    u32 unnorm_dim[RSX_DSP_NUM_TEXTURES][2];
    /* Shadow helpers normalize against the bound resource themselves. Only
     * sampled ordinary UN units need baked dimensions or cache identity. */
    const u32 unnorm_mask = eng_unnorm_units(unnorm_dim) &
        (fixed ? 0xFFFFu : sampled_units) & ~shadow_mask;
    for (u32 u = 0; u < RSX_DSP_NUM_TEXTURES; u++)
        if (!(unnorm_mask & (1u << u)))
            unnorm_dim[u][0] = unnorm_dim[u][1] = 0;

    { static int s_noclear = -1;    /* RSX_FP_NOCLEAR: see rsx_fp_collect_constants */
      if (s_noclear < 0) { const char* e = getenv("RSX_FP_NOCLEAR"); s_noclear = e && e[0] == '1'; }
      if (s_noclear) g.fp_constants.count = 0;
      else memset(&g.fp_constants, 0, sizeof g.fp_constants); }
    const unsigned long long nf0 = g_rsx_fp_nonfinite_constants;
    if (!fixed && rsx_fp_collect_constants(fp_uc, fp_size, &g.fp_constants) < 0)
        return 0;
    const unsigned long long nf = g_rsx_fp_nonfinite_constants - nf0;

    /* Identity: the vertex program's own bytes, the fragment program's
     * STRUCTURE (its inline constants are hoisted into the buffered block, so
     * a constant change must not be a new pipeline), the export-width bit of
     * SHADER_CONTROL, the cube and vertex-texture masks, the input layout,
     * and the structural render state. */
    u64 key = 1469598103934665603ull;
    if (fixed) {
        static const u32 fixed_tag = 0x4E464958u;   /* "XIFN" */
        key = eng_fnv1a(&fixed_tag, sizeof fixed_tag, key);
    } else {
        key = eng_vp_hash(vp_uc, vp_instrs * 16u);   /* = eng_fnv1a from the basis */
        if (memo >= 0) {
            if (s_fp_memo[memo].seed != key) {
                ENG_ST_BEGIN();
                s_fp_memo[memo].hash = rsx_fp_structural_hash(fp_uc, fp_size, key);
                s_fp_memo[memo].seed = key;
                ENG_ST_END(ST_PRESEED);
            }
            key = s_fp_memo[memo].hash;
        } else {
            key = rsx_fp_structural_hash(fp_uc, fp_size, key);
        }
        if (!key) return 0;
        /* Export width and depth export: both change the generated shader. */
        const u32 fp_ctrl_key = fp_ctrl & 0x4Eu; /* colour width and fragment depth export */
        key = eng_fnv1a(&fp_ctrl_key, sizeof fp_ctrl_key, key);
        if (srgb_output) key = eng_fnv1a(&srgb_output, sizeof srgb_output, key);
        if (gamma_units) key = eng_fnv1a(gamma_masks, sizeof gamma_masks, key);
        if (expand_units) key = eng_fnv1a(expand_masks, sizeof expand_masks, key);
        key = eng_fnv1a(&cube_mask, sizeof cube_mask, key);
        if (vol_mask) key = eng_fnv1a(&vol_mask, sizeof vol_mask, key);
        if (shadow_mask) {
            key = eng_fnv1a(&shadow_mask, sizeof shadow_mask, key);
            key = eng_fnv1a(shadow, sizeof shadow, key);
        }
        key = eng_fnv1a(&vtex_mask, sizeof vtex_mask, key);
        key = eng_fnv1a(&unnorm_mask, sizeof unnorm_mask, key);
        if (unnorm_mask) key = eng_fnv1a(unnorm_dim, sizeof unnorm_dim, key);
    }
    key = eng_fnv1a(&layout->mask, sizeof layout->mask, key);
    key = eng_fnv1a(&layout->stride, sizeof layout->stride, key);
    key = eng_fnv1a(&rt_fmt, sizeof rt_fmt, key);
    /* How many colour attachments the pass will have is pipeline identity:
     * a host API matches the two against each other. */
    key = eng_fnv1a(&rt_count, sizeof rt_count, key);
    key = rsx_draw_engine_hash_render_state(rs, key);
    if (s_stats_s > 0) { s_pc.lookups++; if (key == s_pc.prev_key) s_pc.same_key++; s_pc.prev_key = key; }

    /* Keys are unique in pipelines[] (one is added only when not found), so
     * the index finds the same entry the linear scan would. The scan read the
     * whole list on every draw, ~1000+ entries a scene. */
    u32 ix = (u32)(key ^ key >> 29 ^ key >> 47) & (ENG_PIPELINE_INDEX - 1u);
    int reuse = -1;   /* a catalog pipeline with this key, still to be checked */
    if (eng_pipe_fast()) {
        for (u32 s; (s = g.pipeline_index[ix]) != 0; ix = (ix + 1u) & (ENG_PIPELINE_INDEX - 1u))
            if (g.pipelines[s - 1u].key == key) {
                const eng_pipeline* p = &g.pipelines[s - 1u];
                if (p->prov) { reuse = (int)(s - 1u); break; }
                *out_fixed = p->fixed;
                *out_tex_used = p->tex_used;
                *out_fp_hash = p->fp_hash;
                if (memo_early >= 0)
                    eng_pl_keep(&pl_sig, vp_uc, vp_instrs, p->handle, p->tex_used, p->fixed, p->fp_hash, nf,
                                pl_armed ? &pl_was : NULL, &pl_consts);
                return p->handle;
            }
    } else {
        for (u32 i = 0; i < g.n_pipelines; i++)
            if (g.pipelines[i].key == key) {
                *out_fixed = g.pipelines[i].fixed;
                *out_tex_used = g.pipelines[i].tex_used;
                *out_fp_hash = g.pipelines[i].fp_hash;
                return g.pipelines[i].handle;
            }
    }
    if (reuse < 0 && g.n_pipelines >= ENG_MAX_PIPELINES) {
        static u32 logs;
        if (PS3_LOG_FIRST(logs, 4))
            fprintf(stderr, "[rsx engine] pipeline table full (%u): new pipelines fail, their draws drop\n",
                    (unsigned)ENG_MAX_PIPELINES);
        return 0;
    }
    if (reuse < 0) s_pc.creates++;

    u32 handle = 0, dbt_unit = 0, shs = 0;
    int vi = 1, fi = 1;
    u32 nconst = 0;
    if (fixed) {
        snprintf(s_vs_hlsl, sizeof s_vs_hlsl, "%s", kEngFixedVS);
        snprintf(s_ps_hlsl, sizeof s_ps_hlsl, "%s", kEngFixedPS);
    } else {
        vi = rsx_vp_decompile_compact_ex(vp_uc, vp_instrs * 16u, vtex_mask,
                                         layout->mask, s_vs_hlsl,
                                         sizeof s_vs_hlsl);
        fi = rsx_fp_decompile_buffered_shadow_color_expand_vol_ex(fp_uc, fp_size, fp_ctrl, cube_mask, vol_mask,
                shadow_mask ? shadow : NULL, gamma_masks, expand_units ? expand_masks : NULL,
                s_ps_hlsl, sizeof s_ps_hlsl, &nconst);
        if (fi > 0 && nconst != g.fp_constants.count) fi = -1;
        if (fi > 0 && rsx_fp_apply_unnorm_scale(s_ps_hlsl, sizeof s_ps_hlsl, unnorm_mask,
                                                (const u32 (*)[2])unnorm_dim, cube_mask) < 0)
            fi = -1;
        if (fi > 0 && srgb_output &&
            rsx_fp_apply_srgb_output(s_ps_hlsl, sizeof s_ps_hlsl) < 0)
            fi = -1;
        if (fi > 0 && rs->alpha_test_enable &&
            rsx_fp_apply_alpha_test_buffered(s_ps_hlsl, sizeof s_ps_hlsl,
                                             rs->alpha_func) < 0)
            fi = -1;
        /* The highest 2D unit the program leaves free carries the snapshot.
         * With none free the pipeline is built without the test, as before. */
        if (fi > 0 && rs->depth_bounds_test && eng_dbt_emulated()) {
            const u32 busy = eng_sampled_units(s_ps_hlsl) | cube_mask | vol_mask;
            for (int u = RSX_BE_MAX_TEXTURES - 1; u >= 0; u--)
                if (!((busy >> u) & 1u)) {
                    if (rsx_fp_apply_depth_bounds(s_ps_hlsl, sizeof s_ps_hlsl, (u32)u) > 0)
                        dbt_unit = (u32)u + 1u;
                    break;
                }
        }
        /* Shadow units read through fp_shs (after fp_dbt): a depth snapshot bound for
         * a smaller RSX texture is read in that texture's coordinates. */
        if (fi > 0 && shadow_mask && eng_shadow_region_on()) {
            const int r = rsx_fp_apply_shadow_scale(s_ps_hlsl, sizeof s_ps_hlsl, shadow_mask);
            if (r < 0) fi = -1;
            else if (r > 0) shs = 1;
        }
    }
    const u64 text_hash = (vi > 0 && fi > 0) ? eng_cat_text_hash(s_vs_hlsl, s_ps_hlsl) : 0;
    int verified = 0;
    if (reuse >= 0) {
        eng_pipeline* p = &g.pipelines[reuse];
        if (vi > 0 && fi > 0 && p->handle && p->text_hash == text_hash) {
            handle = p->handle;     /* the catalog's build is this program: keep it */
            verified = 1;
            if (s_cat_f) {          /* touched: it stays among the newest */
                const eng_cat_t t = { ENG_CAT_REC_T, p->topo, p->key };
                fwrite(&t, sizeof t, 1, s_cat_f);
            }
        } else {
            if (p->handle) {
                if (p->handle <= ENG_MAX_PIPELINES) s_pipe_entry[p->handle] = 0;
                g.be->pipeline_release(g.be->user, p->handle);
            }
            p->handle = 0;
            static u32 logs;
            if (PS3_LOG_FIRST(logs, 16))
                fprintf(stderr, "[pipe-catalog] pipeline %016llx: the decompiler's text changed since it was recorded, rebuilt\n",
                        (unsigned long long)key);
        }
    }
    if (!verified && vi > 0 && fi > 0) {
        const double t0 = s_cat_f ? eng_cat_ms() : 0.0;
        handle = g.be->pipeline_create(g.be->user, s_vs_hlsl, s_ps_hlsl, rs,
                                       layout, layout->stride, rt_fmt, rt_count);
        if (s_cat_f) {
            const double ms = eng_cat_ms() - t0;
            s_cat_mid.pn++; s_cat_mid.pms += ms; if (ms > s_cat_mid.pmax) s_cat_mid.pmax = ms;
        }
    }
    if (handle && handle <= ENG_MAX_PIPELINES) { s_pipe_dbt[handle] = (u8)dbt_unit; s_pipe_shs[handle] = (u8)shs; }
    const u64 fp_hash = fixed ? 0 : eng_fp_debug_hash(fp_uc, fp_size);
    /* RSX_ENGINE_SHADER_DUMP=<existing dir>: each new pipeline's two programs
     * as <dir>/pipe<handle>.vs.hlsl and .ps.hlsl, the handle being the one
     * RSX_ENGINE_PASS_DUMP's index names, and a guest fragment program's own
     * ucode as .fp.bin with its address and debug hash (the value
     * RSX_FP_DEBUG, RSX_ENGINE_SKIP_FP and RSX_ENGINE_REGDUMP_FP take) in
     * .fp.txt. A failed pipeline is pipe0. */
    {
        const char* dir = getenv("RSX_ENGINE_SHADER_DUMP");
        if (dir && *dir) {
            char path[1024];
            FILE* f;
            snprintf(path, sizeof path, "%s/pipe%u.vs.hlsl", dir, handle);
            if ((f = fopen(path, "w"))) { fputs(s_vs_hlsl, f); fclose(f); }
            snprintf(path, sizeof path, "%s/pipe%u.ps.hlsl", dir, handle);
            if ((f = fopen(path, "w"))) { fputs(s_ps_hlsl, f); fclose(f); }
            if (!fixed && fp_uc && fp_size) {
                u32 fp_loc = 0;
                const u32 fp_off = rsx_dsp_fragment_program(&g.rsx, &fp_loc);
                snprintf(path, sizeof path, "%s/pipe%u.fp.bin", dir, handle);
                if ((f = fopen(path, "wb"))) { fwrite(fp_uc, 1, fp_size, f); fclose(f); }
                snprintf(path, sizeof path, "%s/pipe%u.fp.txt", dir, handle);
                if ((f = fopen(path, "w"))) {
                    fprintf(f, "location %u offset 0x%08X size %u ctrl 0x%08X\n"
                               "fp %016llx\n",
                            fp_loc, fp_off, fp_size, fp_ctrl,
                            (unsigned long long)fp_hash);
                    fclose(f);
                }
            }
        }
    }
    { static u32 logs = 0; if (PS3_LOG_FIRST(logs, 32))
        fprintf(stderr, "[rsx engine] pipeline %016llx: %s vp %d, fp %d,"
                        " %u constants, fp %016llx -> %s\n",
                (unsigned long long)key, fixed ? "built-in" : "guest",
                vi, fi, nconst, (unsigned long long)fp_hash,
                handle ? "ok" : "FAILED (draw dropped)"); }

    const u32 at = reuse >= 0 ? (u32)reuse : g.n_pipelines;
    eng_pipeline* ent = &g.pipelines[at];
    if (reuse < 0) ent->topo = 0;
    ent->key = key;
    ent->fp_hash = fp_hash;
    ent->handle = handle;
    ent->fixed = (u8)fixed;
    ent->text_hash = text_hash;
    ent->prov = 0;
    ent->tex_used = (u16)(fixed ? 0xFFFFu : eng_sampled_units(s_ps_hlsl));
    *out_tex_used = ent->tex_used;
    *out_fp_hash = fp_hash;
    if (handle && handle <= ENG_MAX_PIPELINES) s_pipe_entry[handle] = at + 1u;
    if (handle && !verified) {
        if (reuse >= 0) ent->topo = 0;   /* a new build: its variants are recorded afresh */
        eng_cat_write_p(s_cat_f, ent, rs, layout, layout->stride, (u32)rt_fmt, rt_count, s_vs_hlsl, s_ps_hlsl);
    }
    if (reuse < 0) {
        while (g.pipeline_index[ix]) ix = (ix + 1u) & (ENG_PIPELINE_INDEX - 1u);
        g.pipeline_index[ix] = (u16)(g.n_pipelines + 1u);
        g.n_pipelines++;
    }
    *out_fixed = fixed;
    if (memo_early >= 0)
        eng_pl_keep(&pl_sig, vp_uc, vp_instrs, handle, *out_tex_used, fixed, fp_hash, nf,
                    pl_armed ? &pl_was : NULL, &pl_consts);
    return handle;
}

/* ---- draw accumulation --------------------------------------------------- */

static void dc_reset(void)
{
    dc.n_arr = dc.n_idx = dc.n_packets = 0;
    dc.n_refs = dc.n_source_refs = dc.n_verts = dc.n_cuts = 0;
    dc.gpu_vtx = 0;
    dc.refs_remapped = 0;
    dc.fetch_ok = 1;
    dc.inl = NULL;
    dc.inl_bytes = 0;
    dc.n_immediate = 0;
}

static int dc_push_ref(u32 vertex_id, u32 base_index)
{
    if (dc.n_refs >= dc.cap_refs) {
        const u32 next = dc.cap_refs ? dc.cap_refs * 2u : 4096u;
        rsx_vertex_ref* r = (rsx_vertex_ref*)realloc(dc.refs,
                                                     (size_t)next * sizeof(*r));
        if (!r) return 0;
        dc.refs = r;
        dc.cap_refs = next;
    }
    dc.refs[dc.n_refs].vertex_id = vertex_id;
    dc.refs[dc.n_refs].base_index = base_index;
    dc.n_refs++;
    return 1;
}

/* Room for `more` references past dc.n_refs, grown as dc_push_ref grows. */
static int dc_reserve_refs(u32 more)
{
    const u64 want = (u64)dc.n_refs + more;
    if (want <= dc.cap_refs) return 1;
    if (want > 0xFFFFFFFFull) return 0;
    u64 next = dc.cap_refs ? dc.cap_refs : 4096u;
    while (next < want) next *= 2u;
    if (next > 0xFFFFFFFFull) next = want;
    rsx_vertex_ref* r = (rsx_vertex_ref*)realloc(dc.refs, (size_t)next * sizeof(*r));
    if (!r) return 0;
    dc.refs = r;
    dc.cap_refs = (u32)next;
    return 1;
}

/* A run of big-endian indices appended as references in one pass: the
 * capacity is reserved up front and the loop works on locals, which the
 * per-index dc_push_ref (reloading dc after every store) could not. A
 * restart sentinel still cuts the strip where it stands. */
static int dc_push_index_run(const u8* run, u32 count, int is_u32, u32 base_index,
                             int restart_en, u32 restart_val)
{
    if (!dc_reserve_refs(count)) return 0;
    rsx_vertex_ref* out = dc.refs;
    u32 n = dc.n_refs;
    if (is_u32) {
        for (u32 i = 0; i < count; i++) {
            const u8* p = run + (size_t)i * 4u;
            const u32 index = ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3];
            if (restart_en && index == restart_val) {
                dc.n_refs = n;
                if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts, n)) return 0;
                continue;
            }
            out[n].vertex_id = index; out[n].base_index = base_index; n++;
        }
    } else if (!restart_en) {
        for (u32 i = 0; i < count; i++) {
            out[n + i].vertex_id = (u32)((run[2u * i] << 8) | run[2u * i + 1u]);
            out[n + i].base_index = base_index;
        }
        n += count;
    } else {
        for (u32 i = 0; i < count; i++) {
            const u32 index = (u32)((run[2u * i] << 8) | run[2u * i + 1u]);
            if (index == restart_val) {
                dc.n_refs = n;
                if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts, n)) return 0;
                continue;
            }
            out[n].vertex_id = index; out[n].base_index = base_index; n++;
        }
    }
    dc.n_refs = n;
    return 1;
}

/* RSX_VTX_GPU=1: a draw whose references are consecutive elements, all
 * inside each attribute's stable range, of the types the GPU decode handles,
 * hands the backend the raw guest bytes (draw_gpu_vertices) instead of
 * decoding them here; a backend that declines gets the CPU fetch at the draw.
 * =2 also fetches on the CPU and has the backend compare. */
static int eng_vtx_gpu_mode(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("RSX_VTX_GPU"); s = e ? atoi(e) : 0; }
    return s;
}

static rsx_be_vtx_job s_vjob;

static int eng_vtx_gpu_job(const rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* refs, u32 count)
{
    /* why draws are declined, printed every 60 s while RSX_VTX_GPU is on */
    static unsigned why[8]; static time_t wt;
    if (eng_vtx_gpu_mode()) { const time_t now = time(NULL); if (!wt) wt = now;
        if (now - wt >= 60) { fprintf(stderr, "[vtx-gpu] 60s declined: off/inline/layout %u, refs %u, wrap %u, attr %u; taken %u\n",
                                     why[0], why[1], why[2], why[3], why[4]); memset(why, 0, sizeof why); wt = now; } }
    if (!eng_vtx_gpu_mode() || !g.be->draw_gpu_vertices || plan->inline_data || !count ||
        plan->layout.count > 16u || plan->layout.stride != plan->layout.count * 16u)
        { why[0]++; return 0; }
    const u32 id0 = refs[0].vertex_id, base = refs[0].base_index;
    for (u32 i = 1; i < count; i++)
        if (refs[i].vertex_id != id0 + i || refs[i].base_index != base) { why[1]++; return 0; }
    const u32 e0 = (id0 + base) & 0x000FFFFFu;
    if (e0 + (u64)count - 1u > 0x000FFFFFu) { why[2]++; return 0; }   /* the 20-bit element wraps */
    s_vjob.count = count;
    s_vjob.nslots = plan->layout.count;
    for (u32 s = 0; s < plan->layout.count; s++) {
        const rsx_vertex_fetch_attr* f = &plan->attr[plan->layout.attrs[s]];
        rsx_be_vtx_slot* js = &s_vjob.slot[s];
        memcpy(js->def, f->default_value, sizeof js->def);
        if (!f->desc.type || !f->desc.size) { js->type = 0; js->src = NULL; js->bytes = js->stride = js->n = 0; continue; }
        const u32 type = f->desc.type;
        if (type == RSX_VTX_TYPE_CMP32 || type < RSX_VTX_TYPE_SNORM16 || type > RSX_VTX_TYPE_UINT8 ||
            f->desc.frequency > 1 || !f->stable_base || e0 < f->stable_first ||
            e0 + count - 1u > f->stable_last)
            { why[3]++; return 0; }
        js->type = type;
        js->n = f->desc.size < 4 ? f->desc.size : 4;
        js->stride = f->stride;
        js->src = f->stable_base + (u64)(e0 - f->stable_first) * f->stride;
        js->bytes = (count - 1u) * f->stride + f->elem_size;
    }
    why[4]++;
    return 1;
}

static int dc_reserve_verts(u64 bytes)
{
    if (bytes <= dc.verts_cap) return 1;
    u64 next = dc.verts_cap ? dc.verts_cap : (1u << 20);
    while (next < bytes) next *= 2u;
    u8* v = (u8*)realloc(dc.verts, (size_t)next);
    if (!v) return 0;
    dc.verts = v;
    dc.verts_cap = next;
    return 1;
}

static int dc_reserve_indices(u32 count)
{
    if (count <= g.index_cap) return 1;
    u32 next = g.index_cap ? g.index_cap : 4096u;
    while (next < count) next *= 2u;
    u32* p = (u32*)realloc(g.indices, (size_t)next * sizeof(u32));
    if (!p) return 0;
    g.indices = p;
    g.index_cap = next;
    return 1;
}

/* Turn this group's batches into a reference list, recording a CUT rather
 * than fetching a phantom vertex wherever the restart sentinel appears, then
 * collapse repeated references and fetch each unique vertex once.
 * rsx_live_draw.c's fetch_batches_hoisted (4165-4292). */
static void dc_fetch(const rsx_vertex_layout_plan* layout, int allow_remap)
{
    if (dc.n_immediate) {
        dc.n_source_refs = dc.n_immediate;
        if (!dc_reserve_verts((u64)dc.n_immediate * layout->stride)) {
            dc.fetch_ok = 0;
            return;
        }
        for (u32 v = 0; v < dc.n_immediate; v++)
            for (u32 slot = 0; slot < layout->count; slot++)
                memcpy(dc.verts + (u64)v * layout->stride + slot * 16u,
                       dc.immediate[v][layout->attrs[slot]], 16u);
        dc.n_verts = dc.n_immediate;
        return;
    }
    for (u32 b = 0; b < dc.n_arr && dc.fetch_ok; b++)
        for (u32 i = 0; i < dc.arr[b].count && dc.fetch_ok; i++)
            if (!dc_push_ref(dc.arr[b].first + i, 0)) dc.fetch_ok = 0;

    if (dc.n_idx && dc.fetch_ok) {
        ENG_ST_BEGIN();
        const u32 base_index = rsx_dsp_vertex_data_base_index(&g.rsx);
        rsx_dsp_index_array ia;
        rsx_dsp_get_index_array(&g.rsx, &ia);
        const u32 esz = ia.is_u32 ? 4u : 2u;
        const int restart_en = rsx_dsp_restart_index_enabled(&g.rsx, ia.is_u32);
        const u32 restart_val = rsx_dsp_restart_index(&g.rsx);
        /* The index array's location field is a DMA selector, not the
         * rsx_dsp location enum: 0 selects local memory. */
        const u32 ia_loc = ia.location ? RSX_LOCATION_MAIN : RSX_LOCATION_LOCAL;

        for (u32 b = 0; b < dc.n_idx && dc.fetch_ok; b++) {
            const u32 first = dc.idx[b].first, count = dc.idx[b].count;
            const u64 start = (u64)ia.offset + (u64)first * esz;
            const u64 bytes = (u64)count * esz;
            const u8* run = NULL;
            if (count && start <= 0xFFFFFFFFull && bytes <= 0xFFFFFFFFull &&
                start + bytes <= 0x100000000ull)
                run = eng_guest_ptr(NULL, ia_loc, (u32)start, (u32)bytes);
            if (run) {
                if (!dc_push_index_run(run, count, ia.is_u32, base_index, restart_en, restart_val))
                    dc.fetch_ok = 0;
                continue;
            }
            for (u32 i = 0; i < count && dc.fetch_ok; i++) {
                const u8* p = run ? run + (size_t)i * esz
                    : eng_guest_ptr(NULL, ia_loc, ia.offset + (first + i) * esz, esz);
                if (!p) { dc.fetch_ok = 0; break; }
                const u32 index = ia.is_u32
                    ? (((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3])
                    : (u32)((p[0] << 8) | p[1]);
                if (restart_en && index == restart_val) {
                    if (!rsx_restart_cut_push(&dc.cuts, &dc.n_cuts, &dc.cap_cuts,
                                              dc.n_refs))
                        dc.fetch_ok = 0;
                    continue;
                }
                if (!dc_push_ref(index, base_index)) dc.fetch_ok = 0;
            }
        }
        ENG_ST_END(ST_FIDX);
    }
    if (!dc.fetch_ok) return;

    dc.n_source_refs = dc.n_refs;
    /* Only a topology this engine rebuilds through an index buffer may share
     * repeated references. A line strip or a point list is drawn in the order
     * it arrived, so collapsing duplicates would reorder it. */
    if (allow_remap && dc.n_refs > 1) {
        u32 unique = dc.n_refs;
        int built;
        { ENG_ST_BEGIN(); built = rsx_vertex_remap_build(&dc.ref_remap, dc.refs, dc.n_refs, &unique); ENG_ST_END(ST_FREMAP); }
        if (built) {
            dc.refs_remapped = unique < dc.n_refs;
            dc.n_refs = unique;
        }
    }

    dc.layout = *layout;
    { ENG_ST_BEGIN();
    rsx_vertex_fetch_plan_init(&dc.fetch_plan, &g.rsx, layout, eng_guest_ptr, NULL);
    rsx_vertex_fetch_plan_set_inline(&dc.fetch_plan, &g.rsx, dc.inl, dc.inl_bytes);
    rsx_vertex_fetch_plan_prepare(&dc.fetch_plan, dc.refs, dc.n_refs);
    ENG_ST_END(ST_FPREP); }
    if (!dc_reserve_verts((u64)dc.n_refs * layout->stride)) {
        dc.fetch_ok = 0;
        return;
    }
    dc.n_verts = dc.n_refs;
    if (eng_vtx_gpu_job(&dc.fetch_plan, dc.refs, dc.n_refs)) {
        dc.gpu_vtx = 1;
        if (eng_vtx_gpu_mode() != 2) return;   /* =2 also fetches, for the backend's check */
    }
    int fetched;
    { ENG_ST_BEGIN(); fetched = rsx_vertex_fetch_many(&dc.fetch_plan, dc.refs, dc.n_refs,
                                                    layout->stride ? dc.verts : NULL); ENG_ST_END(ST_FMANY); }
    if (!fetched) {
        dc.fetch_ok = 0;
        return;
    }
    dc.n_verts = dc.n_refs;
}

/* ---- sink ---------------------------------------------------------------- */

static void sink_begin(void* user, const rsx_dispatch* r, u32 prim)
{
    (void)user; (void)prim;
    dc_reset();
    dc.inline_dropped_at_begin = r->inline_dropped;
}

/* A malformed/unbounded group must not render a truncated primitive. */
#define ENG_MAX_IMMEDIATE_VERTICES 4096u
static void sink_immediate_vertex(void* user, const rsx_dispatch* r)
{
    (void)user;
    dc.n_packets++;
    if (!dc.fetch_ok) return;
    if (dc.n_immediate == ENG_MAX_IMMEDIATE_VERTICES) {
        dc.fetch_ok = 0;
        return;
    }
    if (dc.n_immediate == dc.cap_immediate) {
        const u32 next = dc.cap_immediate ? dc.cap_immediate * 2u : 64u;
        void* p = realloc(dc.immediate, (size_t)next * sizeof(*dc.immediate));
        if (!p) { dc.fetch_ok = 0; return; }
        dc.immediate = p;
        dc.cap_immediate = next;
    }
    for (u32 attr = 0; attr < RSX_DSP_NUM_VERTEX_ATTR; attr++) {
        float* value = dc.immediate[dc.n_immediate][attr];
        rsx_dsp_vertex_default(r, attr, value);
        /* The legacy default getter treats all-zero words as unwritten.
         * Direct and folded constants may explicitly set W/alpha to zero. */
        const u32 base = (0x1C00u + attr * 16u) >> 2;
        if (r->seen[base] || r->seen[base + 1] ||
            r->seen[base + 2] || r->seen[base + 3])
            memcpy(value, &r->regs[base], 16u);
    }
    dc.n_immediate++;
}

static void sink_draw_arrays(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dc.n_packets++;
    if (dc.n_arr >= ENG_MAX_BATCHES) return;
    dc.arr[dc.n_arr].first = first;
    dc.arr[dc.n_arr].count = count;
    dc.n_arr++;
}

/* INLINE_ARRAY delivers one finished stream rather than a (first, count)
 * range; queue it as an ordinary consecutive batch and let the fetch plan
 * read the vertices out of it (rsx_vertex_fetch_plan_set_inline). */
static void sink_inline_array(void* user, const rsx_dispatch* r,
                              const u8* data, u32 bytes)
{
    (void)user;
    /* Record source presence even when its declaration or length is invalid,
     * so mixed immediate/inline groups cannot draw a partial prefix. */
    dc.inl = data;
    dc.inl_bytes = bytes;
    const u32 stride = rsx_vertex_inline_layout(r, NULL);
    if (!stride || bytes < stride) return;
    dc.n_packets++;
    if (dc.n_arr >= ENG_MAX_BATCHES) return;
    dc.arr[dc.n_arr].first = 0;
    dc.arr[dc.n_arr].count = bytes / stride;
    dc.n_arr++;
}

static void sink_draw_index(void* user, const rsx_dispatch* r, u32 first, u32 count)
{
    (void)user; (void)r;
    dc.n_packets++;
    if (dc.n_idx >= ENG_MAX_BATCHES) return;
    dc.idx[dc.n_idx].first = first;
    dc.idx[dc.n_idx].count = count;
    dc.n_idx++;
}

/* Which texture units this draw binds, and from where: a unit naming a
 * registered colour surface samples that live target rather than guest
 * memory (every post-process, shadow and reflection in a PS3 title is a
 * render-to-texture read back through a texture unit), a unit naming a
 * tracked zeta in DEPTH24_D8 samples its snapshot, and everything else is a
 * guest upload (rsx_live_draw.c:5986-6062). */
/* A direct copy is valid only for the entire matching physical sample
 * layout. Reinterpreted, swizzled, mipmapped and subrectangle aliases need a
 * different conversion and must not silently sample stale guest bytes. */
static int eng_color_feedback_compatible(const eng_surface* s,
                                          const rsx_dsp_texture* t,
                                          const rsx_dsp_surface* sf)
{
    const u32 base = t->format & 0x9fu;
    u32 bpp = 0, color = 0;
    switch (s->fmt) {
    case RSX_BE_FMT_R8G8B8A8:        if (base == 0x85u) { bpp = 4; color = 8; } break;
    case RSX_BE_FMT_R16G16B16A16F:   if (base == 0x9au) { bpp = 8; color = 11; } break;
    case RSX_BE_FMT_R32G32B32A32F:   if (base == 0x9bu) { bpp = 16; color = 12; } break;
    case RSX_BE_FMT_R32F:            if (base == 0x9cu) { bpp = 4; color = 13; } break;
    default: break;
    }
    if (!bpp || sf->color_format != color || sf->raster_type != 1 ||
        !(t->format & 0x20u) || t->dimension != 2 ||
        t->cubemap || t->mipmaps > 1 || t->depth > 1 ||
        t->width != s->w || t->height != s->h) return 0;
    const u64 row = (u64)s->w * bpp;
    const u64 sp = s->pitch ? s->pitch : row;
    const u64 tp = t->pitch ? t->pitch : row;
    return row && sp >= row && tp == sp;
}

static u32 eng_color_feedback_failure(u32 unit, const eng_surface* s,
                                       const rsx_dsp_texture* t)
{
    static u32 logs;
    rsx_dsp_surface sf;
    rsx_dsp_get_surface(&g.rsx, &sf);
    if (PS3_LOG_FIRST(logs, 8))
        fprintf(stderr, "[rsx engine] rejected colour feedback unit %u: "
                "target %ux%u fmt%u pitch%u, texture %ux%u fmt%02X pitch%u "
                "dim%u levels%u cube%u color%u raster%u\n", unit, s->w, s->h, (u32)s->fmt,
                s->pitch, t->width, t->height, t->format, t->pitch,
                t->dimension, t->mipmaps, t->cubemap, sf.color_format, sf.raster_type);
    return ENG_INVALID;
}

/* With ENG_PHASE_STATS set: each distinct way a draw samples a depth or colour
 * target through guest memory (first 32) -- the texture format it reads the
 * target as, the target's own format and size. */
static void eng_note_via_guest(const char* path, const rsx_dsp_texture* t, u32 src_fmt, u32 w, u32 h)
{
    static int on = -1;
    if (on < 0) on = getenv("ENG_PHASE_STATS") ? 1 : 0;
    if (!on) return;
    static u32 seen[32][4];
    static u32 n_seen;
    const u32 key[4] = { (u32)path[0], t->format, src_fmt, (w << 16) | h };
    for (u32 i = 0; i < n_seen; i++)
        if (!memcmp(seen[i], key, sizeof key)) return;
    if (n_seen >= 32) return;
    memcpy(seen[n_seen++], key, sizeof key);
    fprintf(stderr, "[eng-via-guest] %s: texture fmt 0x%02X %ux%u pitch %u remap 0x%04X at %u:0x%08X, "
            "target fmt %u %ux%u\n", path, t->format, t->width, t->height, t->pitch, t->remap & 0xFFFFu,
            t->location, t->offset, src_fmt, w, h);
}

/* RSX_Z2ARGB=1: a D24S8 depth target sampled as linear A8R8G8B8 (GoW:A's
 * deferred passes read their depth that way to rebuild positions) is built on
 * the GPU (backend depth_to_argb8) instead of being stored into guest memory,
 * which waits for the GPU, converts on the CPU and uploads again. Guest memory
 * keeps its older bytes; it stays marked dirty, so any other reader through
 * eng_zdepth_to_guest still gets a fresh store. RSX_Z2ARGB=2 does both and
 * compares the texels every time, printing a count every 60 s. */
static int eng_z2argb_mode(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("RSX_Z2ARGB"); s = e ? atoi(e) : 0; }
    return s;
}

/* Conversions, textures created, backend failures and the time in the
 * backend op, printed every 60 s while RSX_Z2ARGB is on. */
static struct { unsigned conv, created, failed; unsigned long long us, max_us; time_t t; } s_z2a;

static void eng_z2argb_tick(void)
{
    const time_t now = time(NULL);
    if (!s_z2a.t) s_z2a.t = now;
    if (now - s_z2a.t < 60) return;
    fprintf(stderr, "[z2argb] 60s: %u converted, %u textures created, %u failed, %.1f ms in the backend, max %.2f ms\n",
            s_z2a.conv, s_z2a.created, s_z2a.failed, (double)s_z2a.us / 1000.0, (double)s_z2a.max_us / 1000.0);
    memset(&s_z2a, 0, sizeof s_z2a);
    s_z2a.t = now;
}

static u32 eng_zdepth_argb8(u32 slot, const rsx_dsp_texture* t, u32 base_fmt, int check)
{
    eng_zdepth* z = &g.zdepths[slot];
    eng_z2argb_tick();
    if (!g.be->depth_to_argb8 || !z->handle || base_fmt != 0x85u || z->zfmt != 2u ||
        !(t->format & 0x20u) || t->cubemap || t->dimension != 2 || t->mipmaps > 1 ||
        !t->width || !t->height || t->width > z->w || t->height > z->h || t->pitch != z->pitch)
        return 0;
    const u32 remap = t->remap & 0xFFFFu;
    u32 k = 0;
    while (k < 2 && !(z->argb[k].tex && z->argb[k].format == t->format && z->argb[k].remap == remap &&
                      z->argb[k].w == t->width && z->argb[k].h == t->height)) k++;
    if (k == 2) {
        k = z->argb[0].tex ? 1u : 0u;      /* both taken: the second goes */
        if (z->argb[k].tex) g.be->texture_release(g.be->user, z->argb[k].tex);
        memset(&z->argb[k], 0, sizeof z->argb[k]);
        const u32 tex = g.be->texture_create(g.be->user, RSX_BE_FMT_R8G8B8A8, t->width, t->height,
                                             1, 1, remap, t->format);
        if (!tex) return 0;
        s_z2a.created++;
        z->argb[k].tex = tex; z->argb[k].format = t->format; z->argb[k].remap = remap;
        z->argb[k].w = t->width; z->argb[k].h = t->height;
    }
    if (z->argb[k].serial == z->depth_serial && !check) return z->argb[k].tex;
    const u32 n = t->width * t->height;
    u32* gpu = check ? (u32*)malloc((size_t)n * 4u) : NULL;
    const unsigned long long t0 = eng_now_us();
    const int rc = g.be->depth_to_argb8(g.be->user, z->handle, z->argb[k].tex, gpu);
    const unsigned long long dt = eng_now_us() - t0;
    s_z2a.us += dt; if (dt > s_z2a.max_us) s_z2a.max_us = dt;
    if (rc != 0) { s_z2a.failed++; free(gpu); return 0; }
    s_z2a.conv++;
    z->argb[k].serial = z->depth_serial;
    if (gpu) {
        /* The same texels from guest memory, as the upload would decode them:
         * linear A8R8G8B8 is a byte rotation, guest A,R,G,B to host R,G,B,A. */
        static unsigned s_cmp, s_bad, s_px; static time_t s_t;
        z->guest_dirty = 1;
        unsigned bad = 0, first = ~0u; u32 want = 0;
        int row_fail = -1;
        if (eng_zdepth_to_guest(slot)) {
            for (u32 y = 0; y < t->height; y++) {
                const u8* row = eng_guest_ptr(NULL, z->location, z->offset + y * z->pitch, t->width * 4u);
                if (!row) { bad = n; row_fail = (int)y; break; }
                for (u32 x = 0; x < t->width; x++) {
                    u32 v; memcpy(&v, row + x * 4u, 4);
                    v = (v >> 8) | (v << 24);
                    if (v != gpu[y * t->width + x]) {
                        if (first == ~0u) { first = y * t->width + x; want = v; }
                        bad++;
                    }
                }
            }
            s_cmp++; s_px += bad; if (bad) s_bad++;
            if (bad && s_bad <= 4)
                fprintf(stderr, "[z2argb-check] zeta 0x%08X %ux%u fmt 0x%02X: %u texels differ, first %u: gpu 0x%08X guest 0x%08X (used %ux%u, dirty-row %d)\n",
                        z->offset, t->width, t->height, t->format, bad, first, gpu[first], want,
                        z->used_w, z->used_h, row_fail);
        }
        const time_t now = time(NULL);
        if (!s_t) s_t = now;
        if (now - s_t >= 60) {
            fprintf(stderr, "[z2argb-check] 60s: %u compared, %u with differences, %u texels\n", s_cmp, s_bad, s_px);
            s_t = now; s_cmp = s_bad = s_px = 0;
        }
        free(gpu);
    }
    return z->argb[k].tex;
}

/* RSX_BLIT_SHADOW=1: an unscaled NV3089 blit out of an A8R8G8B8 colour target
 * into memory no engine surface covers is copied on the GPU, into a "strip"
 * texture keyed by its destination rectangle, instead of being read back into
 * guest memory (a wait for the GPU) and copied there by the CPU. A later
 * texture that samples that memory as linear A8R8G8B8 and is fully covered by
 * strips gets them assembled into one texture on the GPU. Guest memory is NOT
 * written: anything else that reads it (a CPU/SPU consumer, a save thumbnail)
 * sees stale bytes -- so it is only for titles whose copies feed nothing but
 * textures (GoW:A's MLAA input, with the MLAA tasks served by HLE). Another
 * NV3089 write over a strip drops it (rsx_draw_engine_blit_shadow_drop).
 * RSX_BLIT_SHADOW=2 keeps the guest-memory copy too and compares every
 * assembled texture with the bytes the copy left there. */
static int eng_blit_shadow_mode(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("RSX_BLIT_SHADOW"); s = e ? atoi(e) : 0; }
    return s;
}

#define ENG_STRIPS 8u
#define ENG_STRIP_COMPS 4u
static struct { u32 loc, off, pitch, w, h, tex; u64 serial; } s_strip[ENG_STRIPS];
static struct { u32 loc, off, pitch, w, h, format, remap, tex; u64 serial; } s_comp[ENG_STRIP_COMPS];
static u32 s_strip_next, s_comp_next;
static u64 s_strip_serial;
static struct { unsigned blits, hits, built, cmp, bad, dropped; time_t t; } s_bs;

static void eng_blit_shadow_tick(void)
{
    const time_t now = time(NULL);
    if (!s_bs.t) s_bs.t = now;
    if (now - s_bs.t < 60) return;
    fprintf(stderr, "[blit-shadow] 60s: %u blits shadowed, %u texture hits, %u assembled, %u dropped;"
            " checked %u, %u with differences\n",
            s_bs.blits, s_bs.hits, s_bs.built, s_bs.dropped, s_bs.cmp, s_bs.bad);
    memset(&s_bs, 0, sizeof s_bs);
    s_bs.t = now;
}

static int eng_ranges_overlap(u32 a0, u32 an, u32 b0, u32 bn)
{
    return (u64)a0 < (u64)b0 + bn && (u64)b0 < (u64)a0 + an;
}

/* Bytes [off, off + pitch * (h - 1) + w * 4) of a strip or a request. */
static u32 eng_rect_bytes(u32 pitch, u32 w, u32 h) { return pitch * (h - 1u) + w * 4u; }

/* Do two rectangles of guest bytes (off, pitch, width in bytes, rows) share
 * a byte? With one pitch, by columns and rows: side-by-side strips of the
 * same rows interleave in address but never touch. */
static int eng_rects_overlap(u32 a, u32 ap, u32 aw, u32 ah, u32 b, u32 bp, u32 bw, u32 bh)
{
    if (!eng_ranges_overlap(a, ap * (ah - 1u) + aw, b, bp * (bh - 1u) + bw)) return 0;
    if (ap != bp || !ap) return 1;
    const u32 ac = a % ap, bc = b % bp, ar = a / ap, br = b / bp;
    return ac < bc + bw && bc < ac + aw && ar < br + bh && br < ar + ah;
}

void rsx_draw_engine_blit_shadow_drop(u32 loc, u32 off, u32 pitch, u32 wbytes, u32 h)
{
    s_nd_events++;
    if (!eng_blit_shadow_mode() || !wbytes || !h) return;
    for (u32 i = 0; i < ENG_STRIPS; i++)
        if (s_strip[i].tex && s_strip[i].loc == loc &&
            eng_rects_overlap(s_strip[i].off, s_strip[i].pitch, s_strip[i].w * 4u, s_strip[i].h,
                              off, pitch, wbytes, h)) {
            g.be->texture_release(g.be->user, s_strip[i].tex);
            memset(&s_strip[i], 0, sizeof s_strip[i]);
            s_bs.dropped++;
        }
}

int rsx_draw_engine_blit_shadow(u32 sloc, u32 soff, u32 spitch, u32 dloc, u32 doff, u32 dpitch,
                                u32 w, u32 h)
{
    const int mode = eng_blit_shadow_mode();
    if (!mode || !g.ready || !g.be->image_copy || !w || !h) return 0;
    eng_blit_shadow_tick();
    u32 xb = 0, y = 0, dxb = 0, dy = 0;
    const u32 slot = eng_surface_at(sloc, soff, spitch, &xb, &y);
    if (slot == ENG_INVALID || g.surfaces[slot].fmt != RSX_BE_FMT_R8G8B8A8 || (xb & 3u)) return 0;
    const eng_surface* s = &g.surfaces[slot];
    const u32 x = xb / 4u;
    if ((u64)x + w > s->w || (u64)y + h > s->h || (u64)w * 4u > dpitch) return 0;
    if (eng_surface_at(dloc, doff, dpitch, &dxb, &dy) != ENG_INVALID) return 0;   /* a surface: the old path */
    u32 k = 0;
    while (k < ENG_STRIPS && !(s_strip[k].tex && s_strip[k].loc == dloc && s_strip[k].off == doff &&
                               s_strip[k].pitch == dpitch && s_strip[k].w == w && s_strip[k].h == h)) k++;
    if (k == ENG_STRIPS) {
        rsx_draw_engine_blit_shadow_drop(dloc, doff, dpitch, w * 4u, h);   /* other shapes over these bytes */
        k = s_strip_next++ % ENG_STRIPS;
        if (s_strip[k].tex) { g.be->texture_release(g.be->user, s_strip[k].tex); s_bs.dropped++; }
        memset(&s_strip[k], 0, sizeof s_strip[k]);
        const u32 tex = g.be->texture_create(g.be->user, RSX_BE_FMT_R8G8B8A8, w, h, 1, 1, 0xAAE4u, 0xA5u);
        if (!tex) return 0;
        s_strip[k].loc = dloc; s_strip[k].off = doff; s_strip[k].pitch = dpitch;
        s_strip[k].w = w; s_strip[k].h = h; s_strip[k].tex = tex;
    }
    if (g.be->image_copy(g.be->user, s->handle, x, y, s_strip[k].tex, 0, 0, w, h, NULL) != 0) {
        g.be->texture_release(g.be->user, s_strip[k].tex);
        memset(&s_strip[k], 0, sizeof s_strip[k]);
        return 0;
    }
    s_strip[k].serial = ++s_strip_serial;
    s_bs.blits++;
    return mode == 1 ? 1 : 2;
}

/* A linear A8R8G8B8 texture fully covered by strips: their contents put
 * together on the GPU, rebuilt when a strip was copied again. 0 = not one. */
static u32 eng_shadow_texture(const rsx_dsp_texture* t, u32 base_fmt)
{
    if (!eng_blit_shadow_mode() || base_fmt != 0x85u || !(t->format & 0x20u) || t->cubemap ||
        t->dimension != 2 || t->mipmaps > 1 || !t->width || !t->height || !t->pitch ||
        (u64)t->width * 4u > t->pitch)
        return 0;
    const u32 span = eng_rect_bytes(t->pitch, t->width, t->height);
    u32 idx[ENG_STRIPS], n = 0;
    u64 area = 0, newest = 0;
    for (u32 i = 0; i < ENG_STRIPS; i++) {
        if (!s_strip[i].tex || s_strip[i].loc != t->location || s_strip[i].pitch != t->pitch ||
            s_strip[i].off < t->offset || s_strip[i].off - t->offset >= span)
            continue;
        const u32 d = s_strip[i].off - t->offset;
        if ((d % t->pitch) & 3u || (d % t->pitch) / 4u + s_strip[i].w > t->width ||
            d / t->pitch + s_strip[i].h > t->height)
            return 0;               /* a strip only partly inside: not this texture's */
        idx[n++] = i;
        area += (u64)s_strip[i].w * s_strip[i].h;
        if (s_strip[i].serial > newest) newest = s_strip[i].serial;
    }
    if (n && area != (u64)t->width * t->height) {
        static unsigned logs; if (logs++ < 4)
            fprintf(stderr, "[blit-shadow] texture %u:0x%08X %ux%u pitch %u: %u strip(s) cover %llu of %u texels\n",
                    t->location, t->offset, t->width, t->height, t->pitch, n, (unsigned long long)area,
                    t->width * t->height);
    }
    if (!n || area != (u64)t->width * t->height) return 0;     /* strips never overlap: equal area = covered */
    const u32 remap = t->remap & 0xFFFFu;
    u32 c = 0;
    while (c < ENG_STRIP_COMPS && !(s_comp[c].tex && s_comp[c].loc == t->location && s_comp[c].off == t->offset &&
                                    s_comp[c].pitch == t->pitch && s_comp[c].w == t->width &&
                                    s_comp[c].h == t->height && s_comp[c].format == t->format &&
                                    s_comp[c].remap == remap)) c++;
    if (c == ENG_STRIP_COMPS) {
        c = s_comp_next++ % ENG_STRIP_COMPS;
        if (s_comp[c].tex) g.be->texture_release(g.be->user, s_comp[c].tex);
        memset(&s_comp[c], 0, sizeof s_comp[c]);
        const u32 tex = g.be->texture_create(g.be->user, RSX_BE_FMT_R8G8B8A8, t->width, t->height,
                                             1, 1, remap, t->format);
        if (!tex) return 0;
        s_comp[c].loc = t->location; s_comp[c].off = t->offset; s_comp[c].pitch = t->pitch;
        s_comp[c].w = t->width; s_comp[c].h = t->height; s_comp[c].format = t->format;
        s_comp[c].remap = remap; s_comp[c].tex = tex;
        { static unsigned logs; if (logs++ < 4)
            fprintf(stderr, "[blit-shadow] texture %u:0x%08X %ux%u pitch %u fmt 0x%02X remap 0x%04X from %u strip(s)\n",
                    t->location, t->offset, t->width, t->height, t->pitch, t->format, remap, n); }
    }
    s_bs.hits++;
    if (s_comp[c].serial == newest) return s_comp[c].tex;
    const int check = eng_blit_shadow_mode() == 2;
    u32* gpu = check ? (u32*)malloc((size_t)t->width * t->height * 4u) : NULL;
    for (u32 j = 0; j < n; j++) {
        const u32 i = idx[j], d = s_strip[i].off - t->offset;
        if (g.be->image_copy(g.be->user, s_strip[i].tex, 0, 0, s_comp[c].tex, (d % t->pitch) / 4u, d / t->pitch,
                             s_strip[i].w, s_strip[i].h, j + 1 == n ? gpu : NULL) != 0) {
            free(gpu);
            return 0;
        }
    }
    s_comp[c].serial = newest;
    s_bs.built++;
    if (gpu) {
        /* The guest-memory copy also ran: the bytes the upload would decode,
         * guest A,R,G,B to host R,G,B,A. */
        unsigned bad = 0, first = ~0u; u32 want = 0;
        for (u32 yy = 0; yy < t->height && bad < 1u << 30; yy++) {
            const u8* row = eng_guest_ptr(NULL, t->location, t->offset + yy * t->pitch, t->width * 4u);
            if (!row) { bad = 1u << 30; break; }
            for (u32 xx = 0; xx < t->width; xx++) {
                u32 v; memcpy(&v, row + xx * 4u, 4);
                v = (v >> 8) | (v << 24);
                if (v != gpu[yy * t->width + xx]) {
                    if (first == ~0u) { first = yy * t->width + xx; want = v; }
                    bad++;
                }
            }
        }
        s_bs.cmp++; if (bad) s_bs.bad++;
        if (bad && s_bs.bad <= 4)
            fprintf(stderr, "[blit-shadow-check] 0x%08X %ux%u: %u texels differ, first %u: gpu 0x%08X guest 0x%08X\n",
                    t->offset, t->width, t->height, bad, first, first != ~0u ? gpu[first] : 0u, want);
        free(gpu);
    }
    return s_comp[c].tex;
}

/* RSX_C2RG16=1: an A8R8G8B8 colour target sampled as linear Y16_X16_FLOAT
 * (GoW:A reads two half-resolution targets that way, ~2 a frame) is
 * regrouped on the GPU (backend color_to_rg16f) instead of being stored into
 * guest memory, which waits for the GPU, converts on the CPU and uploads
 * again. Guest memory keeps its older bytes and stays marked dirty, as with
 * RSX_Z2ARGB. RSX_C2RG16=2 does both and compares every texel. */
static int eng_c2rg16_mode(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("RSX_C2RG16"); s = e ? atoi(e) : 0; }
    return s;
}

static struct { unsigned conv, created, failed, cmp, bad; unsigned long long px; time_t t; } s_c2r;

static u32 eng_color_rg16f(u32 slot, const rsx_dsp_texture* t, u32 base_fmt)
{
    eng_surface* s = &g.surfaces[slot];
    const int check = eng_c2rg16_mode() == 2;
    const time_t now = time(NULL);
    if (!s_c2r.t) s_c2r.t = now;
    if (now - s_c2r.t >= 60) {
        fprintf(stderr, "[c2rg16] 60s: %u converted, %u textures created, %u failed; checked %u, %u with differences, %llu texels\n",
                s_c2r.conv, s_c2r.created, s_c2r.failed, s_c2r.cmp, s_c2r.bad, s_c2r.px);
        memset(&s_c2r, 0, sizeof s_c2r);
        s_c2r.t = now;
    }
    if (!g.be->color_to_rg16f || !s->handle || s->fmt != RSX_BE_FMT_R8G8B8A8 || base_fmt != 0x9Fu ||
        !(t->format & 0x20u) || t->cubemap || t->dimension != 2 || t->mipmaps > 1 ||
        !t->width || !t->height || t->width > s->w || t->height > s->h || t->pitch != s->pitch)
        return 0;
    const u32 remap = t->remap & 0xFFFFu;
    /* Two shapes per target (the same target is read at two sizes): the
     * second is replaced when a third one comes. */
    u32 k = 0;
    while (k < 2 && !(s->rg16[k].tex && s->rg16[k].format == t->format && s->rg16[k].remap == remap &&
                      s->rg16[k].w == t->width && s->rg16[k].h == t->height)) k++;
    if (k == 2) {
        k = s->rg16[0].tex ? 1u : 0u;
        if (s->rg16[k].tex) g.be->texture_release(g.be->user, s->rg16[k].tex);
        memset(&s->rg16[k], 0, sizeof s->rg16[k]);
        const u32 tex = g.be->texture_create(g.be->user, RSX_BE_FMT_R16G16F, t->width, t->height,
                                             1, 1, remap, t->format);
        if (!tex) return 0;
        s_c2r.created++;
        s->rg16[k].tex = tex; s->rg16[k].format = t->format; s->rg16[k].remap = remap;
        s->rg16[k].w = t->width; s->rg16[k].h = t->height; s->rg16[k].serial = ~0ull;
        { static unsigned logs; if (logs++ < 8) fprintf(stderr, "[c2rg16] new texture for 0x%08X: %ux%u fmt 0x%02X remap 0x%04X\n",
                                                        s->offset, t->width, t->height, t->format, remap); }
    }
    if (s->rg16[k].serial == s->color_serial && !check) return s->rg16[k].tex;
    const u32 n = t->width * t->height;
    u32* gpu = check ? (u32*)malloc((size_t)n * 4u) : NULL;
    if (g.be->color_to_rg16f(g.be->user, s->handle, s->rg16[k].tex, gpu) != 0) {
        s_c2r.failed++; free(gpu); return 0;
    }
    s_c2r.conv++;
    s->rg16[k].serial = s->color_serial;
    if (gpu) {
        /* The same texels from guest memory as the upload decodes them: two
         * big-endian halves per word, the first in the low 16 bits. */
        unsigned bad = 0, first = ~0u; u32 want = 0;
        s->guest_dirty = 1;
        if (eng_surface_to_guest(slot)) {
            for (u32 y = 0; y < t->height && bad < n; y++) {
                const u8* row = eng_guest_ptr(NULL, s->location, s->offset + y * s->pitch, t->width * 4u);
                if (!row) { bad = n; break; }
                for (u32 x = 0; x < t->width; x++) {
                    const u8* b = row + x * 4u;
                    const u32 v = ((u32)b[0] << 8 | b[1]) | ((u32)b[2] << 8 | b[3]) << 16;
                    if (v != gpu[y * t->width + x]) {
                        if (first == ~0u) { first = y * t->width + x; want = v; }
                        bad++;
                    }
                }
            }
            s_c2r.cmp++; s_c2r.px += bad; if (bad) s_c2r.bad++;
            if (bad && s_c2r.bad <= 4)
                fprintf(stderr, "[c2rg16-check] surface 0x%08X %ux%u: %u texels differ, first %u: gpu 0x%08X guest 0x%08X\n",
                        s->offset, t->width, t->height, bad, first, first < n ? gpu[first] : 0u, want);
        }
        free(gpu);
    }
    return s->rg16[k].tex;
}

static int eng_surface_to_guest_timed(u32 slot)
{
    ENG_ST_BEGIN();
    const int r = eng_surface_to_guest(slot);
    ENG_ST_END(ST_S2G);
    return r;
}

/* Per unit, for the draw being bound: how a depth snapshot bound in place of a smaller
 * RSX texture is read (fp_shs in the shadow helpers, rsx_fp_apply_shadow_scale). The
 * snapshot is the whole zeta; the guest's texture names only its top-left w x h -- a
 * shadow cascade rendered into part of a shared depth target -- and its normalized
 * coordinates span that part. x,y: the part's share of the snapshot; z: 0 none, 1
 * border (outside [0,1] reads w), 2 clamp to edge, 3 repeat; w: the border depth. */
static float s_shs[RSX_BE_MAX_TEXTURES][4];

static void eng_note_shadow_region(u32 u, const rsx_dsp_texture* t, const eng_zdepth* z)
{
    if (!t->width || !t->height || !z->w || !z->h ||
        (t->width >= z->w && t->height >= z->h))
        return;
    const u32 ws = t->wrap & 0xFu;
    const float mode = (ws == 4u || ws == 5u) ? 1.0f : (ws == 1u || ws == 2u) ? 3.0f : 2.0f;
    s_shs[u][0] = (float)t->width / (float)z->w;
    s_shs[u][1] = (float)t->height / (float)z->h;
    s_shs[u][2] = mode;
    s_shs[u][3] = ((t->border_color & 0xFFFFFFu) == 0xFFFFFFu) ? 1.0f : 0.0f;
}

static u32 sink_bind_textures(const u32* target_slots, u32 n_targets,
                              u32 current_zslot, u32 tex_used,
                              u32 textures[RSX_BE_MAX_TEXTURES],
                              rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES])
{
    u32 mask = 0;
    u32 feedback[RSX_BE_MAX_COLOR_TARGETS] = {0};
    memset(s_shs, 0, sizeof s_shs);
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        /* A unit left enabled by an earlier draw but not sampled by this
         * program is skipped: resolving it would cost a cache check, and an
         * aliased surface a store to guest memory, for nothing. */
        if (!t.enabled || !((tex_used >> u) & 1u)) continue;
        mask |= 1u << u;
        eng_decode_sampler(t.filter, t.wrap, t.control0, &samplers[u], t.border_color, t.format);

        int sampled = -1, own = 0, surface_at = -1;
        for (u32 i = 0; i < g.n_surfaces; i++) {
            if (!g.surfaces[i].handle || g.surfaces[i].location != t.location ||
                g.surfaces[i].offset != t.offset)
                continue;
            surface_at = (int)i;
            /* A current attachment requires an ordered immutable copy; it
             * cannot be sampled directly while the host pass writes it. */
            for (u32 k = 0; k < n_targets; k++) if (target_slots[k] == i) own = 1;
            if (own) break;
            sampled = (int)i;
            break;
        }
        const u32 base_fmt = t.format & RSX_TEX_FMT_BASE_MASK & ~(u32)RSX_TEX_FMT_UNNORM;
        /* The depth at this address, when it was written after any colour
         * there: a depth-only pass names its zeta as a colour target too. */
        int zsrc = -1;
        for (u32 i = 0; i < g.n_zdepths; i++) {
            const eng_zdepth* z = &g.zdepths[i];
            if (z->location != t.location || z->offset != t.offset || !z->depth_serial)
                continue;
            if (surface_at < 0 || z->depth_serial > g.surfaces[surface_at].color_serial)
                zsrc = (int)i;
            break;
        }
        /* What the repeats themselves write: the current targets and the bound
         * zeta. Any other surface or depth target is unchanged between them. */
        if (own || (zsrc >= 0 && (u32)zsrc == current_zslot)) s_tex_plain = 0;
        if (s_stats_s > 0) {
            if (zsrc >= 0) s_pc.tp_z++;
            else if (own) s_pc.tp_own++;
            else if (surface_at >= 0) s_pc.tp_surf++;
            else if (base_fmt == RSX_TEX_FMT_DEPTH24_D8) s_pc.tp_depthfmt++;
            else s_pc.tp_plain++;
        }
        if (zsrc >= 0) {
            /* DEPTH24_D8 keeps the backend's own snapshot when it has one;
             * anything else goes through guest memory. */
            if (base_fmt == RSX_TEX_FMT_DEPTH24_D8 && current_zslot != (u32)zsrc) {
                ENG_ST_BEGIN();
                const u32 snap = eng_zdepth_snapshot((u32)zsrc);
                ENG_ST_END(ST_ZSNAP);
                if (snap) {
                    textures[u] = snap;
                    eng_note_shadow_region(u, &t, &g.zdepths[zsrc]);
                    continue;
                }
            }
            if (eng_z2argb_mode() && current_zslot != (u32)zsrc) {
                ENG_ST_BEGIN();
                const u32 a = eng_zdepth_argb8((u32)zsrc, &t, base_fmt, eng_z2argb_mode() == 2);
                ENG_ST_END(ST_Z2G);
                if (a) { textures[u] = a; continue; }
            }
            int z2g;
            eng_note_via_guest(current_zslot == (u32)zsrc ? "z2guest (bound zeta)" : "z2guest", &t,
                               g.zdepths[zsrc].zfmt, g.zdepths[zsrc].w, g.zdepths[zsrc].h);
            { ENG_ST_BEGIN(); z2g = eng_zdepth_to_guest((u32)zsrc); ENG_ST_END(ST_Z2G); }
            if (z2g) {
                textures[u] = eng_texture_slot(t.location, t.offset, t.format,
                                               t.width, t.height, t.mipmaps, t.pitch,
                                               (int)t.cubemap, t.remap & 0xFFFFu,
                                       (t.dimension == 3 && !t.cubemap) ? t.depth : 1u);
                continue;
            }
        }
        if (own && zsrc < 0 && g.be->color_snapshot) {
            eng_surface* s = &g.surfaces[surface_at];
            rsx_dsp_surface sf;
            rsx_dsp_get_surface(&g.rsx, &sf);
            if (!eng_color_feedback_compatible(s, &t, &sf))
                return eng_color_feedback_failure(u, s, &t);
            u32 k = 0;
            while (k < n_targets && target_slots[k] != (u32)surface_at) k++;
            if (k == n_targets) return eng_color_feedback_failure(u, s, &t);
            if (!feedback[k]) {
                ENG_ST_BEGIN();
                const u32 copy = g.be->color_snapshot(g.be->user, s->handle);
                ENG_ST_END(ST_CSNAP);
                if (!copy) return eng_color_feedback_failure(u, s, &t);
                if (s->feedback) g.be->texture_release(g.be->user, s->feedback);
                s->feedback = feedback[k] = copy;
            }
            if (g.be->surface_view) {
                textures[u] = g.be->surface_view(g.be->user, feedback[k],
                                               t.remap & 0xffffu, t.format);
                if (!textures[u]) return eng_color_feedback_failure(u, s, &t);
            } else if ((t.remap & 0xffffu) == 0xaae4u) {
                textures[u] = feedback[k];
            } else {
                return eng_color_feedback_failure(u, s, &t);
            }
            continue;
        }
        if (sampled >= 0 && zsrc < 0 && eng_c2rg16_mode() && base_fmt == 0x9Fu) {
            u32 c;
            { ENG_ST_BEGIN(); c = eng_color_rg16f((u32)sampled, &t, base_fmt); ENG_ST_END(ST_S2G); }
            if (c) { textures[u] = c; continue; }
        }
        if (sampled >= 0 && zsrc < 0 &&
            g.surfaces[sampled].fmt == RSX_BE_FMT_R8G8B8A8 &&
            eng_format_reinterprets_rgba8(base_fmt) &&
            (eng_note_via_guest("s2guest", &t, (u32)g.surfaces[sampled].fmt,
                                g.surfaces[sampled].w, g.surfaces[sampled].h), 1) &&
            eng_surface_to_guest_timed((u32)sampled)) {
            textures[u] = eng_texture_slot(t.location, t.offset, t.format,
                                           t.width, t.height, t.mipmaps, t.pitch,
                                           (int)t.cubemap, t.remap & 0xFFFFu,
                                       (t.dimension == 3 && !t.cubemap) ? t.depth : 1u);
            continue;
        }
        if (sampled >= 0) {
            const u32 view = g.be->surface_view
                ? g.be->surface_view(g.be->user, g.surfaces[sampled].handle,
                                     t.remap & 0xFFFFu, t.format)
                : 0;
            textures[u] = view ? view : g.surfaces[sampled].handle;
            continue;
        }
        if (zsrc < 0) {
            u32 sh;
            { ENG_ST_BEGIN(); sh = eng_shadow_texture(&t, base_fmt); ENG_ST_END(ST_S2G); }
            if (sh) { textures[u] = sh; s_tex_plain = 0; continue; }
        }
        if (base_fmt == RSX_TEX_FMT_DEPTH24_D8) {
            s_tex_plain = 0;
            for (u32 i = 0; i < g.n_zdepths; i++)
                if (g.zdepths[i].location == t.location &&
                    g.zdepths[i].offset == t.offset && current_zslot != i) {
                    ENG_ST_BEGIN();
                    const u32 snap = eng_zdepth_snapshot(i);
                    ENG_ST_END(ST_ZSNAP);
                    if (snap) textures[u] = snap;
                    break;
                }
            if (textures[u]) continue;
        }
        textures[u] = eng_texture_slot(t.location, t.offset, t.format,
                                       t.width, t.height, t.mipmaps, t.pitch,
                                       (int)t.cubemap, t.remap & 0xFFFFu,
                                       (t.dimension == 3 && !t.cubemap) ? t.depth : 1u);
    }
    return mask;
}

static u32 sink_bind_vertex_textures(
    u32 vtex_mask, u32 textures[RSX_BE_MAX_VERTEX_TEXTURES],
    rsx_be_sampler_desc samplers[RSX_BE_MAX_VERTEX_TEXTURES])
{
    u32 bound = 0;
    for (u32 u = 0; u < RSX_BE_MAX_VERTEX_TEXTURES; u++) {
        if (!((vtex_mask >> u) & 1u)) continue;
        rsx_dsp_vertex_texture vt;
        rsx_dsp_get_vertex_texture(&g.rsx, u, &vt);
        eng_decode_sampler(vt.filter, vt.wrap, vt.control0, &samplers[u], 0u, 0u);
        /* A vertex unit has no crossbar, so its remap is the identity. */
        textures[u] = eng_texture_slot(vt.location, vt.offset, vt.format,
                                       vt.width, vt.height, vt.mipmaps,
                                       vt.pitch, 0, 0xAAE4u, 1u);
        if (textures[u]) bound |= 1u << u;
    }
    return bound;
}

/* ---- per-pass capture (diagnostic) --------------------------------------- */

/* RSX_ENGINE_PASS_DUMP=<existing dir>: capture the draws pass by pass, from
 * the first draw RSX_ENGINE_PASS_DUMP_AFTER seconds after the engine's first
 * one or, without that, from the frame after the first presented frame of at
 * least RSX_ENGINE_PASS_DUMP_MINDRAWS guest draws (default 1).
 *
 * A pass is a run of draws and clears into one target set and zeta. When the
 * set changes, each colour target of the pass that ended is read back to
 * <dir>/passNNN_tK.ppm, its zeta to <dir>/passNNN_z.pgm (stretched from its
 * smallest to its largest value), and <dir>/passes.txt gets one line: the
 * targets, the zeta and its depth range, the draw and clear counts, each
 * clear's scissor, and the first draw's state and textures. Each texture
 * names the colour surface (S) and the depth target (Z) at its address with
 * the serial of their last write -- the newer one is what was sampled -- or G
 * for guest memory. Each presented surface goes to <dir>/presentNNN.ppm,
 * named after the pass count, and the capture runs on across presents until
 * RSX_ENGINE_PASS_DUMP_MAX passes (default 64). RSX_ENGINE_PASS_DUMP_DRAWS=1
 * also writes one line per draw to <dir>/draws.txt: its pipeline, colour
 * mask, blend, depth state and the textures it samples.
 *
 * The question it answers is which pass of a multi-pass frame first comes out
 * wrong. */
static struct {
    int state;              /* 0 unread, 1 waiting, 2 capturing, 3 done/off */
    const char* dir;
    u32 min_draws;
    u32 after_s;
    u32 max_passes;
    time_t start;
    FILE* index;
    FILE* draws_log;        /* RSX_ENGINE_PASS_DUMP_DRAWS: one line a draw */
    u32 pass;
    int open;
    u32 n_targets, targets[RSX_BE_MAX_COLOR_TARGETS], zslot;
    u32 draws, clears;
    char first[2048];
} s_pd;

static int pd_open_index(void)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/passes.txt", s_pd.dir);
    s_pd.index = fopen(path, "w");
    if (!s_pd.index) { s_pd.state = 3; return 0; }
    if (getenv("RSX_ENGINE_PASS_DUMP_DRAWS")) {
        snprintf(path, sizeof path, "%s/draws.txt", s_pd.dir);
        s_pd.draws_log = fopen(path, "w");
    }
    s_pd.state = 2;
    s_pd.open = 0;
    return 1;
}

static const char* pd_trigger(void)
{
    static const char* trig; static int tinit;
    if (!tinit) { tinit = 1; trig = getenv("RSX_ENGINE_PASS_DUMP_TRIGGER"); if (trig && !*trig) trig = NULL; }
    return trig;
}

static int pd_capturing(void)
{
    if (s_pd.state == 0) {
        s_pd.dir = getenv("RSX_ENGINE_PASS_DUMP");
        const char* m = getenv("RSX_ENGINE_PASS_DUMP_MINDRAWS");
        s_pd.min_draws = m ? (u32)strtoul(m, NULL, 0) : 1u;
        const char* a = getenv("RSX_ENGINE_PASS_DUMP_AFTER");
        s_pd.after_s = a ? (u32)strtoul(a, NULL, 0) : 0u;
        const char* x = getenv("RSX_ENGINE_PASS_DUMP_MAX");
        s_pd.max_passes = x ? (u32)strtoul(x, NULL, 0) : 64u;
        s_pd.start = time(NULL);
        s_pd.state = (s_pd.dir && *s_pd.dir) ? 1 : 3;
    }
    if (s_pd.state == 1 && s_pd.after_s &&
        difftime(time(NULL), s_pd.start) >= (double)s_pd.after_s && pd_open_index())
        fprintf(s_pd.index, "from a draw of frame %u, %u s in\n", g.frames, s_pd.after_s);
    /* RSX_ENGINE_PASS_DUMP_TRIGGER=<file>: wait until that file exists (checked once a
     * second), delete it and start the capture there -- a frame of a scene the player is
     * standing in, which a script cannot reach. */
    if (s_pd.state == 1 && !s_pd.after_s) {
        static time_t last;
        const char* trig = pd_trigger();
        if (trig) {
            const time_t now = time(NULL);
            if (now != last) {
                last = now;
                FILE* t = fopen(trig, "rb");
                if (t) {
                    fclose(t);
                    remove(trig);
                    if (pd_open_index())
                        fprintf(s_pd.index, "from the trigger file, frame %u\n", g.frames);
                }
            }
        }
        return s_pd.state == 2;
    }
    return s_pd.state == 2;
}

static float pd_half(u16 h)
{
    const u32 e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v;
    if (e == 0)       v = (float)m / 16777216.0f;              /* m * 2^-24 */
    else if (e == 31) v = m ? 0.0f : 65504.0f;
    else {
        v = (1.0f + (float)m / 1024.0f);
        for (u32 i = e; i > 15; i--) v *= 2.0f;
        for (u32 i = e; i < 15; i++) v *= 0.5f;
    }
    return (h & 0x8000u) ? -v : v;
}

static u8 pd_unit(float v)
{
    if (!(v > 0.0f)) return 0;
    if (v >= 1.0f) return 255;
    return (u8)(v * 255.0f + 0.5f);
}

/* One colour surface to a binary PPM, alpha dropped, floats clamped to [0,1]. */
static void pd_write_surface(u32 slot, const char* path)
{
    const eng_surface* s = &g.surfaces[slot];
    u32 bpp;
    switch (s->fmt) {
    case RSX_BE_FMT_R8G8B8A8:      bpp = 4;  break;
    case RSX_BE_FMT_R16G16B16A16F: bpp = 8;  break;
    case RSX_BE_FMT_R32G32B32A32F: bpp = 16; break;
    case RSX_BE_FMT_R32F:          bpp = 4;  break;
    default: return;
    }
    if (!s->handle || !s->w || !s->h || !g.be->readback) return;
    u8* raw = (u8*)malloc((size_t)s->w * s->h * bpp);
    u8* rgb = (u8*)malloc((size_t)s->w * s->h * 3u);
    FILE* f = (raw && rgb) ? fopen(path, "wb") : NULL;
    if (f) {
        memset(raw, 0, (size_t)s->w * s->h * bpp);
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
        g.be->readback(g.be->user, s->handle, 0, 0, s->w, s->h, raw, s->w * bpp);
        for (size_t i = 0; i < (size_t)s->w * s->h; i++) {
            const u8* p = raw + i * bpp;
            u8* o = rgb + i * 3u;
            if (s->fmt == RSX_BE_FMT_R8G8B8A8) {
                o[0] = p[0]; o[1] = p[1]; o[2] = p[2];
            } else if (s->fmt == RSX_BE_FMT_R16G16B16A16F) {
                for (u32 c = 0; c < 3; c++) {
                    u16 h; memcpy(&h, p + c * 2u, 2);
                    o[c] = pd_unit(pd_half(h));
                }
            } else if (s->fmt == RSX_BE_FMT_R32G32B32A32F) {
                for (u32 c = 0; c < 3; c++) {
                    float v; memcpy(&v, p + c * 4u, 4);
                    o[c] = pd_unit(v);
                }
            } else {
                float v; memcpy(&v, p, 4);
                o[0] = o[1] = o[2] = pd_unit(v);
            }
        }
        fprintf(f, "P6\n%u %u\n255\n", s->w, s->h);
        fwrite(rgb, 1, (size_t)s->w * s->h * 3u, f);
        fclose(f);
    }
    free(raw);
    free(rgb);
}

/* A depth target to a greyscale PPM, stretched from its smallest to its
 * largest value so a shallow range still shows; the range goes to the index. */
static void pd_write_depth(u32 zslot, const char* path, float* lo_out, float* hi_out)
{
    const eng_zdepth* z = &g.zdepths[zslot];
    *lo_out = *hi_out = -1.0f;
    if (!g.be->depth_readback || !z->handle) return;
    const u32 w = z->used_w && z->used_w < z->w ? z->used_w : z->w;
    const u32 h = z->used_h && z->used_h < z->h ? z->used_h : z->h;
    float* d = (float*)malloc((size_t)w * h * sizeof(float));
    u8* px = (u8*)malloc((size_t)w * h);
    FILE* f = (d && px) ? fopen(path, "wb") : NULL;
    if (f) {
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
        if (g.be->depth_readback(g.be->user, z->handle, 0, 0, w, h, d, w * 4u) == 0) {
            float lo = 1.0f, hi = 0.0f;
            for (size_t i = 0; i < (size_t)w * h; i++) {
                if (d[i] < lo) lo = d[i];
                if (d[i] > hi) hi = d[i];
            }
            const float span = hi > lo ? hi - lo : 1.0f;
            for (size_t i = 0; i < (size_t)w * h; i++)
                px[i] = (u8)((d[i] - lo) / span * 255.0f);
            fprintf(f, "P5\n%u %u\n255\n", w, h);
            fwrite(px, 1, (size_t)w * h, f);
            *lo_out = lo; *hi_out = hi;
            /* The exact values too, <path>.f32 (w*h little-endian floats): the stretched
             * 8-bit picture cannot be compared against depth bounds. */
            char raw[1100];
            snprintf(raw, sizeof raw, "%s.f32", path);
            FILE* fr = fopen(raw, "wb");
            if (fr) { fwrite(d, sizeof(float), (size_t)w * h, fr); fclose(fr); }
            /* And the stencil, <path>.s8.pgm, one raw byte a pixel. */
            if (g.be->stencil_readback &&
                g.be->stencil_readback(g.be->user, z->handle, 0, 0, w, h, px, w) == 0) {
                snprintf(raw, sizeof raw, "%s.s8.pgm", path);
                if ((fr = fopen(raw, "wb"))) {
                    fprintf(fr, "P5\n%u %u\n255\n", w, h);
                    fwrite(px, 1, (size_t)w * h, fr);
                    fclose(fr);
                }
            }
        }
        fclose(f);
    }
    free(d);
    free(px);
}

static void pd_close_pass(void)
{
    if (!s_pd.open) return;
    char path[1024];
    fprintf(s_pd.index, "pass %03u: draws %u clears %u |", s_pd.pass, s_pd.draws,
            s_pd.clears);
    for (u32 k = 0; k < s_pd.n_targets; k++) {
        const eng_surface* s = &g.surfaces[s_pd.targets[k]];
        fprintf(s_pd.index, " %c 0x%08X fmt %d %ux%u p%u", 'A' + (int)k, s->offset,
                (int)s->fmt, s->w, s->h, s->pitch);
        snprintf(path, sizeof path, "%s/pass%03u_t%u.ppm", s_pd.dir, s_pd.pass, k);
        pd_write_surface(s_pd.targets[k], path);
    }
    if (s_pd.zslot != ENG_INVALID) {
        float lo, hi;
        snprintf(path, sizeof path, "%s/pass%03u_z.pgm", s_pd.dir, s_pd.pass);
        pd_write_depth(s_pd.zslot, path, &lo, &hi);
        const eng_zdepth* z = &g.zdepths[s_pd.zslot];
        fprintf(s_pd.index, " | zeta 0x%08X p%u depth %.9g..%.9g (%ux%u, written %ux%u)",
                z->offset, z->pitch, lo, hi, z->w, z->h, z->used_w, z->used_h);
    }
    fprintf(s_pd.index, " |%s\n", s_pd.first);
    fflush(s_pd.index);
    s_pd.pass++;
    s_pd.open = 0;
    if (s_pd.pass >= s_pd.max_passes) {
        fprintf(s_pd.index, "stopped after %u passes\n", s_pd.pass);
        fclose(s_pd.index);
        s_pd.index = NULL;
        if (s_pd.draws_log) { fclose(s_pd.draws_log); s_pd.draws_log = NULL; }
        s_pd.state = 3;
        fprintf(stderr, "[rsx engine] pass dump: %u passes written to %s\n",
                s_pd.pass, s_pd.dir);
    }
}

/* A draw (or, with rs NULL, a clear) into this target set: close the pass
 * when the set changed, and describe the first draw of a new one. */
static void pd_note(const u32* targets, u32 n_targets, u32 zslot,
                    const rsx_be_render_state* rs, u32 pipeline, u32 tex_mask)
{
    if (!pd_capturing()) return;
    int same = s_pd.open && s_pd.n_targets == n_targets && s_pd.zslot == zslot;
    for (u32 k = 0; same && k < n_targets; k++) same = s_pd.targets[k] == targets[k];
    if (!same) {
        pd_close_pass();
        s_pd.open = 1;
        s_pd.n_targets = n_targets;
        memcpy(s_pd.targets, targets, n_targets * sizeof(u32));
        s_pd.zslot = zslot;
        s_pd.draws = s_pd.clears = 0;
        s_pd.first[0] = 0;
    }
    if (!rs) {
        s_pd.clears++;
        const u32 h = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
        const u32 v = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
        const size_t n = strlen(s_pd.first);
        snprintf(s_pd.first + n, sizeof s_pd.first - n,
                 " [clear after %u draws, scissor %u,%u %ux%u]",
                 s_pd.draws, h & 0xFFFFu, v & 0xFFFFu, h >> 16, v >> 16);
        return;
    }
    if (s_pd.draws_log) {
        fprintf(s_pd.draws_log, "pass %03u draw %u: pipe %u cmask %08X blend %u %X/%X"
                " dt %u dw %u df %X st %u sf %X ref %02X m %02X/%02X op %X/%X/%X cull %u/%X ff %X at %u |",
                s_pd.pass, s_pd.draws, pipeline,
                rs->color_mask, rs->blend_enable, rs->sf_rgb, rs->df_rgb, rs->depth_test, rs->depth_write,
                rs->depth_func, rs->stencil_enable, rs->s_func,
                rsx_dsp_reg(&g.rsx, M_STENCIL_FUNC_REF) & 0xFFu, rs->s_func_mask & 0xFFu, rs->s_write_mask & 0xFFu,
                rs->s_fail, rs->s_zfail, rs->s_zpass, rs->cull_enable, rs->cull_face,
                rs->front_face, rs->alpha_test_enable);
        { const u32 lo = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MIN), hi = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MAX);
          const u32 sh = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL), sv = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
          float flo, fhi; memcpy(&flo, &lo, 4); memcpy(&fhi, &hi, 4);
          fprintf(s_pd.draws_log, " dbt %u %.7f..%.7f sc %u,%u %ux%u |",
                  rs->depth_bounds_test, flo, fhi, sh & 0xFFFFu, sv & 0xFFFFu, sh >> 16, sv >> 16);
          if (rs->stencil_enable)
              fprintf(s_pd.draws_log, " 2side %u back f %X ref %02X m %02X/%02X op %X/%X/%X |",
                      rs->stencil_two_sided, rs->bs_func, rsx_dsp_reg(&g.rsx, 0x0354u) & 0xFFu,
                      rsx_dsp_reg(&g.rsx, 0x0358u) & 0xFFu, rsx_dsp_reg(&g.rsx, 0x034Cu) & 0xFFu,
                      rs->bs_fail, rs->bs_zfail, rs->bs_zpass); }
        for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
            if (!(tex_mask & (1u << u))) continue;
            rsx_dsp_texture t;
            rsx_dsp_get_texture(&g.rsx, u, &t);
            fprintf(s_pd.draws_log, " t%u %u:0x%08X f%02X %ux%u", u, t.location, t.offset,
                    t.format & 0xFFu, t.width, t.height);
            /* Main-memory textures: is anything there? 256 samples across the
             * first w*h*4 bytes, count of non-zero words. */
            if (t.location == 1 && t.width && t.height) {
                const u32 span = t.width * t.height * 4u;
                const u8* p = eng_guest_ptr(NULL, RSX_LOCATION_MAIN, t.offset, span);
                if (p) {
                    u32 nz = 0, rgb = 0;
                    for (u32 s = 0; s < 256; s++) {
                        const u8* q = p + (((u64)s * (span - 4)) / 255u & ~(u64)3);
                        if (q[0] | q[1] | q[2] | q[3]) nz++;
                        if (q[1] | q[2] | q[3]) rgb++;      /* A8R8G8B8: any colour at all */
                    }
                    fprintf(s_pd.draws_log, "(nz %u/256 rgb %u/256 first %02X%02X%02X%02X ea 0x%08X)", nz, rgb,
                            p[0], p[1], p[2], p[3], vm_base ? (u32)(p - vm_base) : 0u);
                    /* One PPM per distinct main-memory texture (A8R8G8B8 as
                     * stored: A,R,G,B bytes; linear layout assumed). */
                    { static u32 dumped[16]; static u32 nd;
                      u32 k = 0; while (k < nd && dumped[k] != t.offset) k++;
                      if (k == nd && nd < 16 && ((t.format & 0x9Fu) == 0x85u)) {
                          dumped[nd++] = t.offset;
                          char path[1024];
                          snprintf(path, sizeof path, "%s/tex_main_%08X_%ux%u.ppm", s_pd.dir, t.offset,
                                   t.width, t.height);
                          FILE* f = fopen(path, "wb");
                          if (f) {
                              fprintf(f, "P6\n%u %u\n255\n", t.width, t.height);
                              for (u32 i = 0; i < t.width * t.height; i++) fwrite(p + i * 4u + 1u, 1, 3, f);
                              fclose(f);
                          }
                      } }
                } else fprintf(s_pd.draws_log, "(unresolved)");
            }
        }
        /* Geometry inputs: viewport, attribute 0 and its first element as
         * floats (when it is a float stream), and the first draws' transform
         * constants 264..267 -- a draw that never writes a pixel usually has
         * one of these zero or garbage. */
        { rsx_dsp_viewport vp; rsx_dsp_get_viewport(&g.rsx, &vp);
          fprintf(s_pd.draws_log, " | win %08X", rsx_dsp_reg(&g.rsx, 0x1D88));
          fprintf(s_pd.draws_log, " | vp %u,%u %ux%u s(%g %g %g) t(%g %g %g)", vp.x, vp.y, vp.w, vp.h,
                  vp.scale[0], vp.scale[1], vp.scale[2], vp.translate[0], vp.translate[1], vp.translate[2]);
          rsx_dsp_vertex_attr a; rsx_dsp_get_vertex_attr(&g.rsx, 0, &a);
          fprintf(s_pd.draws_log, " | a0 type %u size %u stride %u loc %u off 0x%08X base 0x%X/%u",
                  a.type, a.size, a.stride, a.location, a.offset,
                  rsx_dsp_vertex_data_base_offset(&g.rsx), rsx_dsp_vertex_data_base_index(&g.rsx));
          if (a.type && a.size) {
              const u8* p = eng_guest_ptr(NULL, a.location, a.offset + rsx_dsp_vertex_data_base_offset(&g.rsx), 16);
              if (p) fprintf(s_pd.draws_log, " v0 %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X",
                             p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]);
              else fprintf(s_pd.draws_log, " v0 <unresolved>");
              /* Every other enabled attribute: index, type, size, stride, offset and
               * the first element's bytes (a wrong format shows as stretched textures). */
              for (u32 ai = 1; ai < RSX_DSP_NUM_VERTEX_ATTR; ai++) {
                  rsx_dsp_vertex_attr b; rsx_dsp_get_vertex_attr(&g.rsx, ai, &b);
                  if (!b.type || !b.size) continue;
                  fprintf(s_pd.draws_log, " a%u %u/%u/%u@%X", ai, b.type, b.size, b.stride, b.offset);
                  const u8* r = eng_guest_ptr(NULL, b.location,
                                              b.offset + rsx_dsp_vertex_data_base_offset(&g.rsx), 8);
                  if (r) fprintf(s_pd.draws_log, "=%02X%02X%02X%02X%02X%02X%02X%02X",
                                 r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
              }
              /* Float streams: the first four elements as numbers. */
              if (a.type == 2 && a.stride) {
                  const u8* q = eng_guest_ptr(NULL, a.location,
                                              a.offset + rsx_dsp_vertex_data_base_offset(&g.rsx), a.stride * 4u);
                  for (u32 e = 0; q && e < 4; e++) {
                      fprintf(s_pd.draws_log, " [");
                      for (u32 c = 0; c < a.size && c < 4; c++) {
                          const u8* b = q + e * a.stride + c * 4u;
                          const u32 bits = (u32)b[0] << 24 | (u32)b[1] << 16 | (u32)b[2] << 8 | b[3];
                          float f; memcpy(&f, &bits, 4);
                          fprintf(s_pd.draws_log, "%s%.4g", c ? " " : "", f);
                      }
                      fprintf(s_pd.draws_log, "]");
                  }
              }
          }
          if (s_pd.draws < 2)
              for (u32 c = 264; c < 268; c++) {
                  const u32* k = rsx_dsp_constant(&g.rsx, c);
                  float f[4]; memcpy(f, k, 16);
                  fprintf(s_pd.draws_log, " c%u(%g %g %g %g)", c, f[0], f[1], f[2], f[3]);
              } }
        fprintf(s_pd.draws_log, "\n");
    }
    if (s_pd.draws++) return;

    rsx_dsp_surface sf;
    rsx_dsp_viewport vp;
    rsx_dsp_get_surface(&g.rsx, &sf);
    rsx_dsp_get_viewport(&g.rsx, &vp);
    size_t at = strlen(s_pd.first);
    at += (size_t)snprintf(s_pd.first + at, sizeof s_pd.first - at,
        " rt_sel %u cfmt %u zfmt %u clip %ux%u | pipe %u dt %u dw %u df %u cmask %08X"
        " blend %u %X/%X stencil %u vpz %g+%g |",
        sf.color_target, sf.color_format, sf.depth_format, sf.clip_w, sf.clip_h,
        pipeline, rs->depth_test, rs->depth_write, rs->depth_func, rs->color_mask,
        rs->blend_enable, rs->sf_rgb, rs->df_rgb, rs->stencil_enable,
        vp.scale[2], vp.translate[2]);
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES && at < sizeof s_pd.first; u++) {
        if (!(tex_mask & (1u << u))) continue;
        rsx_dsp_texture t;
        rsx_dsp_get_texture(&g.rsx, u, &t);
        /* S<n> a colour surface, Z<n> a depth target, each with the serial
         * of its last write: the newer one is what the binding sampled. */
        char src[48] = "G";
        size_t sl = 0;
        for (u32 i = 0; i < g.n_surfaces; i++)
            if (g.surfaces[i].handle && g.surfaces[i].location == t.location &&
                g.surfaces[i].offset == t.offset) {
                sl = (size_t)snprintf(src, sizeof src, "S%u@%llu", i,
                                      (unsigned long long)g.surfaces[i].color_serial);
                break;
            }
        for (u32 i = 0; i < g.n_zdepths; i++)
            if (g.zdepths[i].location == t.location && g.zdepths[i].offset == t.offset) {
                snprintf(src + sl, sizeof src - sl, "%sZ%u@%llu", sl ? "," : "", i,
                         (unsigned long long)g.zdepths[i].depth_serial);
                break;
            }
        at += (size_t)snprintf(s_pd.first + at, sizeof s_pd.first - at,
            " t%u %u:0x%08X f%02X %ux%u p%u%s r%04X zf%u %s", u, t.location, t.offset,
            t.format & 0xFFu, t.width, t.height, t.pitch, t.cubemap ? " cube" : "",
            t.remap & 0xFFFFu, t.wrap >> 28, src);
    }
}

/* Called on every present: arm on the first frame heavy enough when no delay
 * was asked for, and while capturing, close the pass in progress and write
 * the presented surface. */
/* A 2D transfer that syncs guest memory with an engine surface, in pass order:
 * a copy out of the scene shows where it falls between the passes. */
static int pd_xfer_wanted(void)
{
    return s_pd.state == 2;
}

static void pd_note_xfer(const char* what, u32 location, u32 offset, u32 w, u32 h,
                         u32 surface, u32 nonzero)
{
    if (s_pd.state != 2) return;
    if (surface == ENG_INVALID)
        fprintf(s_pd.index, "  xfer %s %u:0x%08X %ux%u (after %u draws of pass %03u): no surface\n",
                what, location, offset, w, h, s_pd.draws, s_pd.pass);
    else
        fprintf(s_pd.index, "  xfer %s %u:0x%08X %ux%u (after %u draws of pass %03u): surface 0x%08X,"
                            " %u non-zero pixels\n", what, location, offset, w, h, s_pd.draws,
                s_pd.pass, surface, nonzero);
}

/* A guest draw the engine dropped before it reached the backend, and why: a
 * pass missing from the capture may be one whose every draw was dropped. */
static void pd_note_drop(const char* why, u32 prim)
{
    if (!pd_capturing()) return;
    u32 fp_loc = 0;
    const u32 fp_off = rsx_dsp_fragment_program(&g.rsx, &fp_loc);
    fprintf(s_pd.index, "  drop (%s) prim %u verts %u fp %u:0x%08X (after %u draws of pass %03u)\n",
            why, prim, dc.n_verts, fp_loc, fp_off, s_pd.draws, s_pd.pass);
}

/* Draw-relevant methods in FIFO order (fragment program, BEGIN_END, draw
 * calls) into draws.txt while capturing: shows a draw the FIFO carried that
 * never became a draw line. */
void pd_note_method(u32 method, u32 arg)
{
    if (s_pd.state != 2 || !s_pd.draws_log) return;
    fprintf(s_pd.draws_log, "  m %04X = %08X\n", method, arg);
}

static void pd_present(u32 presented)
{
    if (s_pd.state == 0) pd_capturing();
    if (s_pd.state == 1 && !s_pd.after_s && !pd_trigger()) {
        if (g.guest_draws >= s_pd.min_draws && pd_open_index())
            fprintf(s_pd.index, "frame %u (after a frame of %u guest draws)\n",
                    g.frames + 1, g.guest_draws);
        return;
    }
    if (s_pd.state == 2) {
        char path[1024];
        pd_close_pass();
        if (s_pd.state != 2) return;               /* that pass was the last */
        snprintf(path, sizeof path, "%s/present%03u.ppm", s_pd.dir, s_pd.pass);
        pd_write_surface(presented, path);
        fprintf(s_pd.index, "present%03u: surface %u 0x%08X, %u guest draws\n",
                s_pd.pass, presented, g.surfaces[presented].offset, g.guest_draws);
        fflush(s_pd.index);
    }
}

/* Draw timing diagnostics, both off by default:
 * ENG_SLOW_MS=<ms>     log a BEGIN_END group whose draw takes longer than that,
 *                      with its sizes and where the time went.
 * ENG_PHASE_STATS=<s>  every <s> seconds, the engine's total time in methods and
 *                      in ENDs, split by phase (fetch, pipeline, textures, index
 *                      build, backend, rest), and the slowest END. Names what a
 *                      crawling walker is doing when no single draw is slow. */
static int s_slow_ms = -1;
static unsigned long long s_slow_t[5];
static struct {
    unsigned long long t0, method_us, end_us, max_us, phase_us[6], cpu0;
    u32 methods, ends, drawn;
} s_st;
#define ENG_SLOW_MARK(i) do { if (s_slow_ms > 0 || s_stats_s > 0) s_slow_t[i] = eng_now_us(); } while (0)

static void eng_diag_init(void)
{
    if (s_slow_ms >= 0) return;
    const char* e = getenv("ENG_SLOW_MS");
    const char* st = getenv("ENG_PHASE_STATS");
    s_stats_s = st ? atoi(st) : 0;
    s_slow_ms = e ? atoi(e) : 0;
}

static void sink_end_draw(const rsx_dispatch* r);
static void sink_end(void* user, const rsx_dispatch* r)
{
    (void)user;
    eng_diag_init();
    if (s_slow_ms <= 0 && s_stats_s <= 0) { sink_end_draw(r); return; }
    const unsigned long long t0 = eng_now_us();
    memset(s_slow_t, 0, sizeof s_slow_t);
    const u32 n_idx = dc.n_idx, n_arr = dc.n_arr, n_packets = dc.n_packets;
    u32 idx_total = 0, idx_max = 0;
    for (u32 i = 0; i < n_idx && i < ENG_MAX_BATCHES; i++) {
        idx_total += dc.idx[i].count;
        if (dc.idx[i].count > idx_max) idx_max = dc.idx[i].count;
    }
    sink_end_draw(r);
    const unsigned long long t1 = eng_now_us();
    long long d[6];
    unsigned long long prev = t0;
    for (int i = 0; i < 5; i++) {
        d[i] = s_slow_t[i] ? (long long)(s_slow_t[i] - prev) : -1;
        if (s_slow_t[i]) prev = s_slow_t[i];
    }
    d[5] = (long long)(t1 - prev);
    if (s_stats_s > 0) {
        if (!s_st.t0) { s_st.t0 = t0; s_st.cpu0 = eng_thread_cpu_us(); }
        s_st.ends++;
        if (s_slow_t[4]) s_st.drawn++;
        s_st.end_us += t1 - t0;
        if (t1 - t0 > s_st.max_us) s_st.max_us = t1 - t0;
        for (int i = 0; i < 6; i++) if (d[i] > 0) s_st.phase_us[i] += (unsigned long long)d[i];
        if (t1 - s_st.t0 >= (unsigned long long)s_stats_s * 1000000ull) {
            const double wall = (double)(t1 - s_st.t0) / 1000.0;
            fprintf(stderr, "[eng-stats] %.0f ms: methods %u %.0f ms, ends %u (drawn %u) %.0f ms | "
                    "fetch %.0f pipe %.0f tex %.0f idxbuild %.0f be %.0f rest %.0f | max %.1f ms\n",
                    wall, s_st.methods, s_st.method_us / 1000.0, s_st.ends, s_st.drawn,
                    s_st.end_us / 1000.0, s_st.phase_us[0] / 1000.0, s_st.phase_us[1] / 1000.0,
                    s_st.phase_us[2] / 1000.0, s_st.phase_us[3] / 1000.0, s_st.phase_us[4] / 1000.0,
                    s_st.phase_us[5] / 1000.0, s_st.max_us / 1000.0);
            fprintf(stderr, "[eng-stats]   tex:");
            for (int k = 0; k < ST_N; k++)
                fprintf(stderr, " %s %u/%.0f", s_st_sub_name[k], s_st_sub_n[k], s_st_sub_us[k] / 1000.0);
            fprintf(stderr, " (count/ms)\n");
            {   const unsigned long long cpu = eng_thread_cpu_us();
                fprintf(stderr, "[eng-stats]   cpu: walker %.0f ms of %.0f ms, %.2f us per draw end%s\n",
                        (cpu - s_st.cpu0) / 1000.0, wall, s_st.ends ? (double)(cpu - s_st.cpu0) / s_st.ends : 0.0,
                        s_stats_meth > 0 ? " (methods timed)" : "");
                s_st.cpu0 = cpu; }
            fprintf(stderr, "[eng-stats]   pipe: lookups %u creates %u | memo last %u other %u miss %u | "
                    "same as previous draw: key %u fp 0x08E4 %u ftex %u vp 0x1EA0 %u | distinct fp %u frames %u "
                    "pipelines %u | fill same-structure %u | texhash full small %u big %u %.0f MB | "
                    "cache hits %u checked %u differ %u | merge vs previous draw: draws %u state %u +fp %u +vp %u "
                    "(lists %u, nothing else between %u) | repeat: state %u textures %u checked %u differ %u\n",
                    s_pc.lookups, s_pc.creates, s_pc.memo_last, s_pc.memo_other, s_pc.memo_miss,
                    s_pc.same_key, s_pc.same_fp, s_pc.same_ftex, s_pc.same_vp, s_pc.nfp,
                    (u32)(g.frames - s_pc_frames0), g.n_pipelines, s_pc.fill_samestruct,
                    s_pc.full_small, s_pc.full_big, s_pc.full_bytes / 1048576.0,
                    s_pc.l0_hit, s_pc.l0_checked, s_pc.l0_differ,
                    s_pc.mg_draws, s_pc.mg_state, s_pc.mg_state_fp, s_pc.mg_all, s_pc.mg_list, s_pc.mg_quiet,
                    s_pc.rp_state, s_pc.rp_tex, s_pc.rp_checked, s_pc.rp_differ);
            fprintf(stderr, "[eng-stats]   repeat misses: registers %u constants %u | texture units: depth %u own target %u "
                            "surface %u depth format %u plain %u | registers that differed:",
                    s_pc.rp_regmiss, s_pc.rp_constmiss, s_pc.tp_z, s_pc.tp_own, s_pc.tp_surf, s_pc.tp_depthfmt,
                    s_pc.tp_plain);
            for (int top = 0; top < 10; top++) {
                u32 bi = 0, bc = 0;
                for (u32 i = 0; i < 0x2000u / 4u; i++) if (s_rp_hist[i] > bc) { bc = s_rp_hist[i]; bi = i; }
                if (!bc) break;
                fprintf(stderr, " %04X=%u", bi * 4u, bc);
                s_rp_hist[bi] = 0;
            }
            fprintf(stderr, "\n");
            memset(s_rp_hist, 0, sizeof s_rp_hist);
            s_pc_frames0 = g.frames;
            {   const u32 pf = s_pc.prev_fp, pv = s_pc.prev_vp; const u64 pk = s_pc.prev_key;
                u32 pt[16 * 7 + 16]; memcpy(pt, s_pc.prev_ftex, sizeof pt);
                memset(&s_pc, 0, sizeof s_pc);
                s_pc.prev_fp = pf; s_pc.prev_vp = pv; s_pc.prev_key = pk; memcpy(s_pc.prev_ftex, pt, sizeof pt); }
            memset(s_st_sub_us, 0, sizeof s_st_sub_us);
            memset(s_st_sub_n, 0, sizeof s_st_sub_n);
            {   const unsigned long long c0 = s_st.cpu0;
                memset(&s_st, 0, sizeof s_st);
                s_st.cpu0 = c0; }
            s_st.t0 = t1;
        }
    }
    if (s_slow_ms <= 0 || t1 - t0 < (unsigned long long)s_slow_ms * 1000ull) return;
    static u32 logs;
    if (!PS3_LOG_FIRST(logs, 64)) return;
    fprintf(stderr, "[eng-slow] %.1f ms prim=%u packets=%u arr=%u idx=%u (refs %u, max %u) "
            "source_refs=%u verts=%u stride=%u | us: fetch %lld pipe %lld tex %lld idxbuild %lld be %lld rest %lld\n",
            (t1 - t0) / 1000.0, g.rsx.current_primitive, n_packets, n_arr, n_idx, idx_total, idx_max,
            dc.n_source_refs, dc.n_verts, dc.layout.stride, d[0], d[1], d[2], d[3], d[4], d[5]);
}

/* RSX_DRAW_REPEAT=1: ~77% of GoW:A's draws repeat the previous draw's whole
 * state -- registers, vertex constants, fragment program bytes -- with nothing
 * else reaching the engine in between; only the vertex and index data differ.
 * Such a draw reuses what the previous one resolved: render state, targets,
 * surface and viewport, depth slot, texture handles and samplers, and the two
 * constant buffers. Fetch, the index build, the backend's binds and the draw
 * itself still run. Conditions: the NV4097 registers (less the draw and
 * vertex-array-offset words, which none of that reads), the vertex constants,
 * the layout, the frame and the count of non-draw engine events are all
 * unchanged; for the textures and constants, the pipeline lookup also
 * answered from the previous draw (RSX_PIPE_CACHE, which compares the fragment
 * program's bytes), and every texture unit came from the plain cache path,
 * none from a surface or a depth target. =2 resolves everything as before and
 * compares. */
typedef struct {
    rsx_be_render_state rs;
    u32 targets[RSX_BE_MAX_COLOR_TARGETS], n_targets, zslot;
    rsx_dsp_surface sf;
    rsx_dsp_viewport vp;
    u32 pipeline, tex_used, tex_mask, vtex_mask;
    int fixed;
    u64 fp_hash;
    u32 textures[RSX_BE_MAX_TEXTURES], vtextures[RSX_BE_MAX_VERTEX_TEXTURES];
    rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES], vsamplers[RSX_BE_MAX_VERTEX_TEXTURES];
    u32 fp_bytes;
} eng_rp_state;
#define ENG_RP_REGS (0x2000u / 4u)
static struct {
    int valid, tex_ok;
    u32 nd, frames, lmask, lstride, prim, gen;
    u32 regs[ENG_RP_REGS];
    u32 constants[RSX_DSP_NUM_CONSTANTS][4];
    eng_rp_state st;
    u8 vpcb[ENG_VP_CB_BYTES], fpcb[RSX_DSP_NUM_CONSTANTS * 16u];   /* =2 only */
} s_rp;
static int s_rp_mode = -1;
static int eng_draw_repeat(void)
{
    if (s_rp_mode < 0) { const char* e = getenv("RSX_DRAW_REPEAT"); s_rp_mode = e ? atoi(e) : 0; }
    return s_rp_mode;
}
/* The registers the reused state does not read, which change between
 * otherwise identical draws: NOP (0x0100, the title's markers), the vertex
 * array offsets (0x1680-0x16BC), the draw methods and index array
 * (0x1800-0x182C) and the semaphore offset/release words (0x1D6C-0x1D74, one
 * release per draw here). Byte ranges [lo, hi). */
static const u32 s_rp_skip[][2] = {
    { 0x0100u, 0x0104u }, { 0x1680u, 0x16C0u }, { 0x1800u, 0x1830u }, { 0x1D6Cu, 0x1D78u },
};
static int eng_rp_regs_equal(void)
{
    const u8* a = (const u8*)g.rsx.regs; const u8* b = (const u8*)s_rp.regs;
    u32 at = 0;
    for (u32 k = 0; k < sizeof s_rp_skip / sizeof s_rp_skip[0]; k++) {
        if (memcmp(a + at, b + at, s_rp_skip[k][0] - at)) return 0;
        at = s_rp_skip[k][1];
    }
    return !memcmp(a + at, b + at, 0x2000u - at);
}
static int eng_rp_skipped(u32 byte)
{
    for (u32 k = 0; k < sizeof s_rp_skip / sizeof s_rp_skip[0]; k++)
        if (byte >= s_rp_skip[k][0] && byte < s_rp_skip[k][1]) return 1;
    return 0;
}
static void eng_rp_check(const eng_rp_state* now, const eng_rp_state* was, int tex)
{
    const char* what =
        memcmp(&now->rs, &was->rs, sizeof now->rs) ? "render state" :
        now->n_targets != was->n_targets || memcmp(now->targets, was->targets, sizeof now->targets) ? "targets" :
        now->zslot != was->zslot ? "zslot" :
        memcmp(&now->sf, &was->sf, sizeof now->sf) ? "surface" :
        memcmp(&now->vp, &was->vp, sizeof now->vp) ? "viewport" : NULL;
    if (!what && tex)
        what = now->tex_mask != was->tex_mask || memcmp(now->textures, was->textures, sizeof now->textures) ? "textures" :
               memcmp(now->samplers, was->samplers, sizeof now->samplers) ? "samplers" :
               now->vtex_mask != was->vtex_mask || memcmp(now->vtextures, was->vtextures, sizeof now->vtextures) ? "vertex textures" :
               memcmp(now->vsamplers, was->vsamplers, sizeof now->vsamplers) ? "vertex samplers" :
               now->fp_bytes != was->fp_bytes ? "fp constant size" : NULL;
    s_pc.rp_checked++;
    if (what && ++s_pc.rp_differ <= 32)
        fprintf(stderr, "[draw-repeat-check] repeat differs: %s (textures compared %d)\n", what, tex);
}

static void sink_end_draw(const rsx_dispatch* r)
{
    (void)r;
    if (!g.ready) return;
    if (!dc.n_packets) {
        /* A BEGIN/END pair with no draw method the sinks saw: either an empty
         * pair or a draw method routed nowhere. */
        pd_note_drop("no draw packets", g.rsx.current_primitive);
        return;
    }
    if (!dc.fetch_ok) { pd_note_drop("vertex fetch", g.rsx.current_primitive); return; }
    /* Mixed geometry sources have no defined concatenation here. Reject the
     * whole group instead of drawing partial/misordered geometry. */
    if (dc.n_immediate && (dc.n_arr || dc.n_idx || dc.inl_bytes ||
                          r->inline_dropped != dc.inline_dropped_at_begin)) {
        pd_note_drop("mixed sources", g.rsx.current_primitive);
        return;
    }

    const u32 prim = g.rsx.current_primitive;
    /* Everything that becomes triangles -- lists, strips, fans, quads, quad
     * strips and polygons -- is REBUILT into one triangle list through an
     * index buffer. A host strip topology cannot express a restart cut, which
     * is the whole reason the reference engine rebuilds them, and rebuilding
     * is also what lets repeated references share an uploaded vertex. Points
     * and lines pass through as the guest issued them, and anything left that
     * would need an expansion this engine does not have is dropped rather
     * than drawn as something else. */
    rsx_topology topology = RSX_TOPOLOGY_TRIANGLES;
    int scratch = 0;
    const int rebuild = eng_topology_rebuild(prim, 0, &scratch) != 0;
    if (!rebuild) {
        if (rsx_primitive_needs_expansion(prim)) { pd_note_drop("expansion", prim); return; }
        topology = rsx_primitive_topology(prim);
        if (topology == RSX_TOPOLOGY_UNSUPPORTED) { pd_note_drop("topology", prim); return; }
    }

    rsx_vertex_layout_plan layout;
    eng_vertex_layout(&layout);
    dc_fetch(&layout, rebuild);
    ENG_SLOW_MARK(0);
    if (!dc.n_verts || !dc.fetch_ok) { pd_note_drop("vertex fetch", prim); return; }

    int indexed = 0;
    u32 n_draw = dc.n_source_refs;
    if (rebuild) {
        if (!eng_topology_rebuild(prim, dc.refs_remapped, &indexed)) { pd_note_drop("rebuild", prim); return; }
        n_draw = indexed
            ? rsx_draw_engine_topology_index_count(prim, dc.n_source_refs,
                                                   dc.cuts, dc.n_cuts)
            : dc.n_source_refs - dc.n_source_refs % 3u;
    }
    if (!n_draw) { pd_note_drop("no primitives", prim); return; }

    const int rp_mode = eng_draw_repeat();
    const int rp_cheap = rp_mode && s_rp.valid && s_rp.nd == s_nd_events && s_rp.frames == g.frames &&
                         s_rp.lmask == layout.mask && s_rp.lstride == layout.stride && s_rp.prim == prim;
    /* An equal state generation needs no compare; a different one may still
     * be the same bytes (a register set and set back between the draws). */
    const int rp_gen = rp_cheap && g.rsx.state_gen == s_rp.gen;
    const int rp = rp_gen || (rp_cheap && eng_rp_regs_equal() &&
                              !memcmp(s_rp.constants, g.rsx.constants, sizeof s_rp.constants));
    /* =2 also holds the generation to the byte compare it stands for. */
    if (rp_gen && rp_mode == 2 && !(eng_rp_regs_equal() && !memcmp(s_rp.constants, g.rsx.constants,
                                                                 sizeof s_rp.constants)) &&
        ++s_pc.rp_differ <= 32)
        fprintf(stderr, "[draw-repeat-check] state generation equal but registers or constants differ" "\n");
    const int rp_use = rp && rp_mode != 2;
    /* ENG_PHASE_STATS: which registers keep an otherwise repeating draw from
     * being a repeat. */
    if (s_stats_s > 0 && rp_mode == 2 && !rp && s_rp.valid && s_rp.nd == s_nd_events && s_rp.frames == g.frames &&
        s_rp.lmask == layout.mask && s_rp.lstride == layout.stride && s_rp.prim == prim) {
        if (!eng_rp_regs_equal()) {
            s_pc.rp_regmiss++;
            for (u32 i = 0; i < ENG_RP_REGS; i++)
                if (g.rsx.regs[i] != s_rp.regs[i] && !eng_rp_skipped(i * 4u)) s_rp_hist[i]++;
        } else s_pc.rp_constmiss++;
    }
    s_rp.valid = 0;
    eng_rp_state now;
    memset(&now, 0, sizeof now);

    rsx_be_render_state rs;
    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    u32 n_targets;
    rsx_dsp_surface sf;
    rsx_dsp_viewport vp;
    u32 zslot;
    if (rp_use) {
        s_pc.rp_state++;
        memcpy(&rs, &s_rp.st.rs, sizeof rs);
        memcpy(targets, s_rp.st.targets, sizeof targets);
        n_targets = s_rp.st.n_targets;
        memcpy(&sf, &s_rp.st.sf, sizeof sf);
        memcpy(&vp, &s_rp.st.vp, sizeof vp);
        zslot = s_rp.st.zslot;
    } else {
        rsx_draw_engine_decode_render_state(&g.rsx, &rs);
        n_targets = eng_current_target_set(targets);
        if (!n_targets) { pd_note_drop("no target", prim); return; }
        memset(&sf, 0, sizeof sf);
        memset(&vp, 0, sizeof vp);
        rsx_dsp_get_surface(&g.rsx, &sf);
        rsx_dsp_get_viewport(&g.rsx, &vp);
        zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset, sf.clip_w, sf.clip_h);
    }
    const u32 target = targets[0];

    /* A guest program pair that will not translate has no fallback: the draw
     * is dropped, as the reference engine drops it. Substituting the built-in
     * program would draw the geometry in the wrong colours, which is harder to
     * see than a missing object and hides the translation failure. */
    int pipeline_is_fixed = 1;
    u32 tex_used = 0;
    u64 fp_hash = 0;
    if (s_stats_s > 0) {
        const u32 fp = rsx_dsp_reg(&g.rsx, 0x08E4u), vp0 = rsx_dsp_reg(&g.rsx, 0x1EA0u);
        u32 ftex[16 * 7 + 16];
        for (u32 u = 0; u < 16u; u++) {
            for (u32 k = 0; k < 7u; k++) ftex[u * 7u + k] = rsx_dsp_reg(&g.rsx, 0x1A00u + u * 0x20u + k * 4u);
            ftex[16u * 7u + u] = rsx_dsp_reg(&g.rsx, 0x1840u + u * 4u);
        }
        if (fp == s_pc.prev_fp) s_pc.same_fp++;
        if (vp0 == s_pc.prev_vp) s_pc.same_vp++;
        if (!memcmp(ftex, s_pc.prev_ftex, sizeof ftex)) s_pc.same_ftex++;
        s_pc.prev_fp = fp; s_pc.prev_vp = vp0;
        memcpy(s_pc.prev_ftex, ftex, sizeof ftex);
        u32 h = (fp * 2654435761u) >> 23;
        h = (fp * 2654435761u) >> 18;
        while (s_pc.fp_seen[h] && s_pc.fp_seen[h] != fp + 1u) h = (h + 1u) & 16383u;
        if (!s_pc.fp_seen[h] && s_pc.nfp < 12000u) { s_pc.fp_seen[h] = fp + 1u; s_pc.nfp++; }
    }
    ENG_ST_BEGIN();
    const u32 pipeline = eng_pipeline_get(&layout, &rs,
                                          eng_surface_format(sf.color_format),
                                          n_targets, &pipeline_is_fixed, &tex_used,
                                          &fp_hash);
    ENG_ST_END(ST_PGET);
    ENG_SLOW_MARK(1);
    if (!pipeline) { pd_note_drop("pipeline", prim); return; }
    if (g.be->pipeline_ready && !g.be->pipeline_ready(g.be->user, pipeline, (u32)topology)) {
        pd_note_drop("pipeline compiling", prim);
        return;
    }
    if (fp_hash && eng_debug_fp(fp_hash, pipeline)) return;

    /* Textures before any pipeline or target binding: a cache refresh can
     * replace a resource, and the backend is free to submit while doing it. */
    u32 textures[RSX_BE_MAX_TEXTURES];
    rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES];
    u32 vtextures[RSX_BE_MAX_VERTEX_TEXTURES];
    rsx_be_sampler_desc vsamplers[RSX_BE_MAX_VERTEX_TEXTURES];
    memset(textures, 0, sizeof textures);
    memset(samplers, 0, sizeof samplers);
    memset(vtextures, 0, sizeof vtextures);
    memset(vsamplers, 0, sizeof vsamplers);
    const int rp_tex = rp && s_rp.tex_ok && s_pl_last_hit && pipeline == s_rp.st.pipeline;
    u32 tex_mask, vtex_mask;
    if (rp_tex && rp_use) {
        s_pc.rp_tex++;
        memcpy(textures, s_rp.st.textures, sizeof textures);
        memcpy(samplers, s_rp.st.samplers, sizeof samplers);
        memcpy(vtextures, s_rp.st.vtextures, sizeof vtextures);
        memcpy(vsamplers, s_rp.st.vsamplers, sizeof vsamplers);
        tex_mask = s_rp.st.tex_mask;
        vtex_mask = s_rp.st.vtex_mask;
        /* The full path would have looked these entries up again, in the
         * same order and with nothing in between: the cache's eviction order
         * comes out the same without the lookups. */
    } else {
        s_tex_plain = 1;
        tex_mask = sink_bind_textures(targets, n_targets, zslot, tex_used, textures, samplers);
        if (tex_mask == ENG_INVALID) return;
        vtex_mask = sink_bind_vertex_textures(eng_vtex_mask(), vtextures, vsamplers);
    }
    ENG_SLOW_MARK(2);

    /* RSX_SAMP_LOG=1: each distinct (unit, format, wrap, border colour, filter) once -- what the game asks
     * of its samplers, since the backends ignore the border colour. */
    { static int s_sl = -1; static u64 seen[96]; static u32 nseen;
      if (s_sl < 0) s_sl = getenv("RSX_SAMP_LOG") ? 1 : 0;
      if (s_sl) for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
          if (!((tex_mask >> u) & 1u)) continue;
          rsx_dsp_texture t; rsx_dsp_get_texture(&g.rsx, u, &t);
          const u64 key = ((u64)u << 56) ^ ((u64)t.format << 48) ^ ((u64)(t.wrap & 0x000F0F0Fu) << 24)
                        ^ ((u64)t.border_color * 0x9E3779B1ull) ^ ((u64)t.filter << 8);
          u32 k = 0; while (k < nseen && seen[k] != key) k++;
          if (k == nseen && nseen < 96) {
              seen[nseen++] = key;
              fprintf(stderr, "[samp] unit %u fmt 0x%02X wrap s/t/r %u/%u/%u border 0x%08X filter 0x%08X %ux%u\n",
                      u, t.format, t.wrap & 0xFu, (t.wrap >> 8) & 0xFu, (t.wrap >> 16) & 0xFu,
                      t.border_color, t.filter, t.width, t.height);
          }
      } }

    /* A pipeline running the depth bounds test itself (eng_pipeline_get): the
     * bound zeta's snapshot at its unit, the bounds after fp_alpha. Before the
     * pass has written depth there is no snapshot, and the bounds take every
     * value the unit can read. */
    const u32 dbt_unit = pipeline <= ENG_MAX_PIPELINES ? s_pipe_dbt[pipeline] : 0u;
    float dbt_lo = -INFINITY, dbt_hi = INFINITY;
    /* RSX_DBT_STATS=1: every 2000 draws that ask for the test, how many ran it and why the rest did not. */
    { static int s_ds = -1; static unsigned long n_all, n_run, n_nounit, n_nosnap;
      if (s_ds < 0) s_ds = getenv("RSX_DBT_STATS") ? 1 : 0;
      if (s_ds && rs.depth_bounds_test && eng_dbt_emulated()) {
          n_all++;
          if (!dbt_unit) n_nounit++;
          else if (zslot == ENG_INVALID || !eng_zdepth_snapshot(zslot)) n_nosnap++;
          else n_run++;
          if (n_all % 2000 == 0)
              fprintf(stderr, "[dbt-stats] %lu draws: ran %lu, no free unit %lu, no snapshot %lu\n", n_all, n_run, n_nounit, n_nosnap);
      } }
    if (dbt_unit) {
        const u32 u = dbt_unit - 1u;
        const u32 snap = zslot != ENG_INVALID ? eng_zdepth_snapshot(zslot) : 0u;
        if (snap) {
            textures[u] = snap;
            memset(&samplers[u], 0, sizeof samplers[u]);
            tex_mask |= 1u << u;
            const u32 lo = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MIN), hi = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MAX);
            memcpy(&dbt_lo, &lo, sizeof dbt_lo);
            memcpy(&dbt_hi, &hi, sizeof dbt_hi);
        }
        /* RSX_DBT_LOG=<n>: the first n such draws -- scissor, bounds, zeta. */
        { static int s_dl = -1;
          if (s_dl < 0) { const char* e = getenv("RSX_DBT_LOG"); s_dl = e ? atoi(e) : 0; }
          if (s_dl > 0) {
              s_dl--;
              const u32 sch = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL), scv = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
              fprintf(stderr, "[dbt] frame %u pipe %u unit %u verts %u prim %u scissor %u,%u %ux%u bounds %.6f..%.6f zeta %ux%u snap %u fp %016llx\n",
                      (unsigned)g.frames, pipeline, u, n_draw, (unsigned)prim, sch & 0xFFFFu, scv & 0xFFFFu, sch >> 16, scv >> 16,
                      dbt_lo, dbt_hi, zslot != ENG_INVALID ? g.zdepths[zslot].w : 0u,
                      zslot != ENG_INVALID ? g.zdepths[zslot].h : 0u, snap, (unsigned long long)fp_hash);
          } }
    }

    if (indexed) {
        if (!dc_reserve_indices(n_draw)) { pd_note_drop("index space", prim); return; }
        rsx_draw_engine_write_topology_indices(
            prim, dc.n_source_refs, dc.cuts, dc.n_cuts,
            dc.refs_remapped ? dc.ref_remap.occurrence_to_unique : NULL,
            g.indices);
    }

    /* The transform constant block, then the viewport epilogue the vertex
     * program's tail multiplies by: RSX window coordinates mapped into the
     * host's clip space (rsx_live_draw.c:6143-6156). */
    const float W = sf.clip_w ? (float)sf.clip_w : (float)g.width;
    const float H = sf.clip_h ? (float)sf.clip_h : (float)g.height;
    /* The guest viewport is in pixels; an antialiased surface's clip is in
     * samples (aa_sx x aa_sy per pixel), so map NDC over the pixel extent. */
    const float Wv = W / (float)sf.aa_sx;
    const float Hv = H / (float)sf.aa_sy;
    float xf[8] = { 1, 1, 1, 0, 0, 0, 0, 0 };
    if (vp.scale[0] != 0.0f || vp.translate[0] != 0.0f) {
        xf[0] = vp.scale[0] / (Wv * 0.5f);
        xf[1] = -(vp.scale[1] / (Hv * 0.5f));
        xf[2] = vp.scale[2];
        xf[4] = (vp.translate[0] - Wv * 0.5f) / (Wv * 0.5f);
        xf[5] = -((vp.translate[1] - Hv * 0.5f) / (Hv * 0.5f));
        xf[6] = vp.translate[2];
    }
    const u32 nslots = g.fp_constants.count ? g.fp_constants.count : 1u;
    const u32 pipe_shs = pipeline <= ENG_MAX_PIPELINES ? s_pipe_shs[pipeline] : 0u;
    const u32 fp_bytes = (nslots + 1u + (dbt_unit ? 1u : 0u) + (pipe_shs ? 16u : 0u)) * 16u;   /* + fp_dbt, fp_shs */
    int cb_reused = 0;
    if (rp_tex && rp_use && fp_bytes == s_rp.st.fp_bytes) { cb_reused = 1; goto cb_ready; }
    memcpy(g.vp_cb, g.rsx.constants, RSX_DSP_NUM_CONSTANTS * 16u);
    memcpy(g.vp_cb + RSX_DSP_NUM_CONSTANTS * 16u, xf, sizeof xf);

    /* The buffered fragment constants, then fp_alpha: the layout
     * rsx_fp_decompile_buffered_ex compiled the shader against. */
    if (g.fp_cb_cap < fp_bytes) {
        u8* n = (u8*)realloc(g.fp_cb, fp_bytes);
        if (!n) return;
        g.fp_cb = n;
        g.fp_cb_cap = fp_bytes;
    }
    memset(g.fp_cb, 0, fp_bytes);
    if (g.fp_constants.count)
        memcpy(g.fp_cb, g.fp_constants.values, g.fp_constants.count * 16u);
    { float* alpha = (float*)(g.fp_cb + nslots * 16u);
      alpha[0] = rsx_fp_alpha_ref(rs.alpha_ref_raw, rs.alpha_ref_format);
      /* Window convention for WPOS (rsx_fp_decompiler.c, _rsx_wpos): SET_SHADER_WINDOW
       * height in bits 0-11, origin (1 = BOTTOM) in 12-15, pixel centers (1 = INTEGER)
       * in 16-19. */
      const u32 win = rsx_dsp_reg(&g.rsx, 0x1D88);
      alpha[1] = ((win >> 12) & 0xFu) ? (float)(win & 0xFFFu) : 0.0f;
      alpha[2] = ((win >> 16) & 0xFu) ? -0.5f : 0.0f;
      alpha[3] = 0.0f;
      if (dbt_unit) { alpha[4] = dbt_lo; alpha[5] = dbt_hi; }    /* fp_dbt */
      if (pipe_shs) memcpy(alpha + (dbt_unit ? 8 : 4), s_shs, sizeof s_shs); }   /* fp_shs */
cb_ready:
    if (rp_mode) {
        memcpy(&now.rs, &rs, sizeof rs);
        memcpy(now.targets, targets, sizeof now.targets);
        now.n_targets = n_targets; now.zslot = zslot;
        memcpy(&now.sf, &sf, sizeof sf);
        memcpy(&now.vp, &vp, sizeof vp);
        now.pipeline = pipeline; now.tex_used = tex_used; now.fixed = pipeline_is_fixed; now.fp_hash = fp_hash;
        now.tex_mask = tex_mask; now.vtex_mask = vtex_mask;
        memcpy(now.textures, textures, sizeof now.textures);
        memcpy(now.samplers, samplers, sizeof now.samplers);
        memcpy(now.vtextures, vtextures, sizeof now.vtextures);
        memcpy(now.vsamplers, vsamplers, sizeof now.vsamplers);
        now.fp_bytes = fp_bytes;
        if (rp && rp_mode == 2) eng_rp_check(&now, &s_rp.st, rp_tex);
        if (rp && rp_mode == 2 && rp_tex) {
            s_pc.rp_tex++;
            s_pc.rp_checked++;
            if ((memcmp(s_rp.vpcb, g.vp_cb, ENG_VP_CB_BYTES) ||
                 memcmp(s_rp.fpcb, g.fp_cb, fp_bytes < sizeof s_rp.fpcb ? fp_bytes : sizeof s_rp.fpcb)) &&
                ++s_pc.rp_differ <= 32)
                fprintf(stderr, "[draw-repeat-check] repeat differs: constant buffers\n");
        }
        /* A reused set keeps its flag; a resolved one says whether every unit
         * came from the plain cache path. */
        if (!(rp_tex && rp_use)) s_rp.tex_ok = s_tex_plain;
        if (rp_mode == 2) {
            memcpy(s_rp.vpcb, g.vp_cb, ENG_VP_CB_BYTES);
            memcpy(s_rp.fpcb, g.fp_cb, fp_bytes < sizeof s_rp.fpcb ? fp_bytes : sizeof s_rp.fpcb);
        }
        memcpy(&s_rp.st, &now, sizeof now);
        /* A repeat found both equal already (the skipped register words are
         * never compared). */
        /* A repeat already found both equal (the skipped words are never
         * compared). */
        if (!rp_use) {
            memcpy(s_rp.regs, g.rsx.regs, sizeof s_rp.regs);
            memcpy(s_rp.constants, g.rsx.constants, sizeof s_rp.constants);
        }
        s_rp.gen = g.rsx.state_gen;
        s_rp.nd = s_nd_events; s_rp.frames = g.frames;
        s_rp.lmask = layout.mask; s_rp.lstride = layout.stride; s_rp.prim = prim;
        s_rp.valid = 1;
    }

    pd_note(targets, n_targets, zslot, &rs, pipeline, tex_mask);

    /* The depth attachment is cleared the first time it is bound, so a title
     * that never clears its shadow zeta still tests against something. */
    const u32 depth_handle = (zslot != ENG_INVALID) ? g.zdepths[zslot].handle : 0;
    if (zslot != ENG_INVALID && !g.zdepths[zslot].cleared) {
        g.be->clear_depth_stencil(g.be->user, depth_handle,
                                  RSX_BE_CLEAR_DEPTH | RSX_BE_CLEAR_STENCIL,
                                  1.0f, 0);
        g.zdepths[zslot].cleared = 1;
        g.zdepths[zslot].had_write = 0;
    }

    u32 handles[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i < n_targets; i++) handles[i] = g.surfaces[targets[i]].handle;
    g.be->bind_targets(g.be->user, handles, n_targets, depth_handle);
    g.be->bind_pipeline(g.be->user, pipeline);
    /* A repeat's constant buffers are the previous draw's, which the backend
     * still holds: binding them again only copies the same bytes. */
    if (!cb_reused) {
        g.be->bind_vs_constants(g.be->user, g.vp_cb, ENG_VP_CB_BYTES);
        g.be->bind_ps_constants(g.be->user, g.fp_cb, fp_bytes);
    }
    g.be->bind_textures(g.be->user, textures, samplers, tex_mask);
    g.be->bind_vertex_textures(g.be->user, vtextures, vsamplers, vtex_mask);
    g.be->set_viewport(g.be->user, 0.0f, 0.0f, W, H);

    /* The guest scissor intersected with the surface. The nv40 reset is a
     * full 4096x4096 window and a never-written register reads 0 here, which
     * the w == 0 test treats as no scissor, so only a real game scissor
     * narrows anything (rsx_live_draw.c:6180-6197). */
    {
        u32 sx = 0, sy = 0;
        u32 sw = g.surfaces[target].w, sh = g.surfaces[target].h;
        const u32 h = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL);
        const u32 v = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
        /* The scissor is in pixels: scale it to the sample grid. */
        const u32 gx = (h & 0xFFFFu) * sf.aa_sx, gw = (h >> 16) * sf.aa_sx;
        const u32 gy = (v & 0xFFFFu) * sf.aa_sy, gh = (v >> 16) * sf.aa_sy;
        if (gw > 0 && gh > 0) {
            u32 right = sx + sw, bottom = sy + sh;
            if (gx > sx) sx = gx;
            if (gy > sy) sy = gy;
            if (gx + gw < right)  right = gx + gw;
            if (gy + gh < bottom) bottom = gy + gh;
            sw = right > sx ? right - sx : 0;
            sh = bottom > sy ? bottom - sy : 0;
        }
        g.be->set_scissor(g.be->user, sx, sy, sw, sh);
    }
    /* Dynamic, and deliberately outside the pipeline key. */
    g.be->set_stencil_ref(g.be->user,
                          rsx_dsp_reg(&g.rsx, M_STENCIL_FUNC_REF) & 0xFFu);
    /* SET_BLEND_COLOR (0x031C), A8R8G8B8 for 8-bit targets: GoW:A weights its bloom
     * layers with CONSTANT_ALPHA / CONSTANT_COLOR. */
    if (g.be->set_blend_color) {
        const u32 bc = rsx_dsp_reg(&g.rsx, 0x031Cu);
        const float rgba[4] = { (float)((bc >> 16) & 0xFFu) / 255.0f, (float)((bc >> 8) & 0xFFu) / 255.0f,
                                (float)(bc & 0xFFu) / 255.0f, (float)(bc >> 24) / 255.0f };
        g.be->set_blend_color(g.be->user, rgba);
    }
    /* The bounds too: a deferred renderer moves them between its light and
     * sky passes without changing anything else. Raw floats, as written. */
    if (g.be->set_depth_bounds) {
        const u32 lo = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MIN);
        const u32 hi = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MAX);
        float fmin, fmax;
        memcpy(&fmin, &lo, sizeof fmin);
        memcpy(&fmax, &hi, sizeof fmax);
        g.be->set_depth_bounds(g.be->user, fmin, fmax);
    }
    /* SET_POLY_OFFSET_FILL_ENABLE (0x0A68), SCALE_FACTOR (0x0A78) and BIAS
     * (0x0A7C), raw floats: a shadow-map pass pushes its depths back so the
     * lit surfaces do not shadow themselves. Without it GoW:A's sun mask came
     * out as a moire of self-shadowing over every lit surface (the bridge's
     * "dotted" floor). RSX_POLY_OFFSET=0 ignores it (diagnostic). */
    if (g.be->set_depth_bias) {
        static int on = -1;
        if (on < 0) { const char* e = getenv("RSX_POLY_OFFSET"); on = !(e && *e == '0'); }
        float scale = 0.0f, units = 0.0f;
        if (on && (rsx_dsp_reg(&g.rsx, 0x0A68u) & 1u)) {
            const u32 fs = rsx_dsp_reg(&g.rsx, 0x0A78u), fb = rsx_dsp_reg(&g.rsx, 0x0A7Cu);
            memcpy(&scale, &fs, sizeof scale);
            memcpy(&units, &fb, sizeof units);
            if (!(scale == scale) || scale > 1e6f || scale < -1e6f) scale = 0.0f;
            if (!(units == units) || units > 1e7f || units < -1e7f) units = 0.0f;
            static u32 logs; static float last_s = -1e30f, last_u = -1e30f;
            if ((scale != last_s || units != last_u) && PS3_LOG_FIRST(logs, 8)) {
                fprintf(stderr, "[rsx engine] polygon offset: factor %g units %g\n", scale, units);
                last_s = scale; last_u = units;
            }
        }
        g.be->set_depth_bias(g.be->user, scale, units);
    }

    /* ENG_PHASE_STATS: could this draw have joined the previous one in a single
     * host draw? Same targets, pipeline, textures, samplers, viewport, scissor,
     * stencil ref and depth bounds (mg_state); and the fragment constants too
     * (mg_state_fp); and the vertex constants too (mg_all), counted again when
     * both are triangle lists (mg_list). Counters only. */
    if (s_stats_s > 0) {
        static struct {
            u32 handles[RSX_BE_MAX_COLOR_TARGETS], n_targets, depth, pipeline, tex_mask, vtex_mask, W, H;
            u32 textures[RSX_BE_MAX_TEXTURES], vtextures[RSX_BE_MAX_VERTEX_TEXTURES];
            rsx_be_sampler_desc samplers[RSX_BE_MAX_TEXTURES], vsamplers[RSX_BE_MAX_VERTEX_TEXTURES];
            u32 sc_h, sc_v, sref, dbl, dbh, topo, fp_bytes;
        } cur, prev;
        static u8* prev_vp; static u8* prev_fp; static u32 prev_fp_cap; static int have; static u32 prev_nd;
        memset(&cur, 0, sizeof cur);
        memcpy(cur.handles, handles, n_targets * sizeof(u32));
        cur.n_targets = n_targets; cur.depth = depth_handle; cur.pipeline = pipeline;
        cur.tex_mask = tex_mask; cur.vtex_mask = vtex_mask; cur.W = (u32)W; cur.H = (u32)H;
        memcpy(cur.textures, textures, sizeof cur.textures);
        memcpy(cur.vtextures, vtextures, sizeof cur.vtextures);
        memcpy(cur.samplers, samplers, sizeof cur.samplers);
        memcpy(cur.vsamplers, vsamplers, sizeof cur.vsamplers);
        cur.sc_h = rsx_dsp_reg(&g.rsx, M_SCISSOR_HORIZONTAL); cur.sc_v = rsx_dsp_reg(&g.rsx, M_SCISSOR_VERTICAL);
        cur.sref = rsx_dsp_reg(&g.rsx, M_STENCIL_FUNC_REF) & 0xFFu;
        cur.dbl = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MIN); cur.dbh = rsx_dsp_reg(&g.rsx, M_DEPTH_BOUNDS_MAX);
        cur.topo = (u32)topology; cur.fp_bytes = fp_bytes;
        if (!prev_vp) prev_vp = (u8*)malloc(ENG_VP_CB_BYTES);
        s_pc.mg_draws++;
        if (have && prev_vp && !memcmp(&cur, &prev, sizeof cur)) {
            s_pc.mg_state++;
            if (prev_fp && !memcmp(prev_fp, g.fp_cb, fp_bytes)) {
                s_pc.mg_state_fp++;
                if (!memcmp(prev_vp, g.vp_cb, ENG_VP_CB_BYTES)) {
                    s_pc.mg_all++;
                    if (topology == RSX_TOPOLOGY_TRIANGLES) s_pc.mg_list++;
                    if (topology == RSX_TOPOLOGY_TRIANGLES && s_nd_events == prev_nd) s_pc.mg_quiet++;
                }
            }
        }
        memcpy(&prev, &cur, sizeof cur);
        if (prev_vp) memcpy(prev_vp, g.vp_cb, ENG_VP_CB_BYTES);
        if (prev_fp_cap < fp_bytes) { u8* n = (u8*)realloc(prev_fp, fp_bytes); if (n) { prev_fp = n; prev_fp_cap = fp_bytes; } }
        if (prev_fp && prev_fp_cap >= fp_bytes) memcpy(prev_fp, g.fp_cb, fp_bytes);
        have = prev_fp != NULL;
        prev_nd = s_nd_events;
    }
    const u32 uploaded = indexed ? dc.n_verts : n_draw;
    ENG_SLOW_MARK(3);
    int drawn = 0;
    const double cat_t0 = s_cat_f ? eng_cat_ms() : 0.0;
    if (dc.gpu_vtx && uploaded == s_vjob.count && s_pd.state != 2)   /* a pass dump in progress reads dc.verts */
        drawn = g.be->draw_gpu_vertices(g.be->user, topology, &s_vjob, layout.stride,
                                        indexed ? g.indices : NULL, indexed ? n_draw : 0,
                                        eng_vtx_gpu_mode() == 2 ? dc.verts : NULL) == 0;
    if (!drawn && dc.gpu_vtx && eng_vtx_gpu_mode() != 2 &&
        !rsx_vertex_fetch_many(&dc.fetch_plan, dc.refs, dc.n_refs, layout.stride ? dc.verts : NULL))
        return;
    if (!drawn)
        g.be->draw(g.be->user, topology, dc.verts, uploaded, layout.stride,
                   indexed ? g.indices : NULL, indexed ? n_draw : 0);
    if (s_cat_f) eng_cat_note_draw(pipeline, eng_cat_ms() - cat_t0);
    ENG_SLOW_MARK(4);
    /* RSX_DBT_FLUSH=1 (diagnostic): the batch submitted after every draw that runs the emulated depth
     * bounds test -- does the in-batch order around the light draws matter? */
    { static int s_df = -1; if (s_df < 0) s_df = getenv("RSX_DBT_FLUSH") ? atoi(getenv("RSX_DBT_FLUSH")) : 0;
      if (s_df && dbt_unit && g.be->submit_and_wait) g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE); }
    /* RSX_ENGINE_PASS_DUMP_DRAWS: what this draw did to the depth buffer --
     * min/max after it and how many pixels changed. Names the draw that
     * leaves a buffer at one value (a plane in front of the scene). */
    if (s_pd.state == 2 && s_pd.draws_log && zslot != ENG_INVALID && g.be->depth_readback) {
        const eng_zdepth* z = &g.zdepths[zslot];
        static float* s_prev; static size_t s_prev_n; static u32 s_prev_slot = ENG_INVALID;
        const size_t n = (size_t)z->w * z->h;
        float* d = (float*)malloc(n * sizeof(float));
        if (d && g.be->depth_readback(g.be->user, z->handle, 0, 0, z->w, z->h, d, z->w * 4u) == 0) {
            float lo = 1e30f, hi = -1e30f; size_t changed = 0;
            const int cmp = s_prev && s_prev_n == n && s_prev_slot == zslot;
            for (size_t i = 0; i < n; i++) {
                if (d[i] < lo) lo = d[i];
                if (d[i] > hi) hi = d[i];
                if (cmp && d[i] != s_prev[i]) changed++;
            }
            fprintf(s_pd.draws_log, "    after: topo %u verts %u idx %u | z %.6g..%.6g changed %s%zu",
                    (unsigned)topology, uploaded, indexed ? n_draw : 0u, lo, hi, cmp ? "" : "(first) ", changed);
            /* First triangle: guest index order and its framebuffer winding
             * under constants 264..267 (valid for a plain matrix VS only). */
            if (n_draw >= 3 && layout.slot_by_attr[0] >= 0) {
                float sx[3], sy[3];
                fprintf(s_pd.draws_log, " | tri0 refs");
                for (u32 k = 0; k < 3; k++) {
                    const u32 vi = indexed ? g.indices[k] : k;
                    if (indexed && k < dc.n_refs) fprintf(s_pd.draws_log, " %u", dc.refs[k].vertex_id);
                    const float* v = (const float*)((const u8*)dc.verts + (size_t)vi * layout.stride
                                                    + (size_t)layout.slot_by_attr[0] * 16u);
                    float c[4];
                    for (u32 r = 0; r < 4; r++) {
                        c[r] = 0;
                        for (u32 col = 0; col < 4; col++) {
                            float m; memcpy(&m, &rsx_dsp_constant(&g.rsx, 264 + col)[r], 4);
                            c[r] += (col < 3 ? v[col] : 1.0f) * m;
                        }
                    }
                    const float w = c[3] != 0 ? c[3] : 1;
                    sx[k] = (c[0] * xf[0] + w * xf[4]) / w;
                    sy[k] = (c[1] * xf[1] + w * xf[5]) / w;
                }
                const float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
                fprintf(s_pd.draws_log, " ndc (%.2f,%.2f)(%.2f,%.2f)(%.2f,%.2f) area %+.3g",
                        sx[0], sy[0], sx[1], sy[1], sx[2], sy[2], area);
                /* The same three vertices as uploaded (attr0 x,y): the input a
                 * pass-through VS (fullscreen passes) uses as its position. */
                float rx[3], ry[3];
                for (u32 k = 0; k < 3; k++) {
                    const u32 vi = indexed ? g.indices[k] : k;
                    const float* v = (const float*)((const u8*)dc.verts + (size_t)vi * layout.stride
                                                    + (size_t)layout.slot_by_attr[0] * 16u);
                    rx[k] = v[0]; ry[k] = v[1];
                }
                fprintf(s_pd.draws_log, " raw (%.3g,%.3g)(%.3g,%.3g)(%.3g,%.3g) rawarea %+.3g",
                        rx[0], ry[0], rx[1], ry[1], rx[2], ry[2],
                        (rx[1] - rx[0]) * (ry[2] - ry[0]) - (rx[2] - rx[0]) * (ry[1] - ry[0]));
            }
            fprintf(s_pd.draws_log, "\n");
            free(s_prev); s_prev = d; s_prev_n = n; s_prev_slot = zslot; d = NULL;
        }
        free(d);
    }
    /* Counted only when the draw ran the guest's OWN programs, which is what
     * the test hook means: a draw through the built-in pair is a draw, not
     * evidence that the decompile-translate-compile path worked. */
    if (!pipeline_is_fixed) g.guest_draws++;

    if (rs.color_mask) eng_surfaces_wrote(targets, n_targets);
    if (zslot != ENG_INVALID && rs.depth_test && rs.depth_write) {
        g.zdepths[zslot].had_write = 1;
        /* The next reader resolves the depth this draw left: the depth bounds
         * test in the pixel program reads the snapshot between draws of the
         * same pass, and a stale one dropped the lights wherever later draws
         * (Kratos, the props after the first light) had written. */
        g.zdepths[zslot].snapshot_valid = 0;
        eng_zdepth_wrote(zslot, &sf);
    }
}

static void sink_clear(void* user, const rsx_dispatch* r, u32 mask)
{
    s_nd_events++;
    (void)user; (void)r;
    if (!g.ready) return;
    u32 targets[RSX_BE_MAX_COLOR_TARGETS];
    const u32 n_targets = eng_current_target_set(targets);
    if (!n_targets) return;
    if (pd_capturing()) {
        rsx_dsp_surface zs;
        rsx_dsp_get_surface(&g.rsx, &zs);
        pd_note(targets, n_targets,
                eng_zdepth_get(zs.zeta_location, zs.zeta_offset, zs.clip_w, zs.clip_h),
                NULL, 0, 0);
    }

    if (mask & (RSX_CLEAR_COLOR_R | RSX_CLEAR_COLOR_G |
                RSX_CLEAR_COLOR_B | RSX_CLEAR_COLOR_A)) {
        const u32 c = rsx_dsp_clear_color(&g.rsx);
        const float rgba[4] = {
            (float)((c >> 16) & 0xFF) / 255.0f,
            (float)((c >>  8) & 0xFF) / 255.0f,
            (float)( c        & 0xFF) / 255.0f,
            (float)((c >> 24) & 0xFF) / 255.0f,
        };
        /* Every target the set names, not only A: a deferred pass clears its
         * whole G-buffer in one CLEAR_SURFACE, and B, C and D would otherwise
         * keep whatever the last frame left in them. */
        for (u32 i = 0; i < n_targets; i++)
            g.be->clear_color(g.be->user, g.surfaces[targets[i]].handle, rgba);
        eng_surfaces_wrote(targets, n_targets);
    }
    if (mask & (RSX_CLEAR_DEPTH | RSX_CLEAR_STENCIL)) {
        rsx_dsp_surface sf;
        rsx_dsp_get_surface(&g.rsx, &sf);
        const u32 zslot = eng_zdepth_get(sf.zeta_location, sf.zeta_offset,
                                         sf.clip_w, sf.clip_h);
        if (zslot == ENG_INVALID) return;
        u32 flags = 0;
        if (mask & RSX_CLEAR_DEPTH)   flags |= RSX_BE_CLEAR_DEPTH;
        if (mask & RSX_CLEAR_STENCIL) flags |= RSX_BE_CLEAR_STENCIL;
        /* ZSTENCIL_CLEAR_VALUE is laid out as the zeta is: depth24 << 8 |
         * stencil8 for Z24S8, a bare depth16 for Z16 (SURFACE_FORMAT depth
         * field 1). Read as Z24S8, a Z16 surface's 0xFFFF -- its far plane --
         * would clear it to 255/16777215, nearly the near plane, and every
         * LESS or LEQUAL draw into it would then fail. The nv40 reset
         * 0xFFFFFF00 seeded by rsx_dispatch_init clears to 1.0 / 0. */
        const u32 zs = rsx_dsp_reg(&g.rsx, M_ZSTENCIL_CLEAR);
        const int z16 = sf.depth_format == 1u;
        const float depth = !zs ? 1.0f
                          : z16 ? (float)(zs & 0xFFFFu) / 65535.0f
                                : (float)(zs >> 8) / 16777215.0f;
        { static u32 seen[16]; static u32 ns;
          u32 k = 0; while (k < ns && seen[k] != zs) k++;
          if (k == ns && ns < 16) {
              seen[ns++] = zs;
              fprintf(stderr, "[rsx engine] depth clear raw 0x%08X fmt %u -> %.9g (surface fmt reg 0x%08X)\n",
                      zs, sf.depth_format, depth, rsx_dsp_reg(&g.rsx, 0x0208));
          } }
        g.be->clear_depth_stencil(g.be->user, g.zdepths[zslot].handle, flags,
                                  depth, z16 ? 0 : (u8)(zs & 0xFFu));
        g.zdepths[zslot].cleared = 1;
        if (mask & RSX_CLEAR_DEPTH) {
            /* A clear invalidates the older published depth image; the next
             * texture consumer resolves the newly written pass exactly once. */
            g.zdepths[zslot].had_write = 0;
            g.zdepths[zslot].snapshot_valid = 0;
            eng_zdepth_wrote(zslot, &sf);
        }
    }
}

/* Resolve a flip's buffer id to a REGISTERED surface rather than whatever is
 * currently bound: the live target at that instant is often an offscreen
 * shadow or post-process surface, and copying it presents black despite the
 * scene's draws having executed (rsx_live_draw.c:7557-7570). */
static u32 eng_present_surface(u32 buffer_id)
{
    if (buffer_id < 8 && g.display_buffers[buffer_id].valid) {
        const eng_display_buffer* d = &g.display_buffers[buffer_id];
        for (u32 i = 0; i < g.n_surfaces; i++)
            if (g.surfaces[i].handle && g.surfaces[i].location == d->location &&
                g.surfaces[i].offset == d->offset)
                return i;
    }
    /* The fallback below presents whatever target is bound -- mid-frame, often an
     * intermediate one. Say when it happens. */
    { static unsigned n;
      if (n++ < 12 || (n % 1000) == 0) {
          const u32 cur = eng_current_surface();
          const int dv = buffer_id < 8 && g.display_buffers[buffer_id].valid;
          fprintf(stderr, "[present-fallback] #%u buffer %u display(valid=%d off=0x%08X) -> current 0x%08X\n",
                  n, buffer_id, dv, dv ? g.display_buffers[buffer_id].offset : 0u,
                  cur != ENG_INVALID ? g.surfaces[cur].offset : 0u);
      } }
    return eng_current_surface();
}

static void eng_present(u32 buffer_id)
{
    /* GCM_FLIPCOUNT: every 500 calls, how many presents reached the backend. */
    { static unsigned s_calls, s_unready; static int s_dbg = -1;
      if (s_dbg < 0) s_dbg = getenv("GCM_FLIPCOUNT") ? 1 : 0;
      ++s_calls; if (!g.ready) ++s_unready;
      if (s_dbg && (s_calls % 500u) == 0)
          fprintf(stderr, "[rsx engine] present calls=%u unready=%u frames=%u buffer=%u\n",
                  s_calls, s_unready, g.frames, buffer_id); }
    if (!g.ready) return;
    const u32 target = eng_present_surface(buffer_id);
    if (target == ENG_INVALID) {
        /* Was silent: a title whose flips stop resolving to a surface just
         * stops presenting. Say why, a few times. */
        static unsigned s_n;
        if (s_n++ < 8) {
            const int dv = buffer_id < 8 && g.display_buffers[buffer_id].valid;
            fprintf(stderr, "[rsx engine] present skipped: buffer=%u display(valid=%d loc=%u off=%08X) "
                    "surfaces=%u current=none\n", buffer_id, dv,
                    dv ? g.display_buffers[buffer_id].location : 0,
                    dv ? g.display_buffers[buffer_id].offset : 0, g.n_surfaces);
            for (u32 i = 0; i < g.n_surfaces && i < 16; ++i)
                fprintf(stderr, "[rsx engine]   surface %u loc=%u offset=%08X %ux%u handle=%u\n", i,
                        g.surfaces[i].location, g.surfaces[i].offset, g.surfaces[i].w, g.surfaces[i].h,
                        g.surfaces[i].handle);
        }
        return;
    }
    if (getenv("PS3RECOMP_METAL_FRAME_DUMP") && g.frames % 120 == 0) {
        u32 current = eng_current_surface();
        fprintf(stderr, "[rsx capture] frame=%u buffer=%u selected=%u current=%u draws=%u surfaces=%u\n",
                g.frames, buffer_id, target, current, g.guest_draws, g.n_surfaces);
        for (u32 i = 0; i < g.n_surfaces; ++i)
            fprintf(stderr, "[rsx capture] surface %u loc=%u offset=%08X size=%ux%u handle=%u\n",
                    i, g.surfaces[i].location, g.surfaces[i].offset,
                    g.surfaces[i].w, g.surfaces[i].h, g.surfaces[i].handle);
    }
    pd_present(target);
    g.last_present_surface = target;
    g.be->present(g.be->user, g.surfaces[target].handle);
    g.frames++;
    /* How full the engine's fixed tables are, once a minute of frames: a table that only ever grows
     * (colour and depth targets are keyed by guest address and never freed) runs out in a long session. */
    if (g.frames % 3600u == 0)
        fprintf(stderr, "[eng-tables] frame %llu: surfaces %u/%u, zetas %u/%u, textures %u/%u\n",
                (unsigned long long)g.frames, g.n_surfaces, (unsigned)ENG_MAX_SURFACES,
                g.n_zdepths, (unsigned)ENG_MAX_ZDEPTHS, g.n_textures, (unsigned)ENG_MAX_TEXTURES);
    g.last_guest_draws = g.guest_draws;
    g.guest_draws = 0;
}

static void sink_flip(void* user, const rsx_dispatch* r, u32 arg)
{
    s_nd_events++;
    (void)user; (void)r;
    g.last_flip_buffer = arg & 7u;
    eng_present(g.last_flip_buffer);
}

/* ---- public API ---------------------------------------------------------- */

void rsx_draw_engine_set_backend(const rsx_draw_backend* backend)
{
    g.be = backend;
}

void rsx_draw_engine_set_default(int on)
{
    g.default_on = on;
}

int rsx_draw_engine_enabled(void)
{
    static int cached = -1;
    static int env_seen = 0;
    static int env_on = 0;
    if (!env_seen) {
        const char* e = getenv("PS3RECOMP_RSX_ENGINE");
        env_seen = 1;
        if (e && *e) env_on = (strcmp(e, "dispatch") == 0) ? 1 : -1;
    }
    /* The environment is authoritative in both directions; without it the
     * backend's own default decides. Cached after the first backend has
     * registered, so the FIFO walker pays one compare. */
    const int on = env_on > 0 ? 1 : (env_on < 0 ? 0 : g.default_on);
    if (!g.be) return 0;
    if (cached < 0) cached = on;
    return cached;
}

int rsx_draw_engine_init(u32 width, u32 height)
{
    if (!g.be) return -1;
    g.width  = width  ? width  : 1280;
    g.height = height ? height : 720;
    { const char* e = getenv("PS3RECOMP_ENG_TEXHASH_FULL");
      s_texhash_full_always = e && *e && strcmp(e, "0") != 0; }
    g.vp_cb = (u8*)calloc(1, ENG_VP_CB_BYTES);
    if (!g.vp_cb) return -1;
    if (g.be->init && g.be->init(g.be->user, g.width, g.height) != 0) {
        free(g.vp_cb); g.vp_cb = NULL;
        return -1;
    }
    rsx_dispatch_sink sink;
    memset(&sink, 0, sizeof sink);
    sink.clear            = sink_clear;
    sink.begin            = sink_begin;
    sink.end              = sink_end;
    sink.draw_arrays      = sink_draw_arrays;
    sink.draw_index_array = sink_draw_index;
    sink.inline_array     = sink_inline_array;
    sink.immediate_vertex = sink_immediate_vertex;
    sink.flip             = sink_flip;
    rsx_dispatch_init(&g.rsx, &sink);
    g.ready = 1;
    /* Display buffers the title registered before the backend came up. */
    cellGcm_replay_display_buffers_to_engine();
    fprintf(stderr, "[rsx engine] register-file draw engine up (%ux%u)\n",
            g.width, g.height);
    return 0;
}

void rsx_draw_engine_shutdown(void)
{
    if (!g.be) return;
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_SHUTDOWN);
    for (u32 i = 0; i < g.n_textures; i++)
        if (g.textures[i].handle) g.be->texture_release(g.be->user, g.textures[i].handle);
    for (u32 i = 0; i < g.n_pipelines; i++)
        if (g.pipelines[i].handle) g.be->pipeline_release(g.be->user, g.pipelines[i].handle);
    if (s_cat_f) { fclose(s_cat_f); s_cat_f = NULL; }
    s_cat_state = -1;
    memset(s_pipe_entry, 0, sizeof s_pipe_entry);
    for (u32 i = 0; i < g.n_surfaces; i++) {
        if (g.surfaces[i].feedback) g.be->texture_release(g.be->user, g.surfaces[i].feedback);
        if (g.surfaces[i].handle) g.be->color_target_release(g.be->user, g.surfaces[i].handle);
    }
    for (u32 i = 0; i < g.n_zdepths; i++)
        if (g.zdepths[i].handle) g.be->depth_target_release(g.be->user, g.zdepths[i].handle);
    if (g.ready && g.be->shutdown) g.be->shutdown(g.be->user);

    rsx_draw_engine_set_guest_memory(NULL, NULL);
    free(dc.refs); free(dc.cuts); free(dc.verts); free(dc.immediate);
    rsx_vertex_remap_destroy(&dc.ref_remap);
    memset(&dc, 0, sizeof dc);
    free(g.tex_staging); free(g.vp_cb); free(g.fp_cb); free(g.indices);
    const rsx_draw_backend* be = g.be;
    memset(&g, 0, sizeof g);
    g.be = be;
    s_pl.valid = 0;
    s_rp.valid = 0;
}

void rsx_draw_engine_method(u32 method, u32 arg)
{
    if (!g.ready) return;
    /* The subchannel is a binding slot, not an engine selector, and a title
     * whose SPU-built command lists bind NV4097 elsewhere would otherwise
     * store its state in the wrong register bank.  Standard NV4097 methods
     * (< 0x2000) strip the subchannel with & 0x1FFC.  Driver methods
     * (0xE9xx, 0xEBxx) encode the subchannel bits as part of the address;
     * preserve them with the wider mask. */
    u32 m = (method >= 0xE000u) ? (method & 0xFFFCu) : (method & 0x1FFCu);
    /* RSX_ENGINE_FP_LOG=1: each distinct fragment program the FIFO selects,
     * the first time -- whether a title's material blocks are executed at
     * all, independently of any draw following them. */
    if (m == 0x08E4u || m == 0x1808u || m == 0x1824u || m == 0x1814u) {
        void pd_note_method(u32 method, u32 arg);
        pd_note_method(m, arg);
    }
    if (m == 0x08E4u) {
        static int on = -1;
        if (on < 0) on = getenv("RSX_ENGINE_FP_LOG") ? 1 : 0;
        if (on) {
            static u32 seen[4096]; static u32 n;
            u32 k = 0; while (k < n && seen[k] != arg) k++;
            if (k == n && n < 4096) {
                seen[n++] = arg;
                fprintf(stderr, "[rsx engine] fp select #%u 0x%08X (after %u draws this frame)\n",
                        n, arg, g.guest_draws);
            }
        }
    }
    if (s_stats_s > 0) {
        s_st.methods++;
        if (s_stats_meth < 0) { const char* e = getenv("ENG_PHASE_STATS_METHODS"); s_stats_meth = e && e[0] == '1'; }
        if (s_stats_meth) {
            const unsigned long long t0 = eng_now_us();
            rsx_dispatch_method(&g.rsx, m, arg);
            s_st.method_us += eng_now_us() - t0;
            return;
        }
    }
    rsx_dispatch_method(&g.rsx, m, arg);
}

void rsx_draw_engine_set_display_buffer(u32 buffer_id, u32 location, u32 offset,
                                        u32 pitch, u32 width, u32 height)
{
    s_nd_events++;
    if (buffer_id >= 8) return;
    eng_display_buffer* d = &g.display_buffers[buffer_id];
    d->location = location;
    d->offset = offset;
    d->pitch = pitch;
    d->width = width;
    d->height = height;
    d->valid = width && height;
}

void rsx_draw_engine_flush(void)
{
    s_nd_events++;
    if (g.ready && g.be->submit_and_wait)
        g.be->submit_and_wait(g.be->user, RSX_BE_FLUSH_GUEST_REFERENCE);
}

void rsx_draw_engine_present(void)
{
    s_nd_events++;
    /* A host that drives the flip itself gets whichever buffer the guest's
     * own last flip named, and display buffer 0 before there has been one:
     * a runner that both consumes 0xE944 and calls here would otherwise
     * re-present a double-buffered title's OTHER scanout. Presenting twice is
     * harmless either way -- the engine never clears the surface, so a second
     * present just blits the same image again. */
    eng_present(g.last_flip_buffer);
}

void rsx_draw_engine_present_buffer(u32 buffer_id)
{
    s_nd_events++;
    g.last_flip_buffer = buffer_id & 7u;
    eng_present(g.last_flip_buffer);
}

void rsx_draw_engine_note_flip(u32 buffer_id)
{
    s_nd_events++;
    g.last_flip_buffer = buffer_id & 7u;
}

u32 rsx_draw_engine_guest_draws(void)
{
    return g.last_guest_draws;
}

u32 rsx_draw_engine_readback_center(void)
{
    if (!g.ready || !g.be->readback) return 0;
    const u32 slot = g.last_present_surface;
    if (slot >= g.n_surfaces) return 0;
    const eng_surface* s = &g.surfaces[slot];
    if (!s->handle || s->fmt != RSX_BE_FMT_R8G8B8A8 || !s->w || !s->h) return 0;
    u8 px[4] = { 0, 0, 0, 0 };
    g.be->readback(g.be->user, s->handle, s->w / 2u, s->h / 2u, 1, 1, px, 4);
    /* Decoded rows are R,G,B,A; the guest's own clear colour is A8R8G8B8, and
     * a host comparing the two wants them in the same order. */
    return ((u32)px[3] << 24) | ((u32)px[0] << 16) |
           ((u32)px[1] << 8)  |  (u32)px[2];
}
