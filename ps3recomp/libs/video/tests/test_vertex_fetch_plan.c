/*
 * rsx_vertex_fetch_plan_prepare against the per-attribute walk it replaced.
 *
 * _prepare finds the span of ordinary per-vertex attributes (frequency 0/1)
 * once per draw instead of once per attribute. Random fetch plans -- every
 * vertex type, strides and offsets, frequency dividers in both modes, guest
 * memory that is partly unmapped, INLINE_ARRAY streams -- with random index
 * references, some past the 20-bit element wrap: every attribute's stable
 * range (base, first, last) must be what the old walk gave, and
 * rsx_vertex_fetch_many must produce exactly the bytes and the result of
 * rsx_vertex_fetch_one over the same references.
 *
 * fetch_many's x86 path again, bit for bit, on what it is most likely to get
 * wrong: consecutive-reference draws (RSX_REMAP_RANGE's), NaN payloads,
 * signalling NaNs, denormals, -0 and half specials, and every stable range
 * copied to the very end of a buffer followed by an unmapped page (POSIX),
 * so a load past the bytes _prepare had mapped faults. Also: F16C against
 * the HALF decode on all 65536 inputs, and a rough ns/vertex figure.
 *
 * Build (from the repo root):
 *   clang -std=gnu17 -O2 -Wall -Wextra -Wno-unused-parameter -I include -I libs/video \
 *     -o /tmp/test_vertex_fetch_plan libs/video/tests/test_vertex_fetch_plan.c \
 *     libs/video/rsx_vertex_compact.c libs/video/rsx_dispatch.c
 */
#include "rsx_vertex_compact.h"
#include "rsx_vertex_formats.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define GUARD_PAGES 1
#endif
#if defined(__F16C__)
#include <immintrin.h>
#endif

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails <= 20) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)rng;
}

enum { GUEST = 1 << 16 };
static uint8_t guest[2][GUEST];         /* two locations */
static uint32_t mapped_end[2];          /* bytes past this are unmapped */

static const u8* guest_ptr(void* user, u32 location, u32 offset, u32 min_bytes)
{
    (void)user;
    const u32 loc = location & 1u;
    if ((u64)offset + min_bytes > mapped_end[loc]) return NULL;
    return guest[loc] + offset;
}

/* The walk _prepare used to do: every reference, once per attribute. */
static void ref_prepare(rsx_vertex_fetch_plan* plan, const rsx_vertex_ref* refs, u32 count)
{
    for (u32 slot = 0; slot < plan->layout.count; slot++) {
        const u32 attr = plan->layout.attrs[slot];
        rsx_vertex_fetch_attr* fetch = &plan->attr[attr];
        fetch->stable_base = NULL;
        fetch->stable_first = 0;
        fetch->stable_last = 0;
        if (!refs || !count || !fetch->desc.type || !fetch->desc.size || !fetch->elem_size)
            continue;
        if (plan->inline_data) {
            const u32 stride = plan->inline_stride ? plan->inline_stride : fetch->stride;
            const u32 off = plan->inline_off[attr];
            if (!stride || off + fetch->elem_size > stride)
                continue;
            const u32 verts = plan->inline_bytes / stride;
            if (!verts)
                continue;
            fetch->stride = stride;
            fetch->stable_base = plan->inline_data + off;
            fetch->stable_first = 0;
            fetch->stable_last = verts - 1;
            continue;
        }
        if (!plan->guest_ptr)
            continue;
        u32 first = UINT_MAX, last = 0;
        for (u32 i = 0; i < count; i++) {
            const u32 element = rsx_vertex_element_index(
                refs[i].vertex_id, refs[i].base_index, fetch->desc.frequency,
                (plan->divider_mask >> attr) & 1u);
            if (element < first) first = element;
            if (element > last) last = element;
        }
        const u64 prefix = (u64)plan->base_offset + (u64)fetch->desc.offset;
        const u64 start = prefix + (u64)first * fetch->stride;
        const u64 end = prefix + (u64)last * fetch->stride + fetch->elem_size;
        if (start > UINT_MAX || end > 0x100000000ull || end <= start || end - start > UINT_MAX)
            continue;
        const u8* base = plan->guest_ptr(plan->guest_user, fetch->desc.location,
                                         (u32)start, (u32)(end - start));
        if (!base)
            continue;
        fetch->stable_base = base;
        fetch->stable_first = first;
        fetch->stable_last = last;
    }
}

static void random_plan(rsx_vertex_fetch_plan* plan, int use_inline,
                        const u8* inl, u32 inl_bytes)
{
    memset(plan, 0, sizeof *plan);
    rsx_vertex_layout_plan_init(&plan->layout, rnd() & 0xFFFFu);
    plan->base_offset = (rnd() % 4 == 0) ? rnd() % 256u : 0;
    plan->divider_mask = rnd() & 0xFFFFu;
    plan->guest_ptr = guest_ptr;
    u32 inline_packed = 0;
    for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++) {
        rsx_vertex_fetch_attr* f = &plan->attr[a];
        const u32 r = rnd() % 16;
        f->desc.type = r < 6 ? RSX_VTX_TYPE_FLOAT : r < 7 ? 0 : 1 + rnd() % 7;
        f->desc.size = (rnd() % 8 == 0) ? rnd() % 6 : 1 + rnd() % 4;
        f->elem_size = rsx_vertex_attrib_size(f->desc.type, f->desc.size);
        f->desc.stride = (rnd() % 4 == 0) ? 0 : f->elem_size + (rnd() % 3) * 4u;
        f->stride = f->desc.stride ? f->desc.stride : f->elem_size;
        f->desc.frequency = (rnd() % 4 == 0) ? rnd() % 4 : 0;
        f->desc.offset = rnd() % 4096u;
        f->desc.location = rnd() & 1u;
    }
    if (use_inline) {
        for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++) {
            plan->inline_off[a] = inline_packed;
            if (plan->attr[a].desc.type && plan->attr[a].desc.size)
                inline_packed += plan->attr[a].elem_size;
        }
        plan->inline_stride = inline_packed ? inline_packed : 16u;
        plan->inline_data = inl;
        plan->inline_bytes = inl_bytes;
    }
}

/* rsx_vertex_remap_build against a plain first-use dedup, on both lookups
 * (RSX_REMAP_FAST on and off), each through one long-lived remap object so
 * that stale direct-table entries and a table grown by an earlier big draw
 * are exercised: draws on one base_index over a compact id range (the direct
 * table), mixed base_index, wide and extreme id ranges, big then small. */
static int remap_cases, remap_dense_like, remap_range_taken;
static void test_remap(void)
{
    enum { CASES = 3000, MAXN = 6000 };
    static rsx_vertex_ref src[MAXN], fast_refs[MAXN], slow_refs[MAXN], uniq[MAXN];
    static u32 occ_ref[MAXN];
    rsx_vertex_remap fast_remap, slow_remap;
    memset(&fast_remap, 0, sizeof fast_remap);
    memset(&slow_remap, 0, sizeof slow_remap);
    for (int c = 0; c < CASES; c++) {
        const u32 kind = rnd() % 6;
        const u32 count = (c % 97 == 0) ? MAXN - rnd() % 100 : 1 + rnd() % ((rnd() % 4) ? 400 : 3000);
        const u32 lo = kind == 4 ? 0xFFFFFFFFu - rnd() % 64 : (rnd() % 3 ? rnd() % 70000 : rnd());
        const u32 width = 1 + rnd() % (kind == 0 ? 8 : kind == 1 ? count + 1 : 3 * count + 2000);
        const u32 base = rnd() % 3 ? 0 : rnd();
        for (u32 i = 0; i < count; i++) {
            u32 id;
            switch (kind) {
            case 2:  id = rnd(); break;                                   /* wide */
            case 4:  id = lo + rnd() % (0xFFFFFFFFu - lo + 1u); break;    /* top of u32 */
            case 5:  id = (i & 1) ? lo + rnd() % width : rnd() % 2 ? 0 : 0xFFFFFFFFu; break;
            default: id = lo + rnd() % width; break;
            }
            src[i].vertex_id = id;
            src[i].base_index = (kind == 3 && rnd() % 5 == 0) ? base + 1 + rnd() % 2 : base;
        }
        /* The reference: first use order, by linear search. */
        u32 n_uniq = 0;
        for (u32 i = 0; i < count; i++) {
            u32 k = 0;
            while (k < n_uniq && (uniq[k].vertex_id != src[i].vertex_id ||
                                  uniq[k].base_index != src[i].base_index)) k++;
            if (k == n_uniq) uniq[n_uniq++] = src[i];
            occ_ref[i] = k;
        }
        memcpy(fast_refs, src, count * sizeof *src);
        memcpy(slow_refs, src, count * sizeof *src);
        u32 u_fast = 0, u_slow = 0;
        rsx_vertex_remap_set_fast(1);
        const int ok_fast = rsx_vertex_remap_build(&fast_remap, fast_refs, count, &u_fast);
        rsx_vertex_remap_set_fast(0);
        const int ok_slow = rsx_vertex_remap_build(&slow_remap, slow_refs, count, &u_slow);
        CHECK(ok_fast && ok_slow, "remap case %d: build failed (%d, %d)", c, ok_fast, ok_slow);
        CHECK(u_fast == n_uniq && u_slow == n_uniq,
              "remap case %d kind %u: unique fast %u slow %u, reference %u", c, kind, u_fast, u_slow, n_uniq);
        CHECK(!memcmp(fast_refs, uniq, n_uniq * sizeof *uniq) &&
              !memcmp(slow_refs, uniq, n_uniq * sizeof *uniq),
              "remap case %d kind %u: unique prefix differs", c, kind);
        int occ_ok = 1;
        for (u32 i = 0; i < count; i++)
            occ_ok &= rsx_vertex_remap_index(&fast_remap, i) == occ_ref[i] &&
                      rsx_vertex_remap_index(&slow_remap, i) == occ_ref[i];
        CHECK(occ_ok, "remap case %d kind %u: occurrence map differs", c, kind);
        /* RSX_REMAP_RANGE: any unique order, but every occurrence must name its own vertex. */
        memcpy(fast_refs, src, count * sizeof *src);
        u32 u_range = 0;
        rsx_vertex_remap_set_fast(1); rsx_vertex_remap_set_range(1);
        const int ok_range = rsx_vertex_remap_build(&fast_remap, fast_refs, count, &u_range);
        rsx_vertex_remap_set_range(0);
        int range_ok = ok_range && u_range <= count;
        for (u32 i = 0; i < count && range_ok; i++) {
            const u32 k = rsx_vertex_remap_index(&fast_remap, i);
            range_ok = k < u_range && fast_refs[k].vertex_id == src[i].vertex_id &&
                       fast_refs[k].base_index == src[i].base_index;
        }
        CHECK(range_ok, "remap case %d kind %u: range mode names another vertex", c, kind);
        /* Callers read "unique == count" as "nothing collapsed" and draw the
         * references in order: then the map must be the identity. */
        int order_ok = 1;
        if (ok_range && u_range == count)
            for (u32 i = 0; i < count; i++) order_ok &= rsx_vertex_remap_index(&fast_remap, i) == i;
        CHECK(order_ok, "remap case %d kind %u: nothing collapsed but the order changed", c, kind);
        if (ok_range && u_range != n_uniq) remap_range_taken++;
        remap_cases++;
        if (kind == 0 || kind == 1) remap_dense_like++;
    }
    rsx_vertex_remap_set_fast(-1);
    rsx_vertex_remap_destroy(&fast_remap);
    rsx_vertex_remap_destroy(&slow_remap);
}

/* Big-endian words the decoders could get wrong, mixed with random ones:
 * quiet and signalling NaNs (payloads randomised), denormals, +-0, +-inf,
 * half denormals / NaNs / infinities in either half, s16 extremes. */
static void fill_special(u8* p, u32 bytes)
{
    static const u32 words[] = {
        0x7FC00000u, 0x7F800001u, 0xFFBFFFFFu, 0x7FFFFFFFu, 0xFFC00001u,
        0x00000001u, 0x807FFFFFu, 0x00400000u, 0x80000000u, 0x00000000u,
        0x7F800000u, 0xFF800000u, 0x3F800000u, 0x00010001u, 0x83FF0200u,
        0x7C017E00u, 0xFC00FFFFu, 0x7C00FDFFu, 0x7FFF8000u, 0x80017FFEu,
        0xFFFF0000u, 0x00FF7F80u,
    };
    for (u32 i = 0; i + 4 <= bytes; i += 4) {
        u32 w = rnd() % 3 ? words[rnd() % (sizeof words / sizeof words[0])] : rnd();
        if ((w & 0x7F800000u) == 0x7F800000u && (w & 0x007FFFFFu) && rnd() % 2)
            w ^= rnd() & 0x003FFFFFu;
        p[i] = (u8)(w >> 24); p[i + 1] = (u8)(w >> 16);
        p[i + 2] = (u8)(w >> 8); p[i + 3] = (u8)w;
    }
}

/* Every guest_ptr answer is a copy of the bytes asked for, placed so that
 * it ends where an unmapped page begins: _prepare's stable ranges get a slot
 * each, the one-vertex fallback a passing one. The copy is the same for
 * fetch_one and fetch_many, so they must still agree exactly; a read past
 * min_bytes faults. Without mmap, the slots are plain heap (no fault). */
enum { SLOT_BYTES = GUEST, SLOT_TRANSIENT = RSX_DSP_NUM_VERTEX_ATTR,
       SLOT_INLINE = SLOT_TRANSIENT + 1, SLOTS = SLOT_INLINE + 1 };
static u8* slot_end[SLOTS];
static u32 slot_prepare;
static int in_prepare;

static void guard_init(void)
{
    for (u32 s = 0; s < SLOTS; s++) {
#if GUARD_PAGES
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        const size_t data = ((size_t)SLOT_BYTES + page - 1) / page * page;
        u8* m = (u8*)mmap(NULL, data + page, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED || mprotect(m + data, page, PROT_NONE)) {
            printf("FAIL: guard page setup\n");
            exit(1);
        }
        slot_end[s] = m + data;
#else
        slot_end[s] = (u8*)malloc(SLOT_BYTES) + SLOT_BYTES;
#endif
    }
}

static const u8* guarded_ptr(void* user, u32 location, u32 offset, u32 min_bytes)
{
    (void)user;
    const u32 loc = location & 1u;
    if ((u64)offset + min_bytes > mapped_end[loc] || min_bytes > SLOT_BYTES) return NULL;
    const u32 s = in_prepare ? slot_prepare++ % RSX_DSP_NUM_VERTEX_ATTR : SLOT_TRANSIENT;
    u8* dst = slot_end[s] - min_bytes;
    memcpy(dst, guest[loc] + offset, min_bytes);
    return dst;
}

static int exact_plans, exact_consecutive, exact_fast, exact_inline;
static void test_fetch_many_exact(void)
{
    enum { PLANS = 8000, MAXV = 256 };
    static rsx_vertex_ref refs[MAXV];
    static u8 one[MAXV * 256], many[MAXV * 256];
    guard_init();
    fill_special(guest[0], GUEST);
    fill_special(guest[1], GUEST);
    for (int n = 0; n < PLANS; n++) {
        for (int l = 0; l < 2; l++)
            mapped_end[l] = (rnd() % 4 == 0) ? rnd() % GUEST : GUEST;
        rsx_vertex_fetch_plan plan;
        const int use_inline = rnd() % 6 == 0;
        random_plan(&plan, use_inline, NULL, 0);
        plan.guest_ptr = guarded_ptr;
        if (use_inline) {
            /* The stream ends at the guard page; often exactly on a vertex. */
            const u32 istride = plan.inline_stride;
            u32 verts = 1 + rnd() % MAXV;
            if ((u64)verts * istride + istride > SLOT_BYTES)
                verts = SLOT_BYTES / istride - 1;
            const u32 bytes = verts * istride + (rnd() % 2 ? rnd() % istride : 0);
            u8* data = slot_end[SLOT_INLINE] - bytes;
            fill_special(data, bytes);
            plan.inline_data = data;
            plan.inline_bytes = bytes;
            exact_inline++;
        }

        const u32 count = 1 + rnd() % MAXV;
        const u32 kind = rnd() % 5;
        const u32 base = (rnd() % 4 == 0) ? rnd() % 64u : 0;
        /* Some consecutive runs straddle the 20-bit element wrap. */
        const u32 start = (rnd() % 6 == 0) ? 0x000FFFFFu - rnd() % 300u : rnd() % 512u;
        for (u32 i = 0; i < count; i++) {
            u32 id;
            switch (kind) {
            case 0: case 1: id = start + i; break;                   /* RSX_REMAP_RANGE */
            case 2: id = start + count - 1 - i; break;               /* descending */
            case 3: id = start + i + (rnd() % 16 == 0 ? rnd() % 4 : 0); break;
            default: id = start + rnd() % (count + 16); break;
            }
            refs[i].vertex_id = id;
            refs[i].base_index = base;
        }
        if (kind <= 1) exact_consecutive++;

        in_prepare = 1;
        slot_prepare = 0;
        rsx_vertex_fetch_plan_prepare(&plan, refs, count);
        in_prepare = 0;
        for (u32 slot = 0; slot < plan.layout.count; slot++) {
            const rsx_vertex_fetch_attr* f = &plan.attr[plan.layout.attrs[slot]];
            if (f->stable_base && f->desc.type != RSX_VTX_TYPE_HALF &&
                f->desc.type != RSX_VTX_TYPE_CMP32)
                exact_fast++;
        }

        const u32 stride = plan.layout.stride;
        int ok_one = 1;
        memset(one, 0xCD, sizeof one);
        memset(many, 0xCD, sizeof many);
        for (u32 i = 0; i < count && ok_one; i++)
            ok_one = rsx_vertex_fetch_one(&plan, &refs[i], one + (size_t)i * stride);
        const int ok_many = rsx_vertex_fetch_many(&plan, refs, count, many);
        CHECK(ok_one == ok_many, "exact plan %d: fetch_many %d, fetch_one %d", n, ok_many, ok_one);
        CHECK(!ok_one || !memcmp(one, many, (size_t)count * stride),
              "exact plan %d (kind %u): fetch_many vertices differ from fetch_one", n, kind);
        exact_plans++;
    }
}

/* F16C (vcvtph2ps) against the HALF decode on every input. A mismatch is
 * expected -- the decode flushes half denormals to zero and leaves NaNs
 * unquieted -- and is why fetch_many keeps HALF scalar. */
static void report_f16c(void)
{
#if defined(__F16C__)
    int differ = 0, denormal = 0, snan = 0;
    for (u32 h = 0; h < 65536u; h++) {
        const u8 be[2] = { (u8)(h >> 8), (u8)h };
        float ref[4];
        rsx_vertex_decode_element(RSX_VTX_TYPE_HALF, 1, be, ref);
        const float hw = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
        if (memcmp(&ref[0], &hw, sizeof hw)) {
            differ++;
            if (!(h & 0x7C00u)) denormal++;
            else if ((h & 0x7C00u) == 0x7C00u && !(h & 0x0200u)) snan++;
        }
    }
    printf("F16C: differs from the HALF decode on %d of 65536 inputs "
           "(%d denormal, %d signalling NaN): HALF stays scalar\n", differ, denormal, snan);
#else
    printf("F16C: not enabled in this build (no __F16C__), check skipped\n");
#endif
}

/* Rough ns/vertex: 3000 consecutive vertices, FLOAT3 + UNORM8x4 + FLOAT2
 * interleaved 24 bytes apart -- a typical skinned-free mesh draw. */
static u8 bench_mem[3000 * 24];
static const u8* bench_ptr(void* user, u32 location, u32 offset, u32 min_bytes)
{
    (void)user; (void)location;
    return (u64)offset + min_bytes <= sizeof bench_mem ? bench_mem + offset : NULL;
}

static double now_ns(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void bench(void)
{
    enum { VERTS = 3000, REPS = 2000 };
    static rsx_vertex_ref refs[VERTS];
    static u8 out[VERTS * 48];
    static const u32 attrs[3] = { 0, 3, 8 }, types[3] = {
        RSX_VTX_TYPE_FLOAT, RSX_VTX_TYPE_UNORM8, RSX_VTX_TYPE_FLOAT };
    static const u32 sizes[3] = { 3, 4, 2 }, offsets[3] = { 0, 12, 16 };
    rsx_vertex_fetch_plan plan;
    memset(&plan, 0, sizeof plan);
    rsx_vertex_layout_plan_init(&plan.layout, (1u << 0) | (1u << 3) | (1u << 8));
    plan.guest_ptr = bench_ptr;
    for (u32 k = 0; k < 3; k++) {
        rsx_vertex_fetch_attr* f = &plan.attr[attrs[k]];
        f->desc.type = types[k];
        f->desc.size = sizes[k];
        f->desc.stride = 24;
        f->desc.offset = offsets[k];
        f->elem_size = rsx_vertex_attrib_size(types[k], sizes[k]);
        f->stride = 24;
    }
    for (u32 i = 0; i < sizeof bench_mem; i++) bench_mem[i] = (u8)rnd();
    for (u32 i = 0; i < VERTS; i++) { refs[i].vertex_id = i; refs[i].base_index = 0; }
    rsx_vertex_fetch_plan_prepare(&plan, refs, VERTS);
    double best = 1e30;
    for (int round = 0; round < 5; round++) {
        const double t0 = now_ns();
        for (int r = 0; r < REPS / 5; r++)
            rsx_vertex_fetch_many(&plan, refs, VERTS, out);
        const double t = (now_ns() - t0) / ((double)(REPS / 5) * VERTS);
        if (t < best) best = t;
    }
    volatile u8 sink = out[rnd() % sizeof out];
    (void)sink;
    printf("bench: fetch_many FLOAT3+UNORM8x4+FLOAT2, %d consecutive vertices: %.2f ns/vertex\n",
           VERTS, best);
}

int main(void)
{
    test_remap();
    enum { PLANS = 6000, MAXV = 200 };
    static rsx_vertex_ref refs[MAXV];
    static u8 inl[8192];
    int stable = 0, divided = 0, fetched = 0;
    for (u32 i = 0; i < GUEST; i++) { guest[0][i] = (u8)rnd(); guest[1][i] = (u8)rnd(); }
    for (u32 i = 0; i < sizeof inl; i++) inl[i] = (u8)rnd();

    for (int n = 0; n < PLANS; n++) {
        for (int l = 0; l < 2; l++)
            mapped_end[l] = (rnd() % 3 == 0) ? rnd() % GUEST : GUEST;
        rsx_vertex_fetch_plan plan, ref;
        random_plan(&plan, rnd() % 5 == 0, inl, (u32)(rnd() % sizeof inl));

        const u32 count = 1 + rnd() % MAXV;
        const u32 base = (rnd() % 4 == 0) ? rnd() % 64u : 0;
        /* Some runs sit just below the 20-bit element wrap and cross it. */
        const u32 start = (rnd() % 4 == 0) ? 0x000FFFFFu - rnd() % 400u : 0;
        for (u32 i = 0; i < count; i++) {
            refs[i].vertex_id = start + ((rnd() % 8 == 0) ? rnd() : rnd() % 300u);
            refs[i].base_index = base;
        }
        ref = plan;
        rsx_vertex_fetch_plan_prepare(&plan, refs, count);
        ref_prepare(&ref, refs, count);
        /* The batched fetch against one vertex at a time. */
        {
            static u8 one[MAXV * 256], many[MAXV * 256];
            const u32 stride = plan.layout.stride;
            int ok_one = 1;
            memset(one, 0xCD, sizeof one);
            memset(many, 0xCD, sizeof many);
            for (u32 i = 0; i < count && ok_one; i++)
                ok_one = rsx_vertex_fetch_one(&plan, &refs[i], one + (size_t)i * stride);
            const int ok_many = rsx_vertex_fetch_many(&plan, refs, count, many);
            CHECK(ok_one == ok_many, "plan %d: fetch_many %d, fetch_one %d", n, ok_many, ok_one);
            CHECK(!ok_one || !memcmp(one, many, (size_t)count * stride),
                  "plan %d: fetch_many vertices differ from fetch_one", n);
            if (ok_one) fetched++;
        }
        for (u32 at = 0; at < RSX_DSP_NUM_VERTEX_ATTR; at++) {
            CHECK(plan.attr[at].stable_base == ref.attr[at].stable_base &&
                  plan.attr[at].stable_first == ref.attr[at].stable_first &&
                  plan.attr[at].stable_last == ref.attr[at].stable_last &&
                  plan.attr[at].stride == ref.attr[at].stride,
                  "plan %d attr %u: span %u..%u, walk %u..%u", n, at,
                  plan.attr[at].stable_first, plan.attr[at].stable_last,
                  ref.attr[at].stable_first, ref.attr[at].stable_last);
            if (ref.attr[at].stable_base) {
                stable++;
                if (plan.attr[at].desc.frequency > 1) divided++;
            }
        }
    }

    test_fetch_many_exact();
    report_f16c();
    bench();

    printf("test_vertex_fetch_plan: %d plans, %d stable ranges (%d divided), %d fetched, "
           "%d remap cases (%d compact, %d by range), %d exact plans (%d consecutive, "
           "%d inline, %d fast attributes), %d failure(s)\n",
           PLANS, stable, divided, fetched, remap_cases, remap_dense_like, remap_range_taken,
           exact_plans, exact_consecutive, exact_inline, exact_fast, fails);
    if (stable < PLANS || divided < 100 || fetched < PLANS / 2 || remap_dense_like < 500 ||
        remap_range_taken < 100 || exact_consecutive < 2000 || exact_inline < 500 ||
        exact_fast < 5000) {
        printf("FAIL: too few cases exercised a path\n");
        return 1;
    }
    return fails ? 1 : 0;
}
