/* SPU_FP_NATIVE with SPU_FP_DAZ on x86 (AVX2+FMA): the native fa/fs/fm/fma/fms/fnms under
 * MXCSR round-toward-zero + FTZ + DAZ give the exact per-lane results (spu__*_lane) for
 * random words and for words built around the edge exponents (0, 1, 2, 126-128, 253-255).
 * Build: gcc -O2 -mavx2 -mfma -DSPU_FP_NATIVE=1 test_spu_fp_daz.c -lm */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
int g_spu_fp_check = 0;
void spu_fp_check(int op, u128 a, u128 b, u128 c, u128 r) { (void)op; (void)a; (void)b; (void)c; (void)r; }
static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)s; }
static uint32_t word(void) {
    static const int ex[] = {0, 0, 1, 2, 63, 64, 100, 126, 127, 128, 150, 190, 252, 253, 254, 255, 255};
    const uint32_t r = rnd();
    switch (r & 3) {
    case 0: return rnd();                                                       /* any word */
    case 1: return (r & 0x80000000u) | ((uint32_t)ex[(r >> 8) % 17] << 23) | (rnd() & 0x7FFFFFu);
    case 2: return (r & 0x80000000u) | ((uint32_t)ex[(r >> 8) % 17] << 23) | ((r >> 4) & 1 ? 0 : (rnd() & 0x7FFFFFu)); /* exact powers */
    default: return (r & 0x80000000u) | ((uint32_t)(r >> 8) % 3 ? ((uint32_t)(60 + (rnd() % 8)) << 23) | (rnd() & 0x7FFFFFu) : 0);
    }                                                                           /* small: underflow in fm */
}
int main(void) {
    _mm_setcsr(_mm_getcsr() | 0x6000u | 0x8040u);
    long bad = 0, n = 0;
    for (long it = 0; it < 20000000; it++) {
        u128 a, b, c;
        for (int i = 0; i < 4; i++) { a._u32[i] = word(); b._u32[i] = word(); c._u32[i] = word(); }
        /* fma with c near a*b's negation: cancellation */
        if ((it & 7) == 0) for (int i = 0; i < 4; i++) c._u32[i] = spu__fm_lane(a._u32[i], b._u32[i]) ^ 0x80000000u ^ (rnd() & 3);
        const u128 r[6] = { spu_fa(a, b), spu_fs(a, b), spu_fm(a, b), spu_fma(a, b, c), spu_fms(a, b, c), spu_fnms(a, b, c) };
        for (int i = 0; i < 4; i++) {
            const uint32_t x = a._u32[i], y = b._u32[i], z = c._u32[i];
            const uint32_t e[6] = { spu__fa_lane(x, y), spu__fa_lane(x, y ^ 0x80000000u), spu__fm_lane(x, y),
                                    spu__fma_lane(x, y, z, 0, 0), spu__fma_lane(x, y, z, 0, 0x80000000u),
                                    spu__fma_lane(x, y, z, 0x80000000u, 0) };
            for (int k = 0; k < 6; k++, n++)
                if (r[k]._u32[i] != e[k] && bad++ < 10)
                    printf("op %d: %08X %08X %08X -> %08X, exact %08X\n", k, x, y, z, r[k]._u32[i], e[k]);
        }
    }
    printf("%s (%ld lane results, %ld differ)\n", bad ? "FAIL" : "ok", n, bad);
    return bad != 0;
}
