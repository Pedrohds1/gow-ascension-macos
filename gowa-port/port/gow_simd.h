/* gow_simd.h -- the x86 intrinsics the native EDGE loops are written in, on any host.
 *
 * x86-64 takes <immintrin.h> as before. Elsewhere (arm64) SIMDe implements the same
 * AVX2/FMA/SSE4 intrinsics on NEON, under their own names: a 256-bit op becomes two
 * 128-bit ones, and the float ops follow FPCR, which spu_run_with_halt sets to round
 * toward zero with flush-to-zero (SPU_FP_NATIVE) as it sets MXCSR on x86, so the
 * natives round as they did there. __rdtsc (the check modes' timers) reads the
 * generic timer. */
#ifndef GOW_SIMD_H
#define GOW_SIMD_H

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#else
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <stdint.h>
/* SIMDe 0.8's fmsub/fnmsub/fnmadd are not one fused operation on NEON: under the
 * SPU's round-toward-zero they missed FMA3's single rounding in ~91% of 2 million
 * random vectors (and the A880/6AC8/D4DC natives then wrote 1-ulp-off vertices).
 * Negating an operand is exact, so each is fmadd with a sign flip: 0 differences. */
#define GOW__NEG_PS(x) _mm256_xor_ps((x), _mm256_set1_ps(-0.0f))
#undef _mm256_fmsub_ps
#undef _mm256_fnmsub_ps
#undef _mm256_fnmadd_ps
#define _mm256_fmsub_ps(a, b, c)  _mm256_fmadd_ps((a), (b), GOW__NEG_PS(c))             /* a*b - c */
#define _mm256_fnmsub_ps(a, b, c) _mm256_fmadd_ps(GOW__NEG_PS(a), (b), GOW__NEG_PS(c))  /* -(a*b) - c */
#define _mm256_fnmadd_ps(a, b, c) _mm256_fmadd_ps(GOW__NEG_PS(a), (b), (c))             /* c - a*b */
static inline uint64_t __rdtsc(void)
{
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
/* BMI2 pext, which SIMDe does not cover. The natives only ever pass 0x5555 (the
 * even bits of a 16-bit mask), which folds to four shift-and-mask steps. */
static inline uint32_t _pext_u32(uint32_t src, uint32_t mask)
{
    if (__builtin_constant_p(mask) && mask == 0x5555u) {
        uint32_t x = src & 0x5555u;
        x = (x | (x >> 1)) & 0x3333u;
        x = (x | (x >> 2)) & 0x0F0Fu;
        return (x | (x >> 4)) & 0x00FFu;
    }
    uint32_t r = 0, bit = 1;
    for (; mask; mask &= mask - 1, bit <<= 1)
        if (src & mask & (0u - mask)) r |= bit;
    return r;
}
#endif

#endif /* GOW_SIMD_H */
