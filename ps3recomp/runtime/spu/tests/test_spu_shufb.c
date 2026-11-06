/* spu_shufb (pshufb / NEON tbl path) must match the byte-loop reference for
 * every selector class: a/b sources, 0x00 / 0xFF / 0x80 specials. Build with
 * -msse4.1 on x86 so the SIMD version is the one under test (NEON is always
 * on for arm64). */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
#if !defined(__SSE4_1__) && !defined(__ARM_NEON)
    puts("test_spu_shufb: built without SSE4.1 or NEON, nothing to compare");
    return 0;
#else
    srand(12345);
    for (int it = 0; it < 200000; it++) {
        u128 a, b, c;
        for (int i = 0; i < 16; i++) {
            a._u8[i] = (uint8_t)rand(); b._u8[i] = (uint8_t)rand();
            c._u8[i] = (uint8_t)rand();          /* covers all 256 selector values */
        }
        u128 x = spu_shufb(a, b, c), y = spu_shufb_ref(a, b, c);
        if (memcmp(&x, &y, 16)) {
            printf("test_spu_shufb: MISMATCH at iteration %d\n", it);
            return 1;
        }
    }
    puts("test_spu_shufb: ok (200000 random cases)");
    /* quadword byte rotates/shifts (pshufb path) vs the byte loops, every
     * shift count incl. out-of-range bits of the count operand */
    for (int it = 0; it < 4000; it++) {
        u128 a, b;
        for (int i = 0; i < 16; i++) { a._u8[i] = (uint8_t)rand(); b._u8[i] = (uint8_t)rand(); }
        for (int sh = -300; sh < 300; sh++) {
            b._u32[0] = (uint32_t)sh;
            u128 x[9] = { spu_shlqbyi(a, sh), spu_rotqbyi(a, sh), spu_shlqby(a, b), spu_rotqby(a, b),
                          spu_shlqbybi(a, b), spu_rotqbybi(a, b), spu_rotqmby(a, b), spu_rotqmbybi(a, b),
                          spu_rotqmbyi(a, sh) };
            u128 y[9] = { spu_shlqbyi_ref(a, sh), spu_rotqbyi_ref(a, sh), spu_shlqby_ref(a, b), spu_rotqby_ref(a, b),
                          spu_shlqbybi_ref(a, b), spu_rotqbybi_ref(a, b), spu_rotqmby_ref(a, b), spu_rotqmbybi_ref(a, b),
                          spu_rotqmbyi_ref(a, sh) };
            for (int k = 0; k < 9; k++)
                if (memcmp(&x[k], &y[k], 16)) {
                    printf("test_spu_shufb: qby op %d MISMATCH sh=%d\n", k, sh);
                    return 1;
                }
        }
    }
    puts("test_spu_shufb: qby ok (9 ops x 600 counts x 4000 values)");
    /* fi has one (exact integer) implementation now: no fast/ref pair to compare. */
    return 0;
#endif
}
