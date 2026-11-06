/* clang -std=gnu17 -I runtime/spu -I include \
 * runtime/spu/tests/test_spu_numeric_semantics.c -lm -o /tmp/spu-numeric && /tmp/spu-numeric
 * Covers SPU operations used by Yakuza's geometry and audio programs. */
#include "spu_helpers.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
static int passed, failed;
#define CHECK(c) do { if(c) ++passed; else { ++failed; printf("FAIL line %d: %s\n",__LINE__,#c); } } while(0)
/* Fast path vs the exact per-lane reference (spu_*_ref) over operands that mix
 * ordinary values, zeros of both signs, denormals and exponent-255 words. Both are
 * truncation-exact, so any difference is a bug (fma keeps the 1-ulp near1 slack it
 * had under the older round-to-nearest vector path). The compares have one
 * implementation (no _ref form). */
static uint32_t rng = 12345;
static uint32_t next_word(void) {
    rng = rng * 1664525u + 1013904223u;
    switch ((rng >> 28) & 7) {
    case 0: return (rng & 1) ? 0x80000000u : 0;
    case 1: return rng & 0x807FFFFFu;            /* denormal / zero */
    case 2: return (rng & 0x807FFFFFu) | 0x7F800000u; /* exponent 255 */
    default: rng = rng * 1664525u + 1013904223u;
             return (rng & 0x807FFFFFu) | (((rng >> 8) % 60u + 100u) << 23);
    }
}
static int near1(uint32_t x, uint32_t y) { return x == y || x - y == 1 || y - x == 1; }
static void diff_vector_paths(void)
{
    int bad2 = 0, bad3 = 0;
    for (int n = 0; n < 200000; n++) {
        u128 a, b, c;
        for (int i = 0; i < 4; i++) { a._u32[i] = next_word(); b._u32[i] = next_word(); c._u32[i] = next_word(); }
        u128 x, y;
        x = spu_fa(a, b); y = spu_fa_ref(a, b); bad2 += memcmp(&x, &y, 16) != 0;
        x = spu_fs(a, b); y = spu_fs_ref(a, b); bad2 += memcmp(&x, &y, 16) != 0;
        x = spu_fm(a, b); y = spu_fm_ref(a, b); bad2 += memcmp(&x, &y, 16) != 0;
        u128 v[3][2] = { { spu_fma(a, b, c), spu_fma_ref(a, b, c) },
                         { spu_fms(a, b, c), spu_fms_ref(a, b, c) },
                         { spu_fnms(a, b, c), spu_fnms_ref(a, b, c) } };
        for (int k = 0; k < 3; k++)
            for (int i = 0; i < 4; i++) bad3 += !near1(v[k][0]._u32[i], v[k][1]._u32[i]);
    }
    CHECK(bad2 == 0); CHECK(bad3 == 0);
    if (bad2 || bad3) printf("  fast/ref mismatches: arith %d fma-lanes %d\n", bad2, bad3);
}
int main(void)
{
    u128 a = spu_zero(), r;
    a._f32[0]=1.0f; a._f32[1]=-2.75f; a._f32[2]=42.5f; a._f32[3]=1024.0f;
    r=spu_cflts(a,173);
    CHECK(r._s32[0]==1 && r._s32[1]==-2 && r._s32[2]==42 && r._s32[3]==1024);
    r=spu_cfltu(a,173);
    CHECK(r._u32[0]==1 && r._u32[1]==0 && r._u32[2]==42 && r._u32[3]==1024);
    r=spu_cflts(a,172); CHECK(r._s32[0]==2 && r._s32[1]==-5);
    a=spu_splat_u32(8); r=spu_csflt(a,155); CHECK(r._f32[0]==8.0f);
    r=spu_cuflt(a,156); CHECK(r._f32[0]==16.0f);
    r=spu_csflt(a,154); CHECK(r._f32[0]==4.0f);
    a._u32[0]=0x12345678; a._u32[1]=0x80000000;
    a._u32[2]=0xffffffff; a._u32[3]=7;
    r=spu_xswd(a);
    CHECK(r._u32[0]==0xffffffff && r._u32[1]==0x80000000 && r._u32[2]==0 && r._u32[3]==7);
    a._f32[0]=1.0f; a._f32[1]=99.0f; a._f32[2]=2.0f; a._f32[3]=-99.0f;
    r=spu_fesd(a); CHECK(spu__dget(r,0)==1.0 && spu__dget(r,1)==2.0);
    r=spu_frds(r); CHECK(r._f32[0]==1.0f && r._u32[1]==0 && r._f32[2]==2.0f && r._u32[3]==0);
    /* The estimate/FI pair feeds one Newton refinement in the guest code. */
    const float values[]={1.0f,2.0f,5.3311235f,-76.91874f};
    u128 one=spu_zero(); for(int i=0;i<4;++i) one._f32[i]=1.0f;
    for(int i=0;i<4;++i) a._f32[i]=values[i];
    r=spu_fi(a,spu_frest(a)); r=spu_fma(spu_fnms(a,r,one),r,r);
    for(int i=0;i<4;++i) CHECK(fabsf(r._f32[i]*values[i]-1.0f)<2e-6f);
    for(int i=0;i<4;++i) a._f32[i]=fabsf(values[i]);
    u128 half=one; for(int i=0;i<4;++i) half._f32[i]=0.5f;
    r=spu_fi(a,spu_frsqest(a));
    r=spu_fma(spu_fnms(spu_fm(a,r),r,one),spu_fm(r,half),r);
    for(int i=0;i<4;++i) CHECK(fabsf(r._f32[i]*sqrtf(a._f32[i])-1.0f)<2e-6f);
    /* No NaN/Inf on the SPU: exponent 255 is extended range, denormals are zero,
     * overflow saturates. EDGE's culler ORs masks in with fnms(mask, 0xFFFFFFFF, c). */
    u128 ones=spu_splat_u32(0xffffffff), z=spu_zero(), m=z; m._u32[1]=0xffffffff;
    r=spu_fnms(m,ones,z); CHECK(r._u32[0]==0 && r._u32[1]==0xffffffff && r._u32[2]==0);
    a=z; a._u32[0]=0x7f800000; a._u32[1]=0x00000001;
    u128 b=z; b._f32[0]=4.0f; b._f32[1]=2.0f;
    r=spu_fm(a,b); CHECK(r._u32[0]==0x7fffffff && r._u32[1]==0);
    r=spu_fa(a,z); CHECK(r._u32[0]==0x7f800000);
    b._f32[0]=1e38f; r=spu_fcgt(a,b); CHECK(r._u32[0]==0xffffffff && r._u32[1]==0);
    r=spu_fceq(a,z); CHECK(r._u32[1]==0xffffffff);
    r=spu_cflts(a,173); CHECK(r._s32[0]==0x7fffffff && r._s32[1]==0);
    diff_vector_paths();
printf("SPU numeric semantics: %d passed, %d failed\n",passed,failed);
    return failed?1:0;
}
