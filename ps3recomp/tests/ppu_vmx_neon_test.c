/* ppu_vmx_neon_test.c -- the NEON vaddfp/vsubfp/vmaddfp/vnmsubfp of ppu_vmx.h
 * against their scalar *_ref oracles, bit for bit: random words, then every
 * combination of the special values (signed zeros, denormals, infinities,
 * quiet and signalling NaNs with payloads, extremes), with VSCR[NJ] set and
 * clear, and with the destination aliasing each source.
 *
 *   cc -O2 -I runtime/ppu tests/ppu_vmx_neon_test.c -o /tmp/vmxt && /tmp/vmxt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ppu_vmx.h"

static uint64_t s_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17; return (uint32_t)s_rng; }

static const uint32_t k_special[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x807FFFFFu, 0x00400000u,
    0x00800000u, 0x80800000u, 0x3F800000u, 0xBF800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u,
    0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7FC12345u, 0x7F800001u, 0xFFA00000u, 0x7FBFFFFFu,
    0x34000000u, 0x0C000000u, 0x3F7FFFFFu, 0x40000000u, 0xC0000000u,
};
#define NSP (sizeof k_special / sizeof k_special[0])

static uint32_t pick(void)
{
    const uint32_t r = rnd();
    switch (r & 7) {
    case 0: case 1: return k_special[(r >> 3) % NSP];
    case 2: return (rnd() & 0x807FFFFFu);                         /* denormal or zero */
    case 3: return (rnd() & 0x80000000u) | ((0x60u + (rnd() & 0x3F)) << 23) | (rnd() & 0x7FFFFFu); /* small */
    default: return rnd();
    }
}

static void put(uint8_t* v, const uint32_t w[4]) { for (int i = 0; i < 4; i++) vmx_sw(v, i, w[i]); }

static unsigned long s_fail;
static void check(const char* op, const uint8_t* got, const uint8_t* want, const uint8_t* a,
                  const uint8_t* b, const uint8_t* c, uint32_t vscr)
{
    if (!memcmp(got, want, 16)) return;
    if (s_fail++ < 20) {
        printf("FAIL %s vscr=%08X", op, vscr);
        for (int i = 0; i < 4; i++)
            printf(" | a=%08X b=%08X c=%08X got=%08X want=%08X", vmx_w(a, i), vmx_w(b, i),
                   c ? vmx_w(c, i) : 0u, vmx_w(got, i), vmx_w(want, i));
        printf("\n");
    }
}

static void one(const uint32_t wa[4], const uint32_t wb[4], const uint32_t wc[4], uint32_t vscr)
{
    uint8_t a[16], b[16], c[16], r1[16], r2[16];
    put(a, wa); put(b, wb); put(c, wc);
    vmx_vaddfp(r1, a, b, vscr);  vmx_vaddfp_ref(r2, a, b, vscr);  check("vaddfp", r1, r2, a, b, NULL, vscr);
    vmx_vsubfp(r1, a, b, vscr);  vmx_vsubfp_ref(r2, a, b, vscr);  check("vsubfp", r1, r2, a, b, NULL, vscr);
    vmx_vmaddfp(r1, a, b, c, vscr);  vmx_vmaddfp_ref(r2, a, b, c, vscr);  check("vmaddfp", r1, r2, a, b, c, vscr);
    vmx_vnmsubfp(r1, a, b, c, vscr); vmx_vnmsubfp_ref(r2, a, b, c, vscr); check("vnmsubfp", r1, r2, a, b, c, vscr);
    /* the destination aliasing a source */
    uint8_t x[16], y[16];
    memcpy(x, a, 16); vmx_vmaddfp(x, x, b, c, vscr); vmx_vmaddfp_ref(y, a, b, c, vscr); check("vmaddfp d=a", x, y, a, b, c, vscr);
    memcpy(x, c, 16); vmx_vmaddfp(x, a, b, x, vscr); check("vmaddfp d=c", x, y, a, b, c, vscr);
    memcpy(x, b, 16); vmx_vnmsubfp(x, a, x, c, vscr); vmx_vnmsubfp_ref(y, a, b, c, vscr); check("vnmsubfp d=b", x, y, a, b, c, vscr);
    memcpy(x, a, 16); vmx_vsubfp(x, x, b, vscr); vmx_vsubfp_ref(y, a, b, vscr); check("vsubfp d=a", x, y, a, b, NULL, vscr);
}

int main(void)
{
    unsigned long n = 0;
    const uint32_t vscrs[3] = { 0u, VMX_NJ, VMX_NJ | VMX_SAT };
    for (int v = 0; v < 3; v++) {
        /* every special value against every other, in every lane position */
        for (size_t i = 0; i < NSP; i++)
            for (size_t j = 0; j < NSP; j++)
                for (size_t k = 0; k < NSP; k++) {
                    const uint32_t wa[4] = { k_special[i], k_special[j], k_special[k], k_special[(i + j) % NSP] };
                    const uint32_t wb[4] = { k_special[j], k_special[k], k_special[i], k_special[(j + k) % NSP] };
                    const uint32_t wc[4] = { k_special[k], k_special[i], k_special[j], k_special[(i + k) % NSP] };
                    one(wa, wb, wc, vscrs[v]); n++;
                }
        for (long t = 0; t < 2000000; t++) {
            uint32_t wa[4], wb[4], wc[4];
            for (int l = 0; l < 4; l++) { wa[l] = pick(); wb[l] = pick(); wc[l] = pick(); }
            one(wa, wb, wc, vscrs[v]); n++;
        }
    }
    printf("%lu vectors x 8 checks, %lu failures\n", n, s_fail);
    return s_fail ? 1 : 0;
}
