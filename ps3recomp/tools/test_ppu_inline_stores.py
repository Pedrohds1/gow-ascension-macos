#!/usr/bin/env python3
"""The lifter header's inline guest stores (vm_write*_inl) and trace gate.

Builds a small driver against a ppu_recomp.h made from the lifter's own
HEADER_PREAMBLE and checks that each inline store writes the same big-endian
bytes the out-of-line store would, and that everything the out-of-line store
treats specially still reaches it: vm_inline_st_ok clear, the first 64 KB,
0xE0000000 and up (raw SPU windows), past ppu_vm_size, and a line an SPU holds
reserved. Also that ppu_trace_enter is skipped while ppu_trace_live is 0.

It needs a host C++ compiler (under CI=1 a missing one fails instead of
skipping).

Usage:
    python3 tools/test_ppu_inline_stores.py
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))

DRIVER = r"""
#include "ppu_recomp.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static uint8_t* g_mem;
extern "C" {
uint8_t* vm_base;
uint32_t ppu_vm_size = 0, ppu_hle_inject_base = 0;
int vm_inline_ok = 1, vm_inline_st_ok = 1, ppu_trace_live = 0;
const unsigned char* g_spu_coh_bits = 0;
static int g_slow;     /* out-of-line calls */
uint8_t  (vm_read8)(uint64_t) { return 0; }
uint16_t (vm_read16)(uint64_t) { return 0; }
uint32_t (vm_read32)(uint64_t) { return 0; }
uint64_t (vm_read64)(uint64_t) { return 0; }
void (vm_write8)(uint64_t a, uint8_t v)   { g_slow++; g_mem[(uint32_t)a] = v; }
void (vm_write16)(uint64_t a, uint16_t v) { g_slow++; v = __builtin_bswap16(v); memcpy(g_mem + (uint32_t)a, &v, 2); }
void (vm_write32)(uint64_t a, uint32_t v) { g_slow++; v = __builtin_bswap32(v); memcpy(g_mem + (uint32_t)a, &v, 4); }
void (vm_write64)(uint64_t a, uint64_t v) { g_slow++; v = __builtin_bswap64(v); memcpy(g_mem + (uint32_t)a, &v, 8); }
}
static int g_traced;
void (ppu_trace_enter)(uint32_t, ppu_context*) { g_traced++; }

static int fails, total;
static void check(int ok, const char* what) {
    total++; if (!ok) fails++;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
}
/* one store of each width at ea; returns how many went out of line */
static int store_all(uint32_t ea) {
    int s = g_slow;
    vm_write8(ea, 0x11);
    vm_write16(ea + 2, 0x2233);
    vm_write32(ea + 4, 0x44556677u);
    vm_write64(ea + 8, 0x8899AABBCCDDEEFFull);
    return g_slow - s;
}
static int bytes_ok(uint32_t ea) {
    static const uint8_t want[16] = { 0x11, 0, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF };
    uint8_t got[16]; memcpy(got, g_mem + ea, 16); got[1] = 0;
    return !memcmp(got, want, 16);
}

int main() {
    g_mem = (uint8_t*)calloc(1, 0x100000);           /* the low 1 MB of guest space */
    static uint8_t hi[0x1000];                        /* stands in for 0xE0000000 */
    vm_base = g_mem;

    check(store_all(0x20000) == 0 && bytes_ok(0x20000), "aligned stores inline, big-endian");
    check(store_all(0x30001) == 0, "unaligned stores inline");
    { uint8_t want[15] = { 0x11, 0, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE };
      uint8_t got[15]; memcpy(got, g_mem + 0x30001, 15); got[1] = 0;
      check(!memcmp(got, want, 15), "unaligned inline stores land big-endian"); }
    check(store_all(0x1000) == 4, "first 64 KB goes out of line (null page)");
    check(store_all(0xFFF0) == 4, "just below 0x10000 goes out of line");
    vm_inline_st_ok = 0;
    check(store_all(0x40000) == 4 && bytes_ok(0x40000), "vm_inline_st_ok=0: every store out of line");
    vm_inline_st_ok = 1;
    ppu_vm_size = 0x50008;
    { int s = g_slow;
      vm_write64(0x50000, 1);   check(g_slow == s, "store ending at ppu_vm_size inline");
      vm_write64(0x50004, 1);   check(g_slow == s + 1, "store past ppu_vm_size out of line"); }
    ppu_vm_size = 0;
    /* raw SPU windows: point vm_base so 0xE0000000 lands in `hi` */
    { uint8_t* save = vm_base; vm_base = hi - 0xE0000000u; g_mem = vm_base;
      check(store_all(0xE0000000u) == 4, "0xE0000000+ (raw SPU) out of line");
      vm_base = save; g_mem = save; }
    /* SPU-reserved line: bit for the line holding 0x60080 */
    { static unsigned char bits[(0x100000 >> 7) / 8];
      const uint32_t line = 0x60080u >> 7; bits[line >> 3] |= (unsigned char)(1u << (line & 7));
      g_spu_coh_bits = bits;
      check(store_all(0x60080) == 4 && bytes_ok(0x60080), "reserved line out of line");
      check(store_all(0x60100) == 0, "unreserved neighbour line inline");
      g_spu_coh_bits = 0; }

    ppu_context ctx; memset(&ctx, 0, sizeof ctx);
    ppu_trace_live = 0; ppu_trace_enter(0x10200u, &ctx);
    check(g_traced == 0, "ppu_trace_live=0 skips the tracer");
    ppu_trace_live = 1; ppu_trace_enter(0x10200u, &ctx);
    check(g_traced == 1, "ppu_trace_live=1 calls the tracer");

    printf("%s (%d/%d failed)\n", fails ? "FAIL" : "PASS", fails, total);
    return fails ? 1 : 0;
}
"""


def _cxx():
    return (shutil.which(os.environ.get("CXX", "c++")) or shutil.which("g++")
            or shutil.which("clang++"))


def _header():
    spec = importlib.util.spec_from_file_location(
        "ppu_lifter", os.path.join(TOOLS, "ppu_lifter.py"))
    lifter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifter)
    return lifter.HEADER_PREAMBLE


def main():
    cxx = _cxx()
    if cxx is None:
        if os.environ.get("CI"):
            print("FAIL: no C++ compiler found")
            return 1
        print("skip test_ppu_inline_stores: no C++ compiler")
        return 0
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "ppu_recomp.h"), "w") as fh:
            fh.write(_header())
        src = os.path.join(td, "driver.cpp")
        with open(src, "w") as fh:
            fh.write(DRIVER)
        exe = os.path.join(td, "driver")
        r = subprocess.run([cxx, "-std=gnu++17", "-O2", "-w", "-I" + td, "-o", exe, src],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL: driver did not build")
            print(r.stdout + r.stderr)
            return 1
        r = subprocess.run([exe], capture_output=True, text=True)
        sys.stdout.write(r.stdout + r.stderr)
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
