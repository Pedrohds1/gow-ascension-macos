"""Apply the port's hooks to freshly lifted PPU code (idempotent).

The lifter emits every guest function as plain C++ with a ppu_trace_enter()
call first. A port hook that has to run *instead of* a guest function body
(or veto it) is inserted right after that call; this script does it so a
re-lift keeps the hooks without hand edits.

Usage: python patch_lift.py <recompiled dir>
"""
import glob
import os
import sys

# guest function -> C call returning nonzero to skip the body (see gow_vram_guard.c).
# One address per release: the EU EBOOT's code sits 0x10 below the US one's here.
HOOKS = {
    (0x00370E48,    # BCUS98232
     0x00370E38):   # BCES01741
        ('gow_release_veto', 'int gow_release_veto(uint32_t obj);',
         'if (gow_release_veto((uint32_t)ctx->gpr[4])) return;'),
}


def lifted_address(d, addrs):
    """The one address of `addrs` this lift has a function at."""
    found = []
    for path in sorted(glob.glob(os.path.join(d, 'ppu_recomp_*.cpp'))):
        with open(path, 'rb') as f:
            src = f.read().decode('latin-1')
        found += [a for a in addrs if f'void func_{a:08X}(ppu_context* ctx) {{' in src]
    if len(found) != 1:
        sys.exit(f'expected one of {[hex(a) for a in addrs]} in the lift, found {[hex(a) for a in found]}')
    return found[0]


def main(d):
    for addrs, (name, decl, stmt) in HOOKS.items():
        addr = lifted_address(d, addrs)
        enter = f'ppu_trace_enter(0x{addr:08X}u, ctx);'
        head = f'void func_{addr:08X}(ppu_context* ctx) {{'
        done = False
        for path in sorted(glob.glob(os.path.join(d, 'ppu_recomp_*.cpp'))):
            with open(path, 'rb') as f:
                src = f.read().decode('latin-1')
            i = src.find(head)
            if i < 0:
                continue
            if name in src[i:i + 4000]:
                print(f'{os.path.basename(path)}: func_{addr:08X} already hooked')
                done = True
                continue
            j = src.find(enter, i)
            if 0 <= j and j - i <= 4000:
                j += len(enter)
            else:
                # A lifter that emits no trace call: first thing in the body,
                # before any guest state changes, which is where the trace sits.
                j = i + len(head)
            src = src[:i] + f'extern "C" {decl}\n' + src[i:j] + '\n        ' + stmt + src[j:]
            with open(path, 'wb') as f:
                f.write(src.encode('latin-1'))
            print(f'{os.path.basename(path)}: func_{addr:08X} -> {name}')
            done = True
        if not done:
            sys.exit(f'func_{addr:08X} not found')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'recompiled_v015')
