#!/usr/bin/env python3
"""Quadros normais × longos num PS3_FTRACE: distribuição da duração do quadro (entre pedidos de
flip) e, para os dois grupos, onde vai o tempo — thread principal (ocupada, esperas por ponto de
chamada do jogo), SPUs por workload SPURS (EDGE à parte), andador do RSX (drenagem, espera de
semáforo, apresentação, espera pela GPU).

Uso: scripts/quadros.py <trace.ftr> [limite_ms_longo=20] [limite_ms_normal=17.5]
"""
import collections
import statistics
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../ps3recomp/tools"))
from ftrace_report import (load, SYSCALL, USLEEP, SPU_BUSY, SEMA, DRAIN, FLIP_REQ, PRESENT,  # noqa: E402
                           GPU_WAIT, GPU_ENCODE, SYSCALL_NAMES)

NAMES = {0: "CollisL", 1: "General", 2: "HighestPrio6", 3: "General6", 4: "SysMisc", 5: "ModelServer",
         6: "ModelServerPrep", 7: "BatchJobList", 8: "audio", 9: "RenJobSe9", 10: "RenJobSe10",
         11: "RenJobSe11", 12: "RenJobSe12", 13: "RenderJobs", 14: "MeshProcJobs", 15: "EDGE0",
         16: "EDGE1", 17: "EDGE2", 18: "RsxSync"}


def ov(s0, s1, f0, f1):
    return max(0.0, min(s1, f1) - max(s0, f0))


def main():
    path = sys.argv[1]
    lim_long = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    lim_norm = float(sys.argv[3]) if len(sys.argv) > 3 else 17.5
    _, names, spans = load(path)
    main_tid = [t for t, n in names.items() if n == "main"]
    main_tid = main_tid[0] if main_tid else None
    flips = [s[4] for s in spans if s[0] == FLIP_REQ]
    frames = list(zip(flips, flips[1:]))
    durs = [b - a for a, b in frames]
    if not durs:
        print("sem quadros")
        return
    q = sorted(durs)
    pct = lambda p: q[min(len(q) - 1, int(p * len(q)))]
    print(f"{len(durs)} quadros: média {statistics.mean(durs):.2f} ms ({1000 / statistics.mean(durs):.1f} fps), "
          f"mediana {statistics.median(durs):.2f}, p75 {pct(.75):.1f}, p90 {pct(.9):.1f}, p99 {pct(.99):.1f}, máx {q[-1]:.1f}")
    hist = collections.Counter(min(int(d // 2) * 2, 40) for d in durs)
    print("histograma (ms): " + "  ".join(f"{k}-{k + 2}:{hist[k]}" for k in sorted(hist)))
    longos = [i for i, d in enumerate(durs) if d > lim_long]
    normais = [i for i, d in enumerate(durs) if d <= lim_norm]
    extra = sum(durs[i] - 16.68 for i in longos)
    print(f"longos (> {lim_long:g} ms): {len(longos)} ({100 * len(longos) / len(durs):.0f}%), "
          f"{extra:.0f} ms acima de 16,7 no total; normais (<= {lim_norm:g} ms): {len(normais)}")

    def resume(idx):
        acc = collections.defaultdict(float)
        for i in idx:
            f0, f1 = frames[i]
            acc["dur"] += f1 - f0
            for k, who, a, b, t0, t1 in spans:
                if t1 < f0 or t0 > f1:
                    continue
                o = ov(t0, t1, f0, f1)
                if o <= 0:
                    continue
                if who == main_tid and k == USLEEP:
                    acc[f"main espera usleep lr={b:08X}"] += o   # a = chamadas agrupadas, b = lr do jogo
                    acc["main espera (total)"] += o
                elif who == main_tid and k == SYSCALL:
                    acc[f"main syscall {SYSCALL_NAMES.get(a, a)}"] += o
                    acc["main espera (total)"] += o
                elif k == SPU_BUSY:
                    nm = NAMES.get(b, f"wid{b}")
                    acc["SPU " + ("EDGE" if b in (15, 16, 17) else nm)] += o
                    acc["SPU total"] += o
                elif k == DRAIN:
                    acc["RSX drenagem"] += o
                elif k == SEMA:
                    acc["RSX espera semáforo"] += o
                elif k == PRESENT:
                    acc["RSX apresentação"] += o
                elif k == GPU_WAIT:
                    acc["RSX espera GPU"] += o
        n = max(1, len(idx))
        return {k: v / n for k, v in acc.items()}

    rn, rl = resume(normais), resume(longos)
    keys = sorted(set(rn) | set(rl), key=lambda k: -(rl.get(k, 0) + rn.get(k, 0)))
    print(f"\n{'ms por quadro':44s} {'normal':>8s} {'longo':>8s} {'dif':>8s}")
    for k in keys:
        a, b = rn.get(k, 0.0), rl.get(k, 0.0)
        if max(a, b) < 0.3:
            continue
        print(f"{k:44s} {a:8.2f} {b:8.2f} {b - a:+8.2f}")


if __name__ == "__main__":
    main()
