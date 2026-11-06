#!/usr/bin/env python3
"""Cadeia do quadro a partir de um PS3_FTRACE: para quadros perto da mediana, quando cada
workload SPURS começa e termina (ms desde o pedido de flip), quanto trabalha, e os sonos da
thread principal (usleep por endereço). Uso: cadeia.py <trace> [n_quadros]"""
import sys, statistics, collections
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../ps3recomp/tools"))
from ftrace_report import load, USLEEP, SPU_BUSY, FLIP_REQ, DRAIN, SYSCALL
NAMES = {0:"CollisL",1:"General",2:"HighestPrio6",3:"General6",4:"SysMisc",5:"ModelServer",6:"ModelServerPrep",
         7:"BatchJobList",8:"BrBDblBuf(audio)",9:"RenJobSe9",10:"RenJobSe10",11:"RenJobSe11",12:"RenJobSe12",
         13:"RenderJobs",14:"MeshProcJobs",15:"EDGE0",16:"EDGE1",17:"EDGE2",18:"RsxSync"}
hdr, names, spans = load(sys.argv[1]); nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 3
flips = [s[4] for s in spans if s[0] == FLIP_REQ]
frames = list(zip(flips, flips[1:])); durs = [b - a for a, b in frames]
med = statistics.median(durs)
print(f"{len(frames)} quadros, mediana {med:.1f} ms")
cand = sorted(range(len(frames)), key=lambda i: abs(durs[i] - med))[:nshow]
main = [t for t, n in names.items() if n == "main"][0]
for i in sorted(cand):
    f0, f1 = frames[i]
    print(f"\n== quadro {i}: {f1 - f0:.1f} ms")
    w = collections.defaultdict(lambda: [1e9, -1e9, 0.0, 0])
    for k, who, a, b, t0, t1 in spans:
        if t1 < f0 or t0 > f1: continue
        if k == SPU_BUSY:
            x = w[b]; x[0] = min(x[0], max(t0, f0) - f0); x[1] = max(x[1], min(t1, f1) - f0)
            x[2] += min(t1, f1) - max(t0, f0); x[3] += 1
    for wid, (s, e, busy, n) in sorted(w.items(), key=lambda kv: kv[1][0]):
        if busy < 0.3: continue
        bar = [" "] * 50
        for k, who, a, b, t0, t1 in spans:
            if k == SPU_BUSY and b == wid and t1 >= f0 and t0 <= f1:
                for c in range(int((max(t0, f0) - f0) / (f1 - f0) * 50), min(49, int((min(t1, f1) - f0) / (f1 - f0) * 50)) + 1): bar[c] = "#"
        print(f"  {NAMES.get(wid, wid):18s} {s:5.1f}-{e:5.1f} ms  ocupado {busy:5.1f}  |{''.join(bar)}|")
    bar = ["#"] * 50; sleeps = collections.Counter()
    for k, who, a, b, t0, t1 in spans:
        if who == main and k in (USLEEP, SYSCALL) and t1 >= f0 and t0 <= f1:
            sleeps[f"{b:08X}" if k == USLEEP else f"sc{a}"] += min(t1, f1) - max(t0, f0)
            for c in range(int((max(t0, f0) - f0) / (f1 - f0) * 50), min(49, int((min(t1, f1) - f0) / (f1 - f0) * 50)) + 1): bar[c] = "."
    print(f"  {'main (. = dorme)':18s} {'':17s} {'':15s}|{''.join(bar)}|  " + ", ".join(f"{k} {v:.1f}" for k, v in sleeps.most_common(3)))
    bar = [" "] * 50
    for k, who, a, b, t0, t1 in spans:
        if k == DRAIN and t1 >= f0 and t0 <= f1:
            for c in range(int((max(t0, f0) - f0) / (f1 - f0) * 50), min(49, int((min(t1, f1) - f0) / (f1 - f0) * 50)) + 1): bar[c] = "D"
    print(f"  {'RSX (D = drena)':18s} {'':33s}|{''.join(bar)}|")
