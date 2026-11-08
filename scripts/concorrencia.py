#!/usr/bin/env python3
"""Concorrência por workload SPURS num PS3_FTRACE: quantas threads SPU rodam o mesmo wid ao mesmo
tempo (média ponderada, máximo, % do tempo em cada nível). Mostra quais listas batem no teto de
SPURS_LANES. Uso: concorrencia.py <trace>"""
import sys, collections
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../ps3recomp/tools"))
from ftrace_report import load, SPU_BUSY, FLIP_REQ
hdr, names, spans = load(sys.argv[1])
NAMES = {0:"CollisL",2:"HighestPrio6",3:"General6",7:"BatchJobList",8:"audio",9:"RenJobSe9",10:"RenJobSe10",11:"RenJobSe11",12:"RenJobSe12",13:"RenderJobs",14:"MeshProcJobs",15:"EDGE0",16:"EDGE1",17:"EDGE2"}
ev = collections.defaultdict(list)
for k, w, a, b, t0, t1 in spans:
    if k == SPU_BUSY: ev[b].append((t0, 1, w)); ev[b].append((t1, -1, w))
for wid in sorted(ev):
    xs = sorted(ev[wid]); cur = 0; last = None; hist = collections.Counter(); thr = set()
    for t, d, w in xs:
        if last is not None and cur > 0: hist[cur] += t - last
        cur += d; last = t; thr.add(w)
    tot = sum(hist.values())
    if tot < 50: continue
    avg = sum(c * v for c, v in hist.items()) / tot
    print(f"{str(NAMES.get(wid, wid)):13s} ativo {tot:7.0f} ms  concorrência média {avg:.2f}  máx {max(hist)}  threads {len(thr)}  " +
          " ".join(f"{c}:{100*v/tot:.0f}%" for c, v in sorted(hist.items())))
