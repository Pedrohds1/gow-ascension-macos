#!/usr/bin/env python3
"""Para uma folha (regex) num `sample` do macOS: soma as amostras dela por cadeia de chamadores
imediatos (até 3 níveis acima), em todas as threads.  Uso: scripts/pais.py <sample.txt> '<regex>'"""
import re, sys
from collections import Counter
path, alvo = sys.argv[1], re.compile(sys.argv[2])
LINE = re.compile(r"^(?P<n>\d+) (?P<name>.+?)(?:  \(in (?P<mod>[^)]+)\).*)?$")
in_graph = False; stack = []; out = Counter()
for raw in open(path, encoding="utf-8", errors="replace"):
    l = raw.rstrip("\n")
    if l.startswith("Call graph:"): in_graph = True; continue
    if not in_graph: continue
    if l.startswith("Total number in stack"): break
    if re.match(r"^\s+\d+ Thread_", l): stack = []; continue
    s = l.lstrip(" +!:|")
    m = LINE.match(s)
    if not m: continue
    depth = len(l) - len(s)
    name = m.group("name").split(" + ")[0].strip()
    while stack and stack[-1][0] >= depth: stack.pop()
    stack.append((depth, name, int(m.group("n"))))
    if alvo.search(name):
        # só conta a folha (sem filhos com o mesmo nome logo abaixo): usa a contagem da linha
        pais = " < ".join(x[1][:60] for x in stack[-8:-1][::-1])
        out[pais] += int(m.group("n"))
for k, v in out.most_common(15): print(f"{v:6d}  {k}")
