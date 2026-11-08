#!/usr/bin/env python3
# Uso: scripts/quem_chama.py <sample.txt> <regex das folhas>   (thread principal)
import re, sys
from collections import Counter
# Para a thread "main" de um `sample` do macOS: soma as amostras das folhas-alvo por função do jogo
# (func_XXXXXXXX) mais próxima acima delas na pilha.
path, alvo = sys.argv[1], re.compile(sys.argv[2])
LINE = re.compile(r"^(?P<ind>[ +!:|]*)(?P<n>\d+) (?P<name>.+?)(?:  \(in (?P<mod>[^)]+)\).*)?$")
in_graph = False; in_main = False; stack = []; out = Counter(); outw = Counter()
for raw in open(path, encoding="utf-8", errors="replace"):
    l = raw.rstrip("\n")
    if l.startswith("Call graph:"): in_graph = True; continue
    if not in_graph: continue
    if l.startswith("Total number in stack"): break
    m = LINE.match(l.strip("\n")[4:] if l.startswith("    ") else l)
    if re.match(r"^\s+\d+ Thread_", l):
        in_main = "main-thread" in l or "com.apple.main-thread" in l
        stack = []; continue
    if not in_main: continue
    m = LINE.match(l.lstrip())
    if not m: continue
    depth = len(l) - len(l.lstrip(" +!:|"))
    name = m.group("name").split(" + ")[0].strip()
    n = int(m.group("n"))
    while stack and stack[-1][0] >= depth: stack.pop()
    stack.append((depth, name, n))
    if alvo.search(name):
        anc = [s[1] for s in stack[:-1] if s[1].startswith("func_")]
        key = anc[-1] if anc else "?"
        mid = [s[1] for s in stack[:-1] if not s[1].startswith("func_")][-3:]
        out[key] += n; outw[(key, " > ".join(mid))] += n
for k, v in out.most_common(15): print(f"{v:5d}  {k}")
print("---")
for (k, mid), v in outw.most_common(12): print(f"{v:5d}  {k}  [{mid}]")
