#!/usr/bin/env python3
"""Cobertura da luz do sol por pixel numa captura de passes (com .f32 e .s8.pgm da zeta).

    scripts/cobertura_sol.py <dir> [primeiro-passe-de-luz]

Passes de luz: alvo 0x01D90000 com depth bounds. Cada desenho é meio bloco de 160x90; o pixel
entra se lo <= d <= hi e se passa o estêncil do desenho (lido do draws.txt). Lista, por bloco,
os pixels de geometria que nenhum passe pega -- ficam com o valor de limpeza do alvo.
"""
import re, sys, numpy as np
d = sys.argv[1]; W, H = 1280, 720
first = int(sys.argv[2]) if len(sys.argv) > 2 else None
draw_re = re.compile(r"^pass (\d+) draw (\d+):.*? st (\d) sf ([0-9A-F]+) ref ([0-9A-F]+) m ([0-9A-F]+)/[0-9A-F]+ .*? dbt (\d) ([0-9.]+)\.\.([0-9.]+) ")
raw_re = re.compile(r"raw \(([-0-9.e]+),([-0-9.e]+)\)\(([-0-9.e]+),([-0-9.e]+)\)\(([-0-9.e]+),([-0-9.e]+)\)")
tgt = {}
for line in open(f"{d}/passes.txt"):
    m = re.match(r"pass (\d+): .*?\| A (0x[0-9A-F]+)", line)
    if m: tgt[int(m.group(1))] = m.group(2)
draws = {}; cur = None
for line in open(f"{d}/draws.txt"):
    m = draw_re.match(line)
    if m:
        cur = (int(m.group(1)), int(m.group(3)), int(m.group(4), 16), int(m.group(5), 16),
               int(m.group(6), 16), int(m.group(7)), float(m.group(8)), float(m.group(9)))
        continue
    m = raw_re.search(line)
    if m and cur:
        p = cur[0]; c = cur; cur = None
        if tgt.get(p) != "0x01D90000" or not c[5]: continue
        xs = [float(m.group(i)) for i in (1, 3, 5)]; ys = [float(m.group(i)) for i in (2, 4, 6)]
        r = (round((min(xs)+1)/2*W), round((1-max(ys))/2*H), round((max(xs)+1)/2*W), round((1-min(ys))/2*H))
        draws.setdefault(p, []).append((r, c))
ps = sorted(draws)
if first is None: first = ps[0]
frame = [p for p in ps if first <= p <= first + 10]
z = np.fromfile(f"{d}/pass{frame[0]:03d}_z.pgm.f32", "<f4").reshape(H, W)
sb = open(f"{d}/pass{frame[0]:03d}_z.pgm.s8.pgm", "rb").read()
s = np.frombuffer(sb[-W*H:], np.uint8).reshape(H, W)
geo = z < 0.9999998
cov = np.zeros((H, W), np.int32)
ops = {0x201: np.less, 0x202: np.equal, 0x203: np.less_equal, 0x204: np.greater,
       0x205: np.not_equal, 0x206: np.greater_equal}
for p in frame:
    st = {(c[1], c[2], c[3], c[4]) for _, c in draws[p]}
    print(f"pass {p}: {len(draws[p])} desenhos, estêncil {[(hex(a[1]), hex(a[2]), hex(a[3])) for a in st]}")
    for (x0, y0, x1, y1), c in draws[p]:
        sub = z[y0:y1, x0:x1]; ok = (sub >= c[6]) & (sub <= c[7])
        if c[1]:
            f = ops.get(c[2])
            if f: ok &= f(c[3] & c[4], s[y0:y1, x0:x1] & c[4])
        cov[y0:y1, x0:x1] |= ok
none = geo & (cov == 0)
print(f"geometria {geo.sum()}, sem luz do sol: {none.sum()}")
v, n = np.unique(s[none], return_counts=True)
print("  estêncil desses pixels:", {hex(a): int(b) for a, b in zip(v, n)})
for ty in range(H // 90):
    print("  " + " ".join(f"{int(none[ty*90:(ty+1)*90, tx*160:(tx+1)*160].sum()):6d}" for tx in range(W // 160)))
img = np.zeros((H, W, 3), np.uint8); img[geo & (cov > 0)] = (0, 150, 0); img[none] = (255, 0, 0)
open(f"{d}/cobertura.ppm", "wb").write(b"P6 %d %d 255\n" % (W, H) + img.tobytes())
