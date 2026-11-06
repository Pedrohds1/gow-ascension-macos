#!/usr/bin/env python3
"""Confere, numa captura de passes (RSX_ENGINE_PASS_DUMP com _DRAWS=1), a cobertura das
cascatas de sombra do sol: os limites de depth bounds de cada bloco contra a profundidade real.

    scripts/blocos_dbt.py <dir-da-captura>

Cada desenho dos passes de luz (alvo 0x01D90000, depth bounds ligado) é meio bloco de tela; a
caixa do triângulo dá o retângulo em pixels e o desenho só toca os pixels com lo <= d <= hi.
Soma, por pixel de geometria (d < 1), quantas cascatas o pegam, e lista os blocos com pixels
que nenhuma cascata pega (ficam com o valor limpo do alvo: um bloco liso na imagem).
"""
import re, sys
import numpy as np

d = sys.argv[1]
W, H = 1280, 720
passes = {}
for line in open(f"{d}/passes.txt"):
    m = re.match(r"pass (\d+): draws (\d+) clears \d+ \| A (0x[0-9A-F]+) .*?zeta (0x[0-9A-F]+)", line)
    if m:
        passes[int(m.group(1))] = (int(m.group(2)), m.group(3), m.group(4))

draw_re = re.compile(r"^pass (\d+) draw (\d+):.*? dbt (\d) ([0-9.]+)\.\.([0-9.]+) ")
raw_re = re.compile(r"raw \(([-0-9.e]+),([-0-9.e]+)\)\(([-0-9.e]+),([-0-9.e]+)\)\(([-0-9.e]+),([-0-9.e]+)\)")
cur = None
tiles = {}
for line in open(f"{d}/draws.txt"):
    m = draw_re.match(line)
    if m:
        cur = (int(m.group(1)), int(m.group(3)), float(m.group(4)), float(m.group(5)))
        continue
    m = raw_re.search(line)
    if m and cur:
        p, on, lo, hi = cur
        cur = None
        if not on or p not in passes or passes[p][1] != "0x01D90000":
            continue
        xs = [float(m.group(i)) for i in (1, 3, 5)]
        ys = [float(m.group(i)) for i in (2, 4, 6)]
        x0 = round((min(xs) + 1) / 2 * W); x1 = round((max(xs) + 1) / 2 * W)
        y0 = round((1 - max(ys)) / 2 * H); y1 = round((1 - min(ys)) / 2 * H)
        tiles.setdefault(p, {})[(x0, y0, x1, y1)] = (lo, hi)

if not tiles:
    sys.exit("nenhum passe de luz com depth bounds na captura")
first = min(tiles)
z = np.fromfile(f"{d}/pass{first:03d}_z.pgm.f32", dtype="<f4")
if z.size != W * H:
    sys.exit(f"zeta com {z.size} valores")
z = z.reshape(H, W)
geo = z < 0.9999999
cov = np.zeros((H, W), np.int32)
for p in sorted(tiles):
    los = sorted({v[0] for v in tiles[p].values()})
    his = [v[1] for v in tiles[p].values()]
    print(f"pass {p}: {len(tiles[p])} retângulos, lo {['%.7f' % v for v in los[:3]]}, "
          f"hi {min(his):.7f}..{max(his):.7f}")
    for (x0, y0, x1, y1), (lo, hi) in tiles[p].items():
        sub = z[y0:y1, x0:x1]
        cov[y0:y1, x0:x1] += ((sub >= lo) & (sub <= hi)).astype(np.int32)

none = geo & (cov == 0)
print(f"\npixels de geometria: {geo.sum()}, sem nenhuma cascata: {none.sum()}, "
      f"em 2+ cascatas: {(geo & (cov >= 2)).sum()}")
TW, TH = 160, 90
for ty in range(H // TH):
    for tx in range(W // TW):
        blk = none[ty*TH:(ty+1)*TH, tx*TW:(tx+1)*TW]
        n = int(blk.sum())
        if n:
            g = int(geo[ty*TH:(ty+1)*TH, tx*TW:(tx+1)*TW].sum())
            zs = z[ty*TH:(ty+1)*TH, tx*TW:(tx+1)*TW][blk]
            print(f"  bloco {tx*TW:4d},{ty*TH:3d}: {n:5d} de {g:5d} pixels sem cascata, "
                  f"d {zs.min():.7f}..{zs.max():.7f}")
np.save(f"{d}/cobertura.npy", cov)
