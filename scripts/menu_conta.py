#!/usr/bin/env python3
"""Conta, nos quadros gravados do menu (work/menu-<nome>), os que saem sem o texto do menu."""
import sys, glob, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/f*.ppm"))
def ld(f):
    b = open(f, "rb").read(); return np.frombuffer(b[-1280*720*3:], np.uint8).reshape(720, 1280, 3)
y0, y1, x0, x1 = 653, 667, 480, 733          # "Jogador Individual"
txt = np.array([int((ld(f)[y0:y1, x0:x1].max(axis=2) > 150).sum()) for f in fs])
ref = float(np.median(txt[-60:]))
start = int(np.argmax(txt > ref * 0.6))
menu = txt[start:]
bad = [i + start for i, v in enumerate(menu) if v < ref * 0.3]
print(f"menu: {len(menu)} quadros, sem o texto: {len(bad)} ({100*len(bad)/max(1,len(menu)):.1f}%)")
