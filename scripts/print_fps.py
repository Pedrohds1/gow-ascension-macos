#!/usr/bin/env python3
"""Captura de uma cena com o contador de fps desenhado, a partir de uma execução sem janela.

O jogo não desenha contador na tela; o fps vem do contador de quadros do próprio jogo
(game_frame do heartbeat no run.log): a média nos 10 s em volta do instante da imagem
(frames/tNNN.png, gravadas a cada 10 s).

Uso: .venv/bin/python -I scripts/print_fps.py <pasta da execução> <t> <saída.png> [rótulo]
"""
import re
import sys

from PIL import Image, ImageDraw, ImageFont


def fps_em(run, t, meia=5):
    hb = []
    for l in open(f"{run}/run.log", encoding="utf-8", errors="replace"):
        m = re.search(r"\[heartbeat\] t=(\d+)s flips=\d+ game_frame=(\d+)", l)
        if m:
            hb.append((int(m.group(1)), int(m.group(2))))
    d = dict(hb)
    a, b = t - meia, t + meia
    if a in d and b in d and b > a:
        return (d[b] - d[a]) / (b - a)
    return None


def main():
    run, t, saida = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rotulo = sys.argv[4] if len(sys.argv) > 4 else ""
    img = Image.open(f"{run}/frames/t{t:03d}.png").convert("RGB")
    fps = fps_em(run, t)
    if fps is None:
        sys.exit(f"sem heartbeat em volta de t={t}")
    d = ImageDraw.Draw(img, "RGBA")
    grande = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 34)
    pequena = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    txt = f"{fps:4.1f} FPS"
    w = d.textlength(txt, font=grande)
    w2 = d.textlength(rotulo, font=pequena) if rotulo else 0
    caixa_w = max(w, w2) + 28
    x, y = img.width - caixa_w - 18, 16   # canto superior direito: o HUD do jogo fica à esquerda
    caixa_h = 58 + (24 if rotulo else 0)
    d.rounded_rectangle([x, y, x + caixa_w, y + caixa_h], radius=8, fill=(0, 0, 0, 165))
    cor = (120, 255, 120) if fps >= 59.5 else (255, 225, 90) if fps >= 45 else (255, 120, 100)
    d.text((x + 14, y + 9), txt, font=grande, fill=cor)
    if rotulo:
        d.text((x + 14, y + 56), rotulo, font=pequena, fill=(225, 225, 225))
    if saida.lower().endswith((".jpg", ".jpeg")):
        img.save(saida, quality=88, optimize=True)
    else:
        img.save(saida)
    print(f"{saida}: {fps:.1f} fps em t={t}s")


if __name__ == "__main__":
    main()
