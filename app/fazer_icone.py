"""Ícone do app a partir das imagens do próprio disco (PS3_GAME/PIC1.PNG e ICON0.PNG).
Uso: python app/fazer_icone.py <PS3_GAME> <saida.png>  (1024x1024, cantos no padrão do macOS)."""
import sys
from PIL import Image, ImageDraw, ImageFilter
game, out = sys.argv[1], sys.argv[2]
S = 1024
pic = Image.open(f"{game}/PIC1.PNG").convert("RGB")
w, h = pic.size
side = int(h * 0.92)
cx = int(w * 0.52)
top = 0
crop = pic.crop((cx - side // 2, top, cx + side // 2, top + side)).resize((S, S), Image.LANCZOS)
# escurece a base para o logo
grad = Image.new("L", (1, S))
for y in range(S):
    t = max(0.0, (y - S * 0.55) / (S * 0.45))
    grad.putpixel((0, y), int(200 * t))
grad = grad.resize((S, S))
crop = Image.composite(Image.new("RGB", (S, S), (0, 0, 0)), crop, grad)
logo = Image.open(f"{game}/ICON0.PNG").convert("RGBA")
lw = int(S * 0.86)
logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
canvas = crop.convert("RGBA")
canvas.alpha_composite(logo, ((S - lw) // 2, S - logo.height - int(S * 0.06)))
# máscara do macOS: quadrado de 824 com raio ~185 dentro de 1024, com sombra leve
inner, r = 824, 185
off = (S - inner) // 2
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle((off, off, off + inner, off + inner), r, fill=255)
art = canvas.resize((inner, inner), Image.LANCZOS)
full = Image.new("RGBA", (S, S), (0, 0, 0, 0))
shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(shadow).rounded_rectangle((off, off + 12, off + inner, off + inner + 12), r, fill=(0, 0, 0, 110))
full = Image.alpha_composite(full, shadow.filter(ImageFilter.GaussianBlur(14)))
layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
layer.paste(art, (off, off))
layer.putalpha(Image.composite(layer.getchannel("A"), Image.new("L", (S, S), 0), mask))
full = Image.alpha_composite(full, layer)
full.save(out)
