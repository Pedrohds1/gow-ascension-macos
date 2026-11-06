#!/usr/bin/env python3
"""Tempo próprio (self) das threads de SPU de uma amostra do `sample`, por categoria.
Uso: categorias.py <sample.txt> [--min-busy 0.3]"""
import re, sys
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__file__))
from perfil import parse

CATS = [
    ("nativo EDGE", r"^g(73|53|64|88|6a|o11)_|gow_edge_"),
    ("fragmento EDGE traduzido", r"^edgefrag_"),
    ("job module traduzido", r"^jobmod_"),
    ("blob reloc traduzido", r"^relocblob_"),
    ("imagem SPU traduzida", r"^spu_\w*func_|^img\d+_|_spu_func_"),
    ("interpretador", r"spu_interp|spu_step|spu_dec_|spu_lookup_walk"),
    ("despacho (branch/lookup)", r"spu_indirect_branch|spu_drain_call|spu_lookup|spu_lifted_lookup|spu_hop|_tlv_get_addr"),
    ("canais/MFC/DMA", r"spu_wrch|spu_rdch|mfc_|_platform_mem|spu_overlay_note|spu_coh_|spu_lockline|spu_ls_"),
    ("helpers float/SIMD", r"^spu_(cflts|cfltu|csflt|cuflt|fi|frest|frsqest|fma|fms|shufb|rot)"),
    ("travas do SO", r"psynch|pthread_|os_unfair|_os_lock|swtch|semaphore_|__ulock"),
    ("SPURS/kernel", r"spurs|taskset|g_spu|spu_run|policy"),
]

def selfcounts(t):
    fr = t["frames"]; out = Counter()
    for i, (d, n, name, mod) in enumerate(fr):
        kids = 0
        for j in range(i + 1, len(fr)):
            if fr[j][0] <= d: break
            if fr[j][0] == fr[i + 1][0]: kids += fr[j][1]
        if n - kids > 0: out[name.split("  ")[0].strip()] += n - kids
    return out

def main():
    th = parse(sys.argv[1]); total = Counter(); nthreads = 0
    for t in th:
        sc = selfcounts(t)
        spuish = sum(v for k, v in sc.items() if re.search(r"jobmod_|edgefrag_|^g\d\d_|spu_", k))
        if spuish < 0.2 * t["total"]: continue
        nthreads += 1; total.update(sc)
    cat = Counter(); other = Counter()
    for k, v in total.items():
        for c, rx in CATS:
            if re.search(rx, k): cat[c] += v; break
        else:
            other[k] += v
    allc = sum(total.values())
    print(f"{nthreads} threads de SPU, {allc} amostras")
    for c, v in cat.most_common(): print(f"  {100*v/allc:5.1f}%  {c}")
    print(f"  {100*sum(other.values())/allc:5.1f}%  outros; maiores:")
    for k, v in other.most_common(12): print(f"           {v:5d}  {k[:90]}")
if __name__ == "__main__":
    main()
