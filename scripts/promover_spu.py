#!/usr/bin/env python3
"""Promove os registros da SPU a variáveis locais no C que o spu_lifter gera.

O lifter escreve cada instrução como `ctx->gpr[N] = op(ctx->gpr[M], ...)`. O local
store é um `uint8_t*`, então toda escrita nele pode, para o compilador, ter mudado
qualquer `ctx->gpr`: os registros vivem na memória e são relidos o tempo todo.
Aqui cada `ctx->gpr[N]` de uma função vira um `u128 gN` local:

  - na entrada, `gN = ctx->gpr[N]` para todo registro que a função usa;
  - antes de um ponto de sincronização (chamada a outra função traduzida, canais,
    drain, SPU_DIRECT, SPU_RET, spu_halt, spu_stop, return), os registros que a
    função escreve voltam ao ctx;
  - depois de um ponto que retorna, todos os registros usados são relidos (a
    função chamada pode ter mudado qualquer um). Chamadas seguidas na mesma linha,
    sem registro entre elas, dividem uma gravação e uma releitura.

spu_ls_read128/spu_ls_write128 não tocam registros (SPU_LS_PROBES=0 no build).
Uma função só é promovida quando a conta fecha: (pontos de sincronização x
registros) não passa de RATIO x (acessos a registro).

Uso: promover_spu.py <spu_dir_origem> <spu_dir_destino> [--skip modulo,...] [--ratio R]
Copia a árvore e reescreve os spu_recomp.c do destino; a origem não muda.
"""
import os
import re
import shutil
import sys

PURE = {"spu_ls_read128", "spu_ls_write128"}
VALUE = {"spu_rdch", "spu_rchcnt"}
TERMINAL = {"SPU_DIRECT", "SPU_RET", "SPU_RET_REG", "spu_halt"}
SYNC_NAMED = {"spu_drain_call", "spu_img_restore", "spu_wrch", "spu_stop", "spu_indirect_branch"} | VALUE | TERMINAL
FUNC_START = re.compile(r"^(?:static\s+)?void\s+(\w+)\(spu_context\*\s*ctx\)\s*\{\s*$")
GPR = re.compile(r"ctx->gpr\[(\d+)\]")
GPR_WRITE = re.compile(r"ctx->gpr\[(\d+)\](?:\._\w+\[\d+\])?\s*=(?!=)")
CALL = re.compile(r"\b([A-Za-z_]\w*)\(ctx\b")
RETURN = re.compile(r"\breturn\s*;")


def is_sync(name):
    # gow_edge_*: the EDGE natives the port splices into the d935 lift read and write ctx->gpr
    return (name in SYNC_NAMED or name.startswith("gow_edge_") or
            re.search(r"_spu_func_[0-9A-Fa-f]+$", name) is not None)


def close_paren(s, i):
    """s[i] == '(' -> índice do ')' que fecha."""
    depth = 0
    q = None
    while i < len(s):
        c = s[i]
        if q:
            if c == "\\": i += 1
            elif c == q: q = None
        elif c in "\"'": q = c
        elif c == "(": depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0: return i
        i += 1
    raise ValueError("parêntese sem par")


def sync_events(line):
    """Pontos de sincronização na linha: (início, fim, nome, tipo)."""
    ev = []
    for m in CALL.finditer(line):
        name = m.group(1)
        if name in PURE or not is_sync(name): continue
        a = m.start(1)
        b = close_paren(line, m.end(1)) + 1
        ev.append((a, b, name, "term" if name in TERMINAL else "val" if name in VALUE else "call"))
    for m in RETURN.finditer(line):
        ev.append((m.start(), m.end(), "return", "ret"))
    ev.sort()
    out, end = [], -1
    for e in ev:                      # descarta os aninhados (o externo cobre)
        if e[0] >= end:
            out.append(e); end = e[1]
    return out


def transform_line(line, flush, reload):
    ev = sync_events(line)
    if not ev:
        return GPR.sub(r"g\1", line)
    # grupos de chamadas seguidas sem registro entre elas
    groups, cur = [], [ev[0]]
    for e in ev[1:]:
        between = line[cur[-1][1]:e[0]]
        # only `;` and blanks between them: a goto or a brace (the EDGE native splices) splits
        if (GPR.search(between) or between.replace(";", "").strip() or
                cur[-1][3] in ("term", "ret", "val") or e[3] == "val"):
            groups.append(cur); cur = [e]
        else:
            cur.append(e)
    groups.append(cur)
    out, pos = [], 0
    for g in groups:
        a, b = g[0][0], g[-1][1]
        out.append(GPR.sub(r"g\1", line[pos:a]))
        body = GPR.sub(r"g\1", line[a:b])
        kind = g[-1][3]
        if kind == "ret":
            out.append("{ " + flush + body + " }")
        elif kind == "term":
            out.append("({ " + flush + body + "; })")
        elif kind == "val" and len(g) == 1:
            out.append("({ " + flush + "__typeof__(" + body + ") _pr = " + body + "; " + reload + "_pr; })")
        else:
            out.append("({ " + flush + body + "; " + reload + "})")
        pos = b
    out.append(GPR.sub(r"g\1", line[pos:]))
    return "".join(out)


def promote_function(lines, ratio):
    body = lines[1:-1]
    text = "\n".join(body)
    used = sorted({int(n) for n in GPR.findall(text)})
    if not used:
        return lines, False
    written = sorted({int(n) for n in GPR_WRITE.findall(text)})
    refs = len(GPR.findall(text))
    syncs = sum(len(sync_events(l)) for l in body)
    if syncs * (len(used) + len(written)) > ratio * refs:
        return lines, False
    flush = "".join(f"ctx->gpr[{n}] = g{n}; " for n in written)
    reload = "".join(f"g{n} = ctx->gpr[{n}]; " for n in used)
    decl = "        u128 " + ", ".join(f"g{n} = ctx->gpr[{n}]" for n in used) + ";"
    new = [lines[0], decl] + [transform_line(l, flush, reload) for l in body] + [lines[-1]]
    return new, True


def promote_file(path, ratio):
    src = open(path, encoding="latin-1").read().split("\n")
    out, i, nf, npro = [], 0, 0, 0
    while i < len(src):
        if FUNC_START.match(src[i]):
            j = i + 1
            while j < len(src) and src[j] != "}":
                j += 1
            fn = src[i:j + 1]
            new, did = promote_function(fn, ratio)
            out.extend(new); nf += 1; npro += did
            i = j + 1
        else:
            out.append(src[i]); i += 1
    open(path, "w", encoding="latin-1").write("\n".join(out))
    return nf, npro


def main():
    a = sys.argv[1:]
    src, dst = a[0], a[1]
    skip = set()
    ratio = 4.0
    if "--skip" in a: skip = set(a[a.index("--skip") + 1].split(","))
    if "--ratio" in a: ratio = float(a[a.index("--ratio") + 1])
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tf = tp = 0
    for d in sorted(os.listdir(dst)):
        p = os.path.join(dst, d, "spu_recomp.c")
        if not os.path.isfile(p) or any(d.startswith(s) for s in skip):
            continue
        nf, npro = promote_file(p, ratio)
        tf += nf; tp += npro
    print(f"{tp} de {tf} funções promovidas (razão {ratio}); pulados: {sorted(skip)}")


if __name__ == "__main__":
    main()
