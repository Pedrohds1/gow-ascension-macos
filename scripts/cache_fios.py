#!/usr/bin/env python3
# Confere (e, com --consertar, conserta) o cache de disco da FIOS que o GoW:A grava
# em <save>/hdd1/cache/BCES01741/fios/cache.{idx,dat}.
#
#   .venv/bin/python -I scripts/cache_fios.py saves/jogador            # só confere
#   .venv/bin/python -I scripts/cache_fios.py saves/jogador --consertar
#
# Formato (lido do próprio arquivo): cabeçalho de 512 bytes ("fioscach", número de
# blocos em +0x18, tamanho do bloco em +0x1C = 1 MiB). O .idx tem uma entrada de 512
# bytes por bloco: +0x00 u64 último uso (ns desde 1970), +0x18 u64 deslocamento no
# arquivo de origem, +0x20 u32 bytes válidos, +0x28 16 bytes que identificam o arquivo
# de origem. O .dat guarda os blocos a partir do byte 512. Cada bloco é a cópia de
# 1 MiB alinhado de um arquivo do disco; a verdade é o que o ÍNDICE diz (arquivo e
# deslocamento), porque é por ele que a FIOS procura.
#
# Por que existe (2026-11-08): no cache do Pedro a entrada 636 dizia "CYN.PSARC em
# 0x10200000" (dentro de /WAD/PS3_BE/CYN050.WAD, a zona da cena do Orkos), mas o bloco
# tinha os dados do CYN deslocados 0xD9D bytes, com sobra do HECA.PSARC no começo:
# uma gravação de 1 MiB caiu onde uma leitura de 0xD9D bytes na mesma vaga tinha deixado
# a posição do arquivo (corrida entre lseek e write de duas threads; ver [fs-race] no
# ppu_fs.cpp). Toda sessão lia esse lixo: a cena do Orkos ficava sem som e o cenário da
# zona nunca carregava (só os personagens eram desenhados, sobre restos de quadros).
import os, sys, struct, shutil, time, collections

B = 1 << 20


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    consertar = '--consertar' in sys.argv
    if not args:
        print('uso: cache_fios.py <pasta do save> [--consertar]')
        return 2
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save = args[0]
    fios = os.path.join(save, 'hdd1', 'cache', 'BCES01741', 'fios')
    usrdir = os.path.join(here, 'work', 'disc', 'PS3_GAME', 'USRDIR')
    if not os.path.isdir(usrdir):
        print('disco não montado em', usrdir, '(rode jogar.sh ou teste.sh uma vez, ou monte a ISO)')
        return 2
    pidx, pdat = os.path.join(fios, 'cache.idx'), os.path.join(fios, 'cache.dat')
    if not (os.path.exists(pidx) and os.path.exists(pdat)):
        print('sem cache em', fios)
        return 0
    idx = open(pidx, 'rb').read()
    nblocks = struct.unpack('>I', idx[0x18:0x1C])[0]
    if idx[:8] != b'fioscach' or struct.unpack('>I', idx[0x1C:0x20])[0] != B:
        print('formato inesperado')
        return 2
    entradas = []
    for i in range(nblocks):
        e = idx[512 + i * 512: 512 + (i + 1) * 512]
        valid = struct.unpack('>I', e[0x20:0x24])[0]
        if valid:
            entradas.append((i, struct.unpack('>Q', e[0x18:0x20])[0], valid, e[0x28:0x38]))

    arquivos = []
    for root, _, fs in os.walk(usrdir):
        for f in fs:
            arquivos.append(os.path.join(root, f))

    def ler(p, off, n):
        with open(p, 'rb') as fh:
            fh.seek(off)
            return fh.read(n)

    # Qual arquivo cada identificador representa: o que bate em pelo menos um bloco.
    por_fid = collections.defaultdict(list)
    for ent in entradas:
        por_fid[ent[3]].append(ent)
    fid_arquivo = {}
    with open(pdat, 'rb') as dat:
        for fid, ents in por_fid.items():
            for (i, off, valid, _) in ents[:8]:
                dat.seek(512 + i * B)
                blk = dat.read(min(valid, 4096))
                achou = [p for p in arquivos if os.path.getsize(p) > off and ler(p, off, len(blk)) == blk]
                if len(achou) == 1:
                    fid_arquivo[fid] = achou[0]
                    break

    ok, ruins, sem_arquivo = 0, [], []
    with open(pdat, 'rb') as dat:
        for (i, off, valid, fid) in entradas:
            p = fid_arquivo.get(fid)
            if not p:
                sem_arquivo.append(i)
                continue
            dat.seek(512 + i * B)
            if dat.read(valid) == ler(p, off, valid):
                ok += 1
            else:
                ruins.append((i, p, off, valid))
    print('blocos certos: %d, corrompidos: %d, de arquivo desconhecido: %d' % (ok, len(ruins), len(sem_arquivo)))
    for (i, p, off, valid) in ruins:
        print('  bloco %d: deveria ser %s em 0x%X (%d bytes)' % (i, os.path.basename(p), off, valid))
    if sem_arquivo:
        print('  de arquivo desconhecido:', sem_arquivo[:20])
    if not consertar or not (ruins or sem_arquivo):
        return 1 if (ruins or sem_arquivo) else 0
    if sem_arquivo:
        print('há blocos de arquivo desconhecido: apague o cache inteiro (o jogo refaz) em vez de consertar')
        return 1
    bak = os.path.join(save, 'cache_fios_backup_' + time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(bak)
    shutil.copy2(pidx, bak)
    with open(pdat, 'r+b') as dat:
        for (i, p, off, valid) in ruins:
            dat.seek(512 + i * B)
            open(os.path.join(bak, 'bloco_%d.bin' % i), 'wb').write(dat.read(valid))
            dat.seek(512 + i * B)
            dat.write(ler(p, off, valid))
    print('consertados %d blocos; cópia do índice e dos blocos antigos em %s' % (len(ruins), bak))
    return 0


if __name__ == '__main__':
    sys.exit(main())
