import struct,sys,collections
RS="ate af aref areff be sfr dfr sfa dfa eqr eqa dt dw df ce cf ff cm cmm bm fp16 se s2 dbt sf sfm swm sfail szf szp bsf bsfail bszf bszp".split()
d=open(sys.argv[1],'rb').read()
o=16; recs={}
while o+16<=len(d):
    tag=struct.unpack_from('<I',d,o)[0]
    if tag==0x4F504F54: o+=16; continue
    if tag!=0x50495045 or o+240>len(d): break
    tag,fixed,key,th,fph=struct.unpack_from('<IIQQQ',d,o)
    rs=struct.unpack_from('<34I',d,o+32)
    lay=d[o+168:o+212]
    stride,rtf,rtc,topo,vl,pl=struct.unpack_from('<6I',d,o+212)
    vs=d[o+240:o+240+vl]; ps=d[o+240+vl:o+240+vl+pl]
    o+=240+vl+pl
    recs[key]=(rs,lay,stride,rtf,rtc,hash(vs),hash(ps),fph)
print("pipelines",len(recs))
vs=set(r[5] for r in recs.values()); ps=set(r[6] for r in recs.values())
print("VS",len(vs),"PS",len(ps),"pares",len(set((r[5],r[6]) for r in recs.values())), "fp_hash",len(set(r[7] for r in recs.values())))
print("layouts",len(set(r[1] for r in recs.values())),"rs",len(set(r[0] for r in recs.values())))
for i,n in enumerate(RS):
    vals=collections.Counter(r[0][i] for r in recs.values())
    if len(vals)>1: print(n,len(vals),vals.most_common(6))
for nm,ix in (("stride",2),("rtf",3),("rtc",4)):
    print(nm,collections.Counter(r[ix] for r in recs.values()).most_common(6))
# por par de shaders, quantos pipelines e o que varia
g=collections.defaultdict(list)
for r in recs.values(): g[(r[5],r[6])].append(r)
big=sorted(g.values(),key=len,reverse=True)[:3]
for grp in big:
    print("grupo",len(grp))
    for i,n in enumerate(RS):
        v=set(r[0][i] for r in grp)
        if len(v)>1: print("  ",n,len(v))
    print("   layouts",len(set(r[1] for r in grp)),"stride",len(set(r[2] for r in grp)),"rtf",len(set(r[3] for r in grp)))
print("---- normalizado")
I={n:i for i,n in enumerate(RS)}
def norm(rs,keep_aref=True):
    r=list(rs)
    if not r[I['ate']]: r[I['af']]=r[I['aref']]=r[I['areff']]=0
    elif not keep_aref: r[I['aref']]=0
    if not r[I['be']] and not r[I['bm']]:
        for n in "sfr dfr sfa dfa eqr eqa".split(): r[I[n]]=0
    if not r[I['se']]:
        for n in "s2 sf sfm swm sfail szf szp bsf bsfail bszf bszp".split(): r[I[n]]=0
    elif not r[I['s2']]:
        for n in "bsf bsfail bszf bszp".split(): r[I[n]]=0
    if not r[I['dt']]: r[I['df']]=0
    return tuple(r)
for ka in (True,False):
    s=set((norm(r[0],ka),)+r[1:] for r in recs.values())
    print("keep_aref",ka,len(s))
on=[r for r in recs.values() if r[0][I['ate']]]
print("ate=1 aref distintos",len(set(r[0][I['aref']] for r in on)))
print("be=1 sfa", collections.Counter(r[0][I['sfa']] for r in recs.values() if r[0][I['be']]).most_common(8))
