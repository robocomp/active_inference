import re,sys,csv,collections
sys.path.insert(0,'/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts')
from aff_arbitration import SEL,CAND,REF,producer
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS='c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3'.split()
S=0.35
tot=collections.Counter(); rows=[]
flip_detail=[]
for run in RUNS:
    refused=set()
    for i,ln in enumerate(open(B+run+'/logs/controller.log',errors='replace'),1):
        r=REF.search(ln)
        if r: refused.add(r.group(1)); continue
        m=SEL.search(ln)
        if not m: continue
        win=m.group(1)
        cands=[(n,st,float(g),float(s)) for n,st,g,s in CAND.findall(m.group(4))]
        elig=[c for c in cands if c[1] in('Offered','Executing')]
        concepts={producer(c[0])[0] for c in elig}|{producer(win)[0]}
        refused.discard(win) if False else None
        rows.append((run,i,win,float(m.group(3)),cands,elig,concepts,set(refused)))
        refused.discard(win)
def kind(n):
    c=producer(n)[0]; return 'room' if c=='room' else ('door' if c=='door' else 'object')
for run in RUNS:
    R=[r for r in rows if r[0]==run]
    print(run,len(R),'selections')
print('total selections',len(rows))
# contested: >=2 concepts
con=[r for r in rows if len(r[6])>=2]
print('contested (>=2 concepts)',len(con))
objwin=[r for r in con if kind(r[2])=='object']
print('object wins in contested',len(objwin), 'all object wins (any)',sum(kind(r[2])=='object' for r in rows))
obj_room=[r for r in con if kind(r[2])=='object' and 'room' in r[6]]
room_obj=[r for r in con if r[2]=='afford_room' and any(kind(c[0])=='object' for c in r[5])]
print('object beat room',len(obj_room),'room beat object',len(room_obj))
print('room wins in contested',sum(r[2]=='afford_room' for r in con),'door wins',sum(kind(r[2])=='door' for r in con))
at5=[r for r in objwin if r[3]==5.0]
print('object wins priced at gain==5.0:',len(at5),collections.Counter(producer(r[2])[0] for r in at5))
print('object wins by concept',collections.Counter(producer(r[2])[0] for r in objwin))
print('object wins at 5.0 AND beat room:',sum(1 for r in at5 if 'room' in r[6]))
# recompute without room scale: room candidates (afford_room) score' = s + (1-S)*g  [scale applied only when logged]
def recompute(r,noscale):
    out=[]
    for n,st,g,s in r[5]:
        if n in r[7] and n!=r[2]: continue   # refused skip as in original
        if noscale and n=='afford_room': s=s+(1-S)*g
        out.append((s,n))
    return max(out)[1] if out else None
chg=[];flips=[];mis=0
for r in con:
    base=recompute(r,False)
    if base!=r[2]: mis+=1
    new=recompute(r,True)
    if new!=base:
        chg.append((r[0],r[1],base,new))
        if kind(base)!=kind(new): flips.append((r[0],r[1],base,new))
print('baseline recompute != logged winner:',mis,'of',len(con))
print('winners changed w/o room scale (contested):',len(chg))
print(' object->room flips:',sum(kind(a)=='object' and kind(b)=='room' for _,_,a,b in flips),' other kind flips',collections.Counter((kind(a),kind(b)) for _,_,a,b in flips))
for c in chg: print('  ',c)
# also on all selections incl non-contested
chg2=[r for r in rows if recompute(r,False)!=recompute(r,True)]
print('changes among ALL selections:',len(chg2))
# room scale applied? room candidates' gain vs ... sanity: count contested with room eligible
print('contested with room eligible',sum(1 for r in con if any(c[0]=='afford_room' for c in r[5])))
# floor: raw gains in ai2 logs
