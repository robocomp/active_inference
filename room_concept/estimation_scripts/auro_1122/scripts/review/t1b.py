import re,sys,csv,collections,bisect
sys.path.insert(0,'/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts')
from aff_arbitration import SEL,CAND,REF,producer,line_times
from pathlib import Path
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS='c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3'.split()
S=0.35
def kind(n):
    c=producer(n)[0]; return 'room' if c=='room' else ('door' if c=='door' else 'object')
names=collections.Counter()
rawcache={}
def raw(run,node,t_ms):
    cls=node.split('_')[0]
    key=(run,cls)
    if key not in rawcache:
        d=collections.defaultdict(list)
        p=B+run+f'/{cls}_ai2_log.csv'
        try:
            for r in csv.DictReader(open(p)):
                try: d[r['node']].append((int(r.get('t_ms') or r['pkt_ts']),float(r['nbv_gain_raw']),float(r['nbv_gain_pub'])))
                except: pass
        except FileNotFoundError: pass
        rawcache[key]=d
    L=rawcache[key].get(node)
    if not L: return None
    ts=[x[0] for x in L]; k=bisect.bisect_right(ts,t_ms)-1
    if k<0: return None
    return L[k] if t_ms-L[k][0]<5000 else None
tot=0;res=[];missing=0;pubmis=0
chg=0;flipsn=0;allrows=[]
for run in RUNS:
    at=line_times(Path(B+run))
    t0=int(open(B+run+'/gt_pose.tsv').readlines()[1].split('\t')[0])
    refused=set()
    for i,ln in enumerate(open(B+run+'/logs/controller.log',errors='replace'),1):
        r=REF.search(ln)
        if r: refused.add(r.group(1)); continue
        m=SEL.search(ln)
        if not m: continue
        win=m.group(1)
        cands=[(n,st,float(g),float(s)) for n,st,g,s in CAND.findall(m.group(4))]
        for c in cands:
            if c[0]!='afford_room' and not c[0].startswith('aff_door'): pass
        elig=[c for c in cands if c[1] in('Offered','Executing')]
        concepts={producer(c[0])[0] for c in elig}|{producer(win)[0]}
        if len(concepts)>=2:
            tm=t0+at(i)*1000
            out=[];ok=True;floor_in=False
            for n,st,g,s in elig:
                if n in refused and n!=win: continue
                s2=s
                if n=='afford_room': s2+=0
                if kind(n)=='object' and g==5.0:
                    rr=raw(run,n[4:],tm)
                    if rr is None: ok=False; s2=None
                    else: s2=s-(5.0-rr[1]); floor_in=True
                out.append((s2,n))
            if win in [n for _,n in out] and kind(win)=='object' and float(m.group(3))==5.0:
                rr=raw(run,win[4:],t0+at(i)*1000)
                res.append((run,i,win,rr))
            allrows.append((run,i,win,ok,floor_in,out,float(m.group(3)),kind(win),'afford_room' in [c[0] for c in elig]))
        refused.discard(win)
w5=[r for r in res]
print('floor-priced object winners:',len(w5),'with raw found',sum(1 for r in w5 if r[3]),'missing',sum(1 for r in w5 if not r[3]))
import statistics
raws=[r[3][1] for r in w5 if r[3]]
pubs=[r[3][2] for r in w5 if r[3]]
print('raw gain of floor-priced winners: n',len(raws),'median',statistics.median(raws),'min',min(raws),'max',max(raws),'raw>=5:',sum(x>=5 for x in raws),' raw==0:',sum(x==0 for x in raws), ' pub values',collections.Counter(round(p,3) for p in pubs).most_common(3))
print(sorted(round(x,2) for x in raws))
print(collections.Counter(producer(r[2])[0] for r in w5 if r[3]))
# counterfactual: floor removed, rows where all floor candidates have raw
okrows=[r for r in allrows if r[3]]
chg=0;fl=0;lst=[];nfl=0
for run,i,win,ok,fi,out,g,k,hasroom in allrows:
    if not fi or not ok: continue
    nfl+=1
    new=max(out)[1]
    if new!=win:
        chg+=1; lst.append((run,i,win,new))
        if kind(new)!=kind(win): fl+=1
print('contested rows with >=1 floor-priced eligible object and raw found for all:',nfl,'; rows where some floor cand had no raw:',sum(1 for r in allrows if not r[3]))
print('winner changes w/o floor:',chg,' kind flips:',fl, collections.Counter((kind(a),kind(b)) for _,_,a,b in [(x[0],x[1],x[2],x[3]) for x in lst]))
for l in lst: print(l)
