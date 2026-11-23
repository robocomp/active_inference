import json, collections, statistics
L=[]
for g in ('old','w2'):
    d=json.load(open(f'm1_{g}.json'))
    for r,v in d['runs'].items(): L+=v
print('lives',len(L))
print('by class',collections.Counter(l['cls'] for l in L))
print('verdict',collections.Counter(l['verdict'] for l in L))
rm=[l for l in L if l['removed_by_exist']]
print('exist-removed lives',len(rm),[(l['run'],l['node'],l['verdict']) for l in rm])
# lives under 3 s total (noise)?
def replay(l,DF,variant):
    miss=0
    for k,(sup,fr,lo,rp) in enumerate(l['ticks']):
        if sup: miss=0; continue
        if variant=='A': c=1
        elif variant=='B1': c=fr
        elif variant=='B2': c=lo
        elif variant=='C': c=rp if l['cls']!='chair' else 0
        if c:
            miss+=1
            if miss>=DF: return k*0.1
    return None
res={}
for var in ('A','B1','B2','C'):
  for DF in (90,30):
    cnt=collections.Counter(); who=[]
    for l in L:
        k=replay(l,DF,var)
        if k is not None:
            cnt[l['verdict']]+=1; who.append((l['run'],l['node'],l['verdict'],round(k,1),'also_exist' if l['removed_by_exist'] else ''))
    print(var,DF,dict(cnt),'total',sum(cnt.values()))
    res[(var,DF)]=who
json.dump({f'{k[0]}_{k[1]}':v for k,v in res.items()},open('m1b_who.json','w'))
tot=collections.Counter(l['verdict'] for l in L); print('lives by verdict',tot)
