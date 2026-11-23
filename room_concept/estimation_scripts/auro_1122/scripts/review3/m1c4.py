import json, csv, collections
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
L=[]
for g in ('old','w2'):
    for r,v in json.load(open(f'm1_{g}4.json'))['runs'].items(): L+=v
STALE={('c3a_cab1',c) for c in ('nightstand','bed','toilet')}
L=[l for l in L if (l['run'],l['cls']) not in STALE]
legs={};room={}
for r in set(l['run'] for l in L):
    legs[r]=[(int(x['t0_ms'])/1000,int(x['t1_ms'])/1000,x['arg']) for x in csv.DictReader(open(B+r+'/legs.tsv'),delimiter='\t')]
    room[r]={x['name']:x['room'] for x in csv.DictReader(open(B+r+'/c3a_instances.csv'))}
def exit_t(l):
    rm=room[l['run']].get(l['node']); lg=legs[l['run']]
    if rm is None: return None
    for k,(t0,t1,arg) in enumerate(lg):
        frm='room_1' if k==0 else lg[k-1][2]
        if frm==rm and t0+3>l['born']: return t0+3
    return None
def lastrow(l): return l['born']+max((i for i,t in enumerate(l['ticks']) if t[0]),default=0)*0.1
for l in L:
    nat=l['born']+len(l['ticks'])*0.1
    e=l['end']; why='logged'
    xt=exit_t(l)
    if not l['removed_by_exist'] and abs(l['end']-nat)<0.2:
        # no deletion line observed in the log: cap by room-exit leg, else by last support row
        if xt and xt<e: e=xt; why='leg_exit'
        elif xt is None: e=lastrow(l)+0.1; why='last_row'
        else: why='run_end'
    n=int((e-l['born'])/0.1)+1
    l['ticks']=l['ticks'][:n]; l['why']=why
print(collections.Counter(l['why'] for l in L))
def replay(l,DF,var):
    miss=0
    for k,(sup,fr,lo,rp) in enumerate(l['ticks']):
        if sup: miss=0; continue
        c={'A':1,'B1':fr,'B2':lo}[var]
        if c:
            miss+=1
            if miss>=DF: return k*0.1
    return None
tot=collections.Counter(l['verdict'] for l in L); print('lives',len(L),dict(tot))
rmx=[l for l in L if l['removed_by_exist']]
print('existence-removed lives',len(rmx),collections.Counter(l['verdict'] for l in rmx))
out={}
for var in ('A','B1','B2'):
  for DF in (90,30):
    cnt=collections.Counter(); who=[]
    for l in L:
        k=replay(l,DF,var)
        if k is not None: cnt[l['verdict']]+=1; who.append((l['run'],l['node'],l['verdict'],round(k,1),l['removed_by_exist']))
    # overlap with the actual existence removals
    ov=[w for w in who if w[4]]
    print(var,DF,dict(cnt),'total',sum(cnt.values()),'| overlap with the 11 existence removals',len(ov),[(w[1],w[2]) for w in ov])
    out[f'{var}_{DF}']=who
    if var=='B2': 
        for w in who: print('    ',w)
# ticks in view unsupported stats
json.dump(out,open('m1c_who4.json','w'))
