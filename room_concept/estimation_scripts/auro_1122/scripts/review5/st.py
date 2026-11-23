import json,numpy as np,collections
from scipy.stats import spearmanr
A=json.load(open('A.json'))
def q(x):
    x=[v for v in x if v is not None and v==v]
    return None if not x else (len(x),round(float(np.median(x)),2),round(float(np.percentile(x,25)),2),round(float(np.percentile(x,75)),2))
def sp(p,d):
    pr=[(a,b) for a,b in zip(p,d) if b is not None and b==b]
    if len(pr)<4 or len(set(a for a,_ in pr))<2: return f'n={len(pr)} (too few)'
    r=spearmanr([a for a,_ in pr],[b for _,b in pr]); return f'rho={r.statistic:.2f} p={r.pvalue:.2f} n={len(pr)}'
sets={'0.35':['s1_rs035_3','s1_rs035_4','s1_rs035_5'],'1.0':['s1_rs100_2','s1_rs100_3'],'lam_0p2':['lam_0p2'],'fix 0.35':['s1g_rs035_1','s1g_rs035_2','s1g_rs035_3'],'fix 1.0':['s1g_rs100_1','s1g_rs100_2']}
sets={k:[r for r in v if r in A] for k,v in sets.items()}
g=lambda r,k:(r.get(k) or {})
for sn,rs in sets.items():
    R=[r for k in rs for r in A[k]]
    print('=====',sn,'claims',len(R),'with no ai2 rows for node',sum(not r['rows'] for r in R),'reached',sum(r['reached'] for r in R),dict(collections.Counter(r['endclass'] for r in R)))
    print('  nrows==0 (W0):',sum(1 for r in R if g(r,'d_j').get('nrows')==0),' no-row-at-all:',sum(1 for r in R if not g(r,'d_j')))
    for wn,sel,key in (('W0 all',lambda r:True,'d_j'),('W2 reached->end',lambda r:r.get('d_reach') is not None,'d_reach')):
        S=[r for r in R if sel(r)]
        print(' ',wn,'n',len(S))
        for c in sorted({r['cls'] for r in S})+['ALL']:
            T=[r for r in S if c=='ALL' or r['cls']==c]
            d=[g(r,key).get('tot') for r in T]
            print('    ',c,len(T),'prom',q([r['gain'] for r in T]),'deliv',q(d),'|',sp([r['gain'] for r in T],d))
# adjacency / merge
for sn,rs in sets.items():
    m=0;tot=0
    for k in rs:
        # consecutive object claims of same node among object claims only (room claims between allowed)
        prev=None
        for r in A[k]:
            if prev and prev['node']==r['node']: m+=1
            prev=r
    print(sn,'same node as previous OBJECT claim:',m)
