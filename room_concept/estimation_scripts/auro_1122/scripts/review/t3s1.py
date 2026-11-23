import os,sys,re,csv,math,bisect,statistics,collections
os.environ.setdefault('WBT_PATH','/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
SC='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts'
sys.path.insert(0,SC)
from aff_arbitration import SEL,CAND,REF,producer,line_times
from run_frames import make_to_world
import c3a_objects as C
from pathlib import Path
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS=[a for a in sys.argv[1:] if not a.startswith('--')] or 'c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3'.split()
GT=C.gt_objects(os.environ['WBT_PATH'])
print('GT objects',len(GT),collections.Counter(g[1] for g in GT))
chair3=[g for g in GT if g[0]=='chair(3)' or 'chair' in g[0].lower() and '3' in g[0]]
print([g for g in GT if g[1]=='chair'])
def spearman(a,b):
    import numpy as np
    from scipy.stats import spearmanr
    return spearmanr(a,b)
rows=[]
for run in RUNS:
    R=B+run; at=line_times(Path(R)); 
    gt0=int(open(R+'/gt_pose.tsv').readlines()[1].split('\t')[0])
    tw=make_to_world(R)
    move=None
    if os.path.exists(R+'/chair_move.txt'):
        mv=open(R+'/chair_move.txt').read()
        m=re.search(r'chair moved t_ms (\d+)',mv); m2=re.search(r'CHAIR_3 -> (-?[\d.]+) (-?[\d.]+)',mv)
        if m and m2: move=(int(m.group(1)),float(m2.group(1)),float(m2.group(2)))
    inst={}
    try:
        for r in csv.DictReader(open(R+'/c3a_instances.csv')): inst[r['name']]=r['gt_object']
    except FileNotFoundError: pass
    # claims
    sels=[]; 
    for i,ln in enumerate(open(R+'/logs/controller.log',errors='replace'),1):
        m=SEL.search(ln)
        if m: sels.append((i,m.group(1),float(m.group(3))))
    # ai2 data
    ai={}
    def load(cls):
        if cls in ai: return ai[cls]
        d=collections.defaultdict(list)
        try:
            for r in csv.DictReader(open(R+f'/{cls}_ai2_log.csv')):
                try: d[r['node']].append((int(r.get('t_ms') or r['pkt_ts']),float(r['cx']),float(r['cy']),float(r['std_cx']),float(r['std_cy']),float(r['ex_p']) if r.get('ex_p') not in (None,'') else float('nan')))
                except: pass
        except FileNotFoundError: pass
        for k in d: d[k].sort()
        ai[cls]=d; return d
    for j,(i,win,g) in enumerate(sels):
        cc,kd=producer(win)
        if cc in('room','door') : continue
        node=win[4:]; L=load(cc).get(node)
        t_c=gt0+at(i)*1000
        t_e=gt0+at(sels[j+1][0])*1000 if j+1<len(sels) else None
        rec=dict(run=run,line=i,node=node,cls=cc,gain=g,t_claim=t_c,t_end=t_e)
        if not L: rec['status']='no ai2 rows for node'; rows.append(rec); continue
        ts=[x[0] for x in L]
        kb=bisect.bisect_right(ts,t_c)-1
        if t_e is None: t_e=ts[-1]; rec['t_end']=t_e; rec['end_is_run_end']=1
        ka=bisect.bisect_right(ts,t_e)-1
        if kb<0: rec['status']='node has no row before claim'; rows.append(rec); continue
        b=L[kb]; a=L[ka]
        nrows=ka-kb
        rec.update(status='ok',n_rows_in_window=nrows,lag_before_s=(t_c-b[0])/1000)
        d=0.5*math.log((b[3]**2*b[4]**2)/(a[3]**2*a[4]**2)) if min(a[3],a[4],b[3],b[4])>0 else float('nan')
        H=lambda p: 0.0 if not (0<p<1) else -(p*math.log(p)+(1-p)*math.log(1-p))
        dex=H(b[5])-H(a[5]) if (b[5]==b[5] and a[5]==a[5]) else float('nan')
        rec['delivered_pos']=d; rec['delivered_exist']=dex; rec['p_before']=b[5]; rec['p_after']=a[5]
        d=d+(dex if dex==dex else 0.0)   # total = position info + existence entropy drop (where ex_p is logged)
        rec['delivered']=d; rec['std_before']=(b[3],b[4]); rec['std_after']=(a[3],a[4])
        # GT
        wb=tw(b[1],b[2],b[0]); wa=tw(a[1],a[2],a[0])
        rec['world_after']=wa
        def gtpos(g_):
            nm,c,x,y=g_
            if move and nm=='chair(3)': return (x,y) if t_c<move[0] else (move[1],move[2])
            return (x,y)
        gtn=None
        if node in inst and inst[node]!='' :
            gtn=[g_ for g_ in GT if g_[0]==inst[node]]
            rec['gt_source']='c3a_instances'
        if not gtn:
            cand=[(math.dist(wa,gtpos(g_)),g_) for g_ in GT if g_[1]==cc] if wa else []
            cand=[c_ for c_ in cand if c_[0]<=1.0]
            gtn=[min(cand)[1]] if cand else []
            rec['gt_source']='nearest<=1m'
        if gtn and wa and wb:
            gp=gtpos(gtn[0]); rec['gt']=gtn[0][0]
            rec['err_before']=math.dist(wb,gp); rec['err_after']=math.dist(wa,gp); rec['d_err']=rec['err_after']-rec['err_before']
        else: rec['gt']=None
        rows.append(rec)
import json
ok=[r for r in rows if r.get('status')=='ok']
print('object claims (aff-select -> object):',len(rows),' with ai2 data:',len(ok),'no data:',collections.Counter(r.get('status') for r in rows))
print('claims with >=1 ai2 row inside window (belief updated):',sum(r['n_rows_in_window']>0 for r in ok))
print('end is run end:',sum(1 for r in ok if r.get('end_is_run_end')))
import numpy as np
def q(x): 
    x=[v for v in x if v==v]; 
    return (len(x),float(np.median(x)),float(np.percentile(x,25)),float(np.percentile(x,75))) if x else None
print('\nPER CLASS  n | promised median [IQR] | delivered median [IQR] | d_err median [IQR] (m, neg=better) | n_err')
byc=collections.defaultdict(list)
for r in ok: byc[r['cls']].append(r); byc['ALL'].append(r)
for c,R_ in sorted(byc.items()):
    p=q([r['gain'] for r in R_]); d=q([r['delivered'] for r in R_]); e=q([r.get('d_err',float('nan')) for r in R_])
    print(c,len(R_),'| promised',p[1:] if p else None,'| delivered',d[1:] if d else None,'| derr',e if e else None)
from scipy.stats import spearmanr
for c,R_ in sorted(byc.items()):
    xs=[(r['gain'],r['delivered']) for r in R_ if r['delivered']==r['delivered']]
    if len(xs)>=4 and len(set(x[0] for x in xs))>1:
        rho=spearmanr([x[0] for x in xs],[x[1] for x in xs]); print('spearman',c,len(xs),round(rho.statistic,3),round(rho.pvalue,4))
    xs=[(r['gain'],r['d_err']) for r in R_ if r.get('d_err') is not None]
    if len(xs)>=4 and len(set(x[0] for x in xs))>1:
        rho=spearmanr([x[0] for x in xs],[-x[1] for x in xs]); print('spearman promised vs error-reduction',c,len(xs),round(rho.statistic,3),round(rho.pvalue,4))
# floor-priced vs not
for lab,sel in (('gain==5.0',[r for r in ok if r['gain']==5.0]),('gain<5',[r for r in ok if r['gain']<5.0]),('gain<=ln2',[r for r in ok if r['gain']<=0.6932])):
    print(lab,len(sel),'delivered',q([r['delivered'] for r in sel]),'derr',q([r.get('d_err',float('nan')) for r in sel]), 'promised',q([r['gain'] for r in sel]))
# restricted to windows where belief updated
upd=[r for r in ok if r['n_rows_in_window']>0]
print('updated-only: n',len(upd),'delivered',q([r['delivered'] for r in upd]),'promised',q([r['gain'] for r in upd]))
xs=[(r['gain'],r['delivered']) for r in upd]; print('spearman updated-only',spearmanr(*zip(*xs)))
print('window length s median',statistics.median([(r['t_end']-r['t_claim'])/1000 for r in ok]))
print('delivered<0 count',sum(r['delivered']<0 for r in ok),'delivered==0',sum(r['delivered']==0 for r in ok))
print('GT unmatched',sum(1 for r in ok if r.get('gt') is None))
json.dump(rows,open(os.environ.get('T3_OUT','t3_rows.json'),'w'),default=str)

print('\nSPLIT  position-delivered',q([r['delivered_pos'] for r in ok]),' existence-delivered (logged classes)',q([r['delivered_exist'] for r in ok]))
