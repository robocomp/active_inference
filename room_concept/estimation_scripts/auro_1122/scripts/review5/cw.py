import os,sys,re,csv,math,bisect,json,collections
os.environ.setdefault('WBT_PATH','/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')
SC='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts'
sys.path.insert(0,SC)
from aff_arbitration import SEL,CAND,producer,line_times
from run_frames import make_to_world
import c3a_objects as C
import wbt_rooms as W
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
GT=C.gt_objects(os.environ['WBT_PATH']); GTd={g[0]:g for g in GT}
walls,doors,meshes=W.parse_wbt(W.WBT); occ=W.rasterise(walls,{k:dict(v,width=0.0) for k,v in doors.items()},meshes); x0,_,y0,_=W.BOX
def blocked(a,b):
    n=int(math.hypot(b[0]-a[0],b[1]-a[1])/0.02)
    for i in range(2,n-2):
        x=a[0]+(b[0]-a[0])*i/n; y=a[1]+(b[1]-a[1])*i/n
        if occ[int((y-y0)/W.RES),int((x-x0)/W.RES)]: return True
    return False
def inview(p,yaw,o):
    d=math.dist(p,o)
    if d>4.0: return False
    if abs(math.remainder(math.atan2(o[1]-p[1],o[0]-p[0])-yaw-math.pi/2,2*math.pi))>math.radians(55): return False
    return not blocked(p,o)
H=lambda p: 0.0 if not(0<p<1) else -(p*math.log(p)+(1-p)*math.log(1-p))
def analyse(run):
    R=B+run; at=line_times(Path(R)); tw=make_to_world(R)
    pose=[(int(r['t_ms']),float(r['x']),float(r['y']),float(r['yaw'])) for r in csv.DictReader(open(R+'/gt_pose.tsv'),delimiter='\t')]
    gt0=pose[0][0]; pts=[p[0] for p in pose]
    inst={}
    try:
        for r in csv.DictReader(open(R+'/c3a_instances.csv')): inst[r['name']]=r
    except FileNotFoundError: pass
    L=open(R+'/logs/controller.log',errors='replace').read().splitlines()
    sels=[]
    for i,ln in enumerate(L,1):
        m=SEL.search(ln)
        if m:
            st={n:s for n,s,g,sc in CAND.findall(m.group(4))}
            sels.append(dict(i=i,win=m.group(1),gain=float(m.group(3)),st=st,sc={n:float(sc) for n,s,g,sc in CAND.findall(m.group(4))}))
    cache={}
    def ai(cls):
        if cls in cache: return cache[cls]
        d=collections.defaultdict(list)
        try:
            for r in csv.DictReader(open(R+f'/{cls}_ai2_log.csv')):
                try: d[r['node']].append((int(r.get('t_ms') or r['pkt_ts']),float(r['cx']),float(r['cy']),float(r['std_cx']),float(r['std_cy']),float(r['ex_p']) if r.get('ex_p') not in (None,'') else float('nan'),int(r.get('gated') or 0)))
                except: pass
        except FileNotFoundError: pass
        for k in d: d[k].sort()
        cache[cls]=d; return d
    out=[]
    for k,s in enumerate(sels):
        cc,_=producer(s['win'])
        if cc in('room','door'): continue
        node=s['win'][4:]; i=s['i']; j=sels[k+1]['i'] if k+1<len(sels) else len(L)+1
        seg=L[i:j-1]
        ev=dict(reached=False,locked=False,looked=False,dwell=None,refused=False,none_elig=None,giveup=False)
        own_end=j
        for n,ln in enumerate(seg,i+1):
            if f"{s['win']} reached ->" in ln and not ev['reached']: ev['reached']=True; ev['reach_line']=n
            if 'lock-on LOCKED' in ln: ev['locked']=True
            if 'orient LOOKED' in ln: ev['looked']=True
            if 'lock-on GIVE_UP' in ln: ev['giveup']=True
            for key,pat in (('unreach','reporting Unreachable to the producer'),('noroute','STOPPED NO ROUTE'),('outcontest',f"'{s['win']}' is out of this contest"),('timeout','timed out'),('arrived_policy',f"[policy] {s['win']} (episode"),('wait_door','STOPPED WAITING — door passage'),('hold_rate','POLICY HOLD')):
                if pat in ln and key not in ev: ev[key]=n
            m=re.search(rf"dwell over: '{node}' mask seen (\d+)/(\d+)",ln)
            if m and ev['dwell'] is None: ev['dwell']=int(m.group(1)); own_end=min(own_end,n)
            if f"TARGET REFUSED — '{s['win']}'" in ln and not ev['refused']: ev['refused']=True; own_end=min(own_end,n)
            if 'NONE ELIGIBLE' in ln and ev['none_elig'] is None: ev['none_elig']=n; own_end=min(own_end,n)
        nxt=sels[k+1] if k+1<len(sels) else None
        rec=dict(ev={k:v for k,v in ev.items()},run=run,i=i,node=node,cls=cc,gain=s['gain'],j=j,own_end=own_end,reached=ev['reached'],locked=ev['locked'],looked=ev['looked'],dwell=ev['dwell'],refused=ev['refused'],giveup=ev['giveup'],
                 next_win=nxt['win'] if nxt else None,next_state_of_me=(nxt['st'].get(s['win']) if nxt else None),
                 next_runner_score=(nxt['sc'].get(nxt['win']) if nxt else None),my_next_score=(nxt['sc'].get(s['win']) if nxt else None))
        rec['t_c']=gt0+at(i)*1000; rec['t_j']=gt0+at(j)*1000 if nxt else pose[-1][0]; rec['t_own']=gt0+at(own_end)*1000 if own_end<=len(L) else rec['t_j']
        if nxt is None: rec['run_end']=True
        rec['t_own']=min(rec['t_own'],rec['t_j'])
        Lr=ai(cc).get(node); rec['rows']=bool(Lr)
        if Lr:
            ts=[x[0] for x in Lr]
            def delivered(t0,t1):
                kb=bisect.bisect_right(ts,t0)-1; ka=bisect.bisect_right(ts,t1)-1
                if kb<0: return None
                b=Lr[kb]; a=Lr[ka]
                if min(a[3],a[4],b[3],b[4])<=0: pos=float('nan')
                else: pos=0.5*math.log((b[3]**2*b[4]**2)/(a[3]**2*a[4]**2))
                dex=H(b[5])-H(a[5]) if (b[5]==b[5] and a[5]==a[5]) else float('nan')
                nu=sum(1 for q in Lr[kb+1:ka+1] if not q[6])
                return dict(pos=pos,ex=dex,tot=pos+(dex if dex==dex else 0.0),nrows=ka-kb,nupd=nu,lag=(t0-b[0])/1000)
            rec['t_reach']=(gt0+at(ev['reach_line'])*1000) if 'reach_line' in ev else None
            rec['d_reach']=delivered(rec['t_reach'],rec['t_own']) if rec['t_reach'] else None
            rec['d_j']=delivered(rec['t_c'],rec['t_j']); rec['d_own']=delivered(rec['t_c'],rec['t_own'])
            ref=Lr[-1]; rec['belief_xy']=(ref[1],ref[2],ref[0])
        # GT object position
        gp=None; src=None; ir=inst.get(node)
        if ir and ir.get('verdict')=='match' and ir['gt_object'] in GTd: gp=GTd[ir['gt_object']][2:4]; src='match'
        if gp is None and Lr:
            m_=[ (math.dist(tw(Lr[-1][1],Lr[-1][2],Lr[-1][0]) or (1e9,1e9),(g[2],g[3])),g) for g in GT if g[1]==cc]
            m_=[c for c in m_ if c[0]<=1.0]
            if m_: gp=min(m_,key=lambda c:c[0])[1][2:4]; src='nearest<=1m'
        rec['gt']=gp; rec['gt_src']=src or (ir.get('verdict') if ir else 'no-instance')
        # seconds in view within [t_c,t_own] and [t_c,t_j]
        def siv(t0,t1):
            if gp is None: return None
            k0=max(bisect.bisect_left(pts,t0)-1,0); tot=0.0
            for q in range(k0,len(pose)-1):
                a=pose[q]; b=pose[q+1]
                if a[0]>=t1: break
                lo=max(a[0],t0); hi=min(b[0],t1)
                if hi>lo and inview((a[1],a[2]),a[3],gp): tot+=(hi-lo)/1000
            return tot
        rec['siv_own']=siv(rec['t_c'],rec['t_own']); rec['siv_j']=siv(rec['t_c'],rec['t_j'])
        # robot-to-object min distance in window
        out.append(rec)
    return out
if __name__=='__main__':
    allr={}
    for run in sys.argv[1:]:
        allr[run]=analyse(run)
    json.dump(allr,open(os.path.join(os.path.dirname(__file__),'cw_rows.json'),'w'),default=str)
    print({k:len(v) for k,v in allr.items()})
