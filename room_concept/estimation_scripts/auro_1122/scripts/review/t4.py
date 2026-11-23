import os,sys,re,csv,glob,bisect,collections
SC='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts'
sys.path.insert(0,SC)
from run_frames import make_to_world
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS='c3a_1 c3a_2 c3a_3 c3a_cab1 c3a_w2_1 c3a_w2_2 c3a_w2_3'.split()
for run in RUNS:
    R=B+run
    if not os.path.isdir(R): print(run,'MISSING'); continue
    extra=[os.path.basename(p) for p in glob.glob(R+'/*') if re.search(r'remov|move',os.path.basename(p))]
    legs=[(int(r['leg']),int(r['t0_ms'])/1000,int(r['t1_ms'])/1000,r['arg']) for r in csv.DictReader(open(R+'/legs.tsv'),delimiter='\t')]
    gt0=int(open(R+'/gt_pose.tsv').readlines()[1].split('\t')[0])/1000
    try: tw=make_to_world(R)
    except Exception as e: tw=None; print('to_world fail',e)
    print(f'== {run}: files {extra}; legs {[(l[0],l[3],round(l[1]-gt0),round(l[2]-gt0)) for l in legs]}')
    for f in sorted(glob.glob(R+'/logs/*_concept.log')):
        a=os.path.basename(f).replace('_concept.log','')
        L=open(f,errors='replace').read().splitlines()
        idmap={}
        for ln in L:
            m=re.search(r"BIRTH '(\w+)' id=(\d+)",ln) or re.search(r"RESURRECT (\w+) u?\w* ?id=(\d+)",ln)
            if m: idmap[m.group(2)]=m.group(1)
        for ln in L:
            m=re.match(r'([\d.]+) .*\[existence\] (?:REMOVE id=|removing \w+ id=)(\d+)',ln)
            m2=re.match(r'([\d.]+) .*\[tracker\] DEATH id=(\d+)',ln)
            m3=re.match(r'([\d.]+) .*\[\w+-class\] removing id=(\d+)',ln)
            for mm,kind in ((m,'existence'),(m2,'tracker-miss'),(m3,'shape-argued')):
                if not mm: continue
                t=float(mm.group(1)); uid=mm.group(2); node=idmap.get(uid,'?')
                pos=None
                try:
                    rows=[(int(r.get('t_ms') or r['pkt_ts']),float(r['cx']),float(r['cy'])) for r in csv.DictReader(open(R+f'/{a}_ai2_log.csv')) if r['node']==node]
                    rows.sort(); k=bisect.bisect_right([x[0] for x in rows],t*1000)-1
                    if k>=0:
                        w=tw(rows[k][1],rows[k][2],rows[k][0]) if tw else None
                        pos=(round(w[0],2),round(w[1],2),round((t*1000-rows[k][0])/1000,1)) if w else None
                except FileNotFoundError: pass
                inleg=[l[0] for l in legs if l[1]<=t<=l[2]]
                print(f'  t={t-gt0:7.1f}s agent={a} node={node} uid={uid} cause={kind} world_pos(x,y,lag_s)={pos} during_leg={inleg}')
