import json,re,math,bisect,collections,statistics,sys,csv
import numpy as np
from scipy.stats import spearmanr
sys.argv=['x']
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
import cw
from aff_arbitration import line_times
from pathlib import Path
import os
RUNS=os.environ.get('RUNS','s1_rs035_3 s1_rs035_4 s1_rs035_5 s1_rs100_2 s1_rs100_3 lam_0p2').split()
A={r:cw.analyse(r) for r in RUNS}
PRAG=re.compile(r'^aff_door_\d+_(open|close|cross|approach)$')
def agent_events(run,cls,node,t0,t1):
    ev=[]
    try: L=open(f'{B}{run}/logs/{cls}_concept.log',errors='replace').read().splitlines()
    except: return ev
    for ln in L:
        m=re.match(r'(\d{10}\.\d+) (.*)',ln)
        if not m: continue
        t=float(m.group(1))*1000
        if t0<=t<=t1 and node in m.group(2):
            if 'deleted externally (state=executing)' in m.group(2) or f"node '{node}' deleted" in m.group(2): ev.append('deleted')
            if 'controller completed' in m.group(2) and '→ satisfied' in m.group(2): ev.append('completed')
    return ev
for run in RUNS:
    at=line_times(Path(B+run)); t0=int(open(B+run+'/gt_pose.tsv').readlines()[1].split('\t')[0])
    for r in A[run]:
        r['t_reach']=t0+at(r['ev']['reach_line'])*1000 if 'reach_line' in r['ev'] else None
        ae=agent_events(run,r['cls'],r['node'],r['t_c']-2500,r['t_j']+1500)
        r['agent']=ae
        e=r['ev']; nw=r['next_win'] or ''
        dur=(r['t_j']-r['t_c'])/1000
        if e['refused']: ec='refused: standpoint blocked (occupancy), never moved'
        elif ('unreach' in e or 'noroute' in e) and not r['reached']: ec='unreachable / no route'
        elif r['dwell'] is not None: ec='completed after dwell'
        elif 'deleted' in ae and 'completed' not in ae[:1]: ec='node deleted under claim (belief died)'
        elif PRAG.match(nw): ec='preempted by door mission (required name)'
        elif r.get('run_end') or r['next_win'] is None: ec='run end'
        elif r['next_state_of_me']=='JustCompleted' and dur<1.0: ec='completed <1 s, reason not logged'
        elif r['next_state_of_me']=='JustCompleted': ec='completed, outcome not logged (%.0f s)'%dur
        else: ec='other/unknown'
        r['endclass']=ec
json.dump(A,open('A.json','w'),default=str)
