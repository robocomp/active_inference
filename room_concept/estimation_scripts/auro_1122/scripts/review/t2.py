import re,csv,glob,collections,sys
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS='c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3'.split()
OBJ='chair table nightstand bed toilet hood refrigerator bottle'.split()
tab=collections.defaultdict(collections.Counter)
det=[]
for run in RUNS:
    legs=[(int(r['t0_ms'])/1000,int(r['t1_ms'])/1000) for r in csv.DictReader(open(B+run+'/legs.tsv'),delimiter='\t')]
    for a in OBJ:
        p=B+f'{run}/logs/{a}_concept.log'
        try: L=open(p,errors='replace').read().splitlines()
        except FileNotFoundError: continue
        for ln in L:
            m=re.match(r'([\d.]+) (.*)',ln); 
            t=float(m.group(1)) if m else None
            if re.search(r"\[tracker\] BIRTH '",ln): tab['births'][a]+=1; tab['births']['_all']+=1
            if re.search(r"\[tracker\] RESURRECT",ln): tab['resurrect'][a]+=1; tab['resurrect']['_all']+=1
            if re.search(r"\[existence\] REMOVE id=|\[existence\] removing ",ln): tab['rm_existence'][a]+=1; tab['rm_existence']['_all']+=1; det.append((run,a,'existence',t,ln[:140]))
            if re.search(r"\[tracker\] DEATH",ln): tab['rm_tracker_miss'][a]+=1; tab['rm_tracker_miss']['_all']+=1
            if re.search(r"\[\w+-class\] removing id=",ln): tab['rm_shape_argued'][a]+=1; tab['rm_shape_argued']['_all']+=1; det.append((run,a,'shape',t,ln[:140]))
            m2=re.search(r"deaths=(\d+)",ln)
            if m2 and int(m2.group(1))>0: tab['deaths_counter_gt0'][a]+=1
            mm=re.search(r"\[affordance\] removed node for '(\w+)' \(state=idle\)",ln)
            if mm and t:
                inleg=any(l0-1<=t<=l1+1 for l0,l1 in legs)
                # also within 15 s after leg start
                tab['rm_exit_cleanup' if inleg else 'rm_idle_other'][a]+=1
                tab['rm_exit_cleanup' if inleg else 'rm_idle_other']['_all']+=1
                det.append((run,a,'affnode_removed '+('in-leg' if inleg else 'outside-leg'),t,mm.group(1)))
for k,v in tab.items(): print(k,dict(v))
print()
import json
for d in det:
    if d[2]!='affnode_removed in-leg': print(d)
