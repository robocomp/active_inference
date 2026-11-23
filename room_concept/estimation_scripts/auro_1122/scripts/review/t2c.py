import re,csv,collections
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS='c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3'.split()
OBJ='chair table nightstand bed toilet hood refrigerator bottle'.split()
print('run | births | resurrected | exist-removals | exit-cleanup(in-leg) | promotions | passage-lookup | recogniser swaps | aff-selects (contested/single) | mission steps')
for run in RUNS:
    c=collections.Counter()
    legs=[(int(r['t0_ms'])/1000,int(r['t1_ms'])/1000) for r in csv.DictReader(open(B+run+'/legs.tsv'),delimiter='\t')]
    for a in OBJ:
        try: L=open(B+f'{run}/logs/{a}_concept.log',errors='replace').read().splitlines()
        except: continue
        for ln in L:
            m=re.match(r'([\d.]+) ',ln); t=float(m.group(1)) if m else 0
            if "[tracker] BIRTH '" in ln: c['b']+=1
            if '[tracker] RESURRECT' in ln: c['res']+=1
            if re.search(r"\[existence\] REMOVE id=|\[existence\] removing ",ln): c['ex']+=1
            if re.search(r"\[affordance\] removed node for '\w+' \(state=idle\)",ln) and any(l0-1<=t<=l1+1 for l0,l1 in legs): c['exit']+=1
    for ln in open(B+run+'/logs/ltsm_agent.log',errors='replace'):
        if '[promotion] PROMOTING' in ln: c['prom']+=1
        if '[passage] passage_' in ln and ' match door_' in ln: c['pass']+=1
        if '[recog] SWAP requested' in ln: c['swap']+=1
    rows=list(csv.DictReader(open(B+run+'/aff_arbitration.csv')))
    con=sum(len(r['concepts'].split(';'))>=2 for r in rows)
    ms=collections.Counter()
    for ln in open(B+run+'/logs/controller.log',errors='replace'):
        m=re.search(r'\[mission\] step (\d+):(\w+).*: started$',ln.strip())
        if m: ms[m.group(2)]+=1
    print(run,'|',c['b'],'|',c['res'],'|',c['ex'],'|',c['exit'],'|',c['prom'],'|',c['pass'],'|',c['swap'],'|',f'{con}/{len(rows)-con}','|',dict(ms))
