import re,csv,collections
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
RUNS='c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3'.split()
T=collections.Counter()
for run in RUNS:
    L=open(B+run+'/logs/ltsm_agent.log',errors='replace').read().splitlines()
    for ln in L:
        if '[promotion] PROMOTING' in ln: T['promotions']+=1
        if '[passage] passage_' in ln and ' match door_' in ln: T['passage_match']+=1
        if '[recog]' in ln and 'not eligible' in ln and 'passage in memory' in ln: T['recog_not_eligible_passage']+=1
        if '[recog] episode begins' in ln: T['recog_episode_begins']+=1
        if '[recog] SWAP requested' in ln: T['recog_SWAP_requested']+=1
        if 'is a known door' in ln and 'recognition runs' in ln: T['recog_runs_known_door_no_recall']+=1
        if '[recog] swap to room_id' in ln: T['recog_swap_executed']+=1
        if '[recog] recognition node posted' in ln: T['recog_node_posted']+=1
    # controller
    for ln in open(B+run+'/logs/controller.log',errors='replace'):
        m=re.search(r'\[mission\] step (\d+):(\w+)',ln)
        if m and ln.rstrip().endswith('started'): T['mission_step_'+m.group(2)]+=1
    rows=list(csv.DictReader(open(B+run+'/aff_arbitration.csv')))
    for r in rows:
        nc=len(r['concepts'].split(';'))
        T['aff_select_total']+=1
        T['aff_contested_>=2concepts' if nc>=2 else 'aff_single_concept']+=1
        T['aff_n_eligible_>=2' if int(r['n_eligible'])>=2 else 'aff_n_eligible_1']+=1
        if r['winner_kind'] in('door_open','door_close','door_cross','door_approach'): T['aff_winner_pragmatic_door']+=1
for k in sorted(T): print(k,T[k])
