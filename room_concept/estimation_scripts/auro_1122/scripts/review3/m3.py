import sys,re,csv,collections,json,math
import numpy as np
sys.path.insert(0,'/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts')
import wbt_rooms as W
RM,_,_=W.rooms()
B='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
def true_room(x,y):
    hits=[n for n,r in RM.items() if W.inside(r['poly'],np.array([[x,y]]))[0]]
    if len(hits)==1: return hits[0]
    # doorway / outside: nearest seed
    d=min(W.SEEDS.items(),key=lambda kv: math.hypot(kv[1][0]-x,kv[1][1]-y))
    return (d[0]+'?')
rows=[]
for k in range(62,77):
    run=f'clean{k}'; R=B+run
    L=open(R+'/logs/ltsm_agent.log',errors='replace').read().splitlines()
    gp=[(int(r['t_ms']),float(r['x']),float(r['y']),int(r['ltsm_agent'])) for r in csv.DictReader(open(R+'/gt_pose.tsv'),delimiter='\t')]
    t0=gp[0][0]
    er={r['room']:r for r in csv.DictReader(open(R+'/explore_rooms.csv'))}
    start_truth=true_room(gp[0][1],gp[0][2])
    def pos_at(line):
        best=gp[0]
        for g in gp:
            if g[3]<=line: best=g
            else: break
        return best
    # current provisional name tracking
    prov=None; swaps=[]
    for i,ln in enumerate(L,1):
        m=re.match(r'\[recog\] episode begins: provisional (\S+) behind (\S+)',ln)
        if m: prov=(m.group(1),m.group(2))
        m=re.match(r'\[recog\] SWAP requested: room_id (\d+) \(posterior ([\d.]+)',ln)
        if m: swaps.append(dict(run=run,line=i,K=int(m.group(1)),post=float(m.group(2)),prov=prov,verdict='none',reason='',exec=False))
        if ln.startswith('[recog] swap to room_id') and swaps: swaps[-1]['exec']=True
        m=re.match(r'\[proposal\] proposal_room_\d+ posted under (\S+):',ln)
        if m and swaps: swaps[-1]['posted_under']=m.group(1)
        m=re.match(r'\[swap\] (ACCEPTED|WITHDRAWN): (.*)',ln)
        if m and swaps and swaps[-1]['verdict']=='none':
            swaps[-1]['verdict']=m.group(1); swaps[-1]['reason']=m.group(2)[:90]
    def truth_of(name):
        r=er.get(name)
        if r and r['truth'] not in ('-',''): return r['truth']
        m=re.match(r'room_(\d+)',name)
        if m:
            key='#%d'%(int(m.group(1))-1)
            r=er.get(key)
            if r:
                tb=float(r['born_s'])+5
                g=min(gp,key=lambda g:abs((g[0]-t0)/1000-tb))
                return true_room(g[1],g[2]).rstrip('?')
        return '?'
    for s in swaps:
        s['t_s']=-1
        s['true_here']=truth_of(s['prov'][0]) if s['prov'] else '?'
        s['true_K']=truth_of(f"room_{s['K']}")
        s['correct_claim']= (s['true_here'].rstrip('?')==s['true_K'])
        rows.append(s)
print(len(rows))
print(collections.Counter((s['verdict'],s['correct_claim']) for s in rows))
print(collections.Counter(s['exec'] for s in rows))
for s in rows: print(s['run'],'L',s['line'],'t',s['t_s'],'P',s['prov'],'K',s['K'],'post',s['post'],'here',s['true_here'],'trueK',s['true_K'],'claim_correct',s['correct_claim'],'->',s['verdict'],s['reason'])
json.dump(rows,open('m3.json','w'))
