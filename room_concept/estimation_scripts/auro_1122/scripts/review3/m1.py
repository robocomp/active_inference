"""Miss-counter counterfactual + view analysis of existence removals. usage: m1.py <group: old|w2> -> m1_<group>.json"""
import os, sys, re, csv, glob, json, math, bisect, collections
GROUP = sys.argv[1]
SC = '/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts'
SCR = '/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/1c86b8d6-47c4-4a7b-8f4b-04a583e967d4/scratchpad/review3'
if GROUP == 'old':
    os.environ['WBT_PATH'] = '/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/1c86b8d6-47c4-4a7b-8f4b-04a583e967d4/scratchpad/review/world/threerooms-piso.wbt'
    RUNS = ['c3a_1', 'c3a_2', 'c3a_3', 'c3a_cab1']
else:
    os.environ['WBT_PATH'] = '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt'
    os.environ['C3A_POSE_OVERRIDE'] = 'chair(3)=-7.4323,8.5130'
    RUNS = ['c3a_w2_1', 'c3a_w2_2', 'c3a_w2_3']
sys.path.insert(0, SC)
import wbt_rooms as W
import c3a_objects as C
from run_frames import make_to_world
B = '/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/'
GT = C.gt_objects(os.environ['WBT_PATH'])
walls, doors, meshes = W.parse_wbt(W.WBT)
occ = W.rasterise(walls, {k: dict(v, width=0.0) for k, v in doors.items()}, meshes)
x0, _, y0, _ = W.BOX
def blocked(a, b):
    n = int(math.hypot(b[0]-a[0], b[1]-a[1]) / 0.02)
    for i in range(2, n-2):
        x = a[0]+(b[0]-a[0])*i/n; y = a[1]+(b[1]-a[1])*i/n
        if occ[int((y-y0)/W.RES), int((x-x0)/W.RES)]: return True
    return False

HFOV = 55.0; RANGE = float(os.environ.get('RNG','6')); TICK = 0.1
CLASSES = ['chair', 'table', 'nightstand', 'bed', 'toilet', 'hood', 'refrigerator']
REAL9 = {('c3a_1','chair_94'),('c3a_1','chair_96'),('c3a_1','nightstand_3'),('c3a_2','table_17'),('c3a_3','chair_106'),
         ('c3a_w2_1','nightstand_5'),('c3a_w2_2','chair_214'),('c3a_w2_2','chair_216'),('c3a_w2_2','nightstand_7')}

def load_pose(R):
    return [(int(r['t_ms'])/1000, float(r['x']), float(r['y']), float(r['yaw'])) for r in csv.DictReader(open(R+'/gt_pose.tsv'), delimiter='\t')]
def pose_at(g, ts, t):
    i = bisect.bisect_right(ts, t)-1
    if i < 0 or i+1 >= len(g): return None
    a, b = g[i], g[i+1]
    if b[0]-a[0] > 2.0: return None
    f = (t-a[0])/(b[0]-a[0]); dy = math.remainder(b[3]-a[3], 2*math.pi)
    return (a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2]), a[3]+f*dy)
def inview(p, o, los):
    if p is None: return False
    x, y, yaw = p
    d = math.hypot(o[0]-x, o[1]-y)
    if d > RANGE: return False
    if abs(math.remainder(math.atan2(o[1]-y, o[0]-x)-yaw-math.pi/2, 2*math.pi)) > math.radians(HFOV): return False
    if los and blocked((x, y), o): return False
    return True

out = {'runs': {}}
for run in RUNS:
    R = B+run
    tw = make_to_world(R)
    gp = load_pose(R); gts = [p[0] for p in gp]
    inst_csv = {r['name']: r for r in csv.DictReader(open(R+'/c3a_instances.csv'))}
    res = []
    for cls in CLASSES:
        fcsv = f'{R}/{cls}_ai2_log.csv'; flog = f'{R}/logs/{cls}_concept.log'
        if not os.path.exists(fcsv) or not os.path.exists(flog): continue
        L = open(flog, errors='replace').read().splitlines()
        births = collections.defaultdict(list); idnode = {}; ends = collections.defaultdict(list); exrm = {}
        for ln in L:
            m = re.match(r"([\d.]+) .*\[tracker\] BIRTH '(\w+)' id=(\d+)", ln)
            if m: births[m.group(2)].append(float(m.group(1))); idnode[m.group(3)] = m.group(2)
            m = re.match(r"([\d.]+) \[affordance\] removed node for '(\w+)'", ln)
            if m: ends[m.group(2)].append(float(m.group(1)))
            m = re.match(r'([\d.]+) .*\[existence\] (?:REMOVE id=|removing \w+ id=)(\d+)', ln)
            if m: exrm.setdefault(idnode.get(m.group(2)), []).append(float(m.group(1)))
        t_run_end = float(gp[-1][0])
        rows = list(csv.DictReader(open(fcsv)))
        pernode = collections.defaultdict(list)
        anchor_ts = None; anchor_cy = None; prev_cy = -1
        for r in rows:
            cy = int(r['cycle'])
            if cls == 'chair':
                t = float(r['t_ms'])/1000
            else:
                ts = float(r['pkt_ts'])/1000
                if cy < prev_cy: anchor_ts = None
                if anchor_ts is None or ts != anchor_ts: anchor_ts, anchor_cy = ts, cy; t = ts
                else: t = anchor_ts + (cy-anchor_cy)*TICK
            prev_cy = cy
            pernode[r['node']].append((t, cy, float(r['npts']), r))
        for node, rs in pernode.items():
            bt = sorted(births.get(node, []))
            if not bt: bt = [rs[0][0]]
            for li, b in enumerate(bt):
                nb = bt[li+1] if li+1 < len(bt) else 1e18
                lr = [x for x in rs if b-0.5 <= x[0] < nb-0.5]
                if not lr: continue
                end_c = [e for e in ends.get(node, []) if e > b]
                end = min([e for e in end_c if e < nb] + [t_run_end])
                exr = [e for e in exrm.get(node, []) if b < e <= end+0.5]
                removed_by_exist = bool(exr)
                if exr: end = min(end, exr[0])
                lr = [x for x in lr if x[0] <= end+0.5]
                if not lr: continue
                ws = [tw(float(x[3]['cx']), float(x[3]['cy']), int(x[0]*1000)) for x in lr[::max(1, len(lr)//30)]]
                ws = [w for w in ws if w]
                if not ws: continue
                wx = sorted(w[0] for w in ws)[len(ws)//2]; wy = sorted(w[1] for w in ws)[len(ws)//2]
                ic = inst_csv.get(node)
                verdict = ic['verdict'] if ic else '?'
                gtn = ic['gt_object'] if ic else ''
                g = [o for o in GT if o[0] == gtn and o[1] == cls]
                opos = (g[0][2], g[0][3]) if g else (wx, wy)
                n = int((end-b)/TICK)+1
                sup_ticks = set(int((x[0]-b)/TICK) for x in lr if x[2] > 0)
                rowpres = collections.defaultdict(lambda: [0, 0])
                for x in lr:
                    k = int((x[0]-b)/TICK); rowpres[k][0] += 1; rowpres[k][1] += (x[2] > 0)
                ticks = []
                for k in range(n):
                    t = b+k*TICK
                    p = pose_at(gp, gts, t)
                    fr = inview(p, opos, False); lo = inview(p, opos, True)
                    ticks.append((int(k in sup_ticks), int(fr), int(lo), int(k in rowpres and rowpres[k][1] == 0)))
                res.append(dict(run=run, cls=cls, node=node, life=li, born=b, end=end, removed_by_exist=removed_by_exist,
                                verdict=verdict, gt=gtn, pos=(wx, wy), opos=opos, nrows=len(lr), ticks=ticks,
                                rows=[(x[0], x[2], {k: x[3].get(k) for k in ('ex_pdetect', 'ex_central', 'ex_sndet', 'ex_L', 'ex_streak', 'ex_socc', 'ex_sfree')}) for x in lr] if (run, node) in REAL9 else None))
    out['runs'][run] = res
    print(run, len(res), 'lives', file=sys.stderr)
json.dump(out, open(f'{SCR}/m1_{GROUP}{os.environ.get('RNG','6')}.json', 'w'))
