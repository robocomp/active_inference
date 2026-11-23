import os,sys,csv,math
os.environ['WBT_PATH']=os.path.abspath('world/threerooms-piso.wbt')
sys.path.insert(0,'/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts')
import c3b_report as R
run='/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/c3b_off_hidden1'
legs={int(r['leg']):r for r in csv.DictReader(open(run+'/legs.tsv'),delimiter='\t')}
tret=int(legs[3]['t1_ms']); tleave=int(legs[4]['t0_ms'])
cx,cy=-6.35,8.60
gt=[(int(r['t_ms']),float(r['x']),float(r['y']),float(r['yaw'])) for r in csv.DictReader(open(run+'/gt_pose.tsv'),delimiter='\t')]
print('return t',tret,'leave t',tleave,'return->leave s',(tleave-tret)/1000)
first=None;near=0;rows=[]
for a,b in zip(gt,gt[1:]):
    if b[0]<tret or a[0]>tleave: continue
    d=math.hypot(cx-a[1],cy-a[2]); off=math.degrees(abs(math.remainder(math.atan2(cy-a[2],cx-a[1])-a[3]-math.pi/2,2*math.pi)))
    bl=R.blocked((a[1],a[2]),(cx,cy))
    rows.append((a[0],d,off,bl))
    if d<=2.0:
        if first is None: first=a[0]
        near+=(b[0]-a[0])/1000
print('first within 2 m at +%.1f s after return; leave at +%.1f s => %.1f s before leaving'%((first-tret)/1000,(tleave-tret)/1000,(tleave-first)/1000) if first else 'never within 2m')
print('time within 2 m (return..leave) %.1f s'%near)
print('min distance %.2f m'%min(r[1] for r in rows))
near_rows=[r for r in rows if r[1]<=2.0]
print('while within 2m: n samples',len(near_rows),' blocked by wall',sum(r[3] for r in near_rows),' off-axis>55deg',sum(r[2]>55 for r in near_rows),' in FOV&unblocked',sum((r[2]<=55 and not r[3]) for r in near_rows))
print('within 4m: samples',sum(r[1]<=4 for r in rows),'in FOV (<=55) any wall',sum(r[1]<=4 and r[2]<=55 for r in rows),'unblocked any angle',sum(r[1]<=4 and not r[3] for r in rows),'in view',sum(r[1]<=4 and r[2]<=55 and not r[3] for r in rows))
print('robot pos at first near',[ (round(r[1],2)) for r in near_rows[:1]])
print('---- full tail after return (to end of gt_pose)')
t4s=int(legs[4]['t0_ms']);t4e=int(legs[4]['t1_ms'])
first=None;nr=0;fov=0;allrows=[]
for a,b in zip(gt,gt[1:]):
    if b[0]<tret: continue
    d=math.hypot(cx-a[1],cy-a[2]); off=math.degrees(abs(math.remainder(math.atan2(cy-a[2],cx-a[1])-a[3]-math.pi/2,2*math.pi))); bl=R.blocked((a[1],a[2]),(cx,cy))
    allrows.append((a[0],d,off,bl,a[1],a[2]))
    if d<=2.0:
        if first is None: first=a[0]
        nr+=(b[0]-a[0])/1000
print('within 2 m total after return: %.1f s; first at +%.1f s after return (+%.1f s after leg4 start t0; leg4 t1 at +%.1f s)'%(nr,(first-tret)/1000,(first-t4s)/1000,(t4e-t4s)/1000))
nr2=[r for r in allrows if r[1]<=2.0]
print('samples within 2 m',len(nr2),'blocked',sum(r[3] for r in nr2),'off>55',sum(r[2]>55 for r in nr2),'inview',sum((r[2]<=55 and not r[3]) for r in nr2))
print('window of within-2m: first',round((nr2[0][0]-tret)/1000,1),'last',round((nr2[-1][0]-tret)/1000,1),'robot xy at first',nr2[0][4:6],'at last',nr2[-1][4:6])
print('distance to new spot at leave (leg4 t0):',[round(r[1],2) for r in allrows if abs(r[0]-t4s)<1000][:1])
