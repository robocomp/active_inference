import sys, os, json, math, numpy as np
os.environ["WBT_PATH"]=sys.argv[1]
sys.path.insert(0,"/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts")
import wbt_rooms as W
from shapely.geometry import Polygon
from scipy.optimize import minimize
P="/home/pbustos/drive/Papers/Papers Ongoing/2026/Autonomous Robots /data/fig9/"
GT,_,_=W.rooms()
GT={k:Polygon(v["poly"]).buffer(0) for k,v in GT.items()}
print({k:round(g.area,2) for k,g in GT.items()})
def Rm(a): return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def iou(a,b):
    return a.intersection(b).area/a.union(b).area
def xf(poly,th,t): return Polygon(np.asarray(poly)@Rm(th).T+t).buffer(0)
def fit_room(local,gt,init):
    f=lambda p:-iou(xf(local,p[0],p[1:]),gt)
    best=None
    for d in [(0,0,0),(.05,0,0),(-.05,0,0),(0,.1,0),(0,0,.1)]:
        x0=np.array([init[0]+d[0],init[1]+d[1],init[2]+d[2]])
        r=minimize(f,x0,method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-7,maxiter=600))
        if best is None or r.fun<best.fun: best=r
    return best.x,-best.fun
out={}
for run in sys.argv[2:]:
    d=json.load(open(P+f"{run}_layout_dump.json"))
    TT=d["truth_transform"]; R0=np.array(TT["R"]); t0=np.array(TT["t"]); th0=math.atan2(R0[1,0],R0[0,0])
    res=[]
    for r in d["rooms"]:
        k=d["truth_map"][r["name"]]; gt=GT[k["truth"]]
        loc={}
        for tag in ("before","after"):
            x,y,yw=r[tag+"_pose"]; loc[tag]=(np.asarray(r[tag])-[x,y])@Rm(yw)   # row-vec @ R = R^T v
        dl=np.abs(loc["before"]-loc["after"]).max()
        xa,ya,ywa=r["after_pose"]
        # initial: dump's world transform composed with the after pose
        init=(th0+ywa, R0@np.array([xa,ya])+t0)
        sol,i=fit_room(loc["after"],gt,(init[0],*init[1]))
        res.append(dict(room=r["name"],truth=k["truth"],kind=k["kind"],before=r["before_pose"],after=r["after_pose"],T=[sol[0],sol[1],sol[2]],iou_fit=i,dloc=dl))
    # per-room truth frame T_i (room frame->world). layout pose L_i (room frame->layout frame). Layout->world A rigid fit over rooms.
    def err(tag,A=None):
        # fit A=(thA,tA) minimizing sum_i ||A L_i - T_i||^2 over translation, with rotation by circular mean of (T_th - L_th)
        Ls=[(np.array(r[tag][:2]),r[tag][2]) for r in res]; Ts=[(np.array(r["T"][1:]),r["T"][0]) for r in res]
        # least squares on pose: rotation = circular mean; translation then LS
        th=math.atan2(sum(math.sin(t[1]-l[1]) for l,t in zip(Ls,Ts)),sum(math.cos(t[1]-l[1]) for l,t in zip(Ls,Ts)))
        tt=np.mean([t[0]-Rm(th)@l[0] for l,t in zip(Ls,Ts)],axis=0)
        e=[]
        for l,t in zip(Ls,Ts):
            p=Rm(th)@l[0]+tt; e.append((float(np.linalg.norm(p-t[0])), math.degrees(wrap(th+l[1]-t[1]))))
        return e,th,tt
    for tag in ("before","after"):
        e,th,tt=err(tag)
        for r,(a,b) in zip(res,e): r[tag+"_err"]=(a,b)
    out[run]=res
    print(run)
    for r in res: print("  ",r["room"],r["kind"],r["truth"],"fitIoU %.3f"%r["iou_fit"],"dloc %.1e"%r["dloc"],"before %.3f m %.2f deg"%r["before_err"],"after %.3f m %.2f deg"%r["after_err"])
json.dump(out,open(sys.argv[0].replace("layout_err.py","layout_err.json"),"w"),default=float)
