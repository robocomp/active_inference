import numpy as np,glob,os,sys
def load(f):
    L=[l for l in open(f) if not l.startswith('#')]; h=L[0].strip().split(',')
    a=np.array([[float(x) if x not in('nan','-nan','') else np.nan for x in l.strip().split(',')] for l in L[1:] if len(l.strip().split(','))==len(h)])
    return {k:a[:,i] for i,k in enumerate(h)}
B="/home/pbustos/robocomp/components/active_inference/room_concept/"
files=sorted(glob.glob(B+"tmp/heading/heading_*.csv"))+sorted(glob.glob(B+"estimation_scripts/crossing/c4_*/heading_*.csv"))
print("session,min,cycles,calib_on,episodes_end,stop_cyc_share,t_gyro_s,stop_eff_s,move_eff_s,stop_share_eff,mean_g_stop,mean_g_move,wt_ratio,zupt_dt_share")
for f in files:
    d=load(f)
    if np.nanmax(d['calib_applied'])<=0: continue
    n=len(d['ts_ms']); mins=(d['ts_ms'][-1]-d['ts_ms'][0])/6e4
    if mins<5: continue
    g=d['rest_gain_ro']; t=d['hc_t_gyro']; ok=~np.isnan(g)&~np.isnan(t)
    g,t=g[ok],t[ok]; stop=g<0.5
    eff=t*g
    zs=np.nansum(d['zupt_dt'])/max(np.nansum(d['total_dt']),1e-9)
    print(f"{os.path.relpath(f,B).replace('tmp/heading/heading_','').replace('estimation_scripts/crossing/','')},{mins:.1f},{n},1,{int(np.nanmax(d['calib_episodes']))},{stop.mean():.3f},{t.sum():.0f},{eff[stop].sum():.1f},{eff[~stop].sum():.1f},{eff[stop].sum()/eff.sum():.4f},{g[stop].mean():.3f},{g[~stop].mean():.3f},{g[stop].mean()/g[~stop].mean():.3f},{zs:.3f}")
