import json,math,statistics
L=[]
for g in ('old','w2'):
    for r,v in json.load(open(f'm1_{g}.json'))['runs'].items(): L+=v
R9={('c3a_1','chair_94'),('c3a_1','chair_96'),('c3a_1','nightstand_3'),('c3a_2','table_17'),('c3a_3','chair_106'),('c3a_w2_1','nightstand_5'),('c3a_w2_2','chair_214'),('c3a_w2_2','chair_216'),('c3a_w2_2','nightstand_7')}
for l in L:
    if (l['run'],l['node']) in R9 and l['removed_by_exist']:
        tk=l['ticks']; n=len(tk); w0=max(0,n-300)
        win=tk[w0:]; ab=[t for t in win if not t[0]]
        nab=len(ab)
        f1=sum(t[1] for t in ab)/max(nab,1); f2=sum(t[2] for t in ab)/max(nab,1)
        # support ticks in window
        sup=sum(t[0] for t in win)
        # distance/bearing at removal not stored; ok
        s=f"{l['run']} {l['node']} life={n*0.1:.0f}s window={len(win)*0.1:.0f}s supported_ticks={sup} absence_ticks={nab} ({nab*0.1:.1f}s) in_view_frustum={f1:.2f} in_view_LOS={f2:.2f}"
        # last support before removal
        ls=max((i for i,t in enumerate(tk) if t[0]),default=None)
        s+=f" last_support_before_removal={(n-1-ls)*0.1:.1f}s" if ls is not None else ''
        # logged p_detect over absence rows
        rows=[r for r in (l['rows'] or []) if r[0]>=l['end']-30]
        if rows and rows[0][2].get('ex_pdetect') not in (None,''):
            ab_rows=[r for r in rows if r[1]==0]
            pd=[float(r[2]['ex_pdetect']) for r in ab_rows]; ce=[float(r[2]['ex_central']) for r in ab_rows]
            L0=[float(r[2]['ex_L']) for r in rows]
            dec=[(b,a) for a,b in zip(rows,rows[1:]) if float(b[2]['ex_L'])<float(a[2]['ex_L'])-1e-9]
            s+=f" | rows_in_window={len(rows)} npts0_rows={len(ab_rows)} p_detect median={statistics.median(pd):.2f} [{min(pd):.2f},{max(pd):.2f}] central median={statistics.median(ce):.2f} | rows_with_L_decrease={len(dec)}, L {L0[0]:.2f}->{L0[-1]:.2f}"
        else: s+=' | p_detect/central: not logged for chair (no ex_ columns in chair_ai2_log.csv; chair_existence_log.csv not preserved in run folders)'
        print(s)
