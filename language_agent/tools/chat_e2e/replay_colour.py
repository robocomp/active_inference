import csv,math,sys
PAL=[('red',.75,.10,.10),('orange',.95,.55,.10),('yellow',.9,.9,.1),('green',.15,.55,.2),('blue',.15,.3,.8),('purple',.45,.15,.6),('magenta',.9,.2,.9),('pink',1,.6,.75),('brown',.45,.28,.15),('white',.95,.95,.95),('grey',.5,.5,.5),('black',.05,.05,.05)]
def load_pal():
    import re
    src=open('/home/pbustos/robocomp/components/active_inference.worktrees/object-colour/common/appearance_belief/colour_terms.h').read()
    out=[]
    for m in re.finditer(r'\{\s*"(\w+)"\s*,\s*([\d.]+)f?\s*,\s*([\d.]+)f?\s*,\s*([\d.]+)f?',src): out.append((m[1],float(m[2]),float(m[3]),float(m[4])))
    return out or PAL
PAL=load_pal()
def name(m,v,L,vL):
    lp=[]
    for n,r,g,b in PAL:
        s=r+g+b; l=0
        for x,mu,var in ((m[0],r/s,v[0]+.0016),(m[1],g/s,v[1]+.0016),(L,s/3,vL+.0225)):
            l+=-.5*((x-mu)**2/var+math.log(2*math.pi*var))
        lp.append(l)
    mx=max(lp); z=sum(math.exp(a-mx) for a in lp); i=lp.index(mx); return PAL[i][0],math.exp(0)/z
def run(r,sv=0.0,svL=0.0,drift=0.02,pout=0.0,cap=16):
    m=None; v=None; L=None; vL=None; t=None; names=[]
    for x in r:
        tt=int(x['t_ms'])/1000
        if t is not None:
            q=drift**2*(tt-t); v=[min(a+q,.0841) for a in v]; vL=min(vL+q,.0841)
        t=tt
        c=[float(x['in_r']),float(x['in_g']),float(x['in_b'])]; cv=[float(x['in_var_r']),float(x['in_var_g']),float(x['in_var_b'])]; n=float(x['in_n'])
        l=float(x['in_L']); lv=float(x['in_var_L']); nl=float(x['in_n_L'])
        ne=cap*n/(cap+n); nel=cap*nl/(cap+nl)
        R=[max(a,1e-9)/ne+sv**2 for a in cv]; RL=max(lv,1e-9)/nel+svL**2
        if m is None: m=c[:]; v=[.0841]*3; L=l; vL=.0841
        w=1.0
        if pout>0:
            # predictive density of this frame under the belief vs a broad 'not this object' density
            lin=0
            for k in (0,1):
                s2=v[k]+R[k]; lin+=-.5*((c[k]-m[k])**2/s2+math.log(2*math.pi*s2))
            s2=vL+RL; lin+=-.5*((l-L)**2/s2+math.log(2*math.pi*s2))
            lout=math.log(2.0)+0.0  # uniform over simplex (area 1/2 in r,g) x uniform L on [0,1]
            a=math.log(1-pout)+lin; b=math.log(pout)+lout; mx=max(a,b)
            w=math.exp(a-mx)/(math.exp(a-mx)+math.exp(b-mx))
        for k in range(3):
            pp=1/v[k]; op=w/R[k]; m[k]=(pp*m[k]+op*c[k])/(pp+op); v[k]=1/(pp+op)
        s=sum(max(a,0) for a in m); m=[max(a,0)/s for a in m]
        pp=1/vL; op=w/RL; L=(pp*L+op*l)/(pp+op); vL=1/(pp+op)
        names.append(name(m,v,L,vL))
    return names
def summ(ns):
    d={}
    for n,p in ns: d[n]=d.get(n,0)+1
    return d, round(min(p for n,p in ns),2), round(ns[-1][1],2)
files=sys.argv[1:]
for f in files:
    rows=list(csv.DictReader(open(f)))
    for node in ('chair_34','chair_35'):
        r=[x for x in rows if x['node']==node]
        print('==',f,node,len(r))
        for lab,kw in [('current',{}),('sv=.005',dict(sv=.005,svL=.03)),('sv=.01',dict(sv=.01,svL=.05)),('mix p.05',dict(pout=.05)),('sv.005+mix',dict(sv=.005,svL=.03,pout=.05)),('drift .002',dict(drift=.002)),('sv.005+drift.002',dict(sv=.005,svL=.03,drift=.002))]:
            print(f'  {lab:18s}',*summ(run(r,**kw)))
