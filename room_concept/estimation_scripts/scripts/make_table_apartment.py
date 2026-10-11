import sys,pandas as pd,numpy as np
df=pd.read_csv(sys.argv[1]); st=pd.read_csv(sys.argv[2]).set_index('id')
L=[r'\begin{table}[t]',r'\centering',r'\caption{Room-layout estimation from ten start poses in the simulated apartment (one run per start). IoU of the layout against the ground-truth floor plan after the best rigid alignment (shape only; precision and recall use the trajectory alignment), at promotion (estimating $\to$ localising) and after one minute of refinement. $t_p$: time to promotion; $d_p$: path driven until then; $e$: median position error of plain SLAM.}',r'\label{tab:estimation-campaign}',
r'\begin{tabular}{r r r r r r r r r r}',r'\toprule',
r'start & $t_p$ (s) & $d_p$ (m) & $e$ (cm) & IoU$_p$ & IoU$_r$ & prec$_r$ & rec$_r$ & $|V|_r$ \\',r'\midrule']
for _,r in df.iterrows():
    L.append(f"{int(r.run)} & {r.t_promote_s:.0f} & {r.path_to_promote_m:.1f} & {100*r.pose_err_med_m:.1f} & {r.promoted_iou:.3f} & {r.refined_iou:.3f} & {r.refined_prec:.3f} & {r.refined_rec:.3f} & {int(r.refined_verts)} \\\\")
L.append(r'\midrule')
q=lambda c,f=1,d=3: f"{f*df[c].median():.{d}f}"
L.append(f"median & {q('t_promote_s',1,0)} & {q('path_to_promote_m',1,1)} & {q('pose_err_med_m',100,1)} & {q('promoted_iou')} & {q('refined_iou')} & {q('refined_prec')} & {q('refined_rec')} & {df.refined_verts.median():.0f} \\\\")
m=lambda c,f=1,d=3: f"{f*df[c].min():.{d}f}"
L.append(f"worst & {f'{df.t_promote_s.max():.0f}'} & {f'{df.path_to_promote_m.max():.1f}'} & {f'{100*df.pose_err_med_m.max():.1f}'} & {m('promoted_iou')} & {m('refined_iou')} & {m('refined_prec')} & {m('refined_rec')} & {df.refined_verts.max():.0f} \\\\")
L+= [r'\bottomrule',r'\end{tabular}',r'\end{table}']
open(sys.argv[3],'w').write('\n'.join(L)+'\n')
