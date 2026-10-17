#!/usr/bin/env python3
"""results_shape.csv + rooms.json -> results_table.tex (\\input-ready). usage: make_table.py <campaign dir>"""
import sys, json, numpy as np, pandas as pd
D = sys.argv[1]; df = pd.read_csv(f'{D}/results_shape.csv'); R = {r['id']: r for r in json.load(open(f'{D}/rooms.json'))}
name = {'random': 'M', 'L': 'L', 'T': 'T', 'U': 'U'}
L = [r'\begin{table}[t]', r'\centering',
     r'\caption{Room-layout estimation in ten synthetic rooms (empty, one run each, random start $\geq$0.8 m from walls). '
     r'Type: M = rectangle with random notches and bays, L, T. $|V|_{gt}$, $A$: ground-truth vertices and area. $t_p$: time to promotion '
     r'($t_p\approx0$: the first 360$^\circ$ scan left no standpoint worth visiting). $e$: median SLAM position error. '
     r'IoU after the best rigid alignment of the ground truth to the estimate (shape only; the frame error is reported separately).}',
     r'\label{tab:estimation-synthetic}', r'\begin{tabular}{r c r r r r r r r r}', r'\toprule',
     r'room & type & $|V|_{gt}$ & $A$ (m$^2$) & $t_p$ (s) & $e$ (mm) & IoU$_p$ & IoU$_r$ & $|V|_p$ & $|V|_r$ \\', r'\midrule']
for _, r in df.iterrows():
    g = R[int(r.run)]
    L.append(f"{int(r.run)} & {name[g.get('kind', 'random')]} & {g['verts']} & {g['area']:.1f} & {r.t_promote_s:.0f} & {1000 * r.pose_err_med_m:.0f} & "
             f"{r.promoted_iou:.3f} & {r.refined_iou:.3f} & {int(r.promoted_verts)} & {int(r.refined_verts)} \\\\")
L += [r'\midrule',
      f"median & & & & {df.t_promote_s.median():.0f} & {1000 * df.pose_err_med_m.median():.0f} & {df.promoted_iou.median():.3f} & {df.refined_iou.median():.3f} & & \\\\",
      f"worst & & & & {df.t_promote_s.max():.0f} & {1000 * df.pose_err_med_m.max():.0f} & {df.promoted_iou.min():.3f} & {df.refined_iou.min():.3f} & & \\\\",
      r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{D}/results_table.tex', 'w').write('\n'.join(L) + '\n')
