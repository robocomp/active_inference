#!/usr/bin/env python3
"""results.csv (campaign grader) + results_anchor.csv (regrade_anchor.py) -> results_shape.csv, where promoted_iou /
refined_iou are the SHAPE IoUs (best rigid fit) and the SVD-aligned ones are kept as *_svd, the anchored as *_anchor."""
import sys, pandas as pd
D = sys.argv[1]; r = pd.read_csv(f'{D}/results.csv'); a = pd.read_csv(f'{D}/results_anchor.csv')
r = r.rename(columns={'promoted_iou': 'promoted_iou_svd', 'refined_iou': 'refined_iou_svd'}).merge(a, on='run', how='left')
r['promoted_iou'] = r.prom_shape; r['refined_iou'] = r.ref_shape
r.to_csv(f'{D}/results_shape.csv', index=False, float_format='%.4f')
