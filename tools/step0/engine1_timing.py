#!/usr/bin/env python3
"""Tick rate and stage cost of the countable agents from their timing_log.csv (one row per compute tick).
   engine1_timing.py <checkout_root> <t0_epoch_s> [<t1_epoch_s>]
Rows are cut to [t0, t1]. Prints per class: tick rate, gaps > 150 ms, and per n_inst the median / p95 / max of tick,
inst_total, infer, publish (publish holds the epistemic offer), existence."""
import sys
import pandas as pd

root, t0 = sys.argv[1], float(sys.argv[2]) * 1000
t1 = float(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 1e18
for c in ('bed', 'nightstand', 'toilet'):
    d = pd.read_csv(f'{root}/{c}_concept/etc/timing_log.csv')
    d = d[(d.ts_ms >= t0) & (d.ts_ms <= t1)]
    dt = d.ts_ms.diff().dropna()
    span = (d.ts_ms.iloc[-1] - d.ts_ms.iloc[0]) / 1000
    print(f'{c}: {len(d)} ticks, {span:.1f} s, rate {(len(d) - 1) / span:.2f} Hz, gaps>150ms {(dt > 150).sum()}, max gap {dt.max():.0f} ms')
    for n, g in d.groupby('n_inst'):
        s = ' '.join(f'{k} {g[k].median():.1f}/{g[k].quantile(.95):.1f}/{g[k].max():.1f}'
                     for k in ('tick', 'inst_total', 'infer', 'publish', 'existence'))
        print(f'  n_inst {n} ({len(g)} ticks) median/p95/max ms: {s}')
