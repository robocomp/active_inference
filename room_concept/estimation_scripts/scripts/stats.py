#!/usr/bin/env python3
"""Is the n-start campaign a sound statistic? Wilson CI on success, bootstrap CIs on medians, leave-one-out sensitivity,
paired refined-vs-promoted test, and how well the starts cover the apartment."""
import sys, numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1]); rng = np.random.default_rng(0); B = 20000
n = len(df); ok = df[df.status == 'ok']; k = len(ok)
z = 1.96; c = (k + z*z/2) / (n + z*z); h = z*np.sqrt(k*(n-k)/n + z*z/4) / (n + z*z)
print(f'success {k}/{n}  Wilson 95% [{c-h:.2f}, {c+h:.2f}]')
def boot(x, f):
    x = np.asarray(x); i = rng.integers(0, len(x), (B, len(x))); s = f(x[i], axis=1); return np.percentile(s, [2.5, 97.5])
rows = []
for col in ['promoted_iou', 'refined_iou', 'refined_prec', 'refined_rec', 't_promote_s', 'path_to_promote_m', 'pose_err_med_m']:
    x = ok[col].dropna().values
    lo, hi = boot(x, np.median); mlo, mhi = boot(x, np.mean)
    loo = [np.median(np.delete(x, j)) for j in range(len(x))]
    rows.append(dict(metric=col, n=len(x), median=np.median(x), med_ci=f'[{lo:.3f},{hi:.3f}]', mean=x.mean(), sd=x.std(ddof=1),
                     mean_ci=f'[{mlo:.3f},{mhi:.3f}]', iqr=f'{np.percentile(x,25):.3f}-{np.percentile(x,75):.3f}', min=x.min(), max=x.max(),
                     loo_median_range=f'{min(loo):.3f}-{max(loo):.3f}', shapiro_p=stats.shapiro(x).pvalue if len(x) >= 3 else np.nan))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
d = (ok.refined_iou - ok.promoted_iou).values
w = stats.wilcoxon(d, zero_method='zsplit') if np.any(d) else None
print(f'refined - promoted IoU: median {np.median(d):+.4f}, improved {np.sum(d>0)}/{len(d)}, worse {np.sum(d<0)}, '
      f'Wilcoxon p={w.pvalue:.3f}' if w else 'no change')
# how many runs are needed for the median CI half-width to fall below a given width (resampled from what we have)
x = ok.refined_iou.values
for m in [5, 10, 20, 30]:
    i = rng.integers(0, len(x), (B, m)); s = np.median(x[i], axis=1); print(f'  n={m:2d}: median IoU 95% band width {np.percentile(s,97.5)-np.percentile(s,2.5):.4f}')
