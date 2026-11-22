#!/usr/bin/env python3
"""Read a Model.GnTrace run (<stem>_gn_steps.csv, <stem>_gn_fd.csv beside the ai2 log) and summarise it.

Usage: gn_trace_report.py DIR_OR_STEM [DIR_OR_STEM ...]
  DIR: every *_gn_steps.csv in it is read (with its *_gn_fd.csv and *_ai2_log.csv / ai2_log.csv sibling for `gated`).

Prints, per node:
- the share of unconverged updates on camera+LiDAR (gated 0) vs LiDAR-only (gated 1) cycles;
- for unconverged updates, how the 25 steps went: accepted / rejected, the final λ, the gain ratio ρ
  of accepted steps, and the undamped step in σ units (the largest component, and which one);
- the finite-difference check per channel (camera, LiDAR, θ-only, all):
  the gradient ratio grad_an / grad_fd3 and the curvature ratio curv_an / curv_fd, as medians over components
  whose FD value is not negligible, at θ₀ and at the returned θ.
  grad_marg vs grad_an shows the fresh marginalisation's shrink.
"""
import csv, glob, math, os, statistics as st, sys
from collections import defaultdict

DOF = ['cx', 'cy', 'H', 'w', 'h', 'yaw']
CH = ['cam', 'L', 'th', 'all']


def f(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def med(v):
    v = [x for x in v if math.isfinite(x)]
    return st.median(v) if v else float('nan')


def load(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def report(steps_path):
    stem = steps_path[:-len('_gn_steps.csv')]
    fd_path = stem + '_gn_fd.csv'
    ai2 = stem + '.csv' if os.path.exists(stem + '.csv') else None
    steps = load(steps_path)
    fd = load(fd_path) if os.path.exists(fd_path) else []
    gated = {}
    if ai2:
        for r in load(ai2):
            gated[(r['cycle'], r['node'])] = r.get('gated', '?')
    print('=' * 100)
    print(steps_path, '| fd rows', len(fd), '| ai2', ai2 or '(none: gated unknown)')
    upd = defaultdict(list)
    for r in steps:
        upd[(r['cycle'], r['node'])].append(r)
    by_node = defaultdict(list)
    for k, rs in upd.items():
        by_node[k[1]].append((k, rs))
    for node, items in sorted(by_node.items()):
        print('-' * 100)
        tab = defaultdict(lambda: [0, 0])
        for k, rs in items:
            g = gated.get(k, '?')
            tab[g][1] += 1
            tab[g][0] += rs[0]['gn_conv'] == '0'
        print('%s: updates %d; unconverged by gated (0 = camera+LiDAR, 1 = LiDAR-only): %s' % (
            node, len(items), ', '.join('gated %s %d/%d' % (g, a, b) for g, (a, b) in sorted(tab.items()))))
        unc = [(k, rs) for k, rs in items if rs[0]['gn_conv'] == '0']
        if not unc:
            continue
        acc = [sum(int(r['accepted']) for r in rs) for _, rs in unc]
        lam = [f(rs[-1]['lambda']) for _, rs in unc]
        rho = [f(r['rho']) for _, rs in unc for r in rs if r['accepted'] == '1' and f(r['lambda']) > 0]
        print('  unconverged: accepted steps med %s of 25; final lambda med %.3g (min %.3g max %.3g); rho of accepted med %.3f' % (
            med(acc), med(lam), min(lam), max(lam), med(rho)))
        worst = defaultdict(int)
        d0max = []
        for _, rs in unc:
            for r in rs:
                v = [abs(f(r['d0s_' + d])) for d in DOF]
                if all(math.isfinite(x) for x in v):
                    i = max(range(6), key=lambda j: v[j])
                    worst[DOF[i]] += 1
                    d0max.append(v[i])
        print('  undamped step / sigma (blocks the 0.1 sigma stop): largest component counts %s; |d0|/sigma med %.3g' % (
            dict(worst), med(d0max)))
        # mode: crawl (all accepted, tiny lambda) vs rejection streak
        crawl = sum(1 for (_, rs), a in zip(unc, acc) if a >= 20)
        rej = sum(1 for (_, rs), a in zip(unc, acc) if a <= 10)
        print('  modes: crawl (>= 20 accepted) %d, rejection-heavy (<= 10 accepted) %d, other %d' % (crawl, rej, len(unc) - crawl - rej))
    if not fd:
        return
    print('-' * 100)
    print('finite-difference check (unconverged updates): median ratio analytic / FD over components with |FD| not negligible')
    groups = defaultdict(lambda: defaultdict(list))
    for r in fd:
        key = (r['node'], gated.get((r['cycle'], r['node']), '?'), r['at'], CH[int(r['channel'])])
        ga, gm, g3, g4, ca, cf = (f(r[c]) for c in ('grad_an', 'grad_marg', 'grad_fd3', 'grad_fd4', 'curv_an', 'curv_fd'))
        if abs(g3) > 1e-6 * max(1.0, abs(ga)) and abs(g3) > 1e-3:
            groups[key]['g'].append(ga / g3)
            groups[key]['gm'].append(gm / g3)
            groups[key]['fdc'].append(g4 / g3)   # FD self-consistency (h 1e-4 vs 1e-3)
        if abs(cf) > 1e-3:
            groups[key]['c'].append(ca / cf)
        groups[key]['negc'].append(1.0 if cf < 0 else 0.0)
    print('%-16s %-5s %-3s %-4s %8s %8s %8s %8s %8s %5s' % ('node', 'gated', 'at', 'ch', 'G/FD', 'Gm/FD', 'fd4/fd3', 'W/FD', 'FDc<0', 'n'))
    for key in sorted(groups):
        gr = groups[key]
        print('%-16s %-5s %-3s %-4s %8.3f %8.3f %8.3f %8.3f %8.2f %5d' % (
            key[0][:16], key[1], key[2], key[3], med(gr['g']), med(gr['gm']), med(gr['fdc']), med(gr['c']),
            st.mean(gr['negc']) if gr['negc'] else float('nan'), len(gr['negc'])))


def main():
    paths = []
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, '*_gn_steps.csv')))
        else:
            paths.append(a if a.endswith('_gn_steps.csv') else a + '_gn_steps.csv')
    for p in paths:
        report(p)


if __name__ == '__main__':
    main()
