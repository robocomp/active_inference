#!/usr/bin/env python3
"""Compare a selftest's output against its recorded oracle (refactor guardrail).

Usage:  python3 -I tools/oracle_compare.py ORACLE NEW [--rtol R] [--atol A]

Both files are normalised first (wall-clock "N ms" / "N s" timings and date-time stamps blanked), then
compared line by line. Text must match exactly; numbers must match exactly by default, or within
|a-b| <= atol + rtol*|a| when a tolerance is given (the torch -> Eigen port changes float reduction order,
so it cannot be bit-identical). Exit 0 = match, 1 = differs (first differences printed).
"""
import argparse
import re
import sys

NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')
TIMING = [
    (re.compile(r'\b\d+(?:\.\d+)?\s*ms\b'), '<ms>'),
    (re.compile(r'\d{4}-\d\d-\d\d[_T ]\d\d[-:]\d\d[-:]\d\d'), '<stamp>'),   # no \b: 'joint_2026-...'
]


def normalise(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        for rx, rep in TIMING:
            ln = rx.sub(rep, ln)
        out.append(ln.rstrip())
    return out


def compare_line(a, b, rtol, atol):
    if NUM.sub('#', a) != NUM.sub('#', b):
        return 'text differs'
    for x, y in zip(NUM.findall(a), NUM.findall(b)):
        if x == y:
            continue
        fx, fy = float(x), float(y)
        if abs(fx - fy) > atol + rtol * abs(fx):
            return f'{x} -> {y}'
    return None


VERDICT = re.compile(r'\b(PASS|FAIL|FAILED|FAILURES|ALL PASS)\b|^\[exit ')


def compare_verdicts(old, new):
    """Chaotic benches (1100-frame estimator runs) diverge on a 1-ulp change, so only the verdicts are
    binding. Verdict lines are matched by their text with numbers masked."""
    key = lambda ln: NUM.sub('#', ln).strip()
    vo = [key(l) for l in old if VERDICT.search(l)]
    vn = [key(l) for l in new if VERDICT.search(l)]
    drift = sum(1 for x, y in zip(old, new) if x != y) + abs(len(old) - len(new))
    if vo == vn:
        if drift:
            print(f'  (drift: {drift} non-verdict line(s) differ — informational)')
        return 0
    for x in sorted(set(vo) - set(vn))[:6]: print(f'  verdict lost: {x[:150]}')
    for y in sorted(set(vn) - set(vo))[:6]: print(f'  verdict new:  {y[:150]}')
    if set(vo) == set(vn): print('  verdict order/count changed')
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('oracle'); ap.add_argument('new')
    ap.add_argument('--rtol', type=float, default=0.0)
    ap.add_argument('--atol', type=float, default=0.0)
    ap.add_argument('--verdicts', action='store_true',
                    help='chaotic case: fail only if a PASS/FAIL verdict or the exit code changes; '
                         'report other differences as drift')
    a = ap.parse_args()
    old, new = normalise(a.oracle), normalise(a.new)
    if a.verdicts:
        return compare_verdicts(old, new)
    shown = 0
    if len(old) != len(new):
        print(f'  line count {len(old)} -> {len(new)}')
        shown += 1
    for i, (x, y) in enumerate(zip(old, new), 1):
        why = compare_line(x, y, a.rtol, a.atol)
        if why:
            if shown < 6:
                print(f'  L{i}: {why}\n    - {x.strip()[:150]}\n    + {y.strip()[:150]}')
            shown += 1
    if shown > 6:
        print(f'  ... {shown - 6} more')
    return 1 if shown else 0


if __name__ == '__main__':
    sys.exit(main())
