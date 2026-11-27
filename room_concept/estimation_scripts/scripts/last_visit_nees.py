#!/usr/bin/env python3
"""AuRo test (a), second registered statistic (REGISTERED_TESTS.md, 2026-11-24, A1: "the mean NEES/DOF at the last visit of each
instance", same pass rule). This is first_visit_nees.py UNCHANGED, with one substitution: the visit window is the robot's LAST
stay in the object's room instead of its first. Everything else (instances, truth, node match at the stay's end, frames, full P from
the replay tool and its bit-identity gate, the draw = the node's last update up to the end of the stay, duplicates, the pass rule,
the bootstrap) is the first-visit scorer's own code.
    python3 -I last_visit_nees.py [--scorer PATH] <first_visit_nees.py arguments>...
--scorer: the first_visit_nees.py to use (default: the one next to this file). Batch 1 was scored with the f2d456a3 version.

last stay   the last maximal run of samples inside the room (samples in no polygon, a doorway, are neutral): t_in = its first sample
            inside, t_out = its last sample inside. open = the log ended inside the room. None if the robot was never inside.
"""
import importlib.util, sys
from pathlib import Path


def last_stay(series, room):
    stay, t_in, t_out = None, None, None
    for t, r in series:
        if r == room:
            if t_in is None: t_in = t
            t_out = t
        elif r is not None and t_in is not None:
            stay, t_in, t_out = (t_in, t_out, False), None, None
    return (t_in, t_out, True) if t_in is not None else stay


def main(argv):
    scorer = Path(__file__).with_name('first_visit_nees.py')
    if argv[:1] == ['--scorer']: scorer, argv = Path(argv[1]), argv[2:]
    sys.path.insert(0, str(Path(__file__).parent))          # the scorer's sibling modules (aba_runlog, aba_geometry, ...)
    spec = importlib.util.spec_from_file_location('first_visit_nees', scorer)
    F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
    F.first_stay = last_stay                                  # the one substitution
    def say(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith('first-visit NEES/DOF'):
            a = ('LAST-visit NEES/DOF (last stay in the object room; first_visit_nees.py otherwise unchanged)' + a[0][len('first-visit NEES/DOF'):],) + a[1:]
        print(*a, **k)
    report = F.report
    F.report = lambda recs, classes=F.CLASSES, say_=None: report(recs, classes, say)   # relabel the header only
    print(f'scorer {scorer}')
    return F.main(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
