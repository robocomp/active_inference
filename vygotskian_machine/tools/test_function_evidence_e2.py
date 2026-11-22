"""Tests for the level calibration in function_evidence_e2.py."""
import csv
import tempfile
from pathlib import Path

from function_evidence_e2 import fit_levels, isotonic


def test_isotonic_keeps_monotone_order():
    assert isotonic([0.1, 0.5], [1, 8], [10, 10]) == {0.1: 2 / 12, 0.5: 9 / 12}


def test_isotonic_pools_a_violation():
    m = isotonic([0.25, 0.5, 0.75], [1, 6, 4], [10, 10, 10])
    assert m[0.5] == m[0.75] == (10 + 1) / (20 + 2)
    assert m[0.25] < m[0.5]


def test_fit_levels_pools_pairs_by_level():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rooms.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["house", "split", "room", "function", "sink", "bed"])
            w.writerows([["h", "s", "r1", "kitchen", 1, 0], ["h", "s", "r2", "bedroom", 0, 1],
                         ["h", "s", "r3", "bedroom", 1, 1]])
        vlm = {("sink", "kitchen"): 0.95, ("bed", "bedroom"): 0.95,
               ("sink", "bedroom"): 0.05, ("bed", "kitchen"): 0.05, ("sink", "office"): 0.5}
        cal, raw = fit_levels(vlm, p)
        assert raw == {0.05: (1, 3), 0.95: (3, 3)}   # office is not in the corpus: ignored
        assert cal == {0.05: 2 / 5, 0.95: 4 / 5}
