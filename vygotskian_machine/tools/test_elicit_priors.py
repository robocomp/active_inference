#!/usr/bin/env python3
"""Offline tests for elicit_priors.py (no model needed):  python3 tools/test_elicit_priors.py"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elicit_priors as ep  # noqa: E402


def rec(kind, cls, func, neg, para, p):
    return dict(kind=kind, cls=cls, func=func, negated=neg, para=para, p=p)


class TestElicit(unittest.TestCase):
    def test_article(self):
        self.assertEqual(ep.article("oven"), "an oven")
        self.assertEqual(ep.article(ep.spoken("hood")), "an extractor hood")

    def test_question_count(self):
        vocab = dict(functions=["kitchen", "bedroom"], classes=["bed", "oven", "sink"])
        qs = ep.build_questions(vocab, ["cabinet", "hood"])
        # L4: 2 functions x 3 classes x 3 paraphrases (not-f is derived); L3: (2 classes + base) x 3
        self.assertEqual(sum(q["kind"] == "L4" for q in qs), 18)
        self.assertEqual(sum(q["kind"] == "L3" for q in qs), 9)
        self.assertEqual(len({q["prompt"] for q in qs}), len(qs))  # every prompt distinct

    def test_level4_median_and_spread(self):
        rs = [rec("L4", "oven", "kitchen", False, i, p) for i, p in enumerate([0.95, 0.75, 0.95])]
        rs += [rec("L4", "oven", "bedroom", False, i, 0.05) for i in range(3)]
        rs += [rec("L4", "oven", "office", False, i, 0.25) for i in range(3)]
        l4, _ = ep.summarise(rs, {})
        k = next(r for r in l4 if r["func"] == "kitchen")
        self.assertAlmostEqual(k["pi_cf"], 0.95)
        self.assertAlmostEqual(k["spread_cf"], 0.20)
        self.assertAlmostEqual(k["pi_c_notf"], 0.15)  # uniform mean of bedroom and office
        l4w, _ = ep.summarise(rs, {}, weights={"bedroom": 3.0, "office": 1.0})
        kw = next(r for r in l4w if r["func"] == "kitchen")
        self.assertAlmostEqual(kw["pi_c_notf"], (3 * 0.05 + 0.25) / 4)

    def test_level3_bayes_log_odds(self):
        base = ep.L3_BASE_CLASS
        rs = [rec("L3", "hood", "kitchen unit", False, i, 0.95) for i in range(3)]
        rs += [rec("L3", base, "kitchen unit", False, i, 0.25) for i in range(3)]
        _, l3 = ep.summarise(rs, {"hood": 3.0, "sofa": -1.0})
        self.assertEqual([r["cls"] for r in l3], ["hood"])  # sofa unanswered -> no row
        self.assertAlmostEqual(l3[0]["elicited"], ep.logit(0.95) - ep.logit(0.25))
        self.assertAlmostEqual(l3[0]["spread"], 0.0)

    def test_level3_pairs_paraphrases(self):
        base = ep.L3_BASE_CLASS
        rs = [rec("L3", "cabinet", "kitchen unit", False, i, p) for i, p in enumerate([0.75, 0.5, 0.75])]
        rs += [rec("L3", base, "kitchen unit", False, i, p) for i, p in enumerate([0.5, 0.5, 0.25])]
        _, l3 = ep.summarise(rs, {"cabinet": 0.2})
        per_para = [ep.logit(0.75) - 0.0, 0.0, ep.logit(0.75) - ep.logit(0.25)]
        self.assertAlmostEqual(l3[0]["elicited"], sorted(per_para)[1])
        self.assertAlmostEqual(l3[0]["spread"], max(per_para) - min(per_para))

    def test_ceiling(self):
        self.assertAlmostEqual(ep.logit(0.95) - ep.logit(0.05), 2 * math.log(19))


if __name__ == "__main__":
    unittest.main()
