import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "em_report.py")


def rec(id, level="mission", parent="", t0=1000, t1=2000, outcome="ok", kind="k", author="human", **kw):
    r = {"id": id, "parent": parent, "level": level, "author": author, "kind": kind,
         "t0": t0, "outcome": outcome}
    if t1 is not None:
        r["t1"] = t1
    r.update(kw)
    return r


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.d.cleanup()

    def write(self, name, lines):
        p = os.path.join(self.d.name, name)
        with open(p, "w") as f:
            for l in lines:
                f.write((l if isinstance(l, str) else json.dumps(l)) + "\n")
        return p

    def run_(self, *args):
        p = subprocess.run([sys.executable, "-I", SCRIPT, *args], capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    def clean(self):
        return [rec("20261017-1"),
                rec("20261017-2", "step", "20261017-1"),
                rec("20261017-3", "claim", "20261017-2", outcome="lost")]

    def test_clean_passes(self):
        rc, out, _ = self.run_(self.write("20261017.jsonl", self.clean()))
        self.assertEqual(rc, 0, out)
        self.assertIn("lost", out)

    def test_orphan_fails(self):
        rc, out, _ = self.run_(self.write("a.jsonl", self.clean() + [rec("20261017-9", "step", "nope")]))
        self.assertNotEqual(rc, 0)
        self.assertIn("20261017-9", out)

    def test_duplicate_last_wins(self):
        lines = [rec("20261017-1", t1=None, outcome=""), *self.clean()]
        rc, out, _ = self.run_(self.write("a.jsonl", lines), "--expect-closed")
        self.assertEqual(rc, 0, out)

    def test_truncated_skipped(self):
        rc, out, err = self.run_(self.write("a.jsonl", self.clean() + ['{"id": "20261017-7", "par']))
        self.assertEqual(rc, 0, out)
        self.assertIn("warning", (out + err).lower())

    def test_open_only_with_expect_closed(self):
        p = self.write("a.jsonl", self.clean() + [rec("20261017-4", "step", "20261017-1", t1=None, outcome="")])
        self.assertEqual(self.run_(p)[0], 0)
        rc, out, _ = self.run_(p, "--expect-closed")
        self.assertNotEqual(rc, 0)
        self.assertIn("20261017-4", out)

    def test_t1_before_t0_fails(self):
        rc, out, _ = self.run_(self.write("a.jsonl", self.clean() + [rec("20261017-5", t0=5000, t1=4000)]))
        self.assertNotEqual(rc, 0)
        self.assertIn("20261017-5", out)

    def test_parent_in_also_file(self):
        prev = self.write("prev.jsonl", [rec("20261016-1")])
        p = self.write("a.jsonl", [rec("20261017-1", "step", "20261016-1")])
        self.assertNotEqual(self.run_(p)[0], 0)
        self.assertEqual(self.run_(p, "--also", prev)[0], 0)

    def test_mission_with_parent_fails(self):
        rc, out, _ = self.run_(self.write("a.jsonl", [rec("20261017-1", parent="x")]))
        self.assertNotEqual(rc, 0)

    def test_json_output(self):
        rc, out, _ = self.run_(self.write("a.jsonl", self.clean()), "--json")
        j = json.loads(out)
        self.assertEqual(j["lost"]["by_level"], {"claim": 1})
        self.assertEqual(j["failures"], [])

    # final review, finding 6
    def test_closed_with_empty_outcome_fails(self):
        rc, out, _ = self.run_(self.write("a.jsonl", self.clean() + [rec("20261017-6", "step", "20261017-1", outcome="")]))
        self.assertNotEqual(rc, 0)
        self.assertIn("20261017-6", out)
        self.assertIn("empty outcome", out)

    def test_records_per_mission_key(self):
        lines = [rec("20261017-1", mission_key="m3"),
                 rec("20261017-2", "step", "20261017-1", mission_key="m3", step_id="1", attempt=1),
                 rec("20261017-3", "claim", "20261017-2", mission_key="m3"),
                 rec("20261017-4", author="self", source="self")]
        rc, out, _ = self.run_(self.write("a.jsonl", lines), "--json")
        self.assertEqual(rc, 0, out)
        bm = json.loads(out)["by_mission"]
        self.assertEqual(sorted((v["mission_key"], v["records"]) for v in bm.values()), [("", 1), ("m3", 3)])
        self.assertEqual(bm["20261017-1"], {"mission_key": "m3", "records": 3})
        rc, out, _ = self.run_(self.write("b.jsonl", lines))
        self.assertIn("by_mission", out)

    def test_duplicate_step_record_fails(self):
        step = dict(mission_key="m3", step_id="2.1", attempt=1, iteration=2)
        lines = [rec("20261017-1", mission_key="m3"),
                 rec("20261017-2", "step", "20261017-1", **step),
                 rec("20261017-3", "step", "20261017-1", **step),
                 rec("20261017-4", "step", "20261017-1", **{**step, "iteration": 3})]
        rc, out, _ = self.run_(self.write("a.jsonl", lines))
        self.assertNotEqual(rc, 0)
        self.assertIn("duplicate step", out)
        self.assertIn("20261017-3", out)
        self.assertNotIn("FAIL 20261017-4", out)

    def two_runs(self):
        step = dict(mission_key="c1", step_id="1", attempt=1, iteration=0)
        return [rec("20261017-1", mission_key="c1"),
                rec("20261017-2", "step", "20261017-1", **step),
                rec("20261017-3", mission_key="c1"),
                rec("20261017-4", "step", "20261017-3", **step)]

    def test_controller_restart_same_key_passes(self):
        rc, out, _ = self.run_(self.write("a.jsonl", self.two_runs()), "--json")
        self.assertEqual(rc, 0, out)
        bm = json.loads(out)["by_mission"]
        self.assertEqual(bm, {"20261017-1": {"mission_key": "c1", "records": 2},
                              "20261017-3": {"mission_key": "c1", "records": 2}})

    def test_duplicate_under_same_root_via_claim_chain_fails(self):
        lines = self.two_runs() + [rec("20261017-5", "step", "20261017-3", mission_key="c1", step_id="1",
                                       attempt=1, iteration=0)]
        rc, out, _ = self.run_(self.write("a.jsonl", lines))
        self.assertNotEqual(rc, 0)
        self.assertIn("duplicate step", out)
        self.assertIn("FAIL 20261017-5", out)

    def test_parent_cycle_fails(self):
        lines = [rec("20261017-1", "step", "20261017-2"), rec("20261017-2", "step", "20261017-1")]
        rc, out, _ = self.run_(self.write("a.jsonl", lines))
        self.assertNotEqual(rc, 0)
        self.assertIn("parent cycle", out)
        self.assertIn("20261017-1", out)
        self.assertIn("20261017-2", out)


if __name__ == "__main__":
    unittest.main()
