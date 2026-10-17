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


if __name__ == "__main__":
    unittest.main()
