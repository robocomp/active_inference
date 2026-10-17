#!/usr/bin/env python3
"""
audit_config_keys.py — STAGE 0 baseline for the effective-config table. READ-ONLY.

Answers, per agent, the two questions a config file cannot answer about itself:

  SILENT   a key the CODE reads that the agent's own config.toml never mentions. It runs the code
           default with nothing on the page to say so. This is commit b6c40b4's "absent is not off,
           but it reads as off" — P3Bot ran six older defaults this way.
  ORPHAN   a key IN the config file that no source file reads. Each one is a real defect: a paste
           that never landed (cabinet's ExistenceLegOccupancy, door's four chair-shaped BirthSeat*).

This is a GREP, not the instrument. Keys built by string concatenation ("Platform." + n + "." + key)
are invisible to it, so it deliberately errs towards silence on ORPHANs: a key counts as read if its
full dotted form OR its bare leaf appears as a quoted literal anywhere in the agent's src/ or in
common/. Undercounting orphans is the right failure direction for a baseline — the registry in
config_report.h is what makes the numbers exact.

Acceptance test (--selftest): it must re-find the two orphans already confirmed by hand. A baseline
that cannot rediscover a known defect cannot be trusted to find an unknown one.
"""
from __future__ import annotations
import argparse, glob, os, re, sys
from collections import namedtuple

# Keys consumed by GENERATED code or by enumeration, never by name in the agent's own sources.
# Short, and named in the output, so an exemption that swallows too much is visible rather than silent.
FRAMEWORK_PREFIXES = {
    "Agent.":           "generated/genericworker.cpp reads the agent name/id and the viewer flags",
    "Period.":          "generated/main.cpp drives the compute/emergency periods",
    "Proxies.":         "generated Ice proxy wiring",
    "Endpoints.":       "generated Ice endpoint wiring",
    "Ice.":             "forwarded to the Ice communicator by enumeration over getKeys()",
    "Owns.":            "read by the owned-nodes bookkeeping",
    "Media.":           "media-plane descriptor selection",
    "Component":        "[Component.Debug] Verbose, read by generated code",
}

# Every read dialect in the fleet. (?s) so a call split over lines still matches.
READ_CALL = re.compile(r"""(?sx)
    \b(?: getf | geti | getd | gets | getb | getv
        | load_optional_cast | load_optional_apply | load_optional | load_required )
    \s* (?: < [^;{}]*? > )? \s* \(
    \s* (?: [A-Za-z_][\w:.\[\]()]* \s* , \s* )?      # an optional leading ConfigLoader argument
    " ([^"]+) "
""")
# configLoader.get<T>("K") / cfg.exists("K") — the try/catch and ternary dialects.
DIRECT_CALL = re.compile(r"""(?sx)
    \b\w* (?: onfig ) \w* \s* \. \s* (?: get | exists ) \s* (?: < [^;{}]*? > )? \s* \( \s* " ([^"]+) "
""")
# ★Must match an EMPTY literal too. Without the {0,...} lower bound a `""` argument - which the
# shared presence reader passes as its agent name - is skipped, the scanner resyncs on the WRONG
# quote, and every key literal after it on that line becomes invisible. That silently turned 8
# orphans into 121. Lengths are filtered after matching, where the parity is already correct.
QUOTED = re.compile(r'"([^"\n]{0,200})"')

TOML_SECTION = re.compile(r"^\s*\[([^\]]+)\]")
TOML_KEY = re.compile(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.*?)\s*$")

FileKey = namedtuple("FileKey", "key line value comment")


def parse_toml_keys(path: str) -> list[FileKey]:
    """Line-scan, NOT toml++: comments and ordering are exactly what ConfigLoader throws away."""
    out, section, block = [], "", []
    for n, raw in enumerate(open(path, errors="ignore"), 1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            block = []
            continue
        if stripped.startswith("#"):
            block.append(stripped.lstrip("#").strip())
            continue
        m = TOML_SECTION.match(line)
        if m:
            section, block = m.group(1), []
            continue
        m = TOML_KEY.match(line)
        if m:
            name, rest = m.group(1), m.group(2)
            inline = rest.split("#", 1)[1].strip() if "#" in rest else ""
            value = rest.split("#", 1)[0].strip()
            out.append(FileKey(f"{section}.{name}" if section else name, n, value,
                               inline or (block[-1] if block else "")))
            block = []
    return out


def source_text(agent: str) -> str:
    files = []
    for sub in ("src", "generated"):
        files += glob.glob(os.path.join(agent, sub, "*.cpp")) + glob.glob(os.path.join(agent, sub, "*.h"))
    return "\n".join(open(f, errors="ignore").read() for f in files)


def read_keys(text: str) -> set[str]:
    return set(READ_CALL.findall(text)) | set(DIRECT_CALL.findall(text))


def audit(root: str) -> list[dict]:
    common = "\n".join(open(f, errors="ignore").read()
                       for f in glob.glob(os.path.join(root, "common", "*", "*.cpp"))
                       + glob.glob(os.path.join(root, "common", "*", "*.h")))
    common_quoted = {q for q in QUOTED.findall(common) if len(q) >= 2}
    rows = []
    for agent in sorted(os.listdir(root)):
        cfg = os.path.join(root, agent, "etc", "config.toml")
        src = os.path.join(root, agent, "src")
        if not (os.path.isdir(src) and os.path.exists(cfg)):
            continue
        file_keys = parse_toml_keys(cfg)
        present = {k.key for k in file_keys}
        text = source_text(os.path.join(root, agent))
        read = read_keys(text)
        # Generous "mentioned" test: full dotted key, or the bare leaf, anywhere in this agent or common/.
        quoted = {q for q in QUOTED.findall(text) if len(q) >= 2} | common_quoted
        mentioned = quoted | {q.split(".")[-1] for q in quoted}
        silent = sorted(k for k in read if k not in present)
        orphan = [k for k in file_keys
                  if k.key not in mentioned and k.key.split(".")[-1] not in mentioned
                  and not k.key.startswith(tuple(FRAMEWORK_PREFIXES))]
        undocumented = sum(1 for k in file_keys if not k.comment)
        rows.append(dict(agent=agent, file_keys=len(file_keys), read=len(read),
                         silent=silent, orphan=orphan, undocumented=undocumented))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--agent", help="restrict to one agent")
    ap.add_argument("--selftest", action="store_true", help="must rediscover the two confirmed orphans")
    ap.add_argument("--details", action="store_true", help="name every SILENT key, not just count it")
    a = ap.parse_args()

    rows = audit(a.root)
    if a.agent:
        rows = [r for r in rows if r["agent"] == a.agent]

    if a.selftest:
        want = [("cabinet_concept", "CabinetModel.ExistenceLegOccupancy"),
                ("door_concept", "Tracker.BirthSeatW"),
                ("kinova_controller", "Controller.tip_log")]
        ok = True
        for agent, key in want:
            r = next((x for x in rows if x["agent"] == agent), None)
            found = r is not None and any(o.key == key for o in r["orphan"])
            print(f"  {'PASS' if found else 'FAIL'}  {agent:22} expects orphan {key}")
            ok &= found
        print("\nselftest:", "PASS — the baseline rediscovers every known defect"
              if ok else "FAIL — this baseline cannot find a defect already confirmed by hand; do not build on it")
        return 0 if ok else 1

    print(f"{'agent':24}{'file':>6}{'read':>6}{'SILENT':>8}{'ORPHAN':>8}{'undoc':>7}")
    ts = tr = tsi = to = tu = 0
    for r in sorted(rows, key=lambda x: -len(x["silent"])):
        print(f"{r['agent']:24}{r['file_keys']:6}{r['read']:6}{len(r['silent']):8}{len(r['orphan']):8}{r['undocumented']:7}")
        ts += r["file_keys"]; tr += r["read"]; tsi += len(r["silent"]); to += len(r["orphan"]); tu += r["undocumented"]
    print(f"{'TOTAL':24}{ts:6}{tr:6}{tsi:8}{to:8}{tu:7}")
    print(f"\nSILENT = read by the code, absent from the file  ->  runs the code default, unstated "
          f"({100*tsi//max(tr,1)}% of reads)")
    print("ORPHAN = in the file, read by nothing  ->  a paste that never landed")
    print("exempt prefixes: " + " ".join(sorted(FRAMEWORK_PREFIXES)))

    for r in rows:
        if r["orphan"]:
            print(f"\n{r['agent']} ORPHANS:")
            for o in r["orphan"]:
                print(f"  etc/config.toml:{o.line:<5} {o.key} = {o.value}"
                      + (f"   # {o.comment[:70]}" if o.comment else ""))
        if a.details and r["silent"]:
            print(f"\n{r['agent']} SILENT ({len(r['silent'])}):")
            for k in r["silent"]:
                print(f"  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
