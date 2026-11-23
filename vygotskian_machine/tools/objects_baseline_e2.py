#!/usr/bin/env python3
"""E2 baseline: the VLM asked directly for a room's function from the list of objects in it (no images, no belief
hierarchy), as in Chen et al. 2022 (paper: \\cite{chen2022leveraging}).

For every ScanNet room the annotated classes (the same units the function evidence sees) are listed, and the VLM must
answer with one of the seven functions (JSON-schema enum, temperature 0). Scored like tools/image_baseline_e2.py.
Answers are cached by (model, room) in tools/e2_out/objects_answers.jsonl.

    python3 tools/objects_baseline_e2.py      OLLAMA=http://localhost:11434
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from elicit_priors import spoken  # noqa: E402
from function_evidence_e2 import scores  # noqa: E402

PROMPT = ("A room in a building contains the following objects: {objs}. What is the room used for? "
          'Answer with exactly one of: {opts}. Reply as JSON: {{"function": "<one of them>"}}.')


def ask(url: str, model: str, funcs: list[str], objs: list[str]) -> str:
    body = {
        "model": model, "stream": False, "think": False,
        "format": {"type": "object", "properties": {"function": {"type": "string", "enum": funcs}},
                   "required": ["function"]},
        "options": {"temperature": 0, "seed": 0},
        "messages": [{"role": "user", "content": PROMPT.format(
            objs=", ".join(spoken(c) for c in objs) if objs else "no objects that we can name",
            opts=", ".join(funcs))}],
    }
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                answer = json.loads(json.loads(r.read())["message"]["content"])["function"]
            if answer in funcs:
                return answer
        except (urllib.error.HTTPError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("no valid answer after 5 attempts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rooms", type=Path, default=HERE / "corpora/scannet_rooms.csv")
    ap.add_argument("--vocab", type=Path, default=HERE / "elicit_vocab.toml")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--cache", type=Path, default=HERE / "e2_out/objects_answers.jsonl")
    args = ap.parse_args()
    url = os.environ.get("OLLAMA", "http://localhost:11434").rstrip("/")

    all_funcs = tomllib.loads(args.vocab.read_text())["functions"]
    rooms = list(csv.DictReader(args.rooms.open()))
    classes = list(rooms[0])[4:]
    truth = [r["function"] for r in rooms]
    funcs = sorted(set(truth))
    cache = {}
    if args.cache.exists():
        for rec in map(json.loads, args.cache.read_text().splitlines()):
            cache[(rec["model"], rec["room"])] = rec["answer"]
    with args.cache.open("a") as fh:
        todo = [r for r in rooms if (args.model, r["room"]) not in cache]
        t0 = time.monotonic()
        for n, r in enumerate(todo, 1):
            objs = [c for c in classes if r[c] == "1"]
            a = ask(url, args.model, all_funcs, objs)
            cache[(args.model, r["room"])] = a
            fh.write(json.dumps(dict(model=args.model, room=r["room"], objects=objs, answer=a)) + "\n")
            fh.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {(time.monotonic() - t0) / n:.2f} s/room", flush=True)

    ans = [cache[(args.model, r["room"])] for r in rooms]
    mf1, per = scores([{a} for a in ans], truth, funcs, list(range(len(rooms))))
    acc = sum(a == t for a, t in zip(ans, truth)) / len(rooms)
    other = sum(a not in funcs for a in ans)
    print(f"objects: macro F1 {mf1:.3f}  top-1 {acc:.3f}  (answers outside the corpus functions: {other})  " +
          "  ".join(f"{f[:8]} {p:.2f}/{rc:.2f} ({c})" for f, (p, rc, _, c) in per.items()))


if __name__ == "__main__":
    sys.exit(main())
