#!/usr/bin/env python3
"""E2 (paper §5.3), fifth baseline: the VLM asked for a room's function from images, with no belief hierarchy.

For every corpus room (ScanNet first scans), k colour frames evenly spaced through the scan
(scannet_frames_25k, every 100th frame) go to the VLM in one request, and it must answer with one of the seven
functions (forced by a JSON-schema enum, temperature 0). The answer is a single claim, scored with the same
precision / recall / macro F1 / top-1 as tools/function_evidence_e2.py. Answers are cached by (model, scan, k).

    python3 tools/image_baseline_e2.py [--k 1] [--k 4]      OLLAMA=http://localhost:11434
"""
from __future__ import annotations

import argparse
import base64
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
from function_evidence_e2 import scores  # noqa: E402

PROMPT = ("These {n} photo(s) were taken in one room of a building. What is the room used for? "
          'Answer with exactly one of: {opts}. Reply as JSON: {{"function": "<one of them>"}}.')


def frames(scan_dir: Path, k: int) -> list[Path]:
    fs = sorted((scan_dir / "color").glob("*.jpg"))
    if k == 1:
        return [fs[len(fs) // 2]]
    return [fs[round(i * (len(fs) - 1) / (k - 1))] for i in range(k)]


def ask(url: str, model: str, funcs: list[str], imgs: list[Path]) -> str:
    body = {
        "model": model, "stream": False, "think": False,
        "format": {"type": "object", "properties": {"function": {"type": "string", "enum": funcs}},
                   "required": ["function"]},
        "options": {"temperature": 0, "seed": 0},
        "messages": [{"role": "user", "content": PROMPT.format(n=len(imgs), opts=", ".join(funcs)),
                      "images": [base64.b64encode(p.read_bytes()).decode() for p in imgs]}],
    }
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # Retry: while the model is (re)loading on a nearly full GPU, ollama returns an empty answer or HTTP 500.
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
    ap.add_argument("--frames", type=Path, default=Path("/media/datadisk/scannet/tasks/scannet_frames_25k"))
    ap.add_argument("--vocab", type=Path, default=HERE / "elicit_vocab.toml")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--k", type=int, action="append", help="frames per room (repeatable); default 1 and 4")
    ap.add_argument("--cache", type=Path, default=HERE / "e2_out/image_answers.jsonl")
    args = ap.parse_args()
    url = os.environ.get("OLLAMA", "http://localhost:11434").rstrip("/")
    ks = args.k or [1, 4]

    all_funcs = tomllib.loads(args.vocab.read_text())["functions"]
    rooms = list(csv.DictReader(args.rooms.open()))
    truth = [r["function"] for r in rooms]
    funcs = sorted(set(truth))
    cache = {}
    if args.cache.exists():
        for line in args.cache.read_text().splitlines():
            rec = json.loads(line)
            cache[(rec["model"], rec["scan"], rec["k"])] = rec["answer"]

    with args.cache.open("a") as fh:
        for k in ks:
            todo = [r for r in rooms if (args.model, r["room"], k) not in cache]
            t0 = time.monotonic()
            for n, r in enumerate(todo, 1):
                imgs = frames(args.frames / r["room"], k)
                a = ask(url, args.model, all_funcs, imgs)
                cache[(args.model, r["room"], k)] = a
                fh.write(json.dumps(dict(model=args.model, scan=r["room"], k=k, answer=a,
                                         frames=[p.name for p in imgs])) + "\n")
                fh.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"  k={k}: {n}/{len(todo)}  {(time.monotonic() - t0) / n:.2f} s/room", flush=True)

    for k in ks:
        ans = [cache[(args.model, r["room"], k)] for r in rooms]
        pred = [{a} for a in ans]
        mf1, per = scores(pred, truth, funcs, list(range(len(rooms))))
        acc = sum(a == t for a, t in zip(ans, truth)) / len(rooms)
        other = sum(a not in funcs for a in ans)
        print(f"image k={k}: macro F1 {mf1:.3f}  top-1 {acc:.3f}  (answers outside the corpus functions: {other})  " +
              "  ".join(f"{f[:8]} {p:.2f}/{rc:.2f} ({c})" for f, (p, rc, _, c) in per.items()))


if __name__ == "__main__":
    sys.exit(main())
