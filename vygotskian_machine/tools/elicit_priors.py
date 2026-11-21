#!/usr/bin/env python3
"""E1 (paper §5.2): elicit level-3 and level-4 priors from the VLM, with provenance.

Level 4 — room function. For every class c and function f in elicit_vocab.toml:
    pi_cf    = P(at least one c in the room | the room serves f)
    pi_c,¬f  = sum over g != f of w_g pi_cg / sum w_g   (w uniform until the corpus gives room-type frequencies)
The not-f probability is derived, not asked: asked directly ("a room that is not the kitchen") the model lumps
bedrooms, bathrooms and corridors together and answers inconsistently (mean paraphrase spread 0.28 vs 0.10).

Level 3 — kitchen membership. kitchen_metaconcept carries, per class c, the AUTHORED log-likelihood ratio
    log P(c | kitchen unit) / P(c | not a kitchen unit).
A five-level answer cannot express a small multinomial P(c | kitchen unit), so we ask the posterior and the
base rate instead and use Bayes:
    q_c  = P(kitchen unit | a c against a wall)        q_0 = P(kitchen unit | any furniture/appliance)
    LLR_c = logit(q_c) − logit(q_0)
which is the same likelihood ratio. The ceiling is logit(.95) − logit(.05) ≈ 5.9 nats.

Every question is asked in three paraphrases at temperature 0; the model must answer with one of five phrases
(forced by Ollama's JSON-schema `format`), mapped to 0.05, 0.25, 0.5, 0.75, 0.95. We keep the median over the
paraphrases and the spread (max − min). Every answer is cached in <out>/answers.jsonl with the model, its digest,
the date and the prompt's SHA-256, so a rerun only asks what is new, and the priors can be reviewed and reproduced.

    python3 tools/elicit_priors.py [--model gemma4:26b] [--out tools/e1_out] [--dry-run] [--limit N]
    OLLAMA=http://localhost:11434 (default)

Outputs in <out>/: answers.jsonl (raw, with provenance), level4_priors.csv, level3_kitchen.csv, provenance.json.
The corpus Brier score and the replay of logged kitchen fits are separate steps (they need the corpus and the logs).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

LEVELS = ["almost never", "unlikely", "as likely as not", "likely", "almost always"]
LEVEL_P = dict(zip(LEVELS, [0.05, 0.25, 0.5, 0.75, 0.95]))

SYSTEM = (
    "You answer questions about ordinary homes, as most people would. Answer with exactly one of these phrases: "
    + ", ".join(f'"{l}"' for l in LEVELS)
    + '. Reply as JSON: {"answer": "<phrase>"}.'
)

# Level 4: presence of c given f. Three paraphrases each. (pi_c,notf is derived from the other functions.)
L4_POS = [
    "In a typical home, how often does a room used as a {f} contain at least one {c}?",
    "Think of {f}s in ordinary homes. How likely is it that a given {f} has {a_c} in it?",
    "Someone walks into the {f} of a home. How likely are they to find {a_c} there?",
]
# Level 3: P(kitchen unit | c against a wall); the base question uses c = "piece of furniture or appliance".
L3 = [
    "In a home, {a_c} stands against a wall. How likely is it to be part of a fitted kitchen, the run of built-in kitchen units?",
    "You see {a_c} against a wall in a home. How likely is it that it belongs to the built-in kitchen units?",
    "Of all the {cs} standing against walls in ordinary homes, how often is one part of a fitted kitchen run?",
]
L3_BASE_CLASS = "piece of furniture or appliance"


# How a class is named in a question when its label is ambiguous ("a hood" was read as clothing).
SPOKEN = {"hood": "extractor hood"}


def spoken(c: str) -> str:
    return SPOKEN.get(c, c)


def article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


PLURALS = {"chest of drawers": "chests of drawers", L3_BASE_CLASS: "pieces of furniture or appliances"}


def plural(noun: str) -> str:
    return PLURALS.get(noun, noun + "s")


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def spread(ps: list[float]) -> float:
    return max(ps) - min(ps)


def build_questions(vocab: dict, kitchen_classes: list[str]) -> list[dict]:
    """Every question as a dict: kind, class, function, negated, paraphrase index, prompt."""
    qs = []
    for f in vocab["functions"]:
        for c in vocab["classes"]:
            for i, t in enumerate(L4_POS):
                qs.append(dict(kind="L4", cls=c, func=f, negated=False, para=i,
                               prompt=t.format(f=f, c=spoken(c), a_c=article(spoken(c)))))
    for c in kitchen_classes + [L3_BASE_CLASS]:
        for i, t in enumerate(L3):
            qs.append(dict(kind="L3", cls=c, func="kitchen unit", negated=False, para=i,
                           prompt=t.format(c=spoken(c), cs=plural(spoken(c)), a_c=article(spoken(c)))))
    return qs


def ask(url: str, model: str, prompt: str, timeout: float = 300.0) -> str:
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": {"type": "object", "properties": {"answer": {"type": "string", "enum": LEVELS}},
                   "required": ["answer"]},
        "options": {"temperature": 0, "seed": 0},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    answer = json.loads(content)["answer"]
    if answer not in LEVEL_P:
        raise ValueError(f"answer outside the scale: {answer!r}")
    return answer


def model_digest(url: str, model: str) -> str:
    with urllib.request.urlopen(url + "/api/tags", timeout=10) as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == model:
                return m.get("digest", "")
    raise SystemExit(f"model {model} not served at {url}")


def load_cache(path: Path) -> dict[tuple[str, str], dict]:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[(rec["model"], rec["prompt_sha"])] = rec
    return cache


def summarise(records: list[dict], authored: dict[str, float],
              weights: dict[str, float] | None = None) -> tuple[list[dict], list[dict]]:
    """Medians and spreads per question; pi_c,notf as the weighted mean of pi_cg over g != f (uniform if no
    weights); level-3 log-odds per paraphrase, then median and spread."""
    groups: dict[tuple, dict[int, float]] = {}
    for r in records:
        groups.setdefault((r["kind"], r["cls"], r["func"], r["negated"]), {})[r["para"]] = r["p"]

    med: dict[tuple[str, str], float] = {}
    spr: dict[tuple[str, str], float] = {}
    for (kind, c, f, neg), ps in groups.items():
        if kind == "L4" and not neg:
            med[(c, f)] = statistics.median(ps.values())
            spr[(c, f)] = spread(list(ps.values()))
    l4 = []
    for (c, f) in sorted(med):
        others = [g for (cc, g) in med if cc == c and g != f]
        if not others:
            continue
        w = {g: (weights or {}).get(g, 1.0) for g in others}
        notf = sum(w[g] * med[(c, g)] for g in others) / sum(w.values())
        l4.append(dict(cls=c, func=f, pi_cf=med[(c, f)], spread_cf=spr[(c, f)], pi_c_notf=notf))

    l3 = []
    base = groups.get(("L3", L3_BASE_CLASS, "kitchen unit", False), {})
    for c, a in authored.items():
        qc = groups.get(("L3", c, "kitchen unit", False), {})
        paired = [logit(qc[i]) - logit(base[i]) for i in sorted(qc) if i in base]
        if not paired:
            continue
        l3.append(dict(cls=c, authored=a, elicited=statistics.median(paired), spread=spread(paired),
                       q_c=statistics.median(qc.values()), q_base=statistics.median(base.values())))
    return l4, l3


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--vocab", type=Path, default=HERE / "elicit_vocab.toml")
    ap.add_argument("--out", type=Path, default=HERE / "e1_out")
    ap.add_argument("--dry-run", action="store_true", help="print the question count and three prompts, ask nothing")
    ap.add_argument("--limit", type=int, default=0, help="ask at most N new questions (0 = all)")
    args = ap.parse_args()
    url = os.environ.get("OLLAMA", "http://localhost:11434").rstrip("/")

    vocab = tomllib.loads(args.vocab.read_text())
    kitchen = tomllib.loads((args.vocab.parent / vocab["kitchen_config"]).read_text())
    km = kitchen["KitchenMetaconcept"]
    authored = dict(zip(km["MemberClasses"], km["MemberClassLogOdds"]))
    questions = build_questions(vocab, list(authored))

    if args.dry_run:
        print(f"{len(questions)} questions ({sum(q['kind'] == 'L4' for q in questions)} level 4, "
              f"{sum(q['kind'] == 'L3' for q in questions)} level 3)")
        for q in (questions[0], questions[3], questions[-1]):
            print(" ", q["prompt"])
        return

    args.out.mkdir(parents=True, exist_ok=True)
    answers_path = args.out / "answers.jsonl"
    cache = load_cache(answers_path)
    digest = model_digest(url, args.model)
    todo = [q for q in questions if (args.model, sha(SYSTEM + "\n" + q["prompt"])) not in cache]
    cached = len(questions) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(questions)} questions, {cached} cached, asking {len(todo)} on {args.model}", flush=True)

    t0 = time.monotonic()
    with answers_path.open("a") as fh:
        for n, q in enumerate(todo, 1):
            answer = ask(url, args.model, q["prompt"])
            rec = dict(q, answer=answer, p=LEVEL_P[answer], model=args.model, model_digest=digest,
                       date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       prompt_sha=sha(SYSTEM + "\n" + q["prompt"]))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            cache[(args.model, rec["prompt_sha"])] = rec
            if n % 25 == 0 or n == len(todo):
                el = time.monotonic() - t0
                print(f"  {n}/{len(todo)}  {el / n:.2f} s/question", flush=True)

    current = {sha(SYSTEM + "\n" + q["prompt"]) for q in questions}
    records = [r for (m, h), r in cache.items() if m == args.model and h in current]
    l4, l3 = summarise(records, authored)
    write_csv(args.out / "level4_priors.csv", l4)
    write_csv(args.out / "level3_kitchen.csv", l3)
    (args.out / "provenance.json").write_text(json.dumps(dict(
        model=args.model, model_digest=digest, ollama=url, levels=LEVEL_P, system=SYSTEM,
        templates=dict(L4_POS=L4_POS, L3=L3, L3_BASE_CLASS=L3_BASE_CLASS), spoken=SPOKEN,
        pi_c_notf="uniform mean of pi_cg over the other functions",
        vocab_sha=sha(args.vocab.read_text()), answered=len(records), of=len(questions),
        written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), indent=2))

    if l3:
        print("\nkitchen membership log-odds   authored  elicited  spread")
        for r in l3:
            print(f"  {r['cls']:<26} {r['authored']:>8.2f}  {r['elicited']:>8.2f}  {r['spread']:>6.2f}")
    print(f"\nwrote {args.out}/level4_priors.csv ({len(l4)} rows), level3_kitchen.csv ({len(l3)} rows)")


if __name__ == "__main__":
    sys.exit(main())
