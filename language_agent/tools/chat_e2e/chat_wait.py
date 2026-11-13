#!/usr/bin/env python3
"""Print what language_agent did for the LAST user message, waiting until it has answered (or a card is shown).
usage: chat_wait.py <chat_log_dir> [timeout_s]"""
import glob, json, os, sys, time
logdir = sys.argv[1]; timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 150
log = sorted(glob.glob(os.path.join(logdir, "chat_*.jsonl")), key=os.path.getmtime)[-1]
t_end = time.time() + timeout
while True:
    rows = [json.loads(l) for l in open(log) if l.strip().startswith("{")]
    last = max(i for i, r in enumerate(rows) if r.get("ev") == "user")
    tail = rows[last:]
    done = any(r.get("ev") == "verbatim_reply" or r.get("ev") == "mission_card"
               or (r.get("ev") == "response" and r.get("content") and not r.get("tool_calls"))
               or (r.get("ev") == "fast_path" and r.get("kind") in ("mission_yes", "stop"))
               for r in tail)
    if done or time.time() > t_end:
        break
    time.sleep(1)
time.sleep(2)   # the card/verbatim line and its log rows land together
rows = [json.loads(l) for l in open(log) if l.strip().startswith("{")]
last = max(i for i, r in enumerate(rows) if r.get("ev") == "user")
for r in rows[last:]:
    ev = r.get("ev")
    if ev == "request":
        print("· request to the chat model"); continue
    if ev == "planner_request":
        st = r.get("state", ""); i = st.find("remembered objects")
        print("· planner_request; state has remembered line:", i >= 0, "|", st[i:i + 160].replace("\n", " ") if i >= 0 else "")
        continue
    keep = {k: v for k, v in r.items() if k not in ("t", "history", "tools", "state")}
    print(json.dumps(keep, ensure_ascii=False)[:700])
if not done:
    print("(timed out waiting)")
