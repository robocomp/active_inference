#!/usr/bin/env python3
"""bed_replay_pin.py REPLAY_BIN MANIFEST_DIR [DUMP] — the bed_1 replay pin (F1 round 6.2; docs/superpowers/results/
2026-11-25-bed-depth-diagnosis.md). Replays objects_replay1's bed_1 dump through the current engine with today's staging rules
(--sig2-cam-measured --neighbour-size-std --standoff --permanence) and the birth seed-H rule (--seed-h-rule), then asserts,
with every bound DERIVED (none fitted):
  * at every update the back gap to the staged wall >= -2 sigma_gap, sigma_gap^2 = sigma_standoff^2 + sigma_wall^2
    (the manifest stand-off std and the staged wall's 0.02 m: the wall prior gap - delta ~ N(mu, sigma_standoff^2), delta ~
    N(0, sigma_wall^2));
  * at the end |cy - 3.074| <= 2 sigma_cy (truth: the mesh centre in this run's room frame, rootcause/nees_rows.py);
  * h in [2.04, 2.13] (the coordinator's ruling): the model's h INCLUDES the headboard panel (countable_belief.h), so it is
    graded against the frame, not the mattress (1.971): the camera points support 2.04-2.09 (diagnosis section 0) and the
    Bed.proto frame envelope is 2.12.
Gated: no dump on this machine => SKIP (exit 0).
"""
import csv, math, os, subprocess, sys

DUMP = '/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/objects_replay1/replay/bed_1_1795512057_0001.bin'
replay, mdir = sys.argv[1], sys.argv[2]
dump = sys.argv[3] if len(sys.argv) > 3 else DUMP
if not os.path.exists(dump):
    print(f'bed_replay_pin: SKIP (no dump at {dump})'); sys.exit(0)

std_so = None
for line in open(os.path.join(mdir, 'bed.concept.toml')):
    if line.strip().startswith('standoff_std_m'):
        std_so = float(line.split('=')[1].split('#')[0])
sigma_gap = math.sqrt(std_so ** 2 + 0.02 ** 2)
out = subprocess.run([replay, dump, '--allow-defaults', '--sig2-cam-measured', '--neighbour-size-std', mdir, '--standoff', mdir,
                      '--permanence', mdir, '--seed-h-rule'], capture_output=True, text=True)
rows = list(csv.DictReader(out.stdout.splitlines()))
bad = 0
def check(ok, msg):
    global bad
    print(f'  {msg:70s} {"ok" if ok else "FAIL"}'); bad += 0 if ok else 1
gaps = [(int(r['upd']), float(r['back_gap'])) for r in rows if r['back_gap'] not in ('nan', '')]
worst = min(gaps, key=lambda x: x[1])
last = rows[-1]
cy, scy, h = float(last['cy']), float(last['std_cy']), float(last['h'])
print(f'bed_replay_pin: {len(rows)} updates; seed line: {[l for l in out.stderr.splitlines() if "seed H" in l]}')
check(worst[1] >= -2 * sigma_gap, f'min back gap {worst[1]:+.4f} m (update {worst[0]}) >= -2 sigma_gap = {-2 * sigma_gap:.4f}')
check(abs(cy - 3.074) <= 2 * scy, f'final |cy - 3.074| = {abs(cy - 3.074):.4f} <= 2 sigma_cy = {2 * scy:.4f}')
check(2.04 <= h <= 2.13, f'final h {h:.4f} in [2.04, 2.13] (the frame, not the mattress)')
print('bed_replay_pin:', 'FAILED' if bad else 'OK'); sys.exit(1 if bad else 0)
