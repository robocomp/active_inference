# Does the epistemic explorer earn its place? — A/B on the 594 real layouts

Run 2026-09-14. Reproduce with `analysis/explorer/explorer_ab.py <eig_dir> <heur_dir>`.

## Arms

Same binary (`bin/wall_slam_selftest`, mtime 09-14 01:44, *both* runs after it — no rebuild between),
same corpus, same 900-frame cap, same per-room seeds. The only difference is the viewpoint objective:

| arm | command |
|---|---|
| EIG (default) | `WS_NO7=1 WS_ROOMS=594 WS_LAYOUTS=…/corpus594.txt WS_POLY_ROOM=$OUT/poly` |
| heuristic | the same **plus `WS_NO_INFOGAIN=1`** |

`WS_NO_INFOGAIN` is read at `tools/wall_slam_selftest.cpp:2719` inside the test-8 config, so it does
reach the 594-room arm — checked, not assumed.

## Result: a null on every endpoint

Paired by room, because room-to-room spread (0.95–1.00) is an order of magnitude larger than any
explorer effect and an unpaired test would drown in it. Sentinel rows (IoU 0.000 / Hausdorff 1e9)
filtered and counted separately.

| endpoint | EIG | heuristic | paired difference (EIG − heur) |
|---|---|---|---|
| IoU mean (583 rooms both completed) | 0.9827 | 0.9835 | **−0.0007, 95% CI [−0.0019, +0.0004]** |
| IoU median | 0.9870 | 0.9880 | median +0.0000 |
| rooms each wins | 251 | 263 | 69 ties |
| hard failures / 594 | 8 | 6 | — |
| frames to first publishable map (404 rooms) | med 133 | med 142 | **+5.9, 95% CI [−13.8, +27.9]** |
| never publishable in 900 frames | 129 | 113 | — |

Exploratory (not pre-registered), by ground-truth corner count — no band recovers a difference:

| corners | rooms | ΔIoU | 95% CI |
|---|---|---|---|
| 6 | 324 | −0.0009 | [−0.0022, +0.0004] |
| 8 | 154 | −0.0006 | [−0.0022, +0.0009] |
| 10 | 73 | +0.0008 | [−0.0059, +0.0063] |
| 12+ | 32 | −0.0022 | [−0.0107, +0.0042] |

The heuristic is nominally ahead on every endpoint and significantly ahead on none.

## ★ Why this does not reproduce the earlier win

The 09-04 result that made EIG the default was **50 synthetic rooms, median .947 → .954**, with the
worst L-room .812 → .944. That is a different corpus and a different question. Here **every room ran
to the 900-frame cap in both arms** (594/594 and 593/594) — the early-stop "publishable and nothing
left to ask" never fires. So the final map is measured at saturation: in a room this size, 900 frames
of *any* reasonable policy sees every wall, and the endpoint is physically unable to separate the
policies. The synthetic corpus contained adversarial spurs and alcoves where a policy can genuinely
get stuck; 594 real Manhattan floor plans mostly do not.

★ **An objective cannot be graded on an endpoint its own budget has saturated.** The number that
should have been checked first is not the IoU, it is the 594/594 at the frame cap.

## What to claim

Supported: the uncertainty is *consumed* — it gates publication, and the explorer reads it.
NOT supported, and must not be claimed: that the epistemic term buys accuracy or speed over a simple
heuristic on rooms of this size. Untested: larger or multi-room environments, where a policy has room
to waste time, and a tighter frame budget, which is where a difference would have to show up.
