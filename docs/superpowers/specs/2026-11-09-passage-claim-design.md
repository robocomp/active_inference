# A door passage is one claim, and no claim can hang

Status: DRAFT (2026-11-09); decisions 1-3 of §10 taken by the owner, see §11. The build is not yet approved. Nothing coded. Lines are at 4afebca8 (feat/room-recognition).
Related: `2026-11-05-exploration-tour-design.md`, `2026-11-02-door-crossing-contract.md`, `reachability_belief.h` (F1).

## 1. The problem

12 of 21 tours pass; every pass ended in under 6 min, every failure hit the 1200 s limit. Last passage event of the 9 failures (the log's last line of its kind, a pointer, not a diagnosis): `open` retried
without limit (clean5 18 attempts, clean11 17, clean14); no-mouth / vantage (clean3, clean8); `STUCK` and the drive off
(clean6); the room-switch or a long rate hold in the tail (clean13, 16, 18). Common form: a wait with no exit.

## 2. Every wait on the passage path

Files: ED `exploration_drive.h`, DP `controller_door_passage.h`, PD `controller_policy_drive.cpp`, CS `controller_session.cpp`,
SM `specificworker_mission.cpp`. ⚠ = the wait's exit needs the robot to move, and nothing moves it.

| # | Wait or hold | Waits for | What ends it today | Exit needs motion? |
|---|---|---|---|---|
| 1 | Drive skipped while a passage runs, ED:78 | passage end | passage end only; the MaxRoomS clock runs but is not tested | n/a |
| 2 | Route not built, CS:3869-3876 | map, targets | two failures, `cross_unreachable` (DP:367-376) | no |
| 3 | No target yet, DP:1132-1144 | door_concept publishes | nothing | no (peer) |
| 4 | Route point or re-aim, no route or belief verdict, PD:926-977 | a finite field | 5 s no-route clock (6ca644fe) or hand back, then #2's count | no |
| 5 | Policy `Stuck` tick, PD:605 | next decision | belief hand back, finite T only | yes |
| 6 | Rate hold, PD:171-181 | loop back above MinControlHz | only the rate recovering; no field read, clocks frozen | no (CPU) |
| 7 | Blind, frame, odometry holds, PD:140-200 | signal returns | signal; turn-sign emergency: operator only | no |
| 8 | ⚠ Vantage hold PD:958-961 and mouth hold CS:3821-3833 | the opening seen open | door_concept reading it shut; else nothing | **yes** |
| 9 | ⚠ Open wait at the vantage, DP:1046-1059 | door_concept's `open` | `attempts_` counted, never capped | often |
| 10 | ⚠ Cross on the anchor before the switch, PD:856-903 | anchor, field | Unreachable (finite T) or #11; **T = ∞ has no exit** (the clock runs only while `clearing()`, PD:879) | yes |
| 11 | ⚠ Stale anchor, `policy_stale_wait_`, PD:895-903 | anchor published | 2 plan durations, then reposition | **yes** |
| 12 | Clearing past the switch, PD:879-891 | through | 5 s `ClearingNoRoute` | no |
| 13 | Through, waiting the switch and archive, DP:1122-1130 | room_concept, ltsm | nothing | no (peers) |
| 14 | Tour decision held, SM:1052 / 1062 / 1071 | entry face, proposal, recognition | MaxRoomS for the first and last; **the pending proposal has no bound** | no |
| 15 | Drive's `Cause::Stuck`, ED:87, SM:1090 | — | nothing: the drive turns itself off, and `explore_tour.sh` breaks only on `DONE` | terminal |

Off the tour path: `close_wait_ms_` (DP:727; the tour stops after Cross) and the no-anchor wait PD:1083-1094 (chain mode).

The earlier review's caps are worse than reported. `kMaxCrossUnreachable` (DP:787-797) is unreachable with `UseRoute = true`:
`update()` returns into `update_route` at DP:456 first. `update_route` reads no cross outcome, so a `cross_hand_back` on a tour
reaches no counter. `fail_stuck` is unreachable with `Policy.Enabled = true`: `begin_escape`'s callers lie after the policy branch
returns (CS:7060-7066). The real bound on a tour is `kMaxRouteFailures`, which covers rows 2 and 4 only.

## 3. The claim

`rc::PassageClaim` (a new pure header `passage_claim.h`, no DSR, no Qt) is held by `DoorPassage`, which already has exactly one
passage at a time. It is not an affordance claim (`seen_executing_`), which stays what it is. It holds: the door, the room left,
start time, `T_0`, `T_best` (lowest remaining time reached), lost time, hand-back count, current wait kind, `L`, `dt_obs`, and the
end verdict. The session feeds it the remaining time each cycle; `DoorPassage::update` ticks it. That call sits at
CS:274, upstream of every policy hold, so holds cannot starve it.

| Claim state | Existing DoorPassage state |
|---|---|
| Born | `start()`; `T_0` taken |
| Motion | Look, Approach, Cross with a leg driving, the drive-in past `exited_` |
| Wait(kind) | rows 3, 8, 9, 11, 13 |
| Ended(completed) | `route_through` |
| Ended(abandoned, why) | `finish()` with `failed_`; `last_outcome_ = cross_unreachable` when a route caused it, so missions keep their name [my call] |

Step::Close stays outside this claim (a tour never reaches it). A single command's close runs as a second claim with no
motion term and prior ½, the code's own statement that nothing in the graph tells "not yet" from "never" (DP:12-14).

## 4. Expected duration

`T_c = T_motion + sum S_k`. `T_motion` is the belief's own remaining time: route / v_max + |dyaw| / w_max (reachability_belief.h),
plus the fixed length of the remaining legs. `S_k` is the expected service time of each wait still ahead (door opening, target
published, switch + archive). A wait counts down at real time while it runs: within `S_k` the claim is on schedule. **`S_k` is
measured** as a per-kind median over the 12 passing tours' `controller.log` (test 0, §9), not chosen.

## 5. Abandonment: the reachability belief at the level of the claim

Hypotheses for the claim: U "this passage will not complete from here", R "it will". The evidence is the time the prediction does
not explain:

`lost = elapsed - (T_0 - T_best)`.

Under R, unexplained delay is memoryless with mean `tau`: `P(no progress for lost | R) = exp(-lost / tau)`; under U it is 1. So
`L = L0 + lost / tau + (hand backs) * log(1/h)`. `L0 = logit(p_fail)`, the failed fraction of passages over the 21 tours. `h` is the
measured rate of hand backs in passages that still completed (P(hand back | U) = 1, P(hand back | R) = h). Both come from test 0;
until then `L0 = logit(0.01)` (F1's base rate).

Abandon iff `L > bar`, `bar = log((T_rem + tau) / dt_obs)`. It is F1's decision and F1's price: waiting costs P(U) dt_obs, abandoning
forfeits P(R) T. `tau` is the mean remaining time given R after an overrun, which is what makes the bar well defined at T = 0.

**T = ∞.** `T_rem` is the last finite `T_c`, because an infinite field says the *map* has no route, and under R the map regains one
after a further `tau`. `E[T | R] = T_last + tau` is finite. Survival evidence is folded anyway: it is about time, not the field.
Per-field claims and counts at T = ∞ stay unfolded, as fix round 2 ruled (reachability_belief.h `add`); that ruling stands.

Three properties, each tested in §9:
- **No hang, provably.** `T_best >= 0` gives `lost >= elapsed - T_0`, so `L > bar` by `elapsed = T_0 + tau (bar_max - L0)`. Example:
  T_rem 30 s, dt_obs 0.1 s, L0 -4.6, tau 5 s: `lost = 5 (5.9 + 4.6) ≈ 52 s`.
- **Flicker cannot reset it.** Only a new `T_best` repays lost time, and `T_best` falls at most `T_0` in total. A route that flips
  finite and infinite, or a robot flipping MOVING/STOPPED, buys nothing.
- **Blind holds count, correctly.** Survival evidence is "no completion yet", observed whether or not a field was read. rtD1e's bug
  (13 Blind holds read as "no route") was a per-goal 5 s clock; here they cost 13 dt / tau nats, which is negligible.

**Recovery before abandonment.** Reposition (`DoorPassage::reposition`) is the one action that changes the view from a ⚠ wait. It is the
same rule with a cheaper forfeit: reposition iff `L_a > log((T_repos + tau) / dt_obs)`, where `L_a` is the attempt's belief,
`logit(h) + lost_since_attempt / tau`. A smaller T gives a lower bar, so recovery comes first with no extra level. Each
reposition restarts `L_a` and not `L`, so recovery cannot loop past the claim's bound.

⚑ FLAGGED: `tau` (`kLivenessTauS`, 5 s by default, the value `kNoRouteReportMs` already flagged) is the one liveness constant. It is a
hazard rate and not a timer: the abandon time varies with T, dt_obs and hand backs. No model term supplies it: nothing in the world says how long
a dead wait may take to be recognised.

## 6. What each timer or hold becomes

| Today | After |
|---|---|
| `kMaxRouteFailures`, `kMaxCrossUnreachable`, `attempts_` | deleted; a failed route or Unreachable hand back folds `log(1/h)` and rebuilds; `attempts_` for `open` folds the same |
| `kNoRouteReportMs` on the passage path, `policy_door_route_no_route_`, `policy_clearing_no_route_` | deleted; survival evidence. Non-passage goals keep `advance_no_route_clock` and take `tau` as their constant (CS:6716) [my call] |
| `policy_stale_wait_`, row 8 and 11 waits | Wait kinds; the reposition rule, then abandon |
| `close_wait_ms_` | §3's second claim |
| Rate, Blind, frame holds (rows 6, 7) | stay as safety holds; time accrues to the claim, so they can no longer pass the bound |
| Row 13 | a Wait kind |
| Row 14, the pending proposal | outside any claim (the passage has ended): takes MaxRoomS like its two neighbours |
| ED:78 and MaxRoomS | unchanged: a claim always ends, which removes the concern [my call] |
| `Cause::Stuck`, ED:87 | `Cause::Failed`; §7 |

`kDoorRouteRebuildMs` (2 s) stays: it paces a computation and cannot hang.

## 7. After abandonment

1. The robot stops and the claim's route and anchor are dropped; the passage ends failed.
2. **Body clear of the opening** (`not in_opening_`, not exited): `note_failed(door)`, `AffordancesOnly`, the drive decides again.
   An abandoned door stays skipped for that visit, as today.
3. **Body in the opening, or through without a switch**: no safe next goal exists. The tour ends failed [my call].
4. The tour's end is always logged: `[explore-tour] FAILED <why>` replaces the silent stop, and `explore_tour.sh` breaks on it.
   A start room left with a failed, never-crossed door reports FAILED, not `DONE`; ED:89 currently reports `Done` there.

## 8. What this will not fix

It turns hangs into clean, recoverable failures. It does not fix room recognition errors, so a tour that recognises a room wrongly still
fails. Nor does it fix phantom doors, or a door that truly cannot be crossed. A tour that would have crossed at minute 15 now fails
at minute 3, so the pass rate may fall. **Any change resets the clean-tour count**, including this
one; 3 in a row at a 57% base rate is not yet evidence anyway.

## 9. Tests (written first)

0. **Measure before coding.** From the 21 tours' logs: `S_k` per wait kind, passage count, `p_fail`, `h`, and elapsed vs `T_0` on
   passing passages. If a passing passage's lost time exceeds the bound, the model is wrong and this design returns to the owner.
1. `controller/tests/test_passage_claim.cpp` (pure): the bound from §5 holds for each wait kind; `T = ∞` for ever abandons within
   it; finite/infinite flicker for ever abandons within it; steady progress never abandons; a detour costs its time and no more; a hold
   of length `S_k` costs nothing; repositions cannot exceed the claim's bound.
2. `test_door_passage.cpp`: each Step ends in `Ended`; a failed route rebuilds and folds; route mode counts hand backs (the dead-cap case);
   the abandonment positions of §7 give the right outcome.
3. `test_exploration_drive.cpp`: `Cause::Failed` for a failed Return; `FAILED`, not `Done`, for a start room with a failed door.
4. Replay: the 9 failing tours' timelines through `PassageClaim`: each abandons within the bound at the hold named in §1. The 12 passing
   tours: zero abandonments. Where the logs lack per-tick T, test 0 says so and this becomes a live measure.
5. Live: 5 tours with one record per claim (`T_0`, elapsed, lost, L, bar, outcome). Pass: no run reaches the 1200 s limit, every failure
   names its claim. No clean-tour claim from fewer than 5.

## 10. Open points for the owner

1. `tau`: 5 s reused, or other? It sets every abandon time.
2. Reposition: as in §5, or abandon directly (simpler; loses the ⚠ waits' only recovery)?
3. Body in the opening or through without a switch: end the tour (my default), or attempt a return to a clear side?
4. Should an abandoned door end the tour at once, or may the drive continue with others as §7 has it (my default)?
5. `S_k`, `p_fail`, `h`: fixed from the 21 tours (my default) or updated during a run?
6. Close: the second claim takes about 20 s where 5 s applies today, off the tour path. Acceptable?

## 11. Decisions taken by the owner (2026-11-09)

1. `tau` = 5 s, to be checked against the longest unexplained delay in the passing tours (test 0).
2. Reposition before abandoning, as in §5.
3. The body stopped IN the opening did not occur in any of the 9 failed tours (the nearest stop was 0.79 m from a door). The case that occurred is clean13: through the door, 1.31 m past the plane, and the passage never ended. That case ends the tour with a logged failure. The in-the-opening case is dropped from §7 until one happens.

Open: points 4-6 of §10.
