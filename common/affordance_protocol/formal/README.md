# Formal validation of the affordance protocol

`AffordanceProtocol.tla` is the protocol between one producer (a concept agent) and one consumer
(the controller) over a single affordance node. It is checked with TLC.

## Run

    JAR=$(ls ~/.vscode/extensions/tlaplus.vscode-ide-*/tools/tla2tools.jar | tail -1)
    ./check_all.sh "$JAR"

`check_all.sh` sweeps all eight combinations of the three design choices and reports, for each,
whether a deadlock is reachable. Each combination corresponds to a real pair of code paths.

## The three choices, and where they live in the code

| constant | TRUE | FALSE | code |
|---|---|---|---|
| `LevelTriggered` | producer arms its wait when IT publishes | producer must catch the node in `Executing` | `AffordanceManager::monitor_execution` / `publish_target` |
| `ConsumerRefuses` | declining is reported (node retires) | declining is silent | the useless-spot skip in `ControllerSession` |
| `RearmUnchanged` | re-publishing the same cell re-arms | an unchanged proposal is declined | `publish_target`'s `same_target && !current_pending` |
| `LatchOnDecide` | the producer marks its offer live when it DECIDES | ...only once the offer is ON THE WIRE | `CalibChannel::offer` vs `mark_offered` |

## Publishing is not atomic (added 2026-08-24)

The first version of this module had `node' = "Offered"` in a single step, so the interval between
*deciding* to offer and the offer *becoming visible* did not exist. A real producer stages first:
`room_concept` creates `afford_calib` QUIESCENT and deliberately does not publish on that cycle, so a
consumer can never latch a contract before it is written.

A deadlock lived in exactly that gap and the model could not express it. `CalibChannel::offer()` set
`offer_open_` — "one of my offers is live" — before `publish_target` ran; `ensure_calib_node()`
returned false on the creation cycle; nothing reached the graph; and the latch then blocked every
later offer, releasable only by an outcome no consumer could ever produce. It showed up as
`afford_calib` present in the graph with no `has_intention` edge, for ever. **It was found by looking
at the graph, not by TLC** — which is the second time a defect in this fleet has sat in the gap
between what an agent stages privately and what the shared model can see.

`Publish` is now split into `Decide` / `Emit` / `EmitSkipped`, with a producer-local `armed` and
`staged`. TLC finds the bug in three states:

    State 1  node=Completed  staged=F  armed=F
    State 2  <Decide>        staged=T  armed=T     <- latched on the DECISION
    State 3  <EmitSkipped>   staged=F  armed=T     <- nothing on the wire, nothing can clear it
             Deadlock reached.

Two modelling notes, both of which changed the answer:

* **Reading the outcome and releasing the latch are ONE step**, because `consume_completion_event()`
  and `on_outcome()` are the same branch in the code. Modelling them separately produced a spurious
  deadlock — a defect in the abstraction, not in the system.
* **`Emit` needs STRONG fairness.** Weak fairness only forces an action that stays continuously
  enabled, and `EmitSkipped` disables `Emit` each time it fires, so under WF an adversary can
  stage-and-skip for ever. The real skip happens exactly once, on the node-creation cycle.

## Result (2026-08-24)

    LATCH   LEVEL   REFUSE  REARM   verdict
    TRUE    *       *       *       DEADLOCK   (all eight -- the bug is unconditional; no
                                                combination of the other choices masks it)
    FALSE   TRUE    *       *       SOUND      (all four -- the shipped configuration)
    FALSE   FALSE   *       *       DEADLOCK   (all four -- reproduces the 2026-08-18 finding
                                                that LevelTriggered must be TRUE)

## Result (2026-08-18)

    LEVEL   REFUSE  REARM   verdict
    TRUE    *       *       SOUND      (all four)
    FALSE   TRUE    TRUE    SOUND
    FALSE   TRUE    FALSE   DEADLOCK
    FALSE   FALSE   TRUE    SOUND
    FALSE   FALSE   FALSE   DEADLOCK

**`LevelTriggered` is sufficient on its own.** With edge detection the protocol is only sound if
`RearmUnchanged` compensates — i.e. two independently-reasonable local decisions
("don't re-announce an unchanged proposal", "detect completion by watching the claim") combine into a
system that stops for ever. Neither is wrong by itself, which is why neither was caught by reading
either file.

The counterexample is three states long:

    1. Completed, waiting=F, learned=T      initial
    2. Publish -> Offered,   learned=F, waiting=F     (publishing does not arm the wait)
    3. Refuse  -> Completed, learned=F, waiting=F
       deadlock: Publish needs learned or RearmUnchanged; Observe needs waiting. Neither holds.

That is exactly what the live system did: 549 s with zero completions detected while the consumer was
completing continuously, then 259 s motionless with the node Offered and the producer healthy.

## Why this file exists

Every protocol defect fixed on 2026-08-18 was a liveness defect decidable in a state space of under
thirty states. All of them were instead found by watching a robot stand still, one symptom at a time,
over a full day. Change the protocol here first, check it, then change the code.

---

# `AffordanceExecutionClaim.tla` — can the two agents disagree about what is being executed?

`AffordanceEpoch.tla` asks whether a **report** can be booked against the wrong offer. This asks the
mirror question on the other half of the exchange: **while the approach is still running**, can the
producer and consumer hold different beliefs about which pose the robot is driving to — and can the
producer act on its own belief and destroy a good approach?

## Run

    JAR=$(ls ~/.vscode/extensions/tlaplus.vscode-ide-*/tools/tla2tools.jar | tail -1)
    ./check_claim.sh "$JAR"

## The design choice

| constant | TRUE | FALSE | code |
|---|---|---|---|
| `ClaimCarried` | the consumer publishes the proposal it accepted **and the pose it is actually driving to**, on an `executing` edge it owns; the producer measures progress against that | the consumer's claim is a **boolean** (`active`) — it can say THAT it is executing, never WHAT — so the producer measures against its own latest publication | `AffordanceManager::publish_executing` / `read_executing`; `RoomSceneGraph::break_execution_stall` |

## Result (2026-08-24)

    CLAIM    AgreementObservable    NoFalseAbandon     Terminates   states
    FALSE    VIOLATED               VIOLATED           holds        5
    TRUE     holds                  holds              holds        4518

At a 3× larger bound (`Cells = {c1,c2,c3}`, `MaxSteps = 3`, `Timeout = 3`) `ClaimCarried = TRUE`
completes with **22815 states generated, 7788 distinct, depth 22, no error found**.

**The counterexample is four states long**, and it is the live failure of 2026-08-23 exactly:

    1. Offered.   proposed = c1
    2. Claim.     committed = c2   <- the consumer REPAIRED the pose it was given
                  claim = opaque   <- and the wire cannot say so
    3. Tick.      stall = 1        <- watchdog watches c1; distance to c1 is Far, never improves
    4. Tick.      steps 2 -> 1     <- THE CONSUMER IS DRIVING (idle = 0)
                  stall = 2        <- and the producer's budget expires anyway

State 4 is the whole thing: `idle = 0` (the consumer closed ground on this very tick) with
`stall = Timeout` (the producer is about to abandon it). The budget was spent watching a pose nobody
was driving to. `best` never leaves `Far` because the watched cell is not the committed one.

`AgreementObservable` is the general statement — *no party may judge another by a quantity the other
cannot observe*. Under `ClaimCarried = FALSE` it is violable **by construction**: the old protocol
contains no term that could make it true.

## ⚠ Two harness lessons, both learned the hard way here

1. **`"N states generated"` does not mean the run finished.** The first version of `check_claim.sh`
   copied `check_epoch.sh`'s guard and printed a full, plausible table — `FALSE VIOLATED / TRUE
   holds`, exactly the expected shape — from a run in which TLC had **aborted after four states**
   with `unable to fingerprint` (the `claim` variable compared a record against a string). A crashed
   run still prints that line. `check_claim.sh` now requires TLC's own closing line and greps for the
   abort signatures explicitly; anything else is reported as a broken run, never as a verdict.
   *A table with the shape you expected is the easiest kind of nothing to accept.*
2. **A terminal state is not a deadlock, and `CHECK_DEADLOCK FALSE` is not the fix.** `Retired` had
   no successor, TLC correctly said `Deadlock reached`, and suppressing the check would have hidden
   any genuine deadlock too. The producer really does select another cell, so `Reoffer` models that
   and the deadlock check stays armed.

## What the abstraction keeps and drops

Stated in the module header, and deliberately, because the previous protocol proof on this fleet was
sound about the logic and silent about what actually bit. In particular it **drops the DSR
transport**: attribute writes are atomic here. That is sound *only* because the fix puts the
consumer's half on an edge it alone writes — on a shared node it would be false, since `update_node`
is a whole-node replace on both cortex sync engines and a stale writeback deletes the peer's
attribute graph-wide. **Agreement in this model is not license to move these fields onto the
affordance node.**
