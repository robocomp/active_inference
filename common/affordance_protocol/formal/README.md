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
