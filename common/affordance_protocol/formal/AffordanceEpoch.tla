-------------------------- MODULE AffordanceEpoch --------------------------
(***************************************************************************)
(* CAN A REPORT BE APPLIED TO THE WRONG OFFER?                              *)
(*                                                                          *)
(* AffordanceSelection.tla answers "does the pair keep making progress".     *)
(* It cannot answer this one, and not by oversight: in that model the        *)
(* producer reads an outcome and attributes it to whatever cell the node     *)
(* NAMES AT THAT MOMENT, because the consumer's report carries no identity.  *)
(* A report that refers to a different offer than the one now on the wire is *)
(* therefore not expressible — so the check that matters most for a single   *)
(* shared register cannot be run at all.                                     *)
(*                                                                          *)
(* Every standpoint room will ever publish rides ONE node, `afford_room`.    *)
(* It is a register that is reused thousands of times a run. Four separate   *)
(* live failures on 2026-08-19 were ABA on that register: a value read,      *)
(* the register rewritten, and the stale read applied to the new contents.   *)
(*                                                                          *)
(* This model adds the two things needed to state the question:              *)
(*   1. the consumer captures WHICH cell it is working on, at Claim;         *)
(*   2. the execution lease may reclaim the node WHILE the consumer is still *)
(*      executing (this is real: it exists precisely for a consumer that     *)
(*      died), after which the producer may arm a different cell — and only  *)
(*      THEN does the consumer report.                                       *)
(*                                                                          *)
(* EpochCarried is the design choice under test. FALSE is the code as it     *)
(* stands: the report says only "satisfied", and the producer books it       *)
(* against the current cell. TRUE is the fix in the plan: the report carries *)
(* the id of the goal it belongs to, and the producer ignores a report whose *)
(* id is not the one it is waiting for.                                      *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Cells,          \* the standpoints the producer may name
    Reachable,      \* those the robot can actually stand on
    EpochCarried    \* does the consumer's report say WHICH goal it refers to?

VARIABLES
    node,       \* "Offered" | "Executing" | "Completed"
    proposed,   \* the cell the node names NOW
    gid,        \* producer: the id of the offer it armed
    outcome,    \* "none" | "Satisfied" | "Refused"
    ack,        \* consumer: the id its report refers to (0 = it carries none)
    my_cell,    \* consumer: the cell it captured at Claim and is actually working on
    observed,   \* cells the robot ACTUALLY observed. Physical truth.
    believed,   \* cells the PRODUCER thinks were observed. Must never exceed the truth.
    claimed

vars == <<node, proposed, gid, outcome, ack, my_cell, observed, believed, claimed>>

NoCell == "none"
NoId   == 0

\* Bounded so TLC terminates. Two live ids are enough to express ABA: arm A, arm B, then let A's
\* report arrive. A larger bound finds nothing a smaller one misses here.
MaxId == 3

TypeOK == /\ node     \in {"Offered", "Executing", "Completed"}
          /\ proposed \in Cells \cup {NoCell}
          /\ my_cell  \in Cells \cup {NoCell}
          /\ gid      \in 0..MaxId
          /\ ack      \in 0..MaxId
          /\ outcome  \in {"none", "Satisfied", "Refused"}
          /\ observed \subseteq Cells
          /\ believed \subseteq Cells
          /\ claimed  \in BOOLEAN

Init == /\ node = "Completed" /\ proposed = NoCell /\ my_cell = NoCell
        /\ gid = NoId /\ ack = NoId /\ outcome = "none"
        /\ observed = {} /\ believed = {} /\ claimed = FALSE

(***************************************************************************)
(* PRODUCER                                                                 *)
(***************************************************************************)

\* Arm a cell and stamp it with a fresh id. Re-offering the SAME cell is a new goal — that is
\* settled by AffordanceSelection.tla, where declining an unchanged re-offer deadlocks in all
\* sixteen configurations.
\* ★THE PRODUCER MAY ARM A DIFFERENT CELL, and it must be free to, or ABA cannot arise in the model
\* at all: with a fixed argmax the node always names the same cell, every report matches by accident,
\* and the check passes while proving nothing. (First version of this file did exactly that — both
\* configurations "held", which is the toothless-model failure this project has already paid for
\* twice: an unbounded CHOOSE that generated 0 states, and a grep that read TLC's own error as a
\* verdict. A model that cannot fail is not evidence.)
Publish ==
    /\ node = "Completed"
    /\ gid < MaxId
    /\ \E c \in Cells : proposed' = c
    /\ gid' = gid + 1
    /\ node' = "Offered"
    /\ claimed' = FALSE
    /\ UNCHANGED <<outcome, ack, my_cell, observed, believed>>

\* ★THE LEASE FIRES WHILE THE CONSUMER IS STILL WORKING. This is not a contrived interleaving: the
\* execution lease exists exactly for a consumer that claimed and then died, and the producer cannot
\* tell "died" from "slow" — that is the whole reason it is a lease and not a question.
LeaseReclaim ==
    /\ node = "Executing"
    /\ node' = "Completed"
    /\ outcome' = "none"
    /\ UNCHANGED <<proposed, gid, ack, my_cell, observed, believed, claimed>>

\* Read the report and update the belief. THE ONE LINE THIS WHOLE MODEL EXISTS FOR.
\* Without an epoch the producer has nothing to check the report against, so it books the
\* observation against whatever the node names now. With one, a report from a goal it is no longer
\* waiting on is discarded — which is the only way a reused register can be safe.
Observe ==
    /\ outcome # "none"
    /\ node = "Completed"
    /\ IF EpochCarried
       THEN /\ believed' = IF outcome = "Satisfied" /\ ack = gid
                           THEN believed \cup {proposed} ELSE believed
       ELSE /\ believed' = IF outcome = "Satisfied"
                           THEN believed \cup {proposed} ELSE believed
    /\ outcome' = "none"
    /\ ack' = NoId
    /\ UNCHANGED <<node, proposed, gid, my_cell, observed, claimed>>

(***************************************************************************)
(* CONSUMER                                                                 *)
(***************************************************************************)

\* Take the offer AND CAPTURE WHAT IT IS. Identity carried as data, not re-read later from a
\* register the other side is free to overwrite.
Claim ==
    /\ node = "Offered"
    /\ node' = "Executing"
    /\ my_cell' = proposed
    /\ ack' = gid
    /\ claimed' = TRUE
    /\ UNCHANGED <<proposed, gid, outcome, observed, believed>>

\* Report. The consumer writes the outcome for the cell IT worked on — which may no longer be the
\* cell the node names, if the lease reclaimed it and the producer armed another meanwhile.
\* `ack` is written only when the design carries an epoch; otherwise the report is anonymous.
FinishSatisfied ==
    /\ my_cell \in Reachable
    /\ my_cell # NoCell
    /\ outcome = "none"
    /\ outcome' = "Satisfied"
    /\ observed' = observed \cup {my_cell}
    /\ node' = "Completed"
    /\ ack' = IF EpochCarried THEN ack ELSE NoId
    /\ my_cell' = NoCell
    /\ UNCHANGED <<proposed, gid, believed, claimed>>

FinishRefused ==
    /\ my_cell \notin Reachable
    /\ my_cell # NoCell
    /\ outcome = "none"
    /\ outcome' = "Refused"
    /\ node' = "Completed"
    /\ ack' = IF EpochCarried THEN ack ELSE NoId
    /\ my_cell' = NoCell
    /\ UNCHANGED <<proposed, gid, observed, believed, claimed>>

Next == Publish \/ Observe \/ LeaseReclaim
     \/ Claim \/ FinishSatisfied \/ FinishRefused

\* Fairness on the PRODUCER only — the consumer may stutter for ever, which is the honest assumption
\* about another process. (The same choice AffordanceSelection.tla makes, and for the same reason.)
Spec == Init /\ [][Next]_vars /\ WF_vars(Publish) /\ WF_vars(Observe)

(***************************************************************************)
(* PROPERTIES                                                               *)
(***************************************************************************)

\* ★THE ONE THAT MATTERS. Everything the producer believes it observed, the robot actually observed.
\* A violation is a cell marked explored that nothing ever looked at — the same damage as the live
\* phantom arrivals, arrived at by a different route: not a lying consumer, but an honest report
\* applied to the wrong offer.
BeliefIsHonest == believed \subseteq observed

\* Sanity on the model: only reachable cells can be observed at all.
ObservedAreReachable == observed \subseteq Reachable
=============================================================================
