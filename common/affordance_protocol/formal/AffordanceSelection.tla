--------------------------- MODULE AffordanceSelection ---------------------------
(***************************************************************************)
(* The SELECTION layer: which cell gets offered, and whether the robot ever  *)
(* OBSERVES anything. Written 2026-08-19 after a day in which the wire model *)
(* next door (AffordanceProtocol.tla) said SOUND while the robot stood still *)
(* for hours.                                                                *)
(*                                                                          *)
(* WHY THE WIRE MODEL COULD NOT SEE IT. It has ONE node and no notion of     *)
(* WHICH cell is proposed, so its `learned` flag is commented "and therefore *)
(* may propose a DIFFERENT cell" -- an assumption written into the           *)
(* abstraction, never a checked property. Every failure measured today lives *)
(* in that gap:                                                              *)
(*                                                                          *)
(*   · producer re-offers the SAME cell for ever          (af7ba55)         *)
(*   · consumer refuses instantly, pair spins ~104/min    (2026-08-19)      *)
(*   · producer declines to re-arm an unchanged cell,                        *)
(*     consumer rejects anything not Offered -> DEADLOCK   (15:00 live)      *)
(*   · producer waits 100 s on an offer nobody claimed     (measured)        *)
(*                                                                          *)
(* Its liveness property, []<>(node = "Executing"), is SATISFIED by every one*)
(* of those. Messages keep flowing and nothing is ever observed. So the      *)
(* property here is about OBSERVATION, not about traffic.                    *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Cells,                  \* candidate standpoints
    Reachable,              \* the ones the consumer can actually stand at and observe from
    Favoured,               \* ★the cell the producer's argmax returns. NOT a free choice: the score is
                            \* computed, so the same state yields the same cell every cycle. Modelling
                            \* selection as \E c \in Candidates was WRONG and is why the first two
                            \* versions of this file declared everything SOUND -- TLC simply explored
                            \* the good branch and fairness took it, which the real producer never does.

    RearmUnchanged,         \* TRUE  : producer may re-arm a Completed node with the SAME cell
                            \* FALSE : "an unchanged proposal is not news" -- the live rule until
                            \*         2026-08-19, and one half of the observed deadlock

    ConsumerHoldsRefused,   \* TRUE  : a cell the consumer refused is un-takeable by it for a while
                            \* FALSE : it may re-take instantly -> the ~104/min busy loop

    OfferTimeout,           \* TRUE  : producer retires an offer nobody claimed and picks elsewhere
                            \* FALSE : it waits for ever (measured: 100.6 s on one cell)

    RefusalMovesArgmax      \* ★THE ONE THE FIRST DRAFT MISSED, and the reason it declared all eight
                            \* combinations SOUND while the live pair was deadlocked. The producer's
                            \* choice is an ARGMAX, not a free pick: de-prioritising a cell only
                            \* changes what is offered if it changes which cell WINS.
                            \* TRUE  : a refusal moves the argmax off that cell
                            \* FALSE : it does not -- measured live, gain fell 0.171 -> 0.0004 and the
                            \*         cell still won, because `score` (~0.28) is not dominated by gain.
                            \*         Modelling this as a free choice hides every failure of the day.

VARIABLES
    node,       \* "Offered" | "Executing" | "Completed"
    proposed,   \* the cell currently named by the node
    outcome,    \* "none" | "Satisfied" | "Refused"
    deprio,     \* producer: cells it has de-prioritised after a refusal
    held,       \* consumer: cells it refused and will not re-take yet
    observed,   \* cells actually OBSERVED FROM. The point of the whole exercise.
    claimed     \* has the consumer taken the current offer? (drives OfferTimeout)

vars == <<node, proposed, outcome, deprio, held, observed, claimed>>

\* ★A PLAIN VALUE, not an unbounded CHOOSE. `CHOOSE x : x \notin Cells` is unbounded and TLC
\* refuses it — it errored before generating a single state, and the sweep script reported
\* SOUND for all sixteen rows because it only grepped for failure words. See check_selection.sh.
NoCell == "none"

TypeOK == /\ node     \in {"Offered", "Executing", "Completed"}
          /\ proposed \in Cells \cup {NoCell}
          /\ outcome  \in {"none", "Satisfied", "Refused"}
          /\ deprio   \subseteq Cells
          /\ held     \subseteq Cells
          /\ observed \subseteq Cells
          /\ claimed  \in BOOLEAN

Init == /\ node = "Completed" /\ proposed = NoCell /\ outcome = "none"
        /\ deprio = {} /\ held = {} /\ observed = {} /\ claimed = FALSE

(***************************************************************************)
(* PRODUCER                                                                 *)
(***************************************************************************)

\* Cells it is willing to name. A refusal de-prioritises, but if everything is
\* de-prioritised it must still propose something -- standing still is not an option.
\* ★DETERMINISTIC. The argmax returns Favoured unless de-prioritisation actually moves it, which is
\* exactly the question RefusalMovesArgmax asks. Measured live: gain on the winning cell fell
\* 0.171 -> 0.0004 and it still won, because `score` (~0.28) is not dominated by gain.
Pick ==
    IF RefusalMovesArgmax /\ Favoured \in deprio /\ (Cells \ deprio) # {}
    THEN CHOOSE c \in (Cells \ deprio) : TRUE   \* bounded: fine for TLC
    ELSE Favoured

\* Arm the node. RearmUnchanged decides whether re-naming the SAME cell is allowed
\* once the node is Completed -- the rule that deadlocked the pair live.
Publish ==
    /\ node = "Completed"
    /\ outcome = "none"
    /\ (RearmUnchanged \/ Pick # proposed)
    /\ proposed' = Pick
    /\ node' = "Offered"
    /\ claimed' = FALSE
    /\ UNCHANGED <<outcome, deprio, held, observed>>

\* Read how it ended. Only a refusal teaches the producer anything about WHERE.
Observe ==
    /\ outcome # "none"
    /\ node = "Completed"
    /\ deprio' = IF outcome = "Refused" THEN deprio \cup {proposed} ELSE deprio
    /\ outcome' = "none"
    /\ UNCHANGED <<node, proposed, held, observed, claimed>>

\* ★An offer nobody claimed is one the consumer is declining. Retire it and pick elsewhere.
\* Without this the producer waits on a consumer that has already said no, for ever.
RetireUnclaimed ==
    /\ OfferTimeout
    /\ node = "Offered"
    /\ ~claimed
    /\ node' = "Completed"
    /\ outcome' = "none"
    /\ deprio' = deprio \cup {proposed}
    /\ UNCHANGED <<proposed, held, observed, claimed>>

(***************************************************************************)
(* CONSUMER                                                                 *)
(***************************************************************************)

\* Take the offer -- unless this is a cell it refused and is still holding.
Claim ==
    /\ node = "Offered"
    /\ ~(ConsumerHoldsRefused /\ proposed \in held)
    /\ node' = "Executing"
    /\ claimed' = TRUE
    /\ UNCHANGED <<proposed, outcome, deprio, held, observed>>

\* The hold expiring is what stops a refusal being retried at loop rate.
ReleaseHold ==
    /\ held # {}
    /\ \E c \in held : held' = held \ {c}
    /\ UNCHANGED <<node, proposed, outcome, deprio, observed, claimed>>

\* Drove there and observed. Only a Reachable cell can end this way.
FinishSatisfied ==
    /\ node = "Executing" /\ proposed \in Reachable
    /\ node' = "Completed" /\ outcome' = "Satisfied"
    /\ observed' = observed \cup {proposed}
    /\ UNCHANGED <<proposed, deprio, held, claimed>>

\* Could not stand there. Reported, and remembered so it is not re-taken at once.
FinishRefused ==
    /\ node = "Executing" /\ proposed \notin Reachable
    /\ node' = "Completed" /\ outcome' = "Refused"
    /\ held' = held \cup {proposed}
    /\ UNCHANGED <<proposed, deprio, observed, claimed>>

Next == Publish \/ Observe \/ RetireUnclaimed
     \/ Claim \/ ReleaseHold \/ FinishSatisfied \/ FinishRefused

\* ★★★THE CONSUMER MAY STUTTER FOR EVER (crash-stop). This is the check the model was missing, and
\* the reason both this spec and its predecessor passed while the real pair hung: fairness on the
\* CONSUMER's actions assumes the very thing that fails. A protocol whose liveness needs the consumer
\* to keep acting has no liveness at all — it has a hope. Measured live 2026-08-19: room held one
\* offer for 100.6 s, then 191.6 s, because its only escapes required the consumer to act first.
\* So: fairness on the PRODUCER's actions only. If a property still holds, it holds against a consumer
\* that dies, sleeps, or simply declines — which is the honest assumption for another process.
Fairness == /\ WF_vars(Publish) /\ WF_vars(Observe) /\ WF_vars(RetireUnclaimed)

\* The optimistic spec, kept for comparison: everything is fair, including the other process.
FairnessOptimistic ==
    Fairness /\ WF_vars(Claim) /\ WF_vars(ReleaseHold)
             /\ WF_vars(FinishSatisfied) /\ WF_vars(FinishRefused)

SpecOptimistic == Init /\ [][Next]_vars /\ FairnessOptimistic

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* PROPERTIES                                                               *)
(*                                                                          *)
(* WireProgress is the OLD model's property. Observes is the one that means  *)
(* the robot did its job. The result worth having is that WireProgress can   *)
(* HOLD while Observes FAILS -- "correct in its own terms, livelock in       *)
(* practice", as something a checker can find in seconds.                    *)
(***************************************************************************)

WireProgress == []<>(node = "Executing")

\* ★THE PRODUCER MUST KEEP OFFERING WHATEVER THE CONSUMER DOES. This is the property that separates a
\* protocol from a hope: no reachable state may exist in which the producer's next move requires the
\* consumer to move first. Failure #4 (an offer held 191 s) is exactly a violation.
\* ⚠FIRST ATTEMPT WAS []<>(node = "Offered") AND IT IS SATISFIED BY THE FAILURE: a node stuck at
\* Offered for ever trivially visits Offered infinitely often. Same defect as WireProgress below —
\* a property the failure satisfies proves nothing. The producer must keep CYCLING: to make a new
\* decision it must first see the old one closed, so both states must recur.
ProducerLive == []<>(node = "Offered") /\ []<>(node = "Completed")
Observes     == <>(observed # {})
NoStuckOffer == [](node = "Offered" => <>(node # "Offered"))
ObservedAreReachable == observed \subseteq Reachable   \* sanity on the model itself
=============================================================================
