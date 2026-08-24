---------------------- MODULE AffordanceExecutionClaim ----------------------
(***************************************************************************)
(* CAN THE TWO AGENTS DISAGREE ABOUT WHAT IS BEING EXECUTED?                *)
(*                                                                          *)
(* AffordanceEpoch.tla asks whether a REPORT can be booked against the      *)
(* wrong offer. This asks the mirror question, on the other half of the     *)
(* exchange: while the approach is still running, can the producer and the  *)
(* consumer hold different beliefs about which pose the robot is driving to *)
(* — and can the producer act on its own belief and destroy a good approach?*)
(*                                                                          *)
(* WHY THIS MODEL EXISTS. Measured live 2026-08-23, both logs, one clock:   *)
(*   room:       afford_room EXECUTING (controller-claimed) — target        *)
(*               (2.25,-1.75) d=4.92m best=4.86m no_progress=8.9s/25s       *)
(*   controller: 'afford_room' republished at (-1.25,-0.75), 2.55 m from    *)
(*               the one this approach committed to (-0.75,1.75) — DEFERRED *)
(* Both internally consistent. Both wrong about the other. In an 11.4 min   *)
(* run the robot reached 0.253 m against its own 0.25 m arrival threshold   *)
(* and never once got inside it: 74 target moves, ALL with the robot within *)
(* 1 m, and zero arrivals.                                                  *)
(*                                                                          *)
(* THE DESIGN CHOICE UNDER TEST is ClaimCarried.                            *)
(*   FALSE = the protocol as it stood. The consumer's claim is a BOOLEAN    *)
(*           (`active`), so it can say THAT it is executing and never WHAT. *)
(*           The producer therefore measures approach progress against the  *)
(*           only pose it has: its own latest publication.                  *)
(*   TRUE  = the completed protocol. The consumer publishes the proposal it *)
(*           accepted and the pose it is actually driving to, on an         *)
(*           `executing` edge it owns; the producer measures against that.  *)
(* FALSE MUST VIOLATE NoFalseAbandon. A model in which the known hazard     *)
(* does not reproduce is not modelling it (check_epoch.sh's rule, kept).    *)
(*                                                                          *)
(* ── WHAT THIS ABSTRACTION KEEPS, AND WHAT IT THROWS AWAY ─────────────────*)
(* Stated explicitly because the LAST protocol proof on this fleet was      *)
(* sound about the logic and silent about the thing that actually bit: its  *)
(* abstraction had discarded timing. So, plainly:                           *)
(*  KEPT   - the producer may republish AT ANY TIME, including mid-approach.*)
(*           This is not an anomaly to be modelled away: a viewpoint scored *)
(*           by information gain is SATISFIED by the robot arriving, so     *)
(*           arriving is exactly what moves it.                             *)
(*  KEPT   - the consumer repairs the pose it was given (the real repairs   *)
(*           move it up to 0.3 m, and 5.5 m in the no-path cases), so the   *)
(*           executed pose is NOT the published one even when they agree    *)
(*           about which proposal is in force.                              *)
(*  KEPT   - the producer's watchdog: no improvement for Timeout ticks and  *)
(*           it abandons. This is the action that did the damage.           *)
(*  KEPT   - a consumer that genuinely cannot arrive, so that a fix which   *)
(*           merely disarms the watchdog is caught by NoDeadlock.           *)
(*  DROPPED- real time. Ticks are unitless; the model says nothing about    *)
(*           whether 25 s is the right budget, only about WHICH POSE the    *)
(*           budget is spent watching.                                      *)
(*  DROPPED- the DSR transport. Attribute writes are atomic here. That is   *)
(*           sound ONLY because the fix puts the consumer's half on an edge *)
(*           it alone writes; on a shared node it would be false, because   *)
(*           update_node is a whole-node replace on both cortex sync        *)
(*           engines and a stale writeback deletes the peer's attribute     *)
(*           graph-wide. That hazard is a REASON FOR THE DESIGN, not a      *)
(*           property this model checks — do not read agreement here as     *)
(*           license to move these fields onto the affordance node.         *)
(*  DROPPED- more than one consumer. "At most one claim per consumer" is    *)
(*           enforced in the write, not by this model.                      *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    Cells,          \* the standpoints the producer may name
    ClaimCarried,   \* does the consumer's claim say WHICH proposal and WHICH pose?
    MaxSteps,       \* approach length, in ticks, from a fresh claim
    Timeout         \* ticks without improvement before the producer abandons

VARIABLES
    node,       \* "Offered" | "Executing" | "Retired"
    proposed,   \* the cell the producer names NOW
    pepoch,     \* the producer's proposal counter; bumped on CONTENT change only
    claim,      \* the wire. ★ONE RECORD SHAPE ALWAYS — see NoClaim below.
    committed,  \* consumer: the cell it accepted. PHYSICAL TRUTH, visible to the model always.
    cepoch,     \* consumer: which proposal `committed` came from
    steps,      \* ticks of approach still needed to reach `committed`; 0 = arrived
    idle,       \* consecutive ticks on which the consumer did NOT close ground. Its diligence.
    best,       \* producer's watchdog: closest approach seen to the pose it WATCHES
    stall,      \* producer's watchdog: ticks since `best` improved
    arrived,    \* did the consumer actually complete an approach?
    abandoned   \* did the producer tear down a claim?

vars == <<node, proposed, pepoch, claim, committed, cepoch, steps, idle,
          best, stall, arrived, abandoned>>

NoCell  == "none"
\* ★THE THREE SHAPES OF THE WIRE, ALL ONE RECORD TYPE. Spelling this as a union of a string and a
\* record made TLC abort with "unable to fingerprint" after four states — while the first version of
\* check_claim.sh printed a full, plausible, entirely fictitious table from the wreckage.
\*   "none"   nobody is driving
\*   "opaque" a claim that says THAT but not WHAT — the old `active` boolean
\*   "full"   the completed protocol: which proposal, and which pose
NoClaim == [kind |-> "none",   e |-> 0, c |-> NoCell]
Opaque  == [kind |-> "opaque", e |-> 0, c |-> NoCell]
Full(ep, cell) == [kind |-> "full", e |-> ep, c |-> cell]

\* Bounded so TLC terminates. Two live epochs express "the consumer holds n while the producer offers
\* n+1", which is the entire disagreement; a larger bound finds nothing new.
MaxEpoch == 3

\* A distance the watchdog can never see improve. The robot approaches exactly one cell — the one it
\* is driving to — so its distance to any other cell is, for the watchdog's purposes, not decreasing.
Far == MaxSteps + 1

-----------------------------------------------------------------------------

TypeOK ==
    /\ node      \in {"Offered", "Executing", "Retired"}
    /\ proposed  \in Cells
    /\ pepoch    \in 0..MaxEpoch
    /\ committed \in Cells \cup {NoCell}
    /\ cepoch    \in 0..MaxEpoch
    /\ steps     \in 0..MaxSteps
    /\ idle      \in 0..Timeout
    /\ best      \in 0..Far
    /\ stall     \in 0..Timeout
    /\ arrived   \in BOOLEAN
    /\ abandoned \in BOOLEAN
    /\ claim \in [kind : {"none", "opaque", "full"}, e : 0..MaxEpoch, c : Cells \cup {NoCell}]

Init ==
    /\ node      = "Offered"
    /\ proposed  \in Cells
    /\ pepoch    = 1
    /\ claim     = NoClaim
    /\ committed = NoCell
    /\ cepoch    = 0
    /\ steps     = MaxSteps
    /\ idle      = 0
    /\ best      = Far
    /\ stall     = 0
    /\ arrived   = FALSE
    /\ abandoned = FALSE

-----------------------------------------------------------------------------
(* THE POSE THE PRODUCER'S WATCHDOG IS WATCHING.                             *)
(* The single operator in which the two designs differ, and deliberately the *)
(* only place ClaimCarried appears in the dynamics.                          *)
Watched ==
    IF ClaimCarried /\ claim.kind = "full"
    THEN claim.c        \* what the consumer says it is driving to
    ELSE proposed       \* our own latest publication — all the old protocol had

\* The robot closes on the cell it committed to and on nothing else.
DistTo(cell) == IF cell = committed /\ committed # NoCell THEN steps ELSE Far

-----------------------------------------------------------------------------
(* ONE TICK. The consumer either closes ground or does not, and the producer's watchdog observes —
   in the SAME step, because the whole question is whether those two are about the same pose. Having
   them as independent actions let the watchdog tick twice between drives, which is not a protocol
   defect but an artefact of the interleaving, and it made a legitimate abandon look false. *)
Tick ==
    /\ node = "Executing"
    /\ stall < Timeout
    /\ \E drove \in BOOLEAN :
        /\ IF drove /\ steps > 0
             THEN /\ steps' = steps - 1
                  /\ idle'  = 0
             ELSE /\ steps' = steps
                  /\ idle'  = IF idle < Timeout THEN idle + 1 ELSE idle
        \* The watchdog measures the pose it watches, using the POST-step distance — it observes the
        \* world after the robot has moved, which is the only order a real cycle can have.
        /\ LET d == IF Watched = committed /\ committed # NoCell THEN steps' ELSE Far IN
            IF d < best
              THEN /\ best'  = d
                   /\ stall' = 0
              ELSE /\ best'  = best
                   /\ stall' = stall + 1
    /\ UNCHANGED <<node, proposed, pepoch, claim, committed, cepoch, arrived, abandoned>>

(* PRODUCER *)

\* Republish somewhere else. Legitimate and expected: the gain at a viewpoint collapses as the robot
\* approaches it, so arriving is exactly what moves it. Bumps the epoch because CONTENT changed.
Republish ==
    /\ node \in {"Offered", "Executing"}
    /\ pepoch < MaxEpoch
    /\ \E c \in Cells : /\ c # proposed
                        /\ proposed' = c
    /\ pepoch' = pepoch + 1
    \* The watchdog re-arms on the pose it watches. Under the old design that is our own new cell —
    \* which is how our own republish came to reset a clock that was supposed to be about the consumer.
    /\ best'  = Far
    /\ stall' = 0
    /\ UNCHANGED <<node, claim, committed, cepoch, steps, idle, arrived, abandoned>>

\* The budget is spent: tear the claim down and re-select. THIS IS THE DAMAGING ACTION.
Abandon ==
    /\ node = "Executing"
    /\ stall = Timeout
    /\ abandoned' = TRUE
    /\ node'      = "Offered"
    /\ claim'     = NoClaim
    /\ committed' = NoCell
    /\ cepoch'    = 0
    /\ steps'     = MaxSteps
    /\ idle'      = 0
    /\ best'      = Far
    /\ stall'     = 0
    /\ UNCHANGED <<proposed, pepoch, arrived>>

(* CONSUMER *)

\* Accept the proposal on offer. The pose it will drive to is that proposal AFTER ITS OWN REPAIR, so
\* it may differ from what was published even at the instant of acceptance — modelled by letting it
\* commit to any cell, which is the strongest form of that.
Claim ==
    /\ node = "Offered"
    /\ \E c \in Cells :
        /\ committed' = c
        /\ claim' = IF ClaimCarried THEN Full(pepoch, c) ELSE Opaque
    /\ cepoch' = pepoch
    /\ node'   = "Executing"
    /\ steps'  = MaxSteps
    /\ idle'   = 0
    /\ best'   = Far
    /\ stall'  = 0
    /\ UNCHANGED <<proposed, pepoch, arrived, abandoned>>

\* Arrive, and conclude. The claim comes down with it (INV-5).
Arrive ==
    /\ node = "Executing"
    /\ steps = 0
    /\ arrived'   = TRUE
    /\ node'      = "Retired"
    /\ claim'     = NoClaim
    /\ committed' = NoCell
    /\ cepoch'    = 0
    /\ UNCHANGED <<proposed, pepoch, steps, idle, best, stall, abandoned>>

\* ★THE RUN DOES NOT END AT AN ARRIVAL, AND MODELLING THAT IT DOES IS NOT A CONVENIENCE — it is the
\* difference between a real deadlock check and a suppressed one. Without this, "Retired" has no
\* successor, TLC correctly reports Deadlock reached, and the tempting repair is CHECK_DEADLOCK FALSE
\* — which would ALSO hide any genuine deadlock the protocol has. The producer picks the next cell and
\* the pair goes round again, so that is what the model does.
Reoffer ==
    /\ node = "Retired"
    /\ node'  = "Offered"
    /\ steps' = MaxSteps
    /\ idle'  = 0
    /\ best'  = Far
    /\ stall' = 0
    /\ UNCHANGED <<proposed, pepoch, claim, committed, cepoch, arrived, abandoned>>

Next == Republish \/ Abandon \/ Claim \/ Tick \/ Arrive \/ Reoffer

\* ★FAIRNESS ON THE WATCHDOG, NOT ON THE CONSUMER. "A consumer that cannot get there" needs no extra
\* action and no flag: it is a behaviour in which Tick always chooses drove = FALSE, and those are
\* already in the model. Making Tick and Abandon fair is what guarantees such a behaviour still
\* TERMINATES — the property a "fix" that merely disarmed the watchdog would fail.
Spec == Init /\ [][Next]_vars
             /\ WF_vars(Tick) /\ WF_vars(Abandon) /\ WF_vars(Claim) /\ WF_vars(Arrive)
             /\ WF_vars(Reoffer)

-----------------------------------------------------------------------------
(* THE PROPERTIES                                                            *)

(* INV-4 made formal, and the heart of the matter. Whenever the consumer is executing, the pose the
   producer is judging must BE the pose the consumer is driving to. Under ClaimCarried = FALSE the
   producer judges `proposed`, which it is free to change at will, so this is violable by
   construction — which is exactly the finding: the old protocol contains no term that could make it
   true. This is the formal statement of "no party may judge another by a quantity the other cannot
   observe". *)
AgreementObservable ==
    (node = "Executing" /\ committed # NoCell) => (Watched = committed)

(* THE LIVE FAILURE, STATED AS AN OUTCOME. A consumer that has been closing ground must never be
   abandoned. `idle` is its diligence — consecutive ticks on which it did not close any — so the
   producer reaching its budget while the consumer has been driving means the budget was spent
   watching something else. A genuinely wedged consumer drives idle up in step with stall, and
   abandoning THAT is correct and must stay allowed: an invariant that forbade it would be demanding
   a watchdog that never fires. *)
NoFalseAbandon ==
    ~(node = "Executing" /\ stall = Timeout /\ idle < Timeout)

(* And the fix must not buy agreement with a deadlock. *)
Terminates == <>(arrived \/ abandoned)

=============================================================================
