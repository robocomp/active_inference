---------------------------- MODULE AffordanceProtocol ----------------------------
(***************************************************************************)
(* The DSR affordance protocol between ONE producer (a concept agent) and  *)
(* ONE consumer (the controller), over a single affordance node.           *)
(*                                                                          *)
(* Written 2026-08-18 after a day of fixing this protocol one symptom at a  *)
(* time. Every defect found was a LIVENESS defect -- a reachable state in   *)
(* which the producer waits for news that will never come while the         *)
(* consumer waits for an offer that will never be made. All of them are     *)
(* decidable in a state space of under thirty states. None were found that  *)
(* way; they were found by watching a robot stand still.                    *)
(*                                                                          *)
(* WIRE STATE                                                               *)
(*   epistemic_pending : set by the PRODUCER when it arms the node, cleared *)
(*                       by the CONSUMER when it finishes. The level both   *)
(*                       sides synchronise on.                              *)
(*   active            : the CONSUMER's execution claim.                    *)
(*                                                                          *)
(*   Offered   = pending, not claimed      Executing = pending, claimed     *)
(*   Completed = not pending; aff_outcome records how it ended              *)
(*                                                                          *)
(* PRODUCER-LOCAL STATE                                                     *)
(*   waiting : "a run of mine is outstanding"                               *)
(*   learned : "I have read the last outcome", and therefore may propose a  *)
(*             DIFFERENT cell. Without it the producer can only repeat      *)
(*             itself, which interacts with RearmUnchanged below.           *)
(*   armed   : "one of my offers is LIVE" -- the calib channel's offer_open_ *)
(*             and the general "one live offer at a time" rule.             *)
(*   staged  : "I have decided to offer but it is not on the wire yet".     *)
(*                                                                          *)
(* PUBLISHING IS NOT ATOMIC, AND THAT IS THE POINT OF THIS EXTENSION.        *)
(* The first version of this module had node' = "Offered" in a single step,  *)
(* so the interval between deciding to offer and the offer becoming visible  *)
(* did not exist. A real producer stages first: room_concept creates         *)
(* afford_calib QUIESCENT and deliberately does NOT publish on that cycle,   *)
(* so the contract can never be read late. A deadlock lived in exactly that  *)
(* gap for as long as the model could not express it -- found by looking at  *)
(* a graph, not by TLC. See LatchOnDecide.                                   *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    LevelTriggered,     \* TRUE  : producer arms `waiting` when IT publishes (level)
                        \* FALSE : producer must CATCH the node in Executing  (edge)
    ConsumerRefuses,    \* TRUE  : declining an offer is REPORTED (node -> Completed)
                        \* FALSE : declining is silent (node stays Offered, producer told nothing)
    RearmUnchanged,     \* TRUE  : re-publishing the same cell may re-arm a Completed node
                        \* FALSE : an unchanged proposal is declined as "not news"
    LatchOnDecide       \* TRUE  : the producer marks its offer live when it DECIDES to offer
                        \* FALSE : ... only once the offer is actually ON THE WIRE
                        \* This is the bug and its fix. TRUE was the shipped behaviour until
                        \* 2026-08-24: CalibChannel::offer() set offer_open_ before
                        \* publish_target ran, and ensure_calib_node() skips the publish on the
                        \* cycle it creates the node. The latch then blocked every later offer
                        \* and only an outcome could clear it -- an outcome no consumer could
                        \* ever produce, because nothing had been published. Observed live as
                        \* afford_calib present in the graph with no has_intention edge, for ever.

VARIABLES node, waiting, learned, armed, staged
vars == <<node, waiting, learned, armed, staged>>

NodeStates == {"Offered", "Executing", "Completed"}

TypeOK == /\ node \in NodeStates
          /\ waiting \in BOOLEAN
          /\ learned \in BOOLEAN
          /\ armed   \in BOOLEAN
          /\ staged  \in BOOLEAN

Init == /\ node    = "Completed"   \* nothing outstanding
        /\ waiting = FALSE
        /\ learned = TRUE          \* free to propose
        /\ armed   = FALSE
        /\ staged  = FALSE

(***************************************************************************)
(* PRODUCER                                                                 *)
(***************************************************************************)

\* Put a proposal on the table. A producer that has not learned the last outcome can
\* only repeat itself; RearmUnchanged decides whether repeating re-arms the node.
\* DECIDE. The producer picks a proposal. Nothing is visible to anyone yet.
\* `~armed` is the "one live offer at a time" rule, and it is what makes the latch's
\* placement matter: latch too early and this guard blocks the retry that would fix it.
Decide ==
    /\ node = "Completed"
    /\ ~armed
    /\ ~staged
    /\ (learned \/ RearmUnchanged)
    /\ staged'  = TRUE
    /\ armed'   = IF LatchOnDecide THEN TRUE ELSE armed
    /\ UNCHANGED <<node, waiting, learned>>

\* EMIT. The offer reaches the graph: the node is armed and the has_intention edge written.
Emit ==
    /\ staged
    /\ node' = "Offered"
    /\ staged'  = FALSE
    /\ armed'   = TRUE
    /\ learned' = FALSE
    /\ waiting' = IF LevelTriggered THEN TRUE ELSE waiting

\* EMIT DID NOT HAPPEN. Not a failure injection -- a real code path: ensure_calib_node()
\* returns false on the cycle it CREATES the node, deliberately, so that a consumer can
\* never latch a contract before it is written. publish_target can also decline. Either way
\* the staged proposal evaporates and nothing reached the wire.
\* ★NO FAIRNESS ON THIS ACTION: it is allowed to happen, never required to.
EmitSkipped ==
    /\ staged
    /\ staged' = FALSE
    /\ UNCHANGED <<node, waiting, learned, armed>>



\* EDGE detection only: the producer samples the node and happens to catch the claim.
\* This is the step a fast consumer never leaves open.
PollSawExecuting ==
    /\ ~LevelTriggered
    /\ node = "Executing"
    /\ ~waiting
    /\ waiting' = TRUE
    /\ UNCHANGED <<node, learned, armed, staged>>

\* Read the outcome and de-prioritise that cell, which is what allows a DIFFERENT proposal.
\* ★ONE STEP, NOT TWO. Reading the outcome and releasing the latch are the SAME branch in the
\* implementation -- consume_completion_event() clears waiting_completion_ and on_outcome()
\* clears offer_open_, with nothing between them. Modelling them as separate actions produced a
\* spurious deadlock (Observe fires, waiting goes false, and a ReleaseLatch guarded on waiting is
\* stranded for ever). That was a defect in the abstraction, not in the code -- and worth
\* recording, because the same mistake in the other direction is how a real interleaving hides.
\* ★The producer can only learn an outcome for a run it actually STARTED: waiting is set by
\* publish_target. A latch taken on an offer that never shipped is therefore never released here,
\* which is exactly the deadlock LatchOnDecide = TRUE walks into.
Observe ==
    /\ node = "Completed"
    /\ waiting
    /\ ~learned
    /\ waiting' = FALSE
    /\ learned' = TRUE
    /\ armed'   = FALSE
    /\ UNCHANGED <<node, staged>>

(***************************************************************************)
(* CONSUMER                                                                 *)
(***************************************************************************)

Claim  == /\ node = "Offered"   /\ node' = "Executing" /\ UNCHANGED <<waiting, learned, armed, staged>>
Finish == /\ node = "Executing" /\ node' = "Completed" /\ UNCHANGED <<waiting, learned, armed, staged>>

\* Decline an offer it will not drive to (a standpoint already known unreachable).
\* Reported: the node retires and the producer can see it. Silent: nothing happens at
\* all -- and the consumer will decline the same offer again, for ever.
Refuse == /\ ConsumerRefuses
          /\ node = "Offered"
          /\ node' = "Completed"
          /\ UNCHANGED <<waiting, learned, armed, staged>>

Next == Decide \/ Emit \/ EmitSkipped
        \/ PollSawExecuting \/ Observe \/ Claim \/ Finish \/ Refuse

\* EmitSkipped is deliberately NOT fair: a producer is permitted to drop a staged offer, it is
\* never obliged to. Everything else must eventually happen when it can.
\* ★EMIT IS STRONGLY FAIR, AND THE DIFFERENCE MATTERS. Weak fairness only forces an action that
\* stays CONTINUOUSLY enabled, and EmitSkipped disables Emit each time it fires -- so under WF an
\* adversary may stage-and-skip for ever and the robot never works again. That is not the system:
\* ensure_calib_node() skips exactly ONCE, on the cycle it creates the node, and every later cycle
\* finds the node present and publishes. Strong fairness says precisely that -- an action enabled
\* infinitely often must eventually occur -- so the skip is transient rather than adversarial.
\* ★EmitSkipped itself has NO fairness: it is permitted, never required.
Fairness == /\ WF_vars(Decide)  /\ SF_vars(Emit)
            /\ WF_vars(Observe) /\ WF_vars(PollSawExecuting)
            /\ WF_vars(Claim)   /\ WF_vars(Finish)

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* PROPERTIES                                                               *)
(*                                                                          *)
(* Progress is the whole point: the pair must keep putting fresh proposals   *)
(* on the table and keep executing them. TLC's deadlock check catches the    *)
(* states with no successor at all; Progress catches the subtler case where  *)
(* steps remain possible but the robot never works again.                    *)
(***************************************************************************)

Progress   == []<>(node = "Executing")
FreshOffer == []<>(node = "Offered")

\* A producer that is waiting must always eventually find out how the run ended --
\* otherwise it is blocked on news that cannot arrive.
NewsArrives == [](waiting /\ ~learned => <>learned)

\* THE INVARIANT THE OLD ABSTRACTION COULD NOT STATE. A latch marking "an offer of mine is
\* live" must never be held while nothing of that offer exists on the wire and no path can
\* release it. Expressed as liveness because the state is legal for an instant -- between
\* Decide and Emit -- and only wrong if it persists.
LatchIsBacked == [](armed /\ node = "Completed" /\ ~staged /\ ~waiting => <>(~armed \/ staged))

=============================================================================
