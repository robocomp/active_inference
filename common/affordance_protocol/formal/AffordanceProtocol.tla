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
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    LevelTriggered,     \* TRUE  : producer arms `waiting` when IT publishes (level)
                        \* FALSE : producer must CATCH the node in Executing  (edge)
    ConsumerRefuses,    \* TRUE  : declining an offer is REPORTED (node -> Completed)
                        \* FALSE : declining is silent (node stays Offered, producer told nothing)
    RearmUnchanged      \* TRUE  : re-publishing the same cell may re-arm a Completed node
                        \* FALSE : an unchanged proposal is declined as "not news"

VARIABLES node, waiting, learned
vars == <<node, waiting, learned>>

NodeStates == {"Offered", "Executing", "Completed"}

TypeOK == /\ node \in NodeStates
          /\ waiting \in BOOLEAN
          /\ learned \in BOOLEAN

Init == /\ node    = "Completed"   \* nothing outstanding
        /\ waiting = FALSE
        /\ learned = TRUE          \* free to propose

(***************************************************************************)
(* PRODUCER                                                                 *)
(***************************************************************************)

\* Put a proposal on the table. A producer that has not learned the last outcome can
\* only repeat itself; RearmUnchanged decides whether repeating re-arms the node.
Publish ==
    /\ node = "Completed"
    /\ (learned \/ RearmUnchanged)
    /\ node'    = "Offered"
    /\ learned' = FALSE
    /\ waiting' = IF LevelTriggered THEN TRUE ELSE waiting

\* EDGE detection only: the producer samples the node and happens to catch the claim.
\* This is the step a fast consumer never leaves open.
PollSawExecuting ==
    /\ ~LevelTriggered
    /\ node = "Executing"
    /\ ~waiting
    /\ waiting' = TRUE
    /\ UNCHANGED <<node, learned>>

\* Read the outcome and de-prioritise that cell, which is what allows a DIFFERENT proposal.
Observe ==
    /\ node = "Completed"
    /\ waiting
    /\ ~learned
    /\ waiting' = FALSE
    /\ learned' = TRUE
    /\ UNCHANGED node

(***************************************************************************)
(* CONSUMER                                                                 *)
(***************************************************************************)

Claim  == /\ node = "Offered"   /\ node' = "Executing" /\ UNCHANGED <<waiting, learned>>
Finish == /\ node = "Executing" /\ node' = "Completed" /\ UNCHANGED <<waiting, learned>>

\* Decline an offer it will not drive to (a standpoint already known unreachable).
\* Reported: the node retires and the producer can see it. Silent: nothing happens at
\* all -- and the consumer will decline the same offer again, for ever.
Refuse == /\ ConsumerRefuses
          /\ node = "Offered"
          /\ node' = "Completed"
          /\ UNCHANGED <<waiting, learned>>

Next == Publish \/ PollSawExecuting \/ Observe \/ Claim \/ Finish \/ Refuse

Fairness == /\ WF_vars(Publish) /\ WF_vars(Observe) /\ WF_vars(PollSawExecuting)
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

=============================================================================
