--------------------------------- MODULE RoomTransition ---------------------------------
(***************************************************************************************)
(* The room-transition protocol between four DSR agents of one robot -- room_concept,  *)
(* door_concept, ltsm_agent, controller -- that share one replicated graph.            *)
(*                                                                                     *)
(* A model of the C++ AS IT IS on branch feat/room-recognition (worktree rt-build,      *)
(* 2026-10-30, HEAD be28a70c + d57997dc in, the unpromoted-crossing patch REVERTED),   *)
(* NOT of the design document. Every agent is a set of GUARDED ACTIONS over graph      *)
(* FACTS; each guard cites the line it was read from (room_scene_graph.cpp line        *)
(* numbers are HEAD's). Disagreements with                                             *)
(* docs/superpowers/specs/2026-10-29-room-transition-protocol-design.md: README.md.    *)
(*                                                                                     *)
(* Abstractions (the only places the model is not the code):                           *)
(*  - Geometry and probabilities are nondeterministic choices (environment actions).   *)
(*  - The graph is sequentially consistent: each agent cycle is one atomic step that   *)
(*    reads the latest graph. DDS reordering is NOT modelled (protocol_mock.py has the *)
(*    lag adversary).                                                                  *)
(*  - door_concept offers aff_<door>_cross on every door it fits; a `cross new` leg    *)
(*    may only pick the leg's hole (the tour harness shuts the others).                *)
(*  - A phantom door (a detection of no hole, or a door restored at a stale place from *)
(*    etc/door_identities.csv) is a door of the pseudo-hole "X", which nobody can pass.*)
(*  - Node ids are monotone counters (bounded by MaxRoomIds / MaxDoorIds).             *)
(*                                                                                     *)
(* SCENARIO "tour3": the 6-leg three-room tour. "pingpong": room_1 <-> room_2 through  *)
(* DOOR_1 only, 6 round trips (12 crossings).                                          *)
(*                                                                                     *)
(* Code switches -- what is in the code is TRUE/FALSE as noted:                        *)
(*   FIX_HOLD                 IN (d57997dc): no existence verdict on the door whose    *)
(*                            aff_<door>_cross the controller is `executing`.          *)
(*   FIX_UNPROMOTED_CROSSING  OUT (reverted 2026-10-30): crossings are evaluated in a  *)
(*                            room whose start-up survey has not promoted it.          *)
(* Candidate fixes (NOT in the code; README "Minimal changes"):                        *)
(*   FIX_FIRST_SIGHT          room_concept door_crossing.h:451: a door FIRST SEEN with *)
(*                            the body already past its line is not "crossed".         *)
(*   FIX_COMPLETE_ON_DOOR     controller: a cross completes only on a room switch      *)
(*                            whose crossing fact names THIS door's hole + the room    *)
(*                            the cross started in (the proto's mouth / an exit edge). *)
(*   FIX_NO_STAGED_RECALL_GAP ltsm: recalls under P's doors are kept while a proposal  *)
(*                            for P is pending (room_recall.cpp:146 skips them today). *)
(* Environment switches (proto15):                                                     *)
(*   FALSE_CROSS_ANY  room_concept may also cross through ANY door of a young room the *)
(*                    robot stands in the doorway of (beyond the first-sight mechanism).*)
(*   PHANTOMS         a phantom door may be born where the robot stands in a doorway,  *)
(*                    or restored from a stale identity row of the room's NAME.        *)
(*   STALE_FILE       door_identities.csv starts with a room_2 row from another run.   *)
(*   WRONG_RECOG      ltsm may propose a WRONG remembered room (the fit rejects it).    *)
(***************************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS SCENARIO,
          FIX_HOLD, FIX_UNPROMOTED_CROSSING,
          FIX_FIRST_SIGHT, FIX_COMPLETE_ON_DOOR, FIX_NO_STAGED_RECALL_GAP,
          FALSE_CROSS_ANY, PHANTOMS, STALE_FILE, WRONG_RECOG,
          SURVEY_CAN_PROMOTE,       \* FALSE = no start-up survey finishes while the robot is in the room
          ABSENCE_ANYTIME,          \* TRUE = any fitted door may be judged absent at any time
          MaxAbsent, MaxRejects, MaxFalse, MaxPhantom,
          MaxRoomIds, MaxDoorIds

\* ---------------------------------------------------------------- the world (ground truth)
PRooms    == {1, 2, 3}                    \* physical rooms R1, R2, R3 (threerooms-piso.wbt)
RealHoles == {"D0", "D1", "D2"}           \* DOOR_0 (R1-R3), DOOR_1 (R1-R2), DOOR_2 (R3-R2)
Holes     == RealHoles \cup {"X"}         \* "X": a door node at a place where there is no hole
NoHole    == "none"
Ends(h) == CASE h = "D1" -> {1, 2}
             [] h = "D0" -> {1, 3}
             [] h = "D2" -> {2, 3}
             [] OTHER    -> {}
HolesOf(r)  == {h \in RealHoles : r \in Ends(h)}
Other(h, r) == CHOOSE x \in Ends(h) : x # r

Tour3 == << [rel |-> "new",  hole |-> "D1", name |-> 0, dest |-> 2],   \* 1 -> 2 new
            [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1], \* 2 -> 1 (cross room_1)
            [rel |-> "new",  hole |-> "D0", name |-> 0, dest |-> 3],   \* 1 -> 3 new
            [rel |-> "new",  hole |-> "D2", name |-> 0, dest |-> 2],   \* 3 -> 2, remembered (case 2)
            [rel |-> "room", hole |-> NoHole, name |-> 3, dest |-> 3], \* 2 -> 3 (cross room_3)
            [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1] >>\* 3 -> 1 (cross room_1)
\* 6 round trips through DOOR_1: cross new, then cross room_1 / cross room_2 alternately.
PingPong == [i \in 1..12 |-> IF i = 1 THEN [rel |-> "new", hole |-> "D1", name |-> 0, dest |-> 2]
                             ELSE IF i % 2 = 0 THEN [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1]
                             ELSE [rel |-> "room", hole |-> NoHole, name |-> 2, dest |-> 2]]
Legs  == IF SCENARIO = "pingpong" THEN PingPong ELSE Tour3
NLegs == Len(Legs)
\* the holes perception may detect: ping-pong only uses DOOR_1 (the others are shut and never crossed)
Detectable(r) == IF SCENARIO = "pingpong" THEN HolesOf(r) \cap {"D1"} ELSE HolesOf(r)

NoProp == [p |-> 0, k |-> 0, old |-> 0, st |-> "none"]
Max(S) == CHOOSE x \in S : \A y \in S : x >= y

VARIABLES
  \* world / environment
  pos,          \* the robot's true physical room
  near,         \* the hole the robot is in or at (set while a cross is executing), or NoHole
  doorway,      \* the robot stands in the doorway it last passed (it may stay there after the crossing; proto15)
  lastHole,     \* the last hole the robot physically passed
  passEv,       \* room_concept was evaluating crossings when the robot passed lastHole (PassageMemory accumulated it)
  absences, falseX, phantoms,     \* environment budget used
  \* graph facts
  rooms,        \* live room nodes: id -> [name (room_<k>, k = room_id), truth, proto (self-edge), from, via]
  rcRT,         \* room_concept: the room the robot's LIVE RT edge hangs from
  current,      \* ltsm_agent: the robot->room `current` edge
  doors,        \* door_concept: id -> [hole, room (RT ancestor), mirror (entry mirror not yet adopted), fp (first seen past)]
  passages,     \* ltsm_agent: live passage_N nodes, each = the set of its two `match`ed door ids
  recalls,      \* ltsm_agent: recall_<room_k> nodes hung under a door: [door, name]
  proposal,     \* ltsm_agent's recognition proposal on P (room_concept owns its `state`)
  \* agent-local state
  rcStartup, rcPromoted,          \* room_concept: startup_proto_ / startup_promoted_ of its current room
  hwm,                            \* room_concept: room_name_hwm (etc/room_name_hwm.txt)
  rejects,                        \* correct proposals rejected by the fit so far
  memTruth, memFaces, memPass,    \* ltsm long-term memory: archived room names, faces <<name,hole>>, passages
  carry,                          \* ltsm RoomSwap: carry-overs waiting for the left room's face to be archived
  dcRoom, dcName, lost, idFile,   \* door_concept: followed room (id, name), holes of its doors deleted by others,
                                  \*   etc/door_identities.csv rows [name, hole] (keyed by room NAME, persists across runs)
  ctl, ctlRoom, ctlName, prevName,\* controller: mission leg + door passage; track_room id/name/prev_name
  nextRoom, nextDoor,             \* node id allocators
  \* history (for invariants only)
  issued,       \* name k -> the physical room it was first issued for
  arrived,      \* the robot's true room at each completed leg
  badWrites     \* writes by an agent that does not own the node (rule 1)

vars == << pos, near, doorway, lastHole, passEv, absences, falseX, phantoms, rooms, rcRT, current, doors, passages,
           recalls, proposal, rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, dcName,
           lost, idFile, ctl, ctlRoom, ctlName, prevName, nextRoom, nextDoor, issued, arrived, badWrites >>
envVars  == << pos, near, doorway, lastHole, passEv, absences, falseX, phantoms >>
rcVars   == << rcStartup, rcPromoted, hwm, rejects >>
ltVars   == << memTruth, memFaces, memPass, carry >>
dcVars   == << dcRoom, dcName, lost, idFile >>
ctVars   == << ctl, ctlRoom, ctlName, prevName >>
histVars == << issued, arrived, badWrites >>

\* ---------------------------------------------------------------- derived facts
LiveNames      == {rooms[r].name : r \in DOMAIN rooms}
DoorsOf(r)     == {d \in DOMAIN doors : doors[d].room = r}
RecallsOn(d)   == {x.name : x \in {y \in recalls : y.door = d}}
HasMatch(d)    == \E p \in passages : d \in p
\* door_concept offers the pragmatic affordances (aff_<d>_cross) on the doors it FITS: the doors of the room it
\* follows, an entry mirror only once adopted (specificworker.cpp:2556 init, :2740 update; :3495 removed at release).
Offered(d)     == d \in DOMAIN doors /\ doors[d].room = dcRoom /\ ~doors[d].mirror
\* The controller's `executing` edge controller -> aff_<d>_cross (input_snapshot.cpp:283 reads every such edge).
Executing(d)   == ctl.step = "cross" /\ ctl.door = d /\ Offered(d)
\* room_scene_graph.cpp HEAD:901: `if (PROTO_ROOM_ENABLED and not (startup_proto_ and not startup_promoted_))`;
\* the reverted patch made it `if (PROTO_ROOM_ENABLED)`.
Evaluating     == FIX_UNPROMOTED_CROSSING \/ ~(rcStartup /\ ~rcPromoted)
Young          == rcStartup /\ ~rcPromoted
NewDoor(h, r, m, fp) == [hole |-> h, room |-> r, mirror |-> m, fp |-> fp]

TypeOK ==
  /\ pos \in PRooms /\ near \in Holes \cup {NoHole} /\ lastHole \in Holes \cup {NoHole}
  /\ rcRT \in DOMAIN rooms /\ current \in DOMAIN rooms \cup {0}
  /\ \A d \in DOMAIN doors : doors[d].hole \in Holes
  /\ ctl.leg \in 1..NLegs + 1 /\ ctl.step \in {"idle", "cross"}

Init ==
  \* room_1 is the start room with a GIVEN layout (recog_proto11/15: inserted with its walls, no proto edge), and ltsm
  \* placed `current` on it.
  /\ pos = 1 /\ near = NoHole /\ doorway = FALSE /\ lastHole = NoHole /\ passEv = FALSE
  /\ absences = 0 /\ falseX = 0 /\ phantoms = 0
  /\ rooms = (1 :> [name |-> 1, truth |-> 1, proto |-> FALSE, from |-> 0, via |-> NoHole])
  /\ rcRT = 1 /\ current = 1
  /\ doors = [d \in {} |-> NewDoor("D0", 0, FALSE, FALSE)]
  /\ passages = {} /\ recalls = {} /\ proposal = NoProp
  /\ rcStartup = FALSE /\ rcPromoted = TRUE /\ hwm = 1 /\ rejects = 0
  /\ memTruth = (1 :> 1)                 \* seed_memory(): room 1 goes to memory as soon as it is known
  /\ memFaces = {} /\ memPass = {} /\ carry = {}
  /\ dcRoom = 1 /\ dcName = 1 /\ lost = {}
  \* proto15: "room_2:door_46@(2.92,1.63)" -- a row from an earlier run's room_2 frame; the tour does not clear the file
  /\ idFile = IF STALE_FILE THEN {[name |-> 2, hole |-> "X"]} ELSE {}
  /\ ctl = [leg |-> 1, step |-> "idle", door |-> 0, hole |-> NoHole, croom |-> 0, cname |-> 0, passed |-> FALSE]
  /\ ctlRoom = 1 /\ ctlName = 1 /\ prevName = 0
  /\ nextRoom = 2 /\ nextDoor = 1
  /\ issued = (1 :> 1) /\ arrived = << >> /\ badWrites = {}

\* =================================================================================== room_concept
\* step_proto_room_impl "Before birth" (HEAD room_scene_graph.cpp:914-1147): the birth through door d, into the physical
\* room t0 (the robot's room; for a FALSE crossing the robot is still in the room it was in).
Birth(d, t0) ==
  LET recs  == {k \in RecallsOn(d) : k \notin LiveNames /\ k > 0}   \* :1060 find_recall; :1065 name free, index > 0
      useRc == recs # {}
      k     == IF useRc THEN Max(recs) ELSE Max(LiveNames \cup {hwm}) + 1   \* :1042-1056 one past every room_<k>
      t     == IF useRc THEN issued[k] ELSE t0                      \* a recalled room IS remembered room k
  IN /\ nextRoom <= MaxRoomIds
     /\ rooms' = rooms @@ (nextRoom :> [name |-> k, truth |-> t, proto |-> ~useRc,
                                         from |-> rooms[rcRT].name, via |-> doors[d].hole])
        \* :1104 proto self-edge; :1091 write_mouth (the crossed aperture, in the left room's frame); a recall is
        \* promoted in the same call (rearm_after_crossing :1726 -> :2043)
     /\ rcRT' = nextRoom                      \* :1127 mark_room_rt_not_current, then the RT to the new room
     /\ nextRoom' = nextRoom + 1
     /\ rcStartup' = TRUE                     \* arm_startup_proto :1780
     /\ rcPromoted' = useRc                   \* :1781 false; :2046 true for a recall
     /\ hwm' = Max({hwm, k})                  \* note_room_name_issued
     /\ issued' = IF k \in DOMAIN issued THEN issued ELSE issued @@ (k :> t)
     /\ passEv' = FALSE

RC_Cross ==
  /\ pos # rooms[rcRT].truth                    \* the robot has left the room the RT hangs from ...
  /\ passEv /\ Evaluating                       \* ... and PassageMemory saw the passage (HEAD:901 gates the impl;
                                                \*     the memory restarts per room :970 and at birth :1123)
  /\ \E d \in DoorsOf(rcRT) : doors[d].hole = lastHole   \* :992 only THIS room's doors (RT ancestor) are apertures
  /\ Birth(CHOOSE x \in DoorsOf(rcRT) : doors[x].hole = lastHole, pos)
  /\ UNCHANGED << pos, near, doorway, lastHole, absences, falseX, phantoms, current, doors, passages, recalls, proposal,
                  rejects, ltVars, dcVars, ctVars, nextDoor, arrived, badWrites >>

\* A FALSE crossing (proto15 t=1793366230834, door_46, p_cross 0.954 with p_span 0.08, the robot 8 cm into DOOR_1's
\* doorway and never through door_46). Mechanism, door_crossing.h:451-453: on a door's FIRST SIGHT PassageMemory seeds its
\* "through" mass with the instantaneous reading; a door first seen with the body already past its line has q_N ~ 0, and
\* the three-class posterior (:527-544, W prior eps) then gives A ~ 1. Environment (not fair), at most MaxFalse.
\* FALSE_CROSS_ANY widens it to any door of a young room while the robot stands in a doorway (estimator noise).
RC_FalseCross(d) ==
  /\ d \in DoorsOf(rcRT) /\ Evaluating /\ falseX < MaxFalse
  /\ \/ doors[d].fp /\ ~FIX_FIRST_SIGHT
     \/ FALSE_CROSS_ANY /\ Young /\ doorway
  /\ Birth(d, pos)
  /\ falseX' = falseX + 1
  /\ UNCHANGED << pos, near, doorway, lastHole, absences, phantoms, current, doors, passages, recalls, proposal,
                  rejects, ltVars, dcVars, ctVars, nextDoor, arrived, badWrites >>

\* step_startup_proto :1826 `if (layout_ready and layout_finished_) promote_startup_proto` -> :2043 proto edge removed.
\* Not fair. Gated on the robot being inside, not crossing, and no proposal under judgement (:314 branch).
RC_Promote ==
  /\ SURVEY_CAN_PROMOTE
  /\ Young /\ pos = rooms[rcRT].truth /\ near = NoHole
  /\ ~(proposal.st = "pending" /\ proposal.p = rcRT)
  /\ rooms' = [rooms EXCEPT ![rcRT].proto = FALSE]
  /\ rcPromoted' = TRUE
  /\ UNCHANGED << envVars, rcRT, current, doors, passages, recalls, proposal, rcStartup, hwm, rejects, ltVars, dcVars,
                  ctVars, nextRoom, nextDoor, histVars >>

\* step_proposal (room_scene_graph_late_recall.cpp:185-249) + admit (proposal_adoption.cpp:287-296). The fit is SOUND:
\* a wrong K is always rejected (proto15: 15+ rejections, median |SDF| 1.6 m); a right K may be rejected <= MaxRejects.
RC_Judge ==
  /\ proposal.st = "pending"
  /\ LET P     == proposal.p
         adm   == /\ P = rcRT /\ Young                                  \* provisional_room() :66
                  /\ P \in DOMAIN rooms /\ rooms[P].proto              \* admit :290 proto edge
                  /\ proposal.k \notin LiveNames \ {rooms[P].name}     \* admit :294 NameTaken
         right == adm /\ issued[proposal.k] = rooms[P].truth
     IN \/ /\ ~right                                                   \* :240 / :196 / :211 rejected
           /\ proposal' = [proposal EXCEPT !.st = "rejected"]
           /\ UNCHANGED << rooms, rcPromoted, rejects >>
        \/ /\ right                                                    \* accept_proposal :112-182: rename IN PLACE
           /\ rooms' = [rooms EXCEPT ![P].name = proposal.k, ![P].proto = FALSE]   \* adopt_in_place :337
           /\ rcPromoted' = TRUE
           /\ proposal' = [proposal EXCEPT !.st = "accepted"]
           /\ UNCHANGED rejects
        \/ /\ right /\ rejects < MaxRejects                            \* :211 the fit rejects a right K
           /\ proposal' = [proposal EXCEPT !.st = "rejected"]
           /\ rejects' = rejects + 1
           /\ UNCHANGED << rooms, rcPromoted >>
  /\ UNCHANGED << envVars, rcRT, current, doors, passages, recalls, rcStartup, hwm, ltVars, dcVars, ctVars,
                  nextRoom, nextDoor, histVars >>

\* =================================================================================== ltsm_agent
\* ONE compute() cycle (ltsm_agent specificworker.cpp:343-397), its sub-steps in the code's order:
\*   1. live passages   2. recognition   3. promotion/eviction   4. recalls.
\* K = the room the recogniser proposes this cycle (0: none -- its evidence is not conclusive yet).

\* -- 1. passage_live.cpp step()
PassAlive == {p \in passages : p \subseteq DOMAIN doors}            \* :103 a passage lives as long as both doors
IsProtoMirror(d) == rooms[doors[d].room].proto /\ doors[d].room # current   \* room_resolve.h:182
Entries == {e \in DOMAIN doors :                                     \* :119 doors hung from a proto room ...
              /\ doors[e].room \in DOMAIN rooms /\ rooms[doors[e].room].proto
              /\ current # 0                                         \* :132 never guess without a current room
              /\ doors[e].room # current                             \* :143 a proto that is current pairs nothing
              /\ ~\E p \in PassAlive : e \in p}                      \* :176 a pairing is kept once made
Cands(e) == {d \in DoorsOf(current) : d # e /\ ~IsProtoMirror(d)}    \* :154 the current room's own doors
\* "COUNTERPART = argmin distance": the face at the same place when the current room has one, else the nearest other
Nearest(e) == IF \E d \in Cands(e) : doors[d].hole = doors[e].hole
              THEN CHOOSE d \in Cands(e) : doors[d].hole = doors[e].hole
              ELSE CHOOSE d \in Cands(e) : TRUE
NewPassages == {{e, Nearest(e)} : e \in {x \in Entries : Cands(x) # {}}}

\* -- 4. room_recall.cpp step(): a recall under every live door whose hole memory knows (by place, :137), naming the
\*       passage's OTHER room (:154), unless that room is live (:162). A door of the room whose twin is STAGED by a
\*       pending proposal (RoomSwap case B, room_swap.cpp:143) is skipped (:146), and remembered_hole fails (eviction:1090).
BehindDoor(rm, dr, mt, mp, stg, d) ==
  {o \in DOMAIN mt :
     LET x == rm[dr[d].room].name IN
     /\ dr[d].room \in DOMAIN rm
     /\ x \in DOMAIN mt /\ x # stg
     /\ o # x /\ o \notin {rm[r].name : r \in DOMAIN rm}
     /\ \E m \in mp : \/ (m.a = << x, dr[d].hole >> /\ m.b[1] = o)
                      \/ (m.b = << x, dr[d].hole >> /\ m.a[1] = o)}
Wanted(rm, dr, mt, mp, stg) ==
  UNION {{[door |-> d, name |-> o] : o \in BehindDoor(rm, dr, mt, mp, stg, d)} : d \in DOMAIN dr}

Proposable(K) == /\ rooms[rcRT].proto                                  \* room_recognizer.cpp:352 a proto under the robot
                 /\ K \in DOMAIN memTruth /\ K \notin LiveNames /\ K # rooms[rcRT].name
                 /\ WRONG_RECOG \/ memTruth[K] = rooms[rcRT].truth

LT_Cycle(K) ==
  LET pass1 == PassAlive \cup NewPassages
      \* -- 2. recognition_cycle.h: RoomSwap reads the verdict, or the recogniser proposes
      P0 == proposal.p
      verdictCarry == IF proposal.st = "accepted"                       \* RoomSwap::commit :343: carry-over, P's twin dropped :383
                      THEN {[h |-> rooms[P0].via, from |-> rooms[P0].from, to |-> proposal.k]}
                      ELSE {}
      dropName == IF proposal.st = "accepted" /\ proposal.old # proposal.k THEN proposal.old ELSE 0
      prop1 == CASE proposal.st \in {"accepted", "rejected"} -> NoProp \* adopted / rebased at the rejection
                 [] proposal.st = "pending" /\ (P0 \notin DOMAIN rooms \/ rcRT # P0) -> NoProp \* room_swap.cpp:399/:106 withdrawn
                 [] proposal.st = "none" /\ K # 0 /\ Proposable(K) ->
                        [p |-> rcRT, k |-> K, old |-> rooms[rcRT].name, st |-> "pending"]
                 [] OTHER -> proposal
      mt1 == [n \in DOMAIN memTruth \ {dropName} |-> memTruth[n]]
      mf1 == {f \in memFaces : f[1] # dropName}
      mp1 == {m \in memPass : m.a[1] # dropName /\ m.b[1] # dropName}
      \* -- 3. step_promotion :540-627: due when room_concept's RT room differs from `current` (:546, promotion_due)
      due  == rcRT # current /\ current \in DOMAIN rooms
      old  == current
      new  == rcRT
      pr   == {p \in pass1 : \E f \in p : doors[f].room = new}          \* pair_for(new) :580 (a face whose parent is new)
      pair == IF pr # {} THEN CHOOSE p \in pr : TRUE ELSE {}
      fNew == IF pair # {} THEN CHOOSE f \in pair : doors[f].room = new ELSE 0
      fOld == IF pair # {} /\ Cardinality(pair) = 2 THEN CHOOSE f \in pair : f # fNew ELSE 0
      \* evict() step 4 (room_eviction.cpp:700-930): the crossed door = the live pair's old face when known (:711),
      \* else the old room's door NEAREST THE ROBOT (:714-729) -- the one at the aperture room_concept crossed (its
      \* mouth, rooms[new].via); the far face is the pair's new face (:845), else DERIVED through that door (:890-927).
      oldDs  == DoorsOf(old)
      viaH   == rooms[new].via
      chosen == CASE fOld # 0 /\ fOld \in oldDs                       -> fOld
                  [] \E x \in oldDs : doors[x].hole = viaH           -> CHOOSE x \in oldDs : doors[x].hole = viaH
                  [] oldDs # {}                                      -> CHOOSE x \in oldDs : TRUE
                  [] OTHER                                           -> 0
      newH   == IF fNew # 0 THEN doors[fNew].hole ELSE IF chosen # 0 THEN doors[chosen].hole ELSE NoHole
      gone == IF due THEN DoorsOf(old) ELSE {}                          \* remove_live_room :990 -> sweep_orphans :1543
      mt2  == IF due THEN [n \in DOMAIN mt1 \cup {rooms[old].name, rooms[new].name} |->
                             IF n = rooms[old].name THEN rooms[old].truth
                             ELSE IF n = rooms[new].name THEN rooms[new].truth ELSE mt1[n]]
              ELSE mt1
      mf2  == IF due THEN mf1 \cup {<< rooms[old].name, doors[g].hole >> : g \in gone}
                                  \cup (IF newH # NoHole THEN {<< rooms[new].name, newH >>} ELSE {})
              ELSE mf1
      mp2  == IF due /\ chosen # 0
              THEN mp1 \cup {[a |-> << rooms[old].name, doors[chosen].hole >>, b |-> << rooms[new].name, newH >>]}
              ELSE mp1
      rm2  == IF due THEN [r \in DOMAIN rooms \ {old} |-> rooms[r]] ELSE rooms
      dr2  == [d \in DOMAIN doors \ gone |-> doors[d]]
      \* carry-over is written once the left room's face is in memory (RoomSwap::retry_carries :336)
      cr1  == carry \cup verdictCarry
      done == {c \in cr1 : << c.from, c.h >> \in mf2}
      mp3  == mp2 \cup {[a |-> << c.from, c.h >>, b |-> << c.to, c.h >>] : c \in done}
      mf3  == mf2 \cup {<< c.to, c.h >> : c \in done}
      stg  == IF prop1.st = "pending" /\ ~FIX_NO_STAGED_RECALL_GAP /\ prop1.p \in DOMAIN rm2
                 /\ rm2[prop1.p].name \in DOMAIN mt2
              THEN rm2[prop1.p].name ELSE 0
  IN /\ passages' = pass1
     /\ proposal' = prop1
     /\ current' = IF due THEN new ELSE current                          \* room_eviction.cpp:975-978 the `current` flip
     /\ rooms' = rm2
     /\ doors' = dr2
     /\ lost' = IF due /\ old = dcRoom THEN lost \cup {doors[g].hole : g \in gone} ELSE lost
     /\ memTruth' = mt2 /\ memFaces' = mf3 /\ memPass' = mp3
     /\ carry' = cr1 \ done
     /\ recalls' = Wanted(rm2, dr2, mt2, mp3, stg)                       \* -- 4. after the promotion (:397)
     /\ badWrites' = badWrites \cup
          (IF due THEN {"ltsm deletes room_concept's room node (room_eviction.cpp:1288)"} ELSE {}) \cup
          (IF gone # {} THEN {"ltsm deletes door_concept's door nodes (room_eviction.cpp:1543)"} ELSE {})
     /\ UNCHANGED << envVars, rcRT, rcVars, dcRoom, dcName, idFile, ctVars, nextRoom, nextDoor, issued, arrived >>

\* =================================================================================== door_concept
\* step_room_following :3158-3216 -> release_room :3388: follow ltsm's `current`. The left room's live identities are
\* written to etc/door_identities.csv under the room's NAME (remember_live_identities, specificworker.cpp:3466).
DC_Follow ==
  /\ current # 0 /\ current # dcRoom
  /\ LET new == current
         mir == {m \in DoorsOf(new) : doors[m].mirror}
         ch  == rooms[new].via                                           \* the crossed door = the lost one nearest the robot
     IN IF mir # {}
        THEN /\ doors' = [doors EXCEPT ![CHOOSE m \in mir : TRUE].mirror = FALSE]   \* :3432 adopt the entry mirror
             /\ nextDoor' = nextDoor
        ELSE IF ch \in lost /\ nextDoor <= MaxDoorIds                            \* :3442 recreate the door we came through
                /\ ~\E d \in DoorsOf(new) : doors[d].hole = ch
             THEN /\ doors' = doors @@ (nextDoor :> NewDoor(ch, new, FALSE, FALSE))
                  /\ nextDoor' = nextDoor + 1
             ELSE UNCHANGED << doors, nextDoor >>
  /\ idFile' = idFile \cup {[name |-> dcName, hole |-> h] : h \in lost}
  /\ dcRoom' = current /\ dcName' = rooms[current].name /\ lost' = {}
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars, ltVars, ctVars, nextRoom, histVars >>

\* step_entry_mirrors :3221-3330: one mirror per proto room that is not the followed one (:3282) and was not announced
\* by a recall (:3273 entry_mirror_wanted), from the instance whose aperture holds the proto origin (:3300) -- the door
\* room_concept crossed (rooms[P].via) in the room door_concept follows.
DC_Mirror(P) ==
  /\ P \in DOMAIN rooms /\ rooms[P].proto /\ P # dcRoom
  /\ ~\E m \in DoorsOf(P) : doors[m].mirror
  /\ ~\E x \in recalls : x.name = rooms[P].name
  /\ \E s \in DoorsOf(dcRoom) : ~doors[s].mirror /\ doors[s].hole = rooms[P].via
  /\ nextDoor <= MaxDoorIds
  /\ doors' = doors @@ (nextDoor :> NewDoor(rooms[P].via, P, TRUE, FALSE))
  /\ nextDoor' = nextDoor + 1
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars, ltVars, dcVars, ctVars, nextRoom,
                  histVars >>

\* Perception: a real door of the room the robot is in is detected and born under the followed room (fair). Its first
\* sight is from inside the room (never past its line).
DC_Birth(h) ==
  /\ dcRoom \in DOMAIN rooms /\ pos = rooms[dcRoom].truth /\ h \in Detectable(pos)
  /\ ~\E d \in DoorsOf(dcRoom) : doors[d].hole = h
  /\ nextDoor <= MaxDoorIds
  /\ doors' = doors @@ (nextDoor :> NewDoor(h, dcRoom, FALSE, FALSE))
  /\ nextDoor' = nextDoor + 1
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars, ltVars, dcVars, ctVars, nextRoom,
                  histVars >>

\* A PHANTOM door (environment, <= MaxPhantom): proto15 door_46, a tracker birth at the robot standing in DOOR_1's doorway
\* (later judged L=-4), named from a stale door_identities.csv row of room_2 (an earlier run's frame; the file is keyed by
\* room NAME, door_identity_table.h:57 IdentityRow; the name is restored at door_core.cpp:879, and the file persists). Also a row that is not id-only is RESTORED as a door at its stale place.
\* Born in the doorway, its first sight is with the body PAST its line (fp); restored, it may or may not be.
DC_Phantom ==
  /\ PHANTOMS /\ phantoms < MaxPhantom /\ nextDoor <= MaxDoorIds
  /\ dcRoom \in DOMAIN rooms /\ pos = rooms[dcRoom].truth
  /\ ~\E d \in DoorsOf(dcRoom) : doors[d].hole = "X"
  /\ \/ doorway
     \/ [name |-> rooms[dcRoom].name, hole |-> "X"] \in idFile
  /\ \E fp \in (IF doorway THEN {TRUE} ELSE BOOLEAN) :
        doors' = doors @@ (nextDoor :> NewDoor("X", dcRoom, FALSE, fp))
  /\ nextDoor' = nextDoor + 1
  /\ phantoms' = phantoms + 1
  /\ UNCHANGED << pos, near, doorway, lastHole, passEv, absences, falseX, rooms, rcRT, current, passages, recalls,
                  proposal, rcVars, ltVars, dcVars, ctVars, nextRoom, histVars >>

\* door_existence.cpp update_existence: a fitted door judged absent is REMOVED (:1683-1699).
\* A phantom is eventually judged absent (fair). A real door only by the environment (not fair, <= MaxAbsent): the
\* door of the hole the robot has just passed (its sensors look at it from inside/behind), or any if ABSENCE_ANYTIME.
\* FIX_HOLD (d57997dc, :823-848): no verdict while an `executing` edge names this door's aff_<door>_cross.
DelDoor(d) ==
  /\ doors' = [x \in DOMAIN doors \ {d} |-> doors[x]]
  /\ recalls' = {x \in recalls : x.door # d}        \* the recall's RT edge dies with the door
DC_Absent(d) ==
  /\ Offered(d) /\ doors[d].hole # "X"
  /\ ABSENCE_ANYTIME \/ (near # NoHole /\ doors[d].hole = near /\ ctl.passed)
  /\ absences < MaxAbsent
  /\ FIX_HOLD => ~Executing(d)
  /\ DelDoor(d)
  /\ absences' = absences + 1
  /\ UNCHANGED << pos, near, doorway, lastHole, passEv, falseX, phantoms, rooms, rcRT, current, passages, proposal,
                  rcVars, ltVars, dcVars, ctVars, nextRoom, nextDoor, histVars >>
DC_PhantomGone(d) ==
  /\ Offered(d) /\ doors[d].hole = "X"
  /\ FIX_HOLD => ~Executing(d)
  /\ DelDoor(d)
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, proposal, rcVars, ltVars, dcVars, ctVars, nextRoom, nextDoor,
                  histVars >>

\* =================================================================================== controller
\* controller_world_model.cpp:49 refresh_graph_state -> track_room (mission_door_relation.h:75), on `current`.
CT_Track ==
  /\ current # 0
  /\ \/ /\ current # ctlRoom
        /\ prevName' = IF ctlName # 0 THEN ctlName ELSE prevName
        /\ ctlRoom' = current /\ ctlName' = rooms[current].name
     \/ /\ current = ctlRoom /\ rooms[current].name # ctlName      \* a rename is not a room change
        /\ ctlName' = rooms[current].name /\ UNCHANGED << ctlRoom, prevName >>
  /\ UNCHANGED << envVars, rooms, rcRT, current, doors, passages, recalls, proposal, rcVars, ltVars, dcVars, ctl,
                  nextRoom, nextDoor, histVars >>

\* resolve_door (specificworker_mission.cpp:493-514) + pick_door (mission_door_relation.h:47-67), over
\* door_facts() = doors_of_room(gs.room_id) (:463).
Pick ==
  LET L  == Legs[ctl.leg]
      ds == DoorsOf(ctlRoom)
  IN IF L.rel = "room"
     THEN {d \in ds : L.name \in RecallsOn(d)}                                   \* :53 a recall_<room> child
     ELSE IF prevName # 0 /\ ~\E d \in ds : prevName \in RecallsOn(d)            \* :502 the recall_<prev> guard (11936d55)
          THEN {}
          ELSE {d \in ds : RecallsOn(d) = {} /\ ~HasMatch(d) /\ doors[d].hole = L.hole}  \* :60 (+ harness: other doors shut)

\* 2b9d1d2c/748fbdf3/cac93507 (door_route.h): the route drives the PERCEIVED opening and a robot in the doorway needs no
\* vantage -- a cross may start from the doorway. No protocol guard changes (nothing here required a vantage).
CT_Start ==
  /\ ctl.leg <= NLegs /\ ctl.step = "idle" /\ ctlRoom # 0
  /\ \E d \in Pick :
       /\ Offered(d)                                       \* start_door_passage claims door_concept's offer
       /\ ctl' = [leg |-> ctl.leg, step |-> "cross", door |-> d, hole |-> doors[d].hole, croom |-> ctlRoom,
                  cname |-> ctlName, passed |-> FALSE]
       /\ near' = doors[d].hole
  /\ UNCHANGED << pos, doorway, lastHole, passEv, absences, falseX, phantoms, rooms, rcRT, current, doors, passages,
                  recalls, proposal, rcVars, ltVars, dcVars, ctlRoom, ctlName, prevName, nextRoom, nextDoor, histVars >>

\* The robot physically passes the hole (environment, fair) and stands in its doorway. The route keeps going on its cached
\* targets even if the door node has gone (specificworker_mission.cpp:229-243 re-resolves only on the near side).
CT_Pass ==
  /\ ctl.step = "cross" /\ ~ctl.passed /\ near # NoHole /\ pos \in Ends(near)
  /\ pos' = Other(near, pos)
  /\ lastHole' = near
  /\ doorway' = TRUE
  /\ passEv' = Evaluating
  /\ ctl' = [ctl EXCEPT !.passed = TRUE]
  /\ UNCHANGED << near, absences, falseX, phantoms, rooms, rcRT, current, doors, passages, recalls, proposal, rcVars,
                  ltVars, dcVars, ctlRoom, ctlName, prevName, nextRoom, nextDoor, histVars >>

\* The robot leaves the doorway into the room (environment, fair; it may stay a while -- proto15 turned in it for minutes).
CT_LeaveDoorway ==
  /\ doorway /\ ctl.step = "idle"
  /\ doorway' = FALSE
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, falseX, phantoms, rooms, rcRT, current, doors, passages,
                  recalls, proposal, rcVars, ltVars, dcVars, ctVars, nextRoom, nextDoor, histVars >>

\* controller_door_passage.h:799-803: DONE on a room switch (track_room id, i.e. ltsm's `current`) into a room this
\* door leads to (target_rooms :1169, from the door's passage), or ANY other room when the door has no passage.
\* FIX_COMPLETE_ON_DOOR: only a switch whose crossing fact (the new room's mouth, proto_mouth.h; or R1's exit edge)
\* names this door's hole out of the room the cross started in.
Targets(d) == IF d \in DOMAIN doors
              THEN {doors[e].room : e \in UNION {p \ {d} : p \in {q \in PassAlive : d \in q}}}
              ELSE {}
CT_Done ==
  /\ ctl.step = "cross" /\ ctlRoom # ctl.croom
  /\ Targets(ctl.door) = {} \/ ctlRoom \in Targets(ctl.door)
  /\ FIX_COMPLETE_ON_DOOR => (rooms[ctlRoom].via = ctl.hole /\ rooms[ctlRoom].from = ctl.cname)
  /\ arrived' = Append(arrived, pos)
  /\ ctl' = [leg |-> ctl.leg + 1, step |-> "idle", door |-> 0, hole |-> NoHole, croom |-> 0, cname |-> 0, passed |-> FALSE]
  /\ near' = NoHole
  /\ UNCHANGED << pos, doorway, lastHole, passEv, absences, falseX, phantoms, rooms, rcRT, current, doors, passages,
                  recalls, proposal, rcVars, ltVars, dcVars, ctlRoom, ctlName, prevName, nextRoom, nextDoor, issued,
                  badWrites >>

\* =================================================================================== spec
MemNames == 1..MaxRoomIds
Next ==
  \/ RC_Cross \/ RC_Promote \/ RC_Judge \/ \E d \in DOMAIN doors : RC_FalseCross(d)
  \/ \E K \in {0} \cup MemNames : LT_Cycle(K)
  \/ DC_Follow \/ \E P \in DOMAIN rooms : DC_Mirror(P) \/ \E h \in RealHoles : DC_Birth(h) \/ DC_Phantom
  \/ \E d \in DOMAIN doors : DC_Absent(d) \/ DC_PhantomGone(d)
  \/ CT_Track \/ CT_Start \/ CT_Pass \/ CT_LeaveDoorway \/ CT_Done

\* Every agent keeps cycling (weak fairness on each agent's steps), the robot keeps driving, perception eventually
\* detects a door in view and sees through a phantom. The controller polls its door relation every cycle, so a door that
\* resolves again and again is eventually taken (STRONG fairness on CT_Start). NOT fair: the survey's promotion, false
\* crossings, absence verdicts, phantom births, and which room (if any) the recogniser proposes.
Fairness ==
  /\ WF_vars(RC_Cross) /\ WF_vars(RC_Judge)
  /\ WF_vars(LT_Cycle(0))
  /\ WF_vars(DC_Follow) /\ WF_vars(\E P \in DOMAIN rooms : DC_Mirror(P)) /\ WF_vars(\E h \in RealHoles : DC_Birth(h))
  /\ WF_vars(\E d \in DOMAIN doors : DC_PhantomGone(d))
  /\ WF_vars(CT_Track) /\ SF_vars(CT_Start) /\ WF_vars(CT_Pass) /\ WF_vars(CT_LeaveDoorway) /\ WF_vars(CT_Done)

Spec == Init /\ [][Next]_vars /\ Fairness

\* =================================================================================== properties
\* -- SAFETY ------------------------------------------------------------------------------------------------
\* BUG 1 (recog_proto11 leg 2): while the controller crosses d and the room switch has not been written, d must exist.
ExecCrossDoorLiveBeforeSwitch == (ctl.step = "cross" /\ current = ctl.croom) => ctl.door \in DOMAIN doors
\* The user's literal invariant: executing(cross d) => d exists, for as long as the controller holds the claim.
ExecCrossDoorLive == ctl.step = "cross" => ctl.door \in DOMAIN doors
\* Rule 1 of the design: no agent deletes or rebuilds a node it does not own.
OneOwner == badWrites = {}
\* A live passage matches the two faces of ONE REAL hole, in two DIFFERENT rooms.
PassageJoinsTwoRooms ==
  \A p \in PassAlive : \A a, b \in p : a # b =>
     (doors[a].room # doors[b].room /\ doors[a].hole = doors[b].hole /\ doors[a].hole \in RealHoles)
\* Memory: a remembered passage joins two different remembered rooms through one hole, and those are the hole's ends.
MemoryPassagesSound ==
  \A m \in memPass : /\ m.a[1] # m.b[1] /\ m.a[2] = m.b[2]
                     /\ {issued[m.a[1]], issued[m.b[1]]} = Ends(m.a[2])
LiveNamesUnique == \A r1, r2 \in DOMAIN rooms : r1 # r2 => rooms[r1].name # rooms[r2].name
\* A room name (room_id) always denotes the same physical room, live and in memory.
NameIsIdentity == /\ \A r \in DOMAIN rooms : issued[rooms[r].name] = rooms[r].truth
                  /\ \A k \in DOMAIN memTruth : memTruth[k] = issued[k]
\* One physical room, one name: a room is never re-surveyed under a new name (a duplicate is born by a false crossing
\* into the room the robot is still in, or by a crossing into a room the map should have recognised).
NoDuplicateRoom == \A j, k \in DOMAIN issued : j # k => issued[j] # issued[k]
\* recall_<room_k> under door d names the room that is really behind d.
RecallsPointRight ==
  \A x \in recalls : x.door \in DOMAIN doors /\ doors[x.door].room \in DOMAIN rooms =>
     LET h == doors[x.door].hole
         r == rooms[doors[x.door].room].truth
     IN r \in Ends(h) /\ issued[x.name] = Other(h, r)
\* Each completed leg ended with the robot in the leg's destination.
ArrivedRight == \A i \in 1..Len(arrived) : arrived[i] = Legs[i].dest
\* A recognition verdict never creates or deletes a room node (P becomes K in place, same id).
RenamedInPlace == [][proposal.st = "pending" /\ proposal'.st = "accepted" => DOMAIN rooms' = DOMAIN rooms]_vars

\* -- LIVENESS ----------------------------------------------------------------------------------------------
\* BUG 2: the robot physically past a hole ~> room_concept's room is the robot's room.
CrossingNoticed == [](pos # rooms[rcRT].truth ~> pos = rooms[rcRT].truth)
\* ltsm's `current` follows room_concept's RT room.
CurrentFollows == [](rcRT # current ~> rcRT = current)
\* The tour completes.
TourCompletes == <>(ctl.leg = NLegs + 1)
==========================================================================================
