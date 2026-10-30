--------------------------------- MODULE RoomTransition ---------------------------------
(***************************************************************************************)
(* The room-transition protocol between four DSR agents of one robot -- room_concept,  *)
(* door_concept, ltsm_agent, controller -- that share one replicated graph.            *)
(*                                                                                     *)
(* This is a model of the C++ AS IT IS on branch feat/room-recognition (worktree       *)
(* rt-build, 2026-10-30), NOT of the design document. Every agent is a set of GUARDED  *)
(* ACTIONS over graph FACTS; each guard below cites the line it was read from. Where   *)
(* the code and docs/superpowers/specs/2026-10-29-room-transition-protocol-design.md   *)
(* disagree, the code wins and the disagreement is listed in README.md.                *)
(*                                                                                     *)
(* Abstractions (the only places the model is not the code):                           *)
(*  - Geometry and probabilities are nondeterministic choices. "p_cross >= BirthProb"  *)
(*    becomes "the robot has physically passed hole h and room_concept was evaluating  *)
(*    crossings while it did"; "the door is judged absent" is an environment action    *)
(*    that may fire for a door of the hole the robot stands in (its sensors look at    *)
(*    the robot's own hole from inside it -- recog_proto11 leg 2), at most MaxAbsent   *)
(*    times. Recognition is assumed CORRECT (ltsm proposes K only if K is P's true     *)
(*    room); whether it fires at all is nondeterministic.                              *)
(*  - The graph is sequentially consistent: each agent cycle is one atomic step that   *)
(*    reads the latest graph. DDS reordering is NOT modelled (protocol_mock.py has the *)
(*    lag adversary); interleaving whole agent cycles already exposes the races below. *)
(*  - door_concept offers aff_<door>_cross on every door it fits (its P(passable) is   *)
(*    not modelled); the tour harness closes the other doors of a `cross new` leg, so  *)
(*    a `new` leg may only pick the leg's hole (LegHole).                              *)
(*  - Node ids are monotone counters (bounded by MaxRoomIds / MaxDoorIds).             *)
(*                                                                                     *)
(* Two constants switch today's fixes on/off so both versions can be model-checked:    *)
(*   FIX_HOLD                 door_concept d57997dc: no existence verdict on a door    *)
(*                            whose aff_<door>_cross the controller is `executing`.    *)
(*   FIX_UNPROMOTED_CROSSING  room_concept (uncommitted, room_scene_graph.cpp:905):     *)
(*                            crossings are evaluated in a room whose start-up survey  *)
(*                            has not promoted it yet (old guard HEAD:901).            *)
(***************************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS FIX_HOLD,                 \* BOOLEAN, see above
          FIX_UNPROMOTED_CROSSING,  \* BOOLEAN, see above
          MaxAbsent,                \* how many absence verdicts the environment may produce
          MaxRejects,               \* how many recognition proposals room_concept's fit may reject
          MaxRoomIds,               \* room node ids 1..MaxRoomIds
          MaxDoorIds,               \* door node ids 1..MaxDoorIds
          SURVEY_CAN_PROMOTE,       \* BOOLEAN: FALSE = no start-up survey finishes while the robot is in the room (proto11)
          ABSENCE_ANYTIME           \* BOOLEAN: TRUE = any fitted door may be judged absent at any time (not only the
                                    \*          one the robot has just passed); still at most MaxAbsent verdicts

\* ---------------------------------------------------------------- the world (ground truth)
PRooms == {1, 2, 3}                       \* physical rooms R1, R2, R3 (threerooms-piso.wbt)
Holes  == {"D0", "D1", "D2"}              \* DOOR_0 (R1-R3), DOOR_1 (R1-R2), DOOR_2 (R3-R2)
NoHole == "none"
Ends(h) == CASE h = "D1" -> {1, 2}
             [] h = "D0" -> {1, 3}
             [] h = "D2" -> {2, 3}
HolesOf(r)  == {h \in Holes : r \in Ends(h)}
Other(h, r) == CHOOSE x \in Ends(h) : x # r

\* The tour (recognition_tour.sh): `cross new` x3 and `cross <room>` x3.
Legs == << [rel |-> "new",  hole |-> "D1", name |-> 0, dest |-> 2],   \* 1 -> 2 new
           [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1], \* 2 -> 1 (cross room_1)
           [rel |-> "new",  hole |-> "D0", name |-> 0, dest |-> 3],   \* 1 -> 3 new
           [rel |-> "new",  hole |-> "D2", name |-> 0, dest |-> 2],   \* 3 -> 2, remembered (case 2)
           [rel |-> "room", hole |-> NoHole, name |-> 3, dest |-> 3], \* 2 -> 3 (cross room_3)
           [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1] >>\* 3 -> 1 (cross room_1)
NLegs == Len(Legs)

NoProp == [p |-> 0, k |-> 0, old |-> 0, st |-> "none"]
Max(S) == CHOOSE x \in S : \A y \in S : x >= y

VARIABLES
  \* world / environment
  pos,          \* the robot's true physical room
  near,         \* the hole the robot is in or at (set while a cross is executing), or NoHole
  lastHole,     \* the last hole the robot physically passed
  passEv,       \* room_concept was evaluating crossings when the robot passed lastHole (PassageMemory accumulated it)
  absences,     \* absence verdicts used
  \* graph facts
  rooms,        \* live room nodes: id -> [name (room_<k>, k = room_id), truth, proto (self-edge), from, via]
  rcRT,         \* room_concept: the room the robot's LIVE RT edge hangs from
  current,      \* ltsm_agent: the robot->room `current` edge
  doors,        \* door_concept: id -> [hole, room (RT ancestor), mirror (entry mirror not yet adopted)]
  passages,     \* ltsm_agent: live passage_N nodes, each = the set of its two `match`ed door ids
  recalls,      \* ltsm_agent: recall_<room_k> nodes hung under a door: [door, name]
  proposal,     \* ltsm_agent's recognition proposal on P (room_concept owns its `state`)
  \* agent-local state
  rcStartup, rcPromoted,          \* room_concept: startup_proto_ / startup_promoted_ of its current room
  hwm,                            \* room_concept: room_name_hwm (etc/room_name_hwm.txt)
  rejects,                        \* proposals rejected by the fit so far
  memTruth, memFaces, memPass,    \* ltsm long-term memory: archived room names, faces <<name,hole>>, passages
  carry,                          \* ltsm RoomSwap: carry-overs waiting for the left room's face to be archived
  dcRoom, lost,                   \* door_concept: the room it follows (room_node_id_), holes of its doors deleted by others
  ctl, ctlRoom, ctlName, prevName,\* controller: mission leg + door passage; track_room id/name/prev_name
  nextRoom, nextDoor,             \* node id allocators
  \* history (for invariants only)
  issued,       \* name k -> the physical room it was first issued for
  arrived,      \* the robot's true room at each completed leg
  badWrites     \* writes by an agent that does not own the node (rule 1)

vars == << pos, near, lastHole, passEv, absences, rooms, rcRT, current, doors, passages, recalls, proposal,
           rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost,
           ctl, ctlRoom, ctlName, prevName, nextRoom, nextDoor, issued, arrived, badWrites >>

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
\* room_scene_graph.cpp HEAD:901 (old): `if (PROTO_ROOM_ENABLED and not (startup_proto_ and not startup_promoted_))`;
\* working tree :905 (fix): `if (PROTO_ROOM_ENABLED)`.
Evaluating     == FIX_UNPROMOTED_CROSSING \/ ~(rcStartup /\ ~rcPromoted)

TypeOK ==
  /\ pos \in PRooms /\ near \in Holes \cup {NoHole} /\ lastHole \in Holes \cup {NoHole}
  /\ rcRT \in DOMAIN rooms /\ current \in DOMAIN rooms \cup {0}
  /\ \A d \in DOMAIN doors : doors[d].hole \in Holes
  /\ ctl.leg \in 1..NLegs + 1 /\ ctl.step \in {"idle", "cross"}

Init ==
  \* room_1 is the start room with a GIVEN layout: in recog_proto11 it is inserted with its 31 walls and no proto
  \* edge (graph_trace.csv t=1793362798380), and ltsm placed `current` on it (t=1793362814156).
  /\ pos = 1 /\ near = NoHole /\ lastHole = NoHole /\ passEv = FALSE /\ absences = 0
  /\ rooms = (1 :> [name |-> 1, truth |-> 1, proto |-> FALSE, from |-> 0, via |-> NoHole])
  /\ rcRT = 1 /\ current = 1
  /\ doors = [d \in {} |-> [hole |-> "D0", room |-> 0, mirror |-> FALSE]]
  /\ passages = {} /\ recalls = {} /\ proposal = NoProp
  /\ rcStartup = FALSE /\ rcPromoted = TRUE /\ hwm = 1 /\ rejects = 0
  /\ memTruth = (1 :> 1)                 \* seed_memory(): room 1 goes to memory as soon as it is known (specificworker.cpp:363)
  /\ memFaces = {} /\ memPass = {} /\ carry = {}
  /\ dcRoom = 1 /\ lost = {}
  /\ ctl = [leg |-> 1, step |-> "idle", door |-> 0, croom |-> 0, passed |-> FALSE]
  /\ ctlRoom = 1 /\ ctlName = 1 /\ prevName = 0
  /\ nextRoom = 2 /\ nextDoor = 1
  /\ issued = (1 :> 1) /\ arrived = << >> /\ badWrites = {}

\* =================================================================================== room_concept
\* step_proto_room_impl, "Before birth" branch, working tree room_scene_graph.cpp:947-1159.
RC_Cross ==
  /\ pos # rooms[rcRT].truth                    \* the robot has left the room the RT hangs from ...
  /\ passEv /\ Evaluating                       \* ... and PassageMemory saw the passage (HEAD:901 / :905 gate the
                                                \*     whole impl; memory is cleared at birth :1135 and restarts per room :982)
  /\ \E d \in DoorsOf(rcRT) : doors[d].hole = lastHole   \* :1004 only THIS room's doors (RT ancestor) are apertures;
                                                         \* a door removed for absence is no aperture at all
  /\ nextRoom <= MaxRoomIds
  /\ LET d     == CHOOSE x \in DoorsOf(rcRT) : doors[x].hole = lastHole
         recs  == {k \in RecallsOn(d) : k \notin LiveNames /\ k > 0}   \* :1072 find_recall; :1077 name free, index > 0
         useRc == recs # {}
         k     == IF useRc THEN Max(recs) ELSE Max(LiveNames \cup {hwm}) + 1   \* :1054-1068 one past every room_<k>
         t     == IF useRc THEN issued[k] ELSE pos                    \* a recalled room IS remembered room k
     IN /\ rooms' = rooms @@ (nextRoom :> [name |-> k, truth |-> t, proto |-> ~useRc,
                                            from |-> rooms[rcRT].name, via |-> lastHole])
           \* :1116 proto self-edge; a recall is promoted in the same call (rearm_after_crossing :1738 -> :2055)
        /\ rcRT' = nextRoom                      \* :1139 mark_room_rt_not_current, then the RT to the new room
        /\ nextRoom' = nextRoom + 1
        /\ rcStartup' = TRUE                     \* arm_startup_proto :1792
        /\ rcPromoted' = useRc                   \* :1793 false; :2058 true for a recall
        /\ hwm' = Max({hwm, k})                  \* note_room_name_issued
        /\ issued' = IF k \in DOMAIN issued THEN issued ELSE issued @@ (k :> t)
        /\ passEv' = FALSE
  /\ UNCHANGED << pos, near, lastHole, absences, current, doors, passages, recalls, proposal, rejects,
                  memTruth, memFaces, memPass, carry, dcRoom, lost, ctl, ctlRoom, ctlName, prevName,
                  nextDoor, arrived, badWrites >>

\* step_startup_proto :1838 `if (layout_ready and layout_finished_) promote_startup_proto` -> :2055 proto edge removed.
\* Survey completion is nondeterministic and NOT fair (recog_proto11 leg 2: room_2 was still unpromoted 30 s in).
\* Gated on the robot being inside and not crossing, and on no proposal under judgement (:314 branch).
RC_Promote ==
  /\ SURVEY_CAN_PROMOTE
  /\ rcStartup /\ ~rcPromoted
  /\ pos = rooms[rcRT].truth /\ near = NoHole
  /\ ~(proposal.st = "pending" /\ proposal.p = rcRT)
  /\ rooms' = [rooms EXCEPT ![rcRT].proto = FALSE]
  /\ rcPromoted' = TRUE
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, rcRT, current, doors, passages, recalls, proposal,
                  rcStartup, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost, ctl, ctlRoom,
                  ctlName, prevName, nextRoom, nextDoor, issued, arrived, badWrites >>

\* step_proposal (room_scene_graph_late_recall.cpp:185-249) + admit (proposal_adoption.cpp:287-296).
\* The fit's own duration is collapsed into the verdict step.
RC_Judge ==
  /\ proposal.st = "pending"
  /\ LET P   == proposal.p
         adm == /\ P = rcRT /\ rcStartup /\ ~rcPromoted             \* provisional_room() :66
                /\ P \in DOMAIN rooms /\ rooms[P].proto                \* admit :290 proto edge
                /\ proposal.k \notin LiveNames \ {rooms[P].name}       \* admit :294 NameTaken
     IN \/ /\ ~adm                                                     \* :240 / :196 answered rejected
           /\ proposal' = [proposal EXCEPT !.st = "rejected"]
           /\ UNCHANGED << rooms, rcPromoted, rejects >>
        \/ /\ adm                                                      \* accept_proposal :112-182: rename IN PLACE
           /\ rooms' = [rooms EXCEPT ![P].name = proposal.k, ![P].proto = FALSE]   \* adopt_in_place :337; :1738/2055
           /\ rcPromoted' = TRUE
           /\ proposal' = [proposal EXCEPT !.st = "accepted"]
           /\ UNCHANGED rejects
        \/ /\ adm /\ rejects < MaxRejects                              \* :211 the fit rejects
           /\ proposal' = [proposal EXCEPT !.st = "rejected"]
           /\ rejects' = rejects + 1
           /\ UNCHANGED << rooms, rcPromoted >>
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, rcRT, current, doors, passages, recalls, rcStartup, hwm,
                  memTruth, memFaces, memPass, carry, dcRoom, lost, ctl, ctlRoom, ctlName, prevName,
                  nextRoom, nextDoor, issued, arrived, badWrites >>

\* =================================================================================== ltsm_agent
\* ONE compute() cycle (ltsm_agent specificworker.cpp:343-397), its sub-steps in the code's order:
\*   1. live passages   2. recognition   3. promotion/eviction   4. recalls.
\* `rec` = the recogniser has (or has not) accumulated enough evidence this cycle.

\* -- 1. passage_live.cpp step()
PassAlive == {p \in passages : p \subseteq DOMAIN doors}            \* :103 a passage lives as long as both doors
IsProtoMirror(d) == rooms[doors[d].room].proto /\ doors[d].room # current   \* room_resolve.h:182
Entries == {e \in DOMAIN doors :                                     \* :119 doors hung from a proto room ...
              /\ doors[e].room \in DOMAIN rooms /\ rooms[doors[e].room].proto
              /\ current # 0                                         \* :132 never guess without a current room
              /\ doors[e].room # current                             \* :143 a proto that is current pairs nothing
              /\ ~\E p \in PassAlive : e \in p}                      \* :176 a pairing is kept once made
Cands(e) == {d \in DoorsOf(current) : d # e /\ ~IsProtoMirror(d)}    \* :154 the current room's own doors
\* "COUNTERPART = argmin distance": the face of the same hole when the current room has one, else the nearest other door
Nearest(e) == IF \E d \in Cands(e) : doors[d].hole = doors[e].hole
              THEN CHOOSE d \in Cands(e) : doors[d].hole = doors[e].hole
              ELSE CHOOSE d \in Cands(e) : TRUE
NewPassages == {{e, Nearest(e)} : e \in {x \in Entries : Cands(x) # {}}}

\* -- 4. room_recall.cpp step(): a recall under every live door whose hole memory knows (by place, :137), naming the
\*       passage's OTHER room (:154), unless that room is live (:162).
BehindDoor(rm, dr, mt, mp, d) ==
  {o \in DOMAIN mt :
     LET x == rm[dr[d].room].name IN
     /\ dr[d].room \in DOMAIN rm
     /\ x \in DOMAIN mt                                                  \* room_eviction.cpp:1090 the twin exists
     /\ o # x /\ o \notin {rm[r].name : r \in DOMAIN rm}
     /\ \E m \in mp : \/ (m.a = << x, dr[d].hole >> /\ m.b[1] = o)
                      \/ (m.b = << x, dr[d].hole >> /\ m.a[1] = o)}
Wanted(rm, dr, mt, mp) ==
  UNION {{[door |-> d, name |-> o] : o \in BehindDoor(rm, dr, mt, mp, d)} : d \in DOMAIN dr}

LT_Cycle(rec) ==
  LET pass1 == PassAlive \cup NewPassages
      \* -- 2. recognition_cycle.h: RoomSwap reads the verdict, or the recogniser proposes
      P0 == proposal.p
      verdictCarry == IF proposal.st = "accepted"                       \* RoomSwap::commit :343: carry-over, P's twin dropped :383
                      THEN {[h |-> rooms[P0].via, from |-> rooms[P0].from, to |-> proposal.k]}
                      ELSE {}
      dropName == IF proposal.st = "accepted" /\ proposal.old # proposal.k THEN proposal.old ELSE 0
      recogOK  == /\ proposal.st = "none" /\ rec
                  /\ rooms[rcRT].proto                                   \* room_recognizer.cpp:352 a proto under the robot
                  /\ \E K \in DOMAIN memTruth : K \notin LiveNames /\ memTruth[K] = rooms[rcRT].truth
      prop1 == CASE proposal.st \in {"accepted", "rejected"} -> NoProp \* adopted / rebased at the rejection
                 [] proposal.st = "pending" /\ (P0 \notin DOMAIN rooms \/ rcRT # P0) -> NoProp \* room_swap.cpp:399/:106 withdrawn
                 [] recogOK -> [p |-> rcRT, k |-> CHOOSE K \in DOMAIN memTruth :
                                                    K \notin LiveNames /\ memTruth[K] = rooms[rcRT].truth,
                                old |-> rooms[rcRT].name, st |-> "pending"]
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
      \* else the door of the old room NEAREST THE ROBOT (an argmin over the doors that exist, :714-729). The far face
      \* is the pair's new face (:845), else one DERIVED from the seam through that same door (:890-927).
      oldDs  == DoorsOf(old)
      chosen == CASE fOld # 0 /\ fOld \in oldDs                       -> fOld
                  [] \E x \in oldDs : doors[x].hole = lastHole         -> CHOOSE x \in oldDs : doors[x].hole = lastHole
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
  IN /\ passages' = pass1
     /\ proposal' = prop1
     /\ current' = IF due THEN new ELSE current                          \* room_eviction.cpp:975-978 the `current` flip
     /\ rooms' = rm2
     /\ doors' = dr2
     /\ lost' = IF due /\ old = dcRoom THEN lost \cup {doors[g].hole : g \in gone} ELSE lost
     /\ memTruth' = mt2 /\ memFaces' = mf3 /\ memPass' = mp3
     /\ carry' = cr1 \ done
     /\ recalls' = Wanted(rm2, dr2, mt2, mp3)                           \* -- 4. after the promotion (:397)
     /\ badWrites' = badWrites \cup
          (IF due THEN {"ltsm deletes room_concept's room node (room_eviction.cpp:1288)"} ELSE {}) \cup
          (IF gone # {} THEN {"ltsm deletes door_concept's door nodes (room_eviction.cpp:1543)"} ELSE {})
     /\ UNCHANGED << pos, near, lastHole, passEv, absences, rcRT, rcStartup, rcPromoted, hwm, rejects,
                     dcRoom, ctl, ctlRoom, ctlName, prevName, nextRoom, nextDoor, issued, arrived >>

\* =================================================================================== door_concept
\* step_room_following :3158-3216 -> release_room :3388: follow ltsm's `current`.
DC_Follow ==
  /\ current # 0 /\ current # dcRoom
  /\ LET new == current
         mir == {m \in DoorsOf(new) : doors[m].mirror}
     IN IF mir # {}
        THEN /\ doors' = [doors EXCEPT ![CHOOSE m \in mir : TRUE].mirror = FALSE]   \* :3432 adopt the entry mirror
             /\ nextDoor' = nextDoor
        ELSE IF lastHole \in lost /\ nextDoor <= MaxDoorIds                     \* :3442 recreate the door we came through
                /\ ~\E d \in DoorsOf(new) : doors[d].hole = lastHole
             THEN /\ doors' = doors @@ (nextDoor :> [hole |-> lastHole, room |-> new, mirror |-> FALSE])
                  /\ nextDoor' = nextDoor + 1
             ELSE UNCHANGED << doors, nextDoor >>
  /\ dcRoom' = current /\ lost' = {}
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, rooms, rcRT, current, passages, recalls, proposal,
                  rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, ctl, ctlRoom,
                  ctlName, prevName, nextRoom, issued, arrived, badWrites >>

\* step_entry_mirrors :3221-3330: one mirror per proto room that is not the followed one (:3282) and was not announced
\* by a recall (:3273 entry_mirror_wanted), from the instance whose aperture holds the proto origin (:3300) -- i.e. the
\* door of the crossed hole in the room door_concept follows.
DC_Mirror(P) ==
  /\ P \in DOMAIN rooms /\ rooms[P].proto /\ P # dcRoom
  /\ ~\E m \in DoorsOf(P) : doors[m].mirror
  /\ ~\E x \in recalls : x.name = rooms[P].name
  /\ \E s \in DoorsOf(dcRoom) : ~doors[s].mirror /\ doors[s].hole = rooms[P].via
  /\ nextDoor <= MaxDoorIds
  /\ doors' = doors @@ (nextDoor :> [hole |-> rooms[P].via, room |-> P, mirror |-> TRUE])
  /\ nextDoor' = nextDoor + 1
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, rooms, rcRT, current, passages, recalls, proposal,
                  rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost, ctl,
                  ctlRoom, ctlName, prevName, nextRoom, issued, arrived, badWrites >>

\* Perception: a door of the room the robot is in is detected and born under the followed room (environment, fair).
DC_Birth(h) ==
  /\ dcRoom \in DOMAIN rooms /\ pos = rooms[dcRoom].truth /\ h \in HolesOf(pos)
  /\ ~\E d \in DoorsOf(dcRoom) : doors[d].hole = h
  /\ nextDoor <= MaxDoorIds
  /\ doors' = doors @@ (nextDoor :> [hole |-> h, room |-> dcRoom, mirror |-> FALSE])
  /\ nextDoor' = nextDoor + 1
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, rooms, rcRT, current, passages, recalls, proposal,
                  rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost, ctl,
                  ctlRoom, ctlName, prevName, nextRoom, issued, arrived, badWrites >>

\* door_existence.cpp update_existence: a fitted door judged absent is REMOVED (:1683-1699). Environment, not fair:
\* only for a door of the hole the robot has just passed, whose sensors then look at it from inside/behind.
\* FIX_HOLD (d57997dc, :823-848): no verdict while an `executing` edge names this door's aff_<door>_cross.
DC_Absent(d) ==
  /\ Offered(d)
  /\ ABSENCE_ANYTIME \/ (near # NoHole /\ doors[d].hole = near /\ ctl.passed)   \* past its own hole (0.66 m in proto11)
  /\ absences < MaxAbsent
  /\ FIX_HOLD => ~Executing(d)
  /\ doors' = [x \in DOMAIN doors \ {d} |-> doors[x]]
  /\ recalls' = {x \in recalls : x.door # d}        \* the recall's RT edge dies with the door (trace t=1793362894464)
  /\ absences' = absences + 1
  /\ UNCHANGED << pos, near, lastHole, passEv, rooms, rcRT, current, passages, proposal, rcStartup, rcPromoted,
                  hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost, ctl, ctlRoom, ctlName,
                  prevName, nextRoom, nextDoor, issued, arrived, badWrites >>

\* =================================================================================== controller
\* controller_world_model.cpp:49 refresh_graph_state -> track_room (mission_door_relation.h:75), on `current`.
CT_Track ==
  /\ current # 0
  /\ \/ /\ current # ctlRoom
        /\ prevName' = IF ctlName # 0 THEN ctlName ELSE prevName
        /\ ctlRoom' = current /\ ctlName' = rooms[current].name
     \/ /\ current = ctlRoom /\ rooms[current].name # ctlName      \* a rename is not a room change
        /\ ctlName' = rooms[current].name /\ UNCHANGED << ctlRoom, prevName >>
  /\ UNCHANGED << pos, near, lastHole, passEv, absences, rooms, rcRT, current, doors, passages, recalls, proposal,
                  rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost, ctl,
                  nextRoom, nextDoor, issued, arrived, badWrites >>

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

CT_Start ==
  /\ ctl.leg <= NLegs /\ ctl.step = "idle" /\ ctlRoom # 0
  /\ \E d \in Pick :
       /\ Offered(d)                                       \* start_door_passage claims door_concept's offer
       /\ ctl' = [leg |-> ctl.leg, step |-> "cross", door |-> d, croom |-> ctlRoom, passed |-> FALSE]
       /\ near' = doors[d].hole
  /\ UNCHANGED << pos, lastHole, passEv, absences, rooms, rcRT, current, doors, passages, recalls, proposal,
                  rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost,
                  ctlRoom, ctlName, prevName, nextRoom, nextDoor, issued, arrived, badWrites >>

\* The robot physically passes the hole (environment). The route keeps going on its cached targets even if the door
\* node has gone (controller only re-resolves on the NEAR side, specificworker_mission.cpp:229-243).
CT_Pass ==
  /\ ctl.step = "cross" /\ ~ctl.passed /\ near # NoHole /\ pos \in Ends(near)
  /\ pos' = Other(near, pos)
  /\ lastHole' = near
  /\ passEv' = Evaluating
  /\ ctl' = [ctl EXCEPT !.passed = TRUE]
  /\ UNCHANGED << near, absences, rooms, rcRT, current, doors, passages, recalls, proposal, rcStartup, rcPromoted,
                  hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost, ctlRoom, ctlName, prevName,
                  nextRoom, nextDoor, issued, arrived, badWrites >>

\* controller_door_passage.h:799-803: DONE on a room switch (track_room id, i.e. ltsm's `current`) into a room this
\* door leads to (target_rooms :1169, from the door's passage), or any other room when the door has no passage.
Targets(d) == IF d \in DOMAIN doors
              THEN {doors[e].room : e \in UNION {p \ {d} : p \in {q \in PassAlive : d \in q}}}
              ELSE {}
CT_Done ==
  /\ ctl.step = "cross" /\ ctlRoom # ctl.croom
  /\ Targets(ctl.door) = {} \/ ctlRoom \in Targets(ctl.door)
  /\ arrived' = Append(arrived, pos)
  /\ ctl' = [leg |-> ctl.leg + 1, step |-> "idle", door |-> 0, croom |-> 0, passed |-> FALSE]
  /\ near' = NoHole
  /\ UNCHANGED << pos, lastHole, passEv, absences, rooms, rcRT, current, doors, passages, recalls, proposal,
                  rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry, dcRoom, lost,
                  ctlRoom, ctlName, prevName, nextRoom, nextDoor, issued, badWrites >>

\* =================================================================================== spec
Next ==
  \/ RC_Cross \/ RC_Promote \/ RC_Judge
  \/ \E rec \in BOOLEAN : LT_Cycle(rec)
  \/ DC_Follow \/ \E P \in DOMAIN rooms : DC_Mirror(P) \/ \E h \in Holes : DC_Birth(h)
  \/ \E d \in DOMAIN doors : DC_Absent(d)
  \/ CT_Track \/ CT_Start \/ CT_Pass \/ CT_Done

\* Every agent keeps cycling (weak fairness on each agent's steps), the robot keeps driving, perception eventually
\* detects a door in view. NOT fair: the survey's promotion, absence verdicts, the recogniser's evidence (`rec`).
Fairness ==
  /\ WF_vars(RC_Cross) /\ WF_vars(RC_Judge)
  /\ WF_vars(LT_Cycle(FALSE))
  /\ WF_vars(DC_Follow) /\ WF_vars(\E P \in DOMAIN rooms : DC_Mirror(P)) /\ WF_vars(\E h \in Holes : DC_Birth(h))
  /\ WF_vars(CT_Track) /\ WF_vars(CT_Start) /\ WF_vars(CT_Pass) /\ WF_vars(CT_Done)

Spec == Init /\ [][Next]_vars /\ Fairness

\* =================================================================================== properties
\* -- SAFETY ------------------------------------------------------------------------------------------------
\* BUG 1 (recog_proto11 leg 2): while the controller crosses door d and the room switch has not been written,
\* d must exist (door_concept removed door_55 with aff_door_55_cross executing).
ExecCrossDoorLiveBeforeSwitch == (ctl.step = "cross" /\ current = ctl.croom) => ctl.door \in DOMAIN doors
\* The user's literal invariant: executing(cross d) => d exists, for as long as the controller holds the claim.
ExecCrossDoorLive == ctl.step = "cross" => ctl.door \in DOMAIN doors
\* Rule 1 of the design: no agent deletes or rebuilds a node it does not own.
OneOwner == badWrites = {}
\* A live passage joins the two faces of ONE hole, in two DIFFERENT rooms.
PassageJoinsTwoRooms ==
  \A p \in PassAlive : \A a, b \in p : a # b => (doors[a].room # doors[b].room /\ doors[a].hole = doors[b].hole)
\* Memory: a remembered passage joins two different remembered rooms through one hole, and those are the hole's ends.
MemoryPassagesSound ==
  \A m \in memPass : /\ m.a[1] # m.b[1] /\ m.a[2] = m.b[2]
                     /\ {issued[m.a[1]], issued[m.b[1]]} = Ends(m.a[2])
LiveNamesUnique == \A r1, r2 \in DOMAIN rooms : r1 # r2 => rooms[r1].name # rooms[r2].name
\* A room name (room_id) always denotes the same physical room, live and in memory (adoption renames IN PLACE to the
\* right identity; a recall re-creates the right room).
NameIsIdentity == /\ \A r \in DOMAIN rooms : issued[rooms[r].name] = rooms[r].truth
                  /\ \A k \in DOMAIN memTruth : memTruth[k] = issued[k]
\* recall_<room_k> under door d names the room that is really behind d.
RecallsPointRight ==
  \A x \in recalls : x.door \in DOMAIN doors /\ doors[x.door].room \in DOMAIN rooms =>
     issued[x.name] = Other(doors[x.door].hole, rooms[doors[x.door].room].truth)
\* Each completed leg ended with the robot in the leg's destination.
ArrivedRight == \A i \in 1..Len(arrived) : arrived[i] = Legs[i].dest
\* A recognition verdict never creates or deletes a room node (P becomes K in place, same id).
RenamedInPlace == [][proposal.st = "pending" /\ proposal'.st = "accepted" => DOMAIN rooms' = DOMAIN rooms]_vars

\* -- LIVENESS ----------------------------------------------------------------------------------------------
\* BUG 2: the robot physically past a hole ~> room_concept's room is the robot's room.
CrossingNoticed == [](pos # rooms[rcRT].truth ~> pos = rooms[rcRT].truth)
\* ltsm's `current` follows room_concept's RT room.
CurrentFollows == [](rcRT # current ~> rcRT = current)
\* The six-leg tour completes.
TourCompletes == <>(ctl.leg = NLegs + 1)
==========================================================================================
