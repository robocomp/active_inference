------------------------------- MODULE RoomTransitionExit -------------------------------
(***************************************************************************************)
(* The room-transition protocol REDESIGNED around one fact: the CROSSING IS AN ACTION. *)
(* A model of a DESIGN, not of the C++ (RoomTransition.tla models the code as it is).  *)
(*                                                                                     *)
(* The controller drives the body through a hole it perceives in its own frame (the    *)
(* interaction anchor) and measures, there, when the body is through. At that moment   *)
(* it writes the `exit` fact: an edge ROBOT -> the face it crossed (Lucas Bonilla s   *)
(* TFM edge), carrying the                                                            *)
(* room it predicted behind the face (room_id: the recall it resolved the leg with when *)
(* the cross STARTED, 0 = "new"). One writer, one measurement, written once.           *)
(*                                                                                     *)
(* ENCODING A -- a CHAIN of edges, each dying with its source node (as in the graph):  *)
(*  1 controller   `exit` robot -> OLD_FACE (room_id = prediction)                     *)
(*  2 room_concept on 1 naming a face of ITS room: enters NEW_ROOM, `exit` OLD_FACE -> *)
(*                 NEW_ROOM                                                            *)
(*  3 door_concept on 2: NEW_FACE in NEW_ROOM, `exit` NEW_FACE -> OLD_FACE (the twin)  *)
(*  4 ltsm         on 3: copies OLD_ROOM into memory and writes `exit` OLD_ROOM ->     *)
(*                 NEW_ROOM (archived); it deletes NOTHING it does not own.           *)
(*  5 owners       on 4: door_concept deletes OLD_ROOM's doors once the controller's  *)
(*                 link 1 is gone (OLD_FACE: 2 dies); the controller removes link 1  *)
(*                 when the body is DEEP in the room (the cross target), so link 2 -- *)
(*                 room_concept's "transition open" -- lasts until then (pingpong11).*)
(*                 room_concept deletes OLD_ROOM once nothing else hangs from it.     *)
(* No counters: a fact about leaving room R cannot fire twice, the robot is no longer   *)
(* in R once it has been acted on.                                                     *)
(*                                                                                     *)
(*  room_concept   switches on the exit fact alone: the predicted room if its name is  *)
(*                 free, else a new provisional room. No p_cross, no PassageMemory, no  *)
(*                 survey gate, no first-sight rule, no recall read at crossing time.  *)
(*  door_concept   on the exit fact acknowledged by room_concept: the hole's face in   *)
(*                 the entered room exists, is passable, and is the TWIN of the face  *)
(*                 crossed. Both faces are BACKED by the crossing: a verdict from poor *)
(*                 views (absent / leaf shut) does not overturn them (FACT_HOLD).      *)
(*  ltsm_agent     pairs a passage from the twin only (no nearest-door search) and     *)
(*                 remembers it from the entered room's `via` (written from the fact). *)
(*  controller     tracks room_concept's room; a leg completes when room_concept has   *)
(*                 acknowledged THIS leg's exit fact (no "any room change ends a leg"). *)
(*  everyone       follows room_concept's room (no consumer keys on ltsm's `current`).  *)
(*                                                                                     *)
(* OWNERS DELETE (2026-10-31, user): ltsm used to delete the left room's subtree, 133 *)
(* nodes of room_concept/door_concept/residual_concept per pingpong8 (trace S2).       *)
(* Kept from the code as it is: ltsm's recognition/staging, door_concept's existence  *)
(* hold on the face being executed (d57997dc), name-keyed identities.                  *)
(*                                                                                     *)
(* Adversary (proto15 + pingpong2): phantom doors where the robot stands in a doorway, *)
(* a stale door_identities row, WRONG recognition proposals, and WRONG verdicts about  *)
(* the doors of the hole being / just crossed: ABSENT (deleted) or LEAF SHUT (no cross *)
(* offer, pingpong2 door_2). A wrong verdict may PERSIST (no fair recovery), as live.  *)
(*                                                                                     *)
(* SCENARIO "pingpong": room_1 <-> room_2 through DOOR_1, 6 round trips; "tour3": the  *)
(* six-leg three-room tour.                                                            *)
(***************************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS SCENARIO,
          FIX_HOLD,                 \* code (d57997dc): no verdict on the face the controller is executing
          FACT_HOLD,                \* design: no verdict overturns a face backed by an exit fact
          CLOSE_ON_ARCHIVE,         \* design: a leg completes only once ltsm has archived the left room (`current`
                                    \*   = room_concept's room): the transition is closed before the next one opens
          PHANTOMS, STALE_FILE, WRONG_RECOG,
          SURVEY_CAN_PROMOTE,
          ABSENCE_ANYTIME,          \* TRUE = a wrong verdict may hit any offered door at any time
          MaxDegrade, MaxRejects, MaxPhantom,
          MaxRoomIds, MaxDoorIds

PRooms    == {1, 2, 3}
RealHoles == {"D0", "D1", "D2"}
Holes     == RealHoles \cup {"X"}
NoHole    == "none"
Ends(h) == CASE h = "D1" -> {1, 2}
             [] h = "D0" -> {1, 3}
             [] h = "D2" -> {2, 3}
             [] OTHER    -> {}
HolesOf(r)  == {h \in RealHoles : r \in Ends(h)}
Other(h, r) == CHOOSE x \in Ends(h) : x # r

Tour3 == << [rel |-> "new",  hole |-> "D1", name |-> 0, dest |-> 2],
            [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1],
            [rel |-> "new",  hole |-> "D0", name |-> 0, dest |-> 3],
            [rel |-> "new",  hole |-> "D2", name |-> 0, dest |-> 2],
            [rel |-> "room", hole |-> NoHole, name |-> 3, dest |-> 3],
            [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1] >>
PingPong == [i \in 1..12 |-> IF i = 1 THEN [rel |-> "new", hole |-> "D1", name |-> 0, dest |-> 2]
                             ELSE IF i % 2 = 0 THEN [rel |-> "room", hole |-> NoHole, name |-> 1, dest |-> 1]
                             ELSE [rel |-> "room", hole |-> NoHole, name |-> 2, dest |-> 2]]
Legs  == IF SCENARIO = "pingpong" THEN PingPong ELSE Tour3
NLegs == Len(Legs)
Detectable(r) == IF SCENARIO = "pingpong" THEN HolesOf(r) \cap {"D1"} ELSE HolesOf(r)

NoEx   == 999   \* "no controller exit edge on this face" (a room_id no room has)
NoProp == [p |-> 0, k |-> 0, old |-> 0, st |-> "none"]
Max(S) == CHOOSE x \in S : \A y \in S : x >= y

VARIABLES
  pos, near, doorway, lastHole, degr, phantoms,          \* world / environment
  rooms,        \* room_concept: id -> [name, truth, proto, from, via, face (its `exit` OLD_FACE -> this room; 0 none)]
  rcRT,         \* room_concept: the robot's room
  current,      \* ltsm_agent: `current` (still written; nobody but ltsm reads it)
  doors,        \* door_concept: id -> [hole, room, pass (leaf open enough to offer a cross), fact (backed),
                \*   twin (its `exit` NEW_FACE -> OLD_FACE; 0 none), ex (the CONTROLLER's `exit` edge on it: its room_id,
                \*   NoEx = no edge)]
  passages, recalls, proposal,
  rcStartup, rcPromoted, hwm, rejects,
  memTruth, memFaces, memPass, carry,
  dcRoom, idFile,
  ctl, ctlRoom, ctlName,
  nextRoom, nextDoor,
  issued, arrived

vars == << pos, near, doorway, lastHole, degr, phantoms, rooms, rcRT, current, doors, passages, recalls, proposal, rcStartup, rcPromoted, hwm, rejects, memTruth, memFaces, memPass, carry,
           dcRoom, idFile, ctl, ctlRoom, ctlName, nextRoom, nextDoor, issued, arrived >>
envVars  == << pos, near, doorway, lastHole, degr, phantoms >>
rcVars   == << rcStartup, rcPromoted, hwm, rejects >>
ltVars   == << memTruth, memFaces, memPass, carry >>
dcVars   == << dcRoom, idFile >>
ctVars   == << ctl, ctlRoom, ctlName >>

LiveNames      == {rooms[r].name : r \in DOMAIN rooms}
DoorsOf(r)     == {d \in DOMAIN doors : doors[d].room = r}
RecallsOn(d)   == {x.name : x \in {y \in recalls : y.door = d}}
HasMatch(d)    == \E p \in passages : d \in p
Offered(d)     == d \in DOMAIN doors /\ doors[d].room = dcRoom /\ doors[d].pass
Executing(d)   == ctl.step = "cross" /\ ctl.door = d /\ Offered(d)
NewDoor(h, r, f, tw) == [hole |-> h, room |-> r, pass |-> TRUE, fact |-> f, twin |-> tw, ex |-> NoEx]

TypeOK ==
  /\ pos \in PRooms /\ near \in Holes \cup {NoHole} /\ lastHole \in Holes \cup {NoHole}
  /\ rcRT \in DOMAIN rooms /\ current \in DOMAIN rooms \cup {0}
  /\ \A d \in DOMAIN doors : doors[d].hole \in Holes
  /\ ctl.leg \in 1..NLegs + 1 /\ ctl.step \in {"idle", "cross"}

Init ==
  /\ pos = 1 /\ near = NoHole /\ doorway = FALSE /\ lastHole = NoHole /\ degr = 0 /\ phantoms = 0
  /\ rooms = (1 :> [name |-> 1, truth |-> 1, proto |-> FALSE, from |-> 0, via |-> NoHole, face |-> 0, arch |-> FALSE])
  /\ rcRT = 1 /\ current = 1
  /\ doors = [d \in {} |-> NewDoor("D0", 0, FALSE, 0)]
  /\ passages = {} /\ recalls = {} /\ proposal = NoProp
  /\ rcStartup = FALSE /\ rcPromoted = TRUE /\ hwm = 1 /\ rejects = 0
  /\ memTruth = (1 :> 1) /\ memFaces = {} /\ memPass = {} /\ carry = {}
  /\ dcRoom = 1
  /\ idFile = IF STALE_FILE THEN {[name |-> 2, hole |-> "X"]} ELSE {}
  /\ ctl = [leg |-> 1, step |-> "idle", door |-> 0, hole |-> NoHole, croom |-> 0, pred |-> 0, passed |-> FALSE,
            exited |-> FALSE, deep |-> FALSE]
  /\ ctlRoom = 1 /\ ctlName = 1
  /\ nextRoom = 2 /\ nextDoor = 1
  /\ issued = (1 :> 1) /\ arrived = << >>

\* =================================================================================== room_concept
\* Link 2. The switch, on the controller's `exit` edge on a face of ITS room (the only guard).
RC_Switch ==
  /\ \E d \in DoorsOf(rcRT) : doors[d].ex # NoEx
  /\ nextRoom <= MaxRoomIds
  /\ LET d     == CHOOSE x \in DoorsOf(rcRT) : doors[x].ex # NoEx
         pred  == doors[d].ex
         useRc == pred # 0 /\ pred \notin LiveNames
         k     == IF useRc THEN pred ELSE Max(LiveNames \cup {hwm}) + 1
         t     == IF useRc THEN issued[k] ELSE pos
     IN /\ rooms' = rooms @@ (nextRoom :> [name |-> k, truth |-> t, proto |-> ~useRc, from |-> rooms[rcRT].name,
                                            via |-> doors[d].hole, face |-> d, arch |-> FALSE])
        /\ hwm' = Max({hwm, k})
        /\ issued' = IF k \in DOMAIN issued THEN issued ELSE issued @@ (k :> t)
        /\ rcPromoted' = useRc
  /\ rcRT' = nextRoom /\ nextRoom' = nextRoom + 1 /\ rcStartup' = TRUE
  /\ UNCHANGED << envVars, current, doors, passages, recalls, proposal, rejects, ltVars, dcVars,
                  ctVars, nextDoor, arrived >>

Young == rcStartup /\ ~rcPromoted
RC_Promote ==
  /\ SURVEY_CAN_PROMOTE
  /\ Young /\ pos = rooms[rcRT].truth /\ near = NoHole
  /\ ~(proposal.st = "pending" /\ proposal.p = rcRT)
  /\ rooms' = [rooms EXCEPT ![rcRT].proto = FALSE]
  /\ rcPromoted' = TRUE
  /\ UNCHANGED << envVars, rcRT, current, doors, passages, recalls, proposal, rcStartup,
                  hwm, rejects, ltVars, dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>

RC_Judge ==
  /\ proposal.st = "pending"
  /\ LET P     == proposal.p
         adm   == /\ P = rcRT /\ Young
                  /\ P \in DOMAIN rooms /\ rooms[P].proto
                  /\ proposal.k \notin LiveNames \ {rooms[P].name}
         right == adm /\ issued[proposal.k] = rooms[P].truth
     IN \/ /\ ~right
           /\ proposal' = [proposal EXCEPT !.st = "rejected"]
           /\ UNCHANGED << rooms, rcPromoted, rejects >>
        \/ /\ right
           /\ rooms' = [rooms EXCEPT ![P].name = proposal.k, ![P].proto = FALSE]
           /\ rcPromoted' = TRUE
           /\ proposal' = [proposal EXCEPT !.st = "accepted"]
           /\ UNCHANGED rejects
        \/ /\ right /\ rejects < MaxRejects
           /\ proposal' = [proposal EXCEPT !.st = "rejected"]
           /\ rejects' = rejects + 1
           /\ UNCHANGED << rooms, rcPromoted >>
  /\ UNCHANGED << envVars, rcRT, current, doors, passages, recalls, rcStartup, hwm,
                  ltVars, dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>

\* =================================================================================== ltsm_agent
PassAlive == {p \in passages : p \subseteq DOMAIN doors}
\* pairing from the twin only: a face door_concept made from an exit fact, and the face it is the far side of
NewPassages == {{e, doors[e].twin} : e \in {x \in DOMAIN doors : doors[x].twin \in DOMAIN doors
                                                             /\ ~\E p \in PassAlive : x \in p}}

BehindDoor(rm, dr, mt, mp, stg, d) ==
  {o \in DOMAIN mt :
     LET x == rm[dr[d].room].name IN
     /\ dr[d].room \in DOMAIN rm /\ ~rm[dr[d].room].arch
     /\ x \in DOMAIN mt /\ x # stg
     /\ o # x /\ o \notin {rm[r].name : r \in DOMAIN rm}
     /\ \E m \in mp : \/ (m.a = << x, dr[d].hole >> /\ m.b[1] = o)
                      \/ (m.b = << x, dr[d].hole >> /\ m.a[1] = o)}
Wanted(rm, dr, mt, mp, stg) ==
  UNION {{[door |-> d, name |-> o] : o \in BehindDoor(rm, dr, mt, mp, stg, d)} : d \in DOMAIN dr}

Proposable(K) == /\ rooms[rcRT].proto
                 /\ K \in DOMAIN memTruth /\ K \notin LiveNames /\ K # rooms[rcRT].name
                 /\ WRONG_RECOG \/ memTruth[K] = rooms[rcRT].truth

LT_Cycle(K) ==
  LET pass1 == PassAlive \cup NewPassages
      P0 == proposal.p
      verdictCarry == IF proposal.st = "accepted"
                      THEN {[h |-> rooms[P0].via, from |-> rooms[P0].from, to |-> proposal.k]}
                      ELSE {}
      dropName == IF proposal.st = "accepted" /\ proposal.old # proposal.k THEN proposal.old ELSE 0
      prop1 == CASE proposal.st \in {"accepted", "rejected"} -> NoProp
                 [] proposal.st = "pending" /\ (P0 \notin DOMAIN rooms \/ rcRT # P0) -> NoProp
                 [] proposal.st = "none" /\ K # 0 /\ Proposable(K) ->
                        [p |-> rcRT, k |-> K, old |-> rooms[rcRT].name, st |-> "pending"]
                 [] OTHER -> proposal
      mt1 == [n \in DOMAIN memTruth \ {dropName} |-> memTruth[n]]
      mf1 == {f \in memFaces : f[1] # dropName}
      mp1 == {m \in memPass : m.a[1] # dropName /\ m.b[1] # dropName}
      \* link 4: archive only once door_concept's twin of the crossed face exists (link 3)
      due  == rcRT # current /\ current \in DOMAIN rooms
              /\ \E f \in DoorsOf(rcRT) : doors[f].twin = rooms[rcRT].face /\ rooms[rcRT].face # 0
      old  == current
      new  == rcRT
      viaH == rooms[new].via                    \* written by room_concept FROM the exit fact
      gone == IF due THEN DoorsOf(old) ELSE {}
      mt2  == IF due THEN [n \in DOMAIN mt1 \cup {rooms[old].name, rooms[new].name} |->
                             IF n = rooms[old].name THEN rooms[old].truth
                             ELSE IF n = rooms[new].name THEN rooms[new].truth ELSE mt1[n]]
              ELSE mt1
      mf2  == IF due THEN mf1 \cup {<< rooms[old].name, doors[g].hole >> : g \in gone}
                                  \cup {<< rooms[new].name, viaH >>}
              ELSE mf1
      \* the remembered passage is the crossing itself: (left room, hole) -- (entered room, hole)
      mp2  == IF due THEN mp1 \cup {[a |-> << rooms[old].name, viaH >>, b |-> << rooms[new].name, viaH >>]}
              ELSE mp1
      \* link 4: archived, NOT deleted -- the owners delete (DC_Reap, RC_Reap)
      rm2  == IF due THEN [rooms EXCEPT ![old].arch = TRUE] ELSE rooms
      dr2  == doors
      cr1  == carry \cup verdictCarry
      done == {c \in cr1 : << c.from, c.h >> \in mf2}
      mp3  == mp2 \cup {[a |-> << c.from, c.h >>, b |-> << c.to, c.h >>] : c \in done}
      mf3  == mf2 \cup {<< c.to, c.h >> : c \in done}
      stg  == IF prop1.st = "pending" /\ prop1.p \in DOMAIN rm2 /\ rm2[prop1.p].name \in DOMAIN mt2
              THEN rm2[prop1.p].name ELSE 0
  IN /\ passages' = pass1
     /\ proposal' = prop1
     /\ current' = IF due THEN new ELSE current
     /\ rooms' = rm2
     /\ doors' = dr2
     /\ memTruth' = mt2 /\ memFaces' = mf3 /\ memPass' = mp3
     /\ carry' = cr1 \ done
     /\ recalls' = Wanted(rm2, dr2, mt2, mp3, stg)
     /\ UNCHANGED << envVars, rcRT, rcVars, dcVars, ctVars, nextRoom, nextDoor, issued,
                     arrived >>

\* =================================================================================== door_concept
\* follows room_concept's room (the doors it fits and offers are that room's)
DC_Follow ==
  /\ rcRT # dcRoom
  /\ dcRoom' = rcRT
  /\ UNCHANGED << envVars, rooms, rcRT, current, doors, passages, recalls, proposal,
                  rcVars, ltVars, idFile, ctVars, nextRoom, nextDoor, issued, arrived >>

\* Link 3, on room_concept's `exit` OLD_FACE -> NEW_ROOM (alive while OLD_FACE is): the hole's face in NEW_ROOM,
\* passable, backed by the crossing, `exit` NEW_FACE -> OLD_FACE; OLD_FACE is backed too while it lives.
Pending(r) == /\ rooms[r].face \in DOMAIN doors
              /\ ~\E f \in DoorsOf(r) : doors[f].twin = rooms[r].face
DC_Face ==
  /\ \E r \in DOMAIN rooms : Pending(r)
  /\ LET r   == CHOOSE x \in DOMAIN rooms : Pending(x)
         o   == rooms[r].face
         has == {d \in DoorsOf(r) : doors[d].hole = doors[o].hole}
     IN IF has = {}
        THEN /\ nextDoor <= MaxDoorIds
             /\ doors' = [d \in DOMAIN doors |-> IF d = o THEN [doors[d] EXCEPT !.fact = TRUE, !.pass = TRUE]
                                                 ELSE doors[d]]
                         @@ (nextDoor :> NewDoor(doors[o].hole, r, TRUE, o))
             /\ nextDoor' = nextDoor + 1
        ELSE /\ doors' = [d \in DOMAIN doors |->
                            IF d = o THEN [doors[d] EXCEPT !.fact = TRUE, !.pass = TRUE]
                            ELSE IF d \in has THEN [doors[d] EXCEPT !.fact = TRUE, !.pass = TRUE, !.twin = o]
                            ELSE doors[d]]
             /\ nextDoor' = nextDoor
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars, ltVars,
                  dcVars, ctVars, nextRoom, issued, arrived >>

DC_Birth(h) ==
  /\ dcRoom \in DOMAIN rooms /\ pos = rooms[dcRoom].truth /\ h \in Detectable(pos)
  /\ ~\E d \in DoorsOf(dcRoom) : doors[d].hole = h
  /\ nextDoor <= MaxDoorIds
  /\ doors' = doors @@ (nextDoor :> NewDoor(h, dcRoom, FALSE, 0))
  /\ nextDoor' = nextDoor + 1
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars,
                  ltVars, dcVars, ctVars, nextRoom, issued, arrived >>

DC_Phantom ==
  /\ PHANTOMS /\ phantoms < MaxPhantom /\ nextDoor <= MaxDoorIds
  /\ dcRoom \in DOMAIN rooms /\ pos = rooms[dcRoom].truth
  /\ ~\E d \in DoorsOf(dcRoom) : doors[d].hole = "X"
  /\ \/ doorway
     \/ [name |-> rooms[dcRoom].name, hole |-> "X"] \in idFile
  /\ doors' = doors @@ (nextDoor :> NewDoor("X", dcRoom, FALSE, 0))
  /\ nextDoor' = nextDoor + 1
  /\ phantoms' = phantoms + 1
  /\ UNCHANGED << pos, near, doorway, lastHole, degr, rooms, rcRT, current, passages,
                  recalls, proposal, rcVars, ltVars, dcVars, ctVars, nextRoom, issued, arrived >>

\* A WRONG verdict from poor views on a door of the hole being / just crossed (or any, ABSENCE_ANYTIME): the door is
\* judged ABSENT (deleted; door_55 in proto11) or its leaf SHUT (no cross offer; door_2 in pingpong2). Environment, not
\* fair, <= MaxDegrade, and it may persist: nothing here re-opens it.
DelDoor(d) ==
  /\ doors' = [x \in DOMAIN doors \ {d} |-> doors[x]]
  /\ recalls' = {x \in recalls : x.door # d}
DC_Degrade(d) ==
  /\ Offered(d) /\ doors[d].hole # "X" /\ degr < MaxDegrade
  /\ ABSENCE_ANYTIME \/ doors[d].hole \in {near, lastHole}
  /\ FIX_HOLD => ~Executing(d)
  /\ FACT_HOLD => ~doors[d].fact
  /\ \/ DelDoor(d)
     \/ /\ doors' = [doors EXCEPT ![d].pass = FALSE]
        /\ UNCHANGED recalls
  /\ degr' = degr + 1
  /\ UNCHANGED << pos, near, doorway, lastHole, phantoms, rooms, rcRT, current, passages,
                  proposal, rcVars, ltVars, dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>
DC_PhantomGone(d) ==
  /\ Offered(d) /\ doors[d].hole = "X"
  /\ FIX_HOLD => ~Executing(d)
  /\ DelDoor(d)
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, proposal, rcVars, ltVars,
                  dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>

\* =================================================================================== owners delete (link 5)
\* door_concept: a door of a room ltsm has archived is deleted by its owner (OLD_FACE with it: links 1, 2 die).
Reapable(x) == doors[x].room \in DOMAIN rooms /\ rooms[doors[x].room].arch /\ doors[x].ex = NoEx
DC_Reap ==
  /\ \E d \in DOMAIN doors : Reapable(d)
  /\ LET d == CHOOSE x \in DOMAIN doors : Reapable(x)
     IN DelDoor(d)
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, proposal, rcVars, ltVars,
                  dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>
\* room_concept: an archived room is deleted once no other owner's node hangs from it.
RC_Reap ==
  /\ \E r \in DOMAIN rooms : rooms[r].arch /\ DoorsOf(r) = {}
  /\ LET r == CHOOSE x \in DOMAIN rooms : rooms[x].arch /\ DoorsOf(x) = {}
     IN rooms' = [x \in DOMAIN rooms \ {r} |-> rooms[x]]
  /\ UNCHANGED << envVars, rcRT, current, doors, passages, recalls, proposal, rcVars, ltVars,
                  dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>

\* =================================================================================== controller
\* tracks room_concept's room (id; a rename is not a room change)
CT_Track ==
  /\ \/ /\ rcRT # ctlRoom
        /\ ctlRoom' = rcRT /\ ctlName' = rooms[rcRT].name
     \/ /\ rcRT = ctlRoom /\ rooms[rcRT].name # ctlName
        /\ ctlName' = rooms[rcRT].name /\ UNCHANGED ctlRoom
  /\ UNCHANGED << envVars, rooms, rcRT, current, doors, passages, recalls, proposal,
                  rcVars, ltVars, dcVars, ctl, nextRoom, nextDoor, issued, arrived >>

\* `room` -> a door with a recall_<room> child; `new` -> a door no recall/passage/crossing names (+ harness: its hole).
Pick ==
  LET L  == Legs[ctl.leg]
      ds == DoorsOf(ctlRoom)
  IN IF L.rel = "room"
     THEN {d \in ds : L.name \in RecallsOn(d)}
     ELSE {d \in ds : RecallsOn(d) = {} /\ ~HasMatch(d) /\ ~doors[d].fact /\ doors[d].hole = L.hole}

\* The prediction is taken HERE, before crossing: the recall the leg resolved with (or "new").
CT_Start ==
  /\ ctl.leg <= NLegs /\ ctl.step = "idle" /\ ctlRoom = rcRT
  /\ \E d \in Pick :
       /\ Offered(d)
       /\ ctl' = [leg |-> ctl.leg, step |-> "cross", door |-> d, hole |-> doors[d].hole, croom |-> ctlRoom,
                  pred |-> IF Legs[ctl.leg].rel = "room" THEN Legs[ctl.leg].name ELSE 0, passed |-> FALSE,
                  exited |-> FALSE, deep |-> FALSE]
       /\ near' = doors[d].hole
  /\ UNCHANGED << pos, doorway, lastHole, degr, phantoms, rooms, rcRT, current, doors, passages, recalls, proposal, rcVars, ltVars, dcVars, ctlRoom, ctlName, nextRoom, nextDoor, issued,
                  arrived >>

\* The body passes the hole (environment, fair). The route runs on the anchor, not on the graph door.
CT_Pass ==
  /\ ctl.step = "cross" /\ ~ctl.passed /\ near # NoHole /\ pos \in Ends(near)
  /\ pos' = Other(near, pos) /\ lastHole' = near /\ doorway' = TRUE
  /\ ctl' = [ctl EXCEPT !.passed = TRUE]
  /\ UNCHANGED << near, degr, phantoms, rooms, rcRT, current, doors, passages, recalls,
                  proposal, rcVars, ltVars, dcVars, ctlRoom, ctlName, nextRoom, nextDoor, issued, arrived >>

\* Link 1. The body is through: the `exit` edge controller -> the face crossed (one at a time: the last one goes).
CT_Exit ==
  /\ ctl.step = "cross" /\ ctl.passed /\ ~ctl.exited /\ ctl.door \in DOMAIN doors
  /\ doors' = [d \in DOMAIN doors |-> [doors[d] EXCEPT !.ex = IF d = ctl.door THEN ctl.pred ELSE NoEx]]
  /\ ctl' = [ctl EXCEPT !.exited = TRUE]
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars, ltVars,
                  dcVars, ctlRoom, ctlName, nextRoom, nextDoor, issued, arrived >>

\* The body reaches the cross target, deep in the entered room: the controller removes its link 1.
CT_Deep ==
  /\ ctl.step = "cross" /\ ctl.exited /\ ~ctl.deep
  /\ ctlRoom # ctl.croom            \* link 1 acknowledged: room_concept switched on it (else nobody ever does)
  /\ doors' = [d \in DOMAIN doors |-> IF d = ctl.door THEN [doors[d] EXCEPT !.ex = NoEx] ELSE doors[d]]
  /\ ctl' = [ctl EXCEPT !.deep = TRUE]
  /\ UNCHANGED << envVars, rooms, rcRT, current, passages, recalls, proposal, rcVars, ltVars,
                  dcVars, ctlRoom, ctlName, nextRoom, nextDoor, issued, arrived >>

CT_LeaveDoorway ==
  /\ doorway /\ ctl.step = "idle"
  /\ doorway' = FALSE
  /\ UNCHANGED << pos, near, lastHole, degr, phantoms, rooms, rcRT, current, doors, passages, recalls, proposal, rcVars, ltVars, dcVars, ctVars, nextRoom, nextDoor, issued, arrived >>

\* A leg completes once the controller has written its exit and room_concept has switched (the only cause of a switch
\* now is an exit fact), and -- X3 -- ltsm has archived the room left.
CT_Done ==
  /\ ctl.step = "cross" /\ ctl.exited /\ ctl.deep /\ rcRT # ctl.croom /\ ctlRoom = rcRT
  /\ CLOSE_ON_ARCHIVE => current = rcRT   \* archived; the owners' deletion need not be awaited (checked 10-31)
  /\ arrived' = Append(arrived, pos)
  /\ ctl' = [leg |-> ctl.leg + 1, step |-> "idle", door |-> 0, hole |-> NoHole, croom |-> 0, pred |-> 0,
             passed |-> FALSE, exited |-> FALSE, deep |-> FALSE]
  /\ near' = NoHole
  /\ UNCHANGED << pos, doorway, lastHole, degr, phantoms, rooms, rcRT, current, doors,
                  passages, recalls, proposal, rcVars, ltVars, dcVars, ctlRoom, ctlName, nextRoom, nextDoor, issued >>

\* =================================================================================== spec
MemNames == 1..MaxRoomIds
Next ==
  \/ RC_Switch \/ RC_Promote \/ RC_Judge
  \/ \E K \in {0} \cup MemNames : LT_Cycle(K)
  \/ DC_Follow \/ DC_Face \/ \E h \in RealHoles : DC_Birth(h) \/ DC_Phantom
  \/ \E d \in DOMAIN doors : DC_Degrade(d) \/ DC_PhantomGone(d)
  \/ DC_Reap \/ RC_Reap
  \/ CT_Track \/ CT_Start \/ CT_Pass \/ CT_Exit \/ CT_Deep \/ CT_LeaveDoorway \/ CT_Done

Fairness ==
  /\ WF_vars(RC_Switch) /\ WF_vars(RC_Judge)
  /\ WF_vars(LT_Cycle(0))
  /\ WF_vars(DC_Follow) /\ WF_vars(DC_Face) /\ WF_vars(\E h \in RealHoles : DC_Birth(h))
  /\ WF_vars(\E d \in DOMAIN doors : DC_PhantomGone(d))
  /\ WF_vars(CT_Track) /\ SF_vars(CT_Start) /\ WF_vars(CT_Pass) /\ WF_vars(CT_Exit) /\ WF_vars(CT_Deep) /\ WF_vars(CT_LeaveDoorway)
  /\ WF_vars(CT_Done) /\ WF_vars(DC_Reap) /\ WF_vars(RC_Reap)

Spec == Init /\ [][Next]_vars /\ Fairness

\* =================================================================================== properties
ExecCrossDoorLiveBeforeExit == (ctl.step = "cross" /\ ~ctl.passed) => ctl.door \in DOMAIN doors
PassageJoinsTwoRooms ==
  \A p \in PassAlive : \A a, b \in p : a # b =>
     (doors[a].room # doors[b].room /\ doors[a].hole = doors[b].hole /\ doors[a].hole \in RealHoles)
MemoryPassagesSound ==
  \A m \in memPass : /\ m.a[1] # m.b[1] /\ m.a[2] = m.b[2]
                     /\ {issued[m.a[1]], issued[m.b[1]]} = Ends(m.a[2])
LiveNamesUnique == \A r1, r2 \in DOMAIN rooms : r1 # r2 => rooms[r1].name # rooms[r2].name
NameIsIdentity == /\ \A r \in DOMAIN rooms : issued[rooms[r].name] = rooms[r].truth
                  /\ \A k \in DOMAIN memTruth : memTruth[k] = issued[k]
NoDuplicateRoom == \A j, k \in DOMAIN issued : j # k => issued[j] # issued[k]
RecallsPointRight ==
  \A x \in recalls : x.door \in DOMAIN doors /\ doors[x.door].room \in DOMAIN rooms =>
     LET h == doors[x.door].hole
         r == rooms[doors[x.door].room].truth
     IN r \in Ends(h) /\ issued[x.name] = Other(h, r)
ArrivedRight == \A i \in 1..Len(arrived) : arrived[i] = Legs[i].dest
\* Between legs, room_concept's room is the robot's room.
RoomRightAfterExit == ctl.step = "idle" => pos = rooms[rcRT].truth
RenamedInPlace == [][proposal.st = "pending" /\ proposal'.st = "accepted" => DOMAIN rooms' = DOMAIN rooms]_vars

CrossingNoticed == [](pos # rooms[rcRT].truth ~> pos = rooms[rcRT].truth)
TourCompletes == <>(ctl.leg = NLegs + 1)

\* VACUITY (each must be VIOLATED as an invariant, i.e. reachable): RoomTransitionExit_vacuity.sh
VacTourEnds   == ctl.leg # NLegs + 1
VacAccepted   == proposal.st # "accepted"
VacRejected   == proposal.st # "rejected"
VacDegraded   == degr = 0
VacShutBacked == ~\E d \in DOMAIN doors : doors[d].fact /\ ~doors[d].pass
VacPhantom    == phantoms = 0
VacLeg7       == ctl.leg < 7
VacArchived   == ~\E r \in DOMAIN rooms : rooms[r].arch   \* an archived room still waiting for its owners
==========================================================================================
