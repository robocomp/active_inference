# Changed object: the robot notices a moved chair, decides, asks, and learns

Date: 2026-11-15. Status: design, awaiting the owner's approval. Nothing is built.

## Intent and why

The project claims that the robot **keeps the truth of its representation**. That claim needs evidence: when
something it can detect has changed, the robot must update its map **and** say what changed. The owner moved the
only chair in room 2 (Webots threerooms-piso world, 2026-11-15). Before the move LTOM remembered it as chair_50,
magenta, at (-2.57, -0.61) in the room frame.

The owner's larger goal: *"after fluid mission completion, the robot learns from these conversations, otherwise
the interaction will always repeat the same patterns."* This experiment is the first test of that. The same
situation must lead to different behaviour before and after a conversation, and the reason must be traceable to
what was said.

## The owner's common sense (the model to build)

1. **Identity by count.** If the room held only one chair, a "new" chair is more probably the old one moved than
   a second chair plus the old one gone. Count, class, colour and size act as evidence in a posterior over "same
   object, moved" against "new object, and the old one gone". No match threshold.
2. **Agency.** People can move things. The robot knows one human, the owner (the chat partner). A move can be
   credited to the owner, but only when the owner says so. While the owner is the only known agent, "No, I didn't
   move it" means it is a different chair.

## The scene (owner's words)

"I am not in the scene. I am at the chat, not visible, and the robot knows I am not there since it hasn't seen me.
The robot enters R2 and sees the only chair has moved. It asks."

## Behaviour

### Visit 1: decide, then ask

1. The chair agent sees a magenta chair in room 2 away from chair_50's remembered spot, and it has **seen the old
   spot empty**. It decides straight away using the posterior below. With one chair, the matching colour and the
   empty spot, the answer is "chair_50, moved". It updates the map (chair_50's pose) and **the robot carries on
   with its task**. The robot never blocks on the owner.
2. Who moved it is **unknown** at that point. The robot records "moved, by someone unknown". The belief "the owner
   moves things" does **not** grow from a guess.
3. The question goes to the owner, **one question per change**:
   *"The magenta chair in room 2 is by the window now; it was by the door. Did you move it?"*
   - Owner available (currently in the chat): it is asked at once.
   - Owner away: it stays pending. When the owner next writes in the chat, the robot first answers what they wrote
     and **then raises the question by itself** (owner's choice). For example: "By the way, while you were away
     the magenta chair in room 2 moved to the window. I've assumed it's the same chair. Did you move it?"
4. Answers:
   - **Yes:** the decision stands, and the move is credited to the owner. Two learned beliefs grow: "the owner moves
     things" and "this chair gets moved". Each is stored with the date and the owner's words.
   - **No:** the decision is revised. chair_50 is marked **gone from room 2**. It is kept in LTOM, not deleted, so
     it can match elsewhere later. The chair now seen becomes a new chair.
   - **"I don't know" / no answer:** the decision stands, flagged as uncertain. Nothing is learned about the owner.

### Visit 2: report instead of ask

The owner moves the chair again, and later the robot returns. Because of what it learned on visit 1, it reports
instead of asking: *"The magenta chair has moved again, to the corner. I assume you moved it."* If the owner says
"No, I didn't", it corrects itself and the belief shrinks.

### Inspecting and correcting what was learned (owner: in the chat)

- "What have you learned about me?" or "Why did you think I moved the chair?" make the robot answer from LTOM with
  the **evidence**, never a bare number. For example: "I think you move things: on 15 November you told me you
  moved the magenta chair in room 2, and nothing has contradicted that."
- "No, I didn't move it" adds evidence against the belief. The record keeps both answers.
- "Forget that" drops the belief.

## Goals

- In live Webots, the moved chair is updated in place (same identity) and the robot asks the right question about
  it.
- A yes, a no, or no answer each leaves the map and LTOM in the state described above.
- Visit 2 reports instead of asking, and the difference is traceable to the visit-1 answer in LTOM.
- The owner can ask in the chat what was learned, and correct it.

## Non-goals (this version)

- Other people as agents. Only the owner, for now.
- Object classes other than chairs (the mechanism should not be chair-specific in shape, but only chairs are
  tested).
- Automatic clearing of the chat history (owner: not now).
- Habits beyond the two beliefs in this experiment. The belief store is shaped so others can be added later
  without changing it.

## Anti-goals

- **Asking about the wrong chair** (the owner's wince). The question must name the chair that actually moved,
  describe its old and new places correctly, and never be raised about a chair that did not move. The known trap
  is in LTOM today: **chair_44 is a stale duplicate at chair_50's old spot, with no colour.** A colourless
  duplicate, a flickering detection, or a spot never actually seen empty must not produce a question.
- Asking before the old spot has been seen empty. A chair seen elsewhere is not yet a change.
- Blocking the task while waiting for an answer.
- Nagging: one pending question per owner message, and no repeated asks about old news.
- Crediting the owner on a guess. Only an answer from the owner (or something seen) teaches the robot about them.
- Thresholds or fixed counts where a model term can do the job (CLAUDE.md).

## Constraints

- Agents talk **only through memory**: the shared graph and LTOM. No files cross agents.
- CONCEPT_AGENT_LIFECYCLE.md: birth is tracker-only. Absence is weighted by P(detect) and line of sight, walls
  included. Occlusion holds.
- New graph attributes go through cortex `REGISTER_TYPE`. The **owner reinstalls cortex**; never `sudo cp` it.
  Use type-attributed access only, never `runtime_checked_*`.
- DSR signals are Queued, never DirectConnection. Parse numbers with `std::from_chars`. Use `and`/`or`/`not`.
  Builds at -j8 (-j4 if another build is running). Stop agents with SIGTERM.
- Live runs need Webots, which is CLAIMED and FREED with the peer sessions.

## How (the parts already decided)

**The split (agreed with the owner).**
- **The chair agent** makes the "moved or new" decision, inside its existing remembered-or-new choice
  (`rc::recall::decide`, `common/recall_match/recall_match.h`). Today that choice is a MAP over recalled candidates
  against "new", using **position only**: `location_log_lik` is the mixture (1−π_move)·Gaussian + π_move/room_area,
  with `Tracker.RecallPiMove`=0.5 and `RecallLogPriorNew`=0. It is extended with **colour, size, the room's chair
  count, and the empty old spot** as likelihood or prior terms.
- **LTOM** holds the learned beliefs and the pending question, and records the answers and "gone from room".
- **language_agent** puts the question to the owner and sends the answer back.

**The path.** Every step goes through the graph.
1. The chair agent marks the live chair with the open question "is this <recalled uid>?" and its probability.
2. language_agent sees the mark (in the same tick loop as `track_object_changes`) and asks, or holds the question
   until the owner next writes. The question is phrased from the graph (colour, room, old and new places), not
   made up by the LLM.
3. The answer travels chat → LTOM the way naming claims do today (`chat_intention` claim node →
   `ltom_agent/src/human_graph.h` `read_claim` → LTOM's answer → `check_claim_answer`), as a new claim kind.
4. LTOM applies it: identity kept or chair_50 marked gone from room 2, plus the belief update. The chair agent reads
   the result back through its next recall, and the mark clears.

**What is known to be missing today.**
- Recall nodes (`ltom_recall_<uid>` metaconcepts) carry no colour or size.
- LTOM continuity is by uid only and has no "moved" or "gone" concept. A `death` end is ambiguous.
- human_1 has no presence history.
- There is no path for a question from the agents to the person. The claim card only runs chat → LTOM, and only
  for name/owns/disowns.

## Left to the builder (my calls, marked)

- **Where the question mark lives:** an attribute on the live chair node. This is my call; it needs a cortex
  attribute and an owner reinstall.
- **Belief representation:** dated evidence records (answer, date, the owner's words) per belief, with the belief
  strength computed from them, for example as a Beta over yes/no. That keeps beliefs probabilistic, visible, and
  undoable. My call.
- **Ask vs report** follows from the identity/agency uncertainty and the cost of asking, not from a fixed count.
  With one "yes" this should already tip visit 2 towards reporting, and "I assume" stays in the sentence. My call.
- **Dropping a question nobody answered:** its worth fades as later evidence settles identity (for example, no
  second magenta chair ever appears). It is asked only while the answer would still change something. No fixed
  retry count. My call.
- **"Forget that":** drops the belief but keeps a dated note of the request, so the same old conversation does not
  re-teach it. My call.
- **Mid-mission:** if the owner is in the middle of giving a mission, the question waits until the mission is sent.
  My call.
- **How the empty old spot is observed for a recall:** the lifecycle's absence model (P(detect), line of sight)
  applied to the recalled position. The mechanism is the builder's.
- **The chair_44 duplicate:** the decision is made **robust** to it rather than relying on a clean-up. The room's
  chair count is an *expected* count: each remembered chair is weighted by its existence probability
  (`obj_exist_prob`, already the recall's `log_prior`), so a stale colourless duplicate adds little. Two recalls at
  the same spot compete for the same detection as usual, and the question names the winner only. Cleaning chair_44
  out of LTOM is allowed but not required. My call.
- **Colour and size evidence:** use the colour posterior the chair agent already publishes (object colour names,
  11-11), compared against the colour LTOM remembered, as a likelihood term. A detection with no colour posterior
  yet is **no evidence**, not evidence against. LTOM supplies the remembered colour and size on the recall node;
  the chair agent does not invent them. My call.
- **Moved vs gone in LTOM:** a dated `moved` event on the object's history, old pose kept, and a separate dated
  `gone_from_room` record. A `death` end is not reused. The visit-1 pose edit is reversible: the old pose stays in
  the history, and "No" restores chair_50 to it and marks it gone from room 2. The exact schema is the builder's,
  within these semantics. My call.
- **Availability, without a presence window:** the question goes into the chat as soon as the decision is made.
  If the owner's next message is not an answer to it, the robot replies to that message and then raises the
  question once more by itself. No time window or presence heartbeat is needed. My call; it meets the owner's
  "raise it by itself".
- **Ask vs report:** belief = Beta over the dated yes/no records. The robot reports when the expected cost of a
  wrong "I assume" is lower than the cost of a question. The costs are set so one "yes" tips it, as the owner's
  visit-2 behaviour requires. My call.
- **Who owns the pending question:** LTOM, because it survives restarts as memory. The mark on the chair node is
  only the trigger. LTOM closes the question when it is answered, when it has faded, or when the chair is replaced
  after a "No" (the question closes with it and does not move to the new chair). A restarted language_agent re-reads
  the open questions from LTOM. My call.
- **"Forget that":** the note suppresses re-learning from the old conversation only. A fresh "yes" later counts as
  usual. My call.

## The live experiment

- **Setup:** fixture with chair_50 remembered in room 2. The world already has the chair moved (any room_2 run after
  2026-11-15 sees the new position).
- **Run A (owner in the chat):** enter room 2. Expect an immediate, correct question. Answer "yes". Check the pose
  update and both beliefs in LTOM, each with date and words.
- **Run B (owner away):** enter room 2 with no chat activity. Expect the decision to be made and the task to go on.
  Then write in the chat and expect the question raised after the reply.
- **Run C ("no"):** expect chair_50 marked gone from room 2 and a new chair born.
- **Visit 2:** move the chair again in Webots. Expect a report with "I assume", then "no" to shrink the belief.
- **Inspection:** "What have you learned about me?" returns the evidence.
- **Pass:** every question names the chair that moved, and nothing is asked about a chair that did not move.
