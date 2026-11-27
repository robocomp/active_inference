# Day 5 plan: the VLM, the language agent, and what language models bring to a robot

Decision of the lecturer: Days 1-4 built the representation; Day 5 adds the vision-language model and the language agent, shows many real queries and commands with what the robot answered or did, and opens a discussion of what language models provide.
Material: `day5_examples.md` (29 entries; 24 from the logs, 5 not logs). Draft deck: `day5.tex` (old one saved as `day5_before_rework.tex`).

## Timing (3 hours)

| time | block | frames (main deck) |
|---|---|---|
| 0:00-0:10 | Opening: what language adds to beliefs that already exist | 1-2 |
| 0:10-0:35 | The vision-language model: what it reads, what it cannot ground | 3-6 |
| 0:35-1:25 | The language agent: digest, questions, memory, commands, missions, conditions, imagining, naming | 7-20 |
| 1:25-1:35 | Claims and failures: the cases that went wrong, and why they were stopped | 21-24 |
| 1:35-1:45 | break | |
| 1:45-2:35 | Lab 5 (reduced to 50 min, see below) | 25 |
| 2:35-2:55 | Discussion: three open questions | 26-28 |
| 2:55-3:00 | Open directions, the week, closing | 29-31 |

The discussion moves after the lab (the old deck put talking before the lab). If the lecturer prefers discussion before the break, swap blocks 3 and 5 and the lab gets 60 min.

## Outline, one line per frame (title = message; what it shows)

Section 0, opening
1. **Today language reads and changes the beliefs we already have.** Three boxes: VLM, language agent, discussion; one line that every example carries its log file and where it ran.
2. **Four days built beliefs that agents hold on one graph.** The graph with its holders (room, pose, doors, chairs, tables, memories). `\ask`: which could the robot already answer in numbers; answer: all, what was missing was a way to ask.

Section 1, seeing with a vision-language model
3. **A vision-language model turns a picture and a question into words.** Pipeline frame + question -> model -> text; chat block A1 ("what do you see?" -> look -> green door, wooden floor).
4. **The picture can hold what the beliefs do not.** Chat A2 (bedroom with a bed in the picture; "is the bed for one or two?" -> no bed perceived). `\ask`: whose answer should the robot give?
5. **A colour read from one frame depends on what was asked.** Chat A3 (yellow chair, then black and grey). Lesson: a reading of one frame is a hypothesis.
6. **The model describes; the beliefs measure.** Two columns: what the model does well / what a belief holds, with the questions each answers. `\ask`: which should say the table is 0.80 by 0.74 m?

Section 2, the language agent
7. **The language model acts only through tools.** Reused architecture frame (tool count removed, see open questions).
8. **The model reads a text digest of the beliefs, refreshed every 0.5 s.** The real digest of 11-10 (shortened) with three callouts: P(open) is the door agent's, remembered rooms come from the long-term map, only this room is perceived.
9. **Questions about the scene are answered from a belief.** Two chat blocks side by side: door (A4) and table sizes (A5).
10. **Each question goes to the agent that holds the belief.** Reused table (who holds which belief) with `P(see | v)`.
11. **A fact about the past is repeated, not rephrased.** Chat B1 before the fix (reworded, "may have moved" lost, chair_39 moved to room_1) and after (verbatim).
12. **When the memory is empty the robot says so.** Chats B3: 42 missions, 35 succeeded; and "I do not remember doing anything today". `\ask`: what should it say when the memory is empty?
13. **A gate in code decides which commands wait for a yes.** Chat C1 (back up a bit, runs at once, blocked at the third) next to C2 (cross card with Run button).
14. **The model proposes a plan; the controller runs it.** Reused flow frame with the mission language.
15. **One request, one plan, two reports.** Chat C3: "go to chair_34" -> card (3 steps) -> "yes" -> arrived and no chair seen -> two steps skipped, closing.
16. **A condition in a mission reads a belief.** Three mission cards D1-D3, labelled "planner's examples, not a log", and the live sim result as one line.
17. **To imagine is to predict observations under a change.** Reused imagining frame.
18. **The same question about an imagined chair gives two answers, and says why.** Chat E1 twice (P 0.00 outside the view; P 0.97 at 2.5 m, too close). `\ask`.
19. **A "what if" needs a place in the world.** Chat E2 (could you open a refrigerator; with a servo; "I do not know which room I am in").
20. **A name given in words lived only in the chat.** Chat F1 (front door alias; the gate refuses) and the design on the right (designed, not built for use).

Section 3, claims and failures
21. **"The crossing is complete" had no tool behind it.** Chat C7 (cross, then "go" moved back 0.35 m; the robot claimed the crossing was done).
22. **Every claim must come from a tool that read a belief.** Reused claims frame.
23. **Three failures, each stopped by a different part.** C4 (controller refused: no approach affordance), C5 (door_21 not in the graph; asks to replan), C9 (model server stopped; nothing moved).
24. **Three failures nothing stopped.** A6 (stale "not moving"), G2 (frame rate with no source), G3 (confirmation typed in the chat). a last line: what code could catch each.
25. **What is measured, and what is not.** Reused; numbers updated to the eval of 10-30 (tools 307/315, missions 99/102) with the doc as source. Then the lab frame.

Section 4, lab, discussion, close
26. **Lab 5** (reused frame; see the lab section below).
27. **What does a language model give a robot?** Three columns: a way to ask, a way to compose, common sense from outside. `\ask`: which one could the beliefs not give.
28. **What can it not give?** Four boxes linked to the examples (measure, remember, know the present, decide). `\ask`: where did a wrong sentence come out?
29. **Where must the beliefs stay grounded?** Layers sensors -> beliefs -> language; reading is open, writing goes through a check and a yes. `\ask`: what if the model wrote a belief directly?
30. **Open directions** (reused).
31. **The week in five lines**, then the closing result (reused text).
Backup: second sight (2 reused frames), the inner voice (2 reused frames, work in progress).

## Reused from the current day5.tex / new

- Reused unchanged: architecture, proposes/controller acts, question-as-query table, imagining, claims, open directions, lab frame, last result.
- Reused with edits: opening (rewritten), what is measured (numbers updated, source added), inner voice frames and second-sight frames (moved to backup).
- New: all chat-style frames (A1-A6, B1-B3, C1-C9 as chosen, D, E1-E2, F1), digest frame, VLM frames, failures frames, three discussion frames, the week frame.
- Dropped from the main line: the second-sight section (it was announced for Day 5 at the end of Day 4, see open questions) and the inner-voice frames (proposals, nothing built).

## The lab

`labs/src/day5.py` (unchanged): parts A (second sight: three remembered chairs and "new") and B (a toy language agent: tools on a toy graph, a scripted model, a claim check, a mission with a P(open) condition, optional local model).
- Part A no longer matches the deck's story (the section was moved to backup). Keep it only if second sight is kept in the lecture.
- Part B still fits and is the right core: tools that read beliefs, a claim check, a mission with a condition. It should stay.
- Proposed new lab: students write queries against a recorded graph. A recorded graph snapshot (the apartment, with 2 rooms, 2 doors, 3 chairs, memory of chairs 15 h old) plus the digest generator. Tasks: (1) write five questions that are answered from the digest, five that need a tool, five that must be refused; (2) run the claim check on a recorded set of model answers (with the bed and yellow-chair cases among them) and mark each as supported or not; (3) write a mission with a `watch` and an `if` and run the toy interpreter; (4) optional: connect a local model.
- Needed for that: the recorded graph and a set of model answers as files (from the logs), and a small toy interpreter for `watch`/`if`. Not written yet.

## What the lecturer should check before the day

See the report; the open questions are repeated in the delivery message.
