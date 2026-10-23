---
description: exploring a room: budget, the explored() condition, and what it does not measure
---
- explore ROOM always needs budget_s (for example budget_s=60s); without it the mission is refused.
- Add until=explored(ROOM)>=0.9 to stop early once the walls have been seen. The budget still ends it if that never happens.
- explored() measures how much of the room's BOUNDARY (walls) has been seen. It says nothing about the objects inside: a room can be explored with its furniture still unseen.
- To find out what is in a room, explore it, then use report "..." while inside; the report lists the objects seen there.
- Typical budgets: 60s to look around a small room, 300s to map a large one.
- After exploring a new room, add report "<topic>" as its own step (the chat then describes what is in the room), then go back with cross or go_to anchor:NAME.
