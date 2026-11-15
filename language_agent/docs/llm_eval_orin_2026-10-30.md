# LLM evaluation on the Jetson AGX Orin — 2026-10-30

Server: Jetson AGX Orin 32 GB at 192.168.1.34, JetPack 7.2.1 / Ubuntu 24.04, MAXN, Ollama.
Evals run from the `feat/vlm-missions` worktree, one model loaded at a time:

    OLLAMA=http://192.168.1.34:11434 python3 tests/check_llm_tools.py <model> 3
    OLLAMA=http://192.168.1.34:11434 python3 tests/check_llm_missions.py <model> 3 --form text

- **tools**: tool choice, 105 cases x 3 repeats = 315 (the suite was 279 in earlier runs).
- **missions**: mission writing by the planner, 34 cases x 3 = 102.
- Both scripts send `think: false`, `num_ctx` 16384.

## Results

| model | tools /315 | latency/call (median, mean) | missions /102 | memory (/api/ps, all GPU) |
|---|---|---|---|---|
| **gemma4:26b** (current choice) | **307** | **0.89 s, 1.37 s** | **99** (2026-10-24, final fix wave) | 18 GB |
| qwen3.8:latest (27B dense) | 299 | 2.21 s, 3.84 s | **102** (perfect) | 17.5 GB |
| gpt-oss:20b | 287 | 6.4 s, 8.70 s | 82 (median 21.8 s/call) | 12.9 GB |
| gemma4:31b | crash | — | crash | — |
| nemotron-3.5-lightning:30b | 238 | 1.1 s, 2.25 s | 39 | 25.5 GB |
| nemotron-3-nano:30b | 183 | 1.7 s, 2.42 s | 33 | 24.3 GB |

## Failures by model

- **gemma4:26b, 8 tool failures:**
  - `draw the graph` x3: wrote its own `draw` script instead of the intended tool;
  - `where are you going (moving)` x3: answered "stationary" while the STATE says it is moving;
  - `cross it (door is shut)` x1;
  - `approach the chair from behind` x1.
- **qwen3.8:latest (27B dense), 16 tool failures, missions 102/102:**
  - `where are you going (moving)` x3: same miss as gemma4:26b;
  - `plan: wait and return` x3, `plan: cruza y vuelve (es)` x3, `plan: cross, explore, come back` x3, `plan: condition` x3: plan-style cases;
  - `abilities table, room known` x1.
- **gpt-oss:20b:**
  - It reasons even with `think: false`, so it is about 7x slower than gemma4:26b on tools and about 20 s per mission.
  - Mission failures: the output does not start with `mission`.
  - Top tool failures: where is the fridge, what do you look like, bar chart.
- **gemma4:31b:** the Orin's Ollama server died at about call 34, twice: first RemoteDisconnected, then connection refused. The likely cause is OOM, from a dense 31B plus the 16k context in 32 GB.
- **nemotron-3.5-lightning:30b, nemotron-3-nano:30b:**
  - They are fast but weak at tool choice: working memory, what do you see, episode questions.
  - Most mission failures are `anchor 'start' not defined`: they leave out the `anchor` step.
- **Nemotron 3 / 3.5 Super (120B):** 80 GB or more, so they don't fit on the 32 GB Orin and were not tested.

## Conclusion

Keep **gemma4:26b** for chat: it is the most accurate on tools (307 vs 299) and 2.5x faster per call (0.89 s vs 2.21 s median).
qwen3.8:latest (27B dense) is the only model with a perfect missions score (102/102 vs 99), so it is a candidate for the planner, at about 3.7 s per mission call.
gpt-oss:20b is the only plausible second choice, but its forced reasoning makes it too slow for interactive chat.

Logs: `/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chat-agent/a2f2272b-f33f-4143-8105-0ec4edc68e6a/scratchpad/evals/` (temporary).
