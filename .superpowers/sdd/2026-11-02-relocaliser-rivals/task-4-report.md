# Task 4 report: background proposer
Files: room_concept/src/reloc_proposer.{h,cpp}, room_concept/tools/reloc_proposer_test.cpp (build lines in header).
Tests: one_in_flight, stale_epoch, never_blocks (200 ms fake; submit+submit+poll 1 us), real-search smoke on empty job. Pass plain and with -fsanitize=thread (no reports).
Deviations: submit() also takes the room polygon (search() needs it; raster has none). `around` seeds the search AND enters Params::prior_modes with sampling_cov (weight 1) - flag: this adds a continuity term to mode weights; drop that line if Task 5 wants a pure seed. submit refuses only while RUNNING; an unpolled older result is replaced by the new job. RED step not run separately (test and impl written together).
Not added to CMake. Destructor requests stop; joins after an in-flight search finishes.

## Fix round 1
Removed the prior_modes push (incumbent prior belongs to birth_lambda); sampling_cov unused, commented in .h/.cpp. Seeds still {around}. Worker wraps search in try/catch (empty Result, running_ cleared) and checks stop_requested after the wait. New tests: no_incumbent_prior (prior_modes empty at the search, 1 seed), exception_clears_running. Plain and TSAN runs: all passed, no TSAN reports.
