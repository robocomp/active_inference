# Regression of the revised standpoint rule (commit bbcff1e3, 2026-10-11)

The revised rule (sec:la-explore, the U-room fix) re-run on the two earlier campaigns, same rooms, same starts, one attempt each.

| | before (rule of 22d763b9 / bdb6095) | after (bbcff1e3) |
|---|---|---|
| **Furnished synthetic, 10 rooms** | campaign v5: 10/10 | 10/10 |
| refined IoU median / worst | .994 / .990 | .994 / .990 |
| promoted IoU median / worst | .984 / .954 | .988 / .967 |
| time to promotion median | 25 s | 39 s |
| **Apartment, 10 starts** | campaign v4: 10/10 (start 9 on its repeat) | 10/10, first attempt |
| refined IoU median / worst | .977 / .973 | .978 / .973 |
| promoted IoU median / worst | .966 / .898 | .968 / .960 |
| time to promotion median / worst | 41 s / 184 s | 43 s / 63 s |
| promoted IoU spread (sd) | .022 | .004 |

- Refined accuracy is unchanged in both sets. What the rule changes is the moment of promotion: rooms that were promoted before their last notch had been seen (synthetic rooms 2 and 7, apartment start 0) now wait for it, so the promoted layout is better and much more consistent.
- The price is exploration: in the furnished rooms 14 s more median time to promotion and about 3 m more path; in the apartment almost none (42.6 vs 40.9 s), and the slowest start drops from 184 s to 63 s.
- One small loss: furnished L room 1, refined .999 -> .995 (a notch around a cabinet).
- Furnished T room 8 promoted the full T (.998). The early promotion of the edge case did not occur, but it did not occur in the previous full campaigns either, so this neither confirms nor rules out that the rule cures it.
- These numbers can replace the earlier tables for both sets (same protocol, newer build), or be quoted as a regression check alongside them.

Files: `apartment/` and `furnished/` each hold `results_shape.csv`, `results_table.tex` (labels `tab:estimation-campaign-bbcff1e`, `tab:estimation-synthetic-furnished-bbcff1e`), `stats.txt`, `layouts_panel_furniture.png`.
