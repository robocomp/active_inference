# Proposed paragraph — real-robot corroboration of the corner-channel finding

Target location: §V-E, immediately after the "How much of this is one run" paragraph (the one
citing the two simulated tours, ratios 5.6 and 8.7). Style/format matched to the surrounding text.
Anonymized per the run's own notes: no robot name, laboratory, institution or building.

---

## Proposed LaTeX paragraph

```latex
\emph{A real-robot run reproduces the same shape.} The argument above rests on simulated tours; we
also ran the same pipeline and the same script, unmodified, on a differential-drive robot with a 3D
LiDAR in an apartment-scale indoor space, over \SI{295}{\meter} across three chained recording
sessions (a session boundary forces a brief re-localization transient, which the stationary-stretch
detector correctly excludes rather than mistaking for rest). Over \num{6} stationary stretches and
\num{46} corner-stretch pairs, the repeating part has a median of \SI{0.0624}{\meter} against
\SI{0.0046}{\meter} that scatters, a ratio of \num{13.4}, with the bias exceeding the scatter in
\SI{98}{\percent} of pairs---inside the \num{5.6}--\num{8.7} band the simulated tours already
established. The viewpoint test agrees on the more important point: across \num{10} corners seen
from three or more separate stretches, the spread \emph{between} stretches
(\SI{0.0459}{\meter}) is \num{9.9}$\times$ the spread \emph{within} one (\SI{0.0046}{\meter}), ruling
out a map offset on hardware the same way it does in simulation. One methodological note the run
itself surfaced: with only three stretches available, the between-stretch statistic can appear
small simply because too few genuinely different viewpoints were sampled, not because the error is
map-attached---the fix is deliberately visiting distinct viewpoints, not more distance.
\end{quote}
```

(Drop the stray `\end{quote}` if the surrounding block isn't inside a `quote`/`quotation` environment
— copied from the working draft as a placeholder for wherever this lands structurally; check against
the actual environment the neighbouring paragraphs use before pasting.)

---

## Suggested table

```latex
\begin{table}[t]
\centering
\caption{Corner-channel repeat/scatter split, simulated vs.\ real.}
\label{tab:real-tour}
\begin{tabular}{lccccc}
\toprule
Tour & Stretches & Pairs & Repeat & Scatter & Ratio \\
\midrule
Simulated (tour 1) & --  & -- & 0.0403\,m & 0.0063\,m & 5.6$\times$ \\
Simulated (tour 2, \S\ref{...}) & 34 & 386 & 0.0574\,m & 0.0066\,m & 8.7$\times$ \\
Real robot (this work) & 6 & 46 & 0.0624\,m & 0.0046\,m & 13.4$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

Note: "Simulated (tour 1)" is the 5.6x figure the existing text already cites in prose without a
stretch/pair count in this document — fill those two cells from wherever that run's own log lives, or
drop the row and keep only the tour-2/real comparison if that count is not readily at hand before the
deadline.

## Suggested figure

`figures/fig_repeat_vs_scatter.pdf` (also `.png` for a quick look, both copied from
`real_apartment_COMBINADO/`) — log-log scatter
of scatter (x) vs.\ repeat (y) for all 46 corner-stretch pairs, with the $y=x$ reference line. 45 of
46 points (98%) sit above it, i.e.\ the bias exceeds the noise for that pair. Regenerate with
`real_apartment_COMBINADO/pairs.csv` (per-pair bias/noise, dumped via the instrumented copy of the
script, `split_and_nis_dump_pairs.py`, `DUMP_PAIRS_TO=... python3 split_and_nis_dump_pairs.py ...`)
if the underlying data changes.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{fig/fig_repeat_vs_scatter.pdf}
\caption{Real-tour corner channel: each point is one corner observed in one stationary stretch.
Points above the dashed $y=x$ line have a repeatable bias larger than their own scatter (45/46,
\SI{98}{\percent}).}
\label{fig:real-tour-scatter}
\end{figure}
```

## Numbers used, traceable to source

| Quantity | Value | Source |
|---|---|---|
| Distance | 295.10 m | sum of 3 sessions, `pose_trace.csv` per session |
| Sessions chained | 3 | `real_apartment_0914_2006/`, `..._sesion2/`, `..._sesion3/` |
| Stationary stretches | 6 | `RESULTS_combinado_3sesiones.txt` |
| Corner-stretch pairs | 46 | idem |
| Repeat (bias) median | 0.0624 m | idem |
| Scatter (noise) median | 0.0046 m | idem |
| Ratio | 13.4x | idem |
| Bias > scatter | 98% of pairs | idem |
| Corners in ≥3 stretches | 10 | idem |
| Within-stretch scatter | 0.0046 m | idem |
| Between-stretch spread | 0.0459 m | idem |
| Viewpoint ratio | 9.9x | idem |
| Verdict | viewpoint-attached | idem |

Compare against the existing simulated-tour numbers already in the paper (§V-E): 8.7x / 6.0x (run
`final_0914`) and the mentioned second tour at 5.6x. All four numbers (5.6, 8.7, 9.9, 13.4) tell the
same qualitative story; only the magnitude moves between runs, which the paper's own "how much of
this is one run" paragraph already anticipates and argues is expected.

## Caveat to flag to whoever integrates this (do not silently omit)

The 6 stretches / 46 pairs sample is much thinner than the simulated tour's 34/386. The headline
ratio (13.4x) and the bias-dominance percentage (98%) are on a similar footing to the simulated
numbers, but the viewpoint test's between-stretch spread is estimated from only 10 corners across 6
viewpoints — enough to flip sign once (see `EXPERIMENT_LOG.md`, sessions 1+2 alone gave the opposite
verdict from too few viewpoints) and worth stating as a smaller-N real-hardware corroboration rather
than a replacement for the simulated result.

## Not included, and why

- No rotation-band breakdown (NIS/dof per angular-rate band): `layout_trace.csv` was not generated by
  this run, and that breakdown needs it. The two headline quantities (repeat/scatter split, viewpoint
  test) do not require it — verified by running the unmodified script on this exact data.
- No robot/lab identity, per the run's own anonymization note.
