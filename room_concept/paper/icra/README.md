# ICRA submission — calibrated uncertainty for active room-layout estimation

`main.tex` uses `IEEEtran` (conference, letterpaper, 10pt) — the current IEEE conference format.
`ieeeconf.cls` is also here in case the CFP asks for the older RAS/PaperCept template; switch the
`\documentclass` line only.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

⚠ Every number in §V is traceable. Sources:
- 594-room accuracy + calibration: one run, `WS_NO7=1 WS_ROOMS=594 WS_LAYOUTS=datasets/matterport_layout/corpus594.txt WS_POLY_ROOM=…`, analysed by the Hungarian matcher. See `datasets/matterport_layout/RESULTS_594.md`.
- Real-sensor split: `datasets/mp3d_fpe/RESULTS.md`.
- Robot tour: `tmp/TOUR_CALIBRATION.md`, raw in `tmp/corner_probe.csv` (one row per candidate).
Do not quote a number that is not in one of those four files.

## State

Compiles (3 pp. text). **Missing before submission:** `refs.bib` is empty (every citation in §II is
currently a prose placeholder with no `\cite`); no figures; author block anonymous.

## Figures the argument needs (in priority order)

1. **The calibration curve** — declared $\sigma$ (x) vs. realised error (y), one point per matched
   corner, with the $y=x$ line and the same plot before/after $\sigma_m$. This is the paper's claim
   in one image, and it is the figure a reviewer will look at first. Data: the Hungarian matcher.
2. **The motion-banded $\nis$** — $|\Delta\theta|$ per frame vs. $\nis$, showing the channel is worst
   at rest. Data: `tmp/corner_probe.csv`.
3. **Room mosaic** — truth vs. estimate on a sample of the 594, already generated as
   `datasets/matterport_layout/mosaic_cells_120.html`; needs a vector export.
4. **System/teaser** — the layout with its per-corner $\sigma$ discs and per-edge bands, published vs.
   withheld corners in different colour. The live 2-D canvas already draws this.

## Open honesty items, deliberately in the text

- §V-A observations are simulated; §V-C validates the noise model, not the pipeline.
- The negative dataset result is stated as a limitation, not hidden.
- The pose-covariance defect is reported as measured-but-unfixed (`TODO_PER_WALL_OFFSET.md`).
