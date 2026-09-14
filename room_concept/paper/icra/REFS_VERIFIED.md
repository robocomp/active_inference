# Reference check — 2026-09-14

All **36** cited entries checked against the publisher record: existence, author list, venue, year.
Sources were arXiv abs pages, proceedings sites (NeurIPS/ECVA/CVF/IEEE Xplore/Springer/ACM DL) and DOIs.
No cited work turned out not to exist. Nine entries carried wrong or missing data; those are below.

## Corrections made

| key | was | is | evidence |
|---|---|---|---|
| `hu2021mvlayoutnet` | year 2021, no venue | **ACM MM 2022** | ACM DL 10.1145/3503161.3548071 — the 2021 came from the arXiv stamp, the paper appeared a year later |
| `su2022gprnet` | year 2022, no venue | **CVPRW 2023** (OmniCV) | CVF open access, `Su_GPR-Net_..._CVPRW_2023_paper.pdf` |
| `luperto2020exploration` | year 2020, no venue | **AAMAS 2021**, pp. 836–843 | the 2020 arXiv version was the ARMS 2020 workshop; the conference paper is AAMAS 2021 |
| `howardjenkins2019unconstrained` | year 2019, no venue | **ACCV 2018** | authors' own page names it `howardjenkins_etal_accv2018`; arXiv was posted after the conference |
| `fayyazsanaviu2rle` | **no year at all** | **CVPRW 2023** | CVF open access, CIVILS workshop |
| `ahmed2023aslamreview` | `@misc`, venue unknown | **Sensors 23(19):8097, 2023** | arXiv journal_ref; the bib note had misread it as "Preprints" |
| `zou2021manhattan` | author `{Zou, Chuhang and others}` | full 8-author list, **IJCV 129(5):1410–1431** | doi 10.1007/s11263-020-01426-8 |
| `placed2023survey` | no volume/pages | **T-RO 39(3):1686–1705** | doi 10.1109/TRO.2023.3248510 |
| `solarte2022dfpe` | `@misc`, no venue | **RA-L 7(4):8746–8753, 2022** | arXiv journal_ref "IEEE RA-L 2022" |

Venue filled in (was a bare year, no other error): `liu2018floornet` ECCV 2018 · `chenfloorsp` ICCV 2019 ·
`chen2022heat` CVPR 2022 · `yue2023roomformer` CVPR 2023 · `liu2024polyroom` ECCV 2024 · `xufrinet` ECCV 2024 ·
`phung2026raster2seq` SIGGRAPH 2026 · `hutchcroftcovispose` ECCV 2022 · `wangpsmnet` CVPR 2022 ·
`lambertsalve` ECCV 2022 · `avetisyan2024scenescript` ECCV 2024 · `poggiuncertainty` CVPR 2020 ·
`ericson2024floorist` RA-L 2024 · `ho2025mapex` ICRA 2025 pp. 13074–13080 · `wang2025cogniplan` CoRL 2025 ·
`solarte2024raycasting` ECCV 2024.

Claimed venue **confirmed correct**, no change: `su2023slibonet` NeurIPS 2023 · `chen2023polydiffuse` NeurIPS 2023 ·
`liu2025cage` NeurIPS 2025 · `solarte2022mlc` NeurIPS 2022 · `bieri2025houselayout3d` NeurIPS 2025 D&B ·
`chang2017matterport3d` 3DV 2017 · `placed2022enough` IFAC IAV 2022 · `liu2024point2building` ISPRS J. 215 ·
`jarzabek2022uncertainty` ISPRS Archives XLIII-B2-2022.

Genuinely unpublished, and now labelled `arXiv:NNNN.NNNNN` instead of printing a bare year that looks
like a dropped field: `ye2025floorsam` (no venue on arXiv), `luperto2024mapcompleteness` ("under review
at IEEE RA-L").

## ★ Two BibTeX traps this file hit, both SILENT in the LaTeX build

1. **`%` is not a comment character inside an entry block.** 29 entries carried a
   `% UNVERIFIED — note = {...}` line, and BibTeX answered each with "You're missing a field name /
   I'm skipping whatever remains of this entry". The fields happened to sit above the comment so the
   output survived, but the parser was being handed a broken file 29 times and nothing in the `.tex`
   run said so — only `main.blg` did.
2. **BibTeX scans for the at-sign everywhere, comments included.** A comment naming a key after an
   at-sign was parsed as a second, empty copy of that entry ("Repeated entry"). Writing a comment
   *warning about* this reintroduced it, because the warning contained the character.

`main.blg` is now clean: no errors, no warnings.

## How to re-check

`grep -c "^@" refs.bib` for the count; the authoritative cited list is `\citation{...}` in `main.aux`,
**not** a grep for `\cite{` in `main.tex` — the grep missed five keys here and they were nearly
shipped unverified.
