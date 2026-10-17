# Room-layout estimation campaigns

Everything needed to run, grade and tabulate the room-layout estimation experiments, in the apartment and in synthetic worlds.

## Folder layout
- `scripts/`: the shared tools.
- `apartment/`: the apartment world (`webots-shadow/worlds/piso/piso.wbt`).
  - `furniture.json`: furniture footprints extracted from the world.
  - `campaign_v4/`: the campaign run on bdb6095.
- `synthetic/`: generated worlds of 10 rooms placed 30 m apart.
  - `campaign_v1_empty/`: empty Manhattan rooms.
  - `campaign_v2_LT/`: rooms 1 and 6 are L-shaped, rooms 3 and 8 T-shaped.
  - `rerun_r2/`: the room-2 refiner diagnosis.

A campaign directory holds:
- `starts.csv`: columns id, x, y, theta.
- `rooms.json`: the ground-truth polygons in world coordinates.
- `run_<id>/`: per-run logs. These are git-ignored.
- The results.

## Running a campaign
1. **Synthetic only — generate the world:**
   ```
   python3 scripts/make_world.py synthetic/<campaign> ~/robocomp/components/webots-shadow/worlds/piso/piso.wbt
   ```
   This writes `rooms.json`, `starts.csv` and `synthetic_rooms.wbt`. Copy the `.wbt` to `webots-shadow/worlds/piso/` and load it in Webots.

2. **Run:**
   ```
   scripts/run_campaign.sh <campaign_dir> [first] [last]
   ```
   For each start it teleports the robot (`tp.py`), starts room_concept, residual and controller, and waits for promotion plus 60 s of refinement. Then it archives the logs and `refine_input.bin`.

3. **Grade:**
   - Raw grading, with ground truth aligned to the estimate by an SVD fit of the whole trajectory:
     - apartment: `scripts/grade_campaign.py <dir>`
     - synthetic: `scripts/grade_synthetic.py <dir>`
     Both write `results.csv` and `layouts_panel.png`.
   - Shape and anchored IoU, the numbers to quote:
     ```
     scripts/regrade_anchor.py <dir>
     ```
     It writes `results_anchor.csv` and a panel in the shape-aligned frame. Set `FURNITURE=apartment/furniture.json` to get `layouts_panel_furniture.png` instead.
   - Then `scripts/merge_shape.py <dir>`, which writes `results_shape.csv`.

4. **Statistics and tables:**
   - `scripts/stats.py <dir>/results_shape.csv`
   - `scripts/make_table_apartment.py <dir>/results_shape.csv <dir>/starts.csv <out.tex>`
   - `scripts/make_table_synthetic.py <dir>`

5. **Furniture footprints:**
   ```
   scripts/furniture.py <world.wbt> <out.json>
   ```

## Replaying the refiner offline
Each run stores the refiner's exact inputs in `run_<id>/refine_input.bin`, written by `step_layout_refiner`. A replay harness reads that file and calls `rc::refine::LayoutRefiner::propose()` frame by frame. A harness of this kind was used for the room-2 diagnosis, but it is not in this folder.
