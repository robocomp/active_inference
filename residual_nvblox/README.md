# residual_nvblox

Experimental variant of `residual_concept` whose occupancy evidence comes from **NVIDIA nvblox** (GPU projective
LiDAR integration, 5 cm voxels) instead of the in-house DDA ray carving. Everything downstream of the grid columns —
explainers (floor, ceiling, walls, robot, modelled objects), speckle/component filters, belief field and the DSR
publication — is the `residual_concept` code, unchanged.

**It replaces `residual_concept` in the graph**: same `[Agent] name = "residual_concept"` and `id = 14`, so consumers
(rerun_logger, controller, netmon) see no difference. **Never run both at once** (id collision = CRDT actor clash).

## Layout

| Path | What |
|---|---|
| `nvblox_bridge/` | `libnvblox_bridge.so`: the only CUDA/nvblox code. Its header exposes no nvblox, Eigen or CUDA type, so the agent never sees nvblox's Eigen (see the ABI note in `src/CMakeLists.txt`). |
| `nvblox_bridge/nvblox_lidar_rings.patch` | Patch to nvblox v0.0.10 (required): per-ring elevation table + `RingFraction` nearest-pixel reach along elevation. |
| `src/` | Agent. nvblox-specific: `residual_nvblox_map.*` (`rc::NvbloxMap`: mapper, per-sweep integration, column sampling, live window / 3D dump), `residual_nvblox_view.*` (the window), `residual_config.*` and `OccupancyGrid::ingest_columns()`. `specificworker.cpp` keeps only the nvblox path of `residual_concept`'s compute (the stock ray-carving / ZED / floor-plane / tracker methods are gone). The fitter, scene graph, tracker, ZED ingestor and floor-plane fit of `residual_concept` are not in this agent. |
| `etc/config.toml` | Config: `[Publish]` (rate, field low-pass), `[Readout]` (explainers), `[Grid]`, `[Lidar]`, `[Nvblox]`, `[SelfMask]`, `[Diag]`. Every key in it is read, and every key read is in it. |
| `etc/self_mask_bpearl.bin` | Bpearl self-mask calibrated on shadow (see below). |
| `tools/` | `nvblox_rerun_view.py` (3D viewer), `cloud_deleted.py` (pre/post cloud plot), `grid_dynamics_probe.cpp` (stock grid bench). |
| `generated/` | robocompdsl output, still named `residual_concept` (copied from it). The binary is renamed to `residual_nvblox` in the top `CMakeLists.txt`. |

## Dependencies

- Everything `residual_concept` needs (RoboComp, DSR/cortex, Qt6, `../common` ai_common libraries).
- CUDA 13.2, gflags, glog.
- nvblox **v0.0.10, commit `24eee49`**, with `nvblox_bridge/nvblox_lidar_rings.patch` applied, installed without sudo:

  ```bash
  git clone https://github.com/nvidia-isaac/nvblox.git ~/software/nvblox && cd ~/software/nvblox
  git checkout 24eee49
  git apply ~/robocomp/components/active_inference/residual_nvblox/nvblox_bridge/nvblox_lidar_rings.patch
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DCMAKE_INSTALL_PREFIX=$HOME/software/nvblox/install \
        -DBUILD_TESTING=OFF -DBUILD_EXPERIMENTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_PYTORCH_WRAPPER=OFF -DBUILD_RENDERER=OFF
  make -C build -j8 nvblox_lib && cmake --install build
  ```

  `120` = RTX 5080 Laptop (shadow); `89` = RTX 4090. Building all targets crashes nvcc on the `esdf_query` example —
  only `nvblox_lib` is needed.
- Python tools: `rerun-sdk` 0.37, numpy, matplotlib (`~/software/pip_venv` on shadow).

## Build

The bridge first (the agent's CMake fails with a hint if `nvblox_bridge/lib/libnvblox_bridge.so` is missing):

```bash
cmake -S nvblox_bridge -B nvblox_bridge/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
      [-DNVBLOX_PREFIX=$HOME/software/nvblox/install] [-DCMAKE_CUDA_ARCHITECTURES=89]
cmake --build nvblox_bridge/build -j
cmake -B build && make -C build -j8          # → bin/residual_nvblox
```

The bridge defaults to Release; a non-Release build makes sampling ~45× slower (90 ms instead of 2).

## Run

```bash
./run_capped.sh                          # bin/residual_nvblox etc/config.toml in a memory-capped cgroup
```

or through netmon with `utils/cognitive_stabilization_nvblox.toml` (component `residual nvblox`).

## First-time setup (new robot or new install)

In this order — each step assumes the previous one:

1. **Dependencies**: CUDA, then nvblox with the patch (see *Dependencies*). Pick `CMAKE_CUDA_ARCHITECTURES` for the GPU.
2. **Build** the bridge, then the agent (see *Build*).
3. **LiDAR drivers** (`lidar3d_dds/etc/config_bpearl_shadow.toml`, `config_helios_flip.toml`). The agent expects the
   RAW clouds — it handles the floor and the robot body itself:
   - `[Floor] z = -1000.0` and `[Top] z = 10000.0` on both (no floor / top crop; the old crop was 110 / 2100 mm).
   - Bpearl: `[MeshFilter] enabled = false`, `[Footprint] radius = 0.0`, `[Skirt] radius = 0.0`. If the footprint /
     skirt crop is kept instead, mirror its numbers in `Nvblox.BpearlBlind*` / `Nvblox.BpearlSkirt*`.
   - Bpearl in single-return mode (dual return doubles the mixed points along the body edge).
   - Helios keeps `[MeshFilter] enabled = true` (it sees no body of its own; with the filter off it is harmless too).
4. **Extrinsics — the floor must come out at z = 0.** FloorPush / FloorFill work with bands of 4–6 cm over the room
   plane z = 0, so a mount height or tilt error of a couple of cm turns the floor into obstacles (or hides low ones).
   The agent places each cloud with the DSR RT chain `Shadow → body → <sensor>` from robot_concept's JSON (on shadow
   `robot_concept/shadow_helios_flip.json`; DSR euler convention R = Rx·Ry·Rz). Check it on a flat floor with the
   robot still and ≥ 2 m free around it:
   - set `Nvblox.DumpClouds = true`, start the agent, wait a few seconds;
   - side view: `~/software/pip_venv/bin/python tools/cloud_deleted.py --out side.png --radius 3` — the floor points
     (lower plot, z vs range) must lie flat on z = 0 at every range;
   - number: fit a plane to the sensor's floor points in the room frame (0.8–3 m, |z| < 0.15 m):

     ```python
     import numpy as np
     b = open("/dev/shm/nvblox_cloud_pre_1.bin", "rb").read()   # 1 = bpearl, 0 = helios
     T = np.frombuffer(b, np.float32, 12).reshape(3, 4); n = np.frombuffer(b, np.int32, 1, 48)[0]
     P = np.frombuffer(b, np.float32, 3 * n, 52).reshape(-1, 3) @ T[:, :3].T + T[:, 3]   # room frame
     r = np.hypot(*(P[:, :2] - T[:2, 3]).T)
     F = P[(r > 0.8) & (r < 3) & (np.abs(P[:, 2]) < 0.15)]
     a, b_, c = np.linalg.lstsq(np.c_[F[:, :2], np.ones(len(F))], F[:, 2], rcond=None)[0]
     print(f"offset {c*100:.1f} cm, tilt x {np.degrees(a):.2f} deg, tilt y {np.degrees(b_):.2f} deg, {len(F)} pts")
     ```

   - correct the sensor's `translation` z by −offset and its `rotation` (rx, ry) by the tilt, restart robot_concept
     (the fleet) and repeat until offset < 1 cm and tilt < 0.2°. Measured on shadow 2026-10-19: bpearl z 0.67 → 0.745,
     rotation (0, π, 0) → (0.0105, 3.16079, 0); body z 0.045 → 0.05 (the casing stands 5 cm off the floor).
     The bpearl is mounted upside down (ry ≈ π), so a tilt in the room maps to its rx/ry with flipped signs: apply a
     small step and re-measure to confirm the direction.
     The helios grazes the floor at range, so its fit is less reliable; check it, but trust the bpearl.
   - `Nvblox.DumpClouds = false` again.
5. **Self-mask** of the bpearl (see *Self-mask calibration*) — AFTER the extrinsics: the mask is learnt against the
   floor, so redo it whenever the extrinsics, the body or the driver filtering change. Without
   `etc/self_mask_bpearl.bin` the agent says so in the log and runs unmasked (the body's edge returns become obstacles
   around the robot).
6. **First run** through netmon (`utils/cognitive_stabilization_nvblox.toml`). Check in the log: `[self-mask] bpearl:
   N pixels masked`, `[nvblox] … sweeps helios=… bpearl=…` growing, `[map-time]` at a few ms, and in the live window
   no latched cells on open floor or around the robot.

## Per-sweep pipeline

For each LiDAR (helios, bpearl), in `nvb::Mapper::integrate_lidar`:

1. **Self-mask** (bpearl): range-image pixels (ring × azimuth) that see the robot's own body. Inside
   `SelfMask.ClearRadiusM` a masked pixel keeps only floor returns (pushed under the floor, they still clear);
   everything else in it is dropped. Outside the radius the mask does nothing.
2. **FloorPush**: by height over the floor and horizontal range r:
   `z < FloorClearZ0M + FloorClearSlope·r` → certain floor, moved `FloorFillBelowM` under the floor along its ray
   (clears, never marks); else a **wall foot** (a ring neighbour at the same horizontal range ±5 cm and > 2 cm higher)
   is kept; else `z < FloorPushZ0M + FloorPushSlope·r` → ambiguous, dropped (neither marks nor clears).
3. **FloorFill**: an empty pixel whose ray meets the floor within `MaxRangeM` gets a virtual return under the floor.
   Pixels with a real return in the ring above or below, or masked, are skipped (they would cut through obstacles).
4. nvblox projective occupancy update (nearest-pixel fallback with `NearestRayDistVox` / `RingFraction`).

Then each cycle the voxel columns are sampled at the grid's 64 z-bins of every cell, from `ReadMinZM` up
(`OccupancyGrid::ingest_columns`), and the stock latch/release, explainers and publication run on that.
The 2D ESDF (`EsdfZMinM`..`EsdfZMaxM`) is computed only for the live window / 3D dump; it is not published.

The floor is the room plane z = 0, so the floor bands absorb pose, tilt and extrinsic error. They were taken to the
minimum the noise allows (σ ≈ 1–4 cm); going lower produced small artefacts.

## Self-mask calibration

Needed after any change to the body, the sensor mount or the bpearl driver filtering (MeshFilter, returns mode):

1. Robot still, ≥ `SelfMask.ClearRadiusM` (1 m) of free floor all around.
2. `SelfMask.Calibrate = true`, start the agent: it learns over `Sweeps` (100) sweeps, maps nothing meanwhile, and
   writes `etc/self_mask_<sensor>.bin`.
3. Set `Calibrate = false` again.

Only the bpearl is masked (`Sensors = "bpearl"`): the helios mask learned walls in a narrow corridor.

## Diagnostics

| Config | Output | Reader |
|---|---|---|
| `Nvblox.Display` | live window: nvblox P, height, ESDF, residual | — |
| `Nvblox.Dump3D` | `/dev/shm/residual_nvblox_3d.bin` every display refresh | `DISPLAY=:1 ~/software/pip_venv/bin/python tools/nvblox_rerun_view.py` (Rerun on port 9881; 9877 is rerun_logger's) |
| `Nvblox.DumpClouds` | `/dev/shm/nvblox_cloud_{pre,post}_<id>.bin` every ~2 s | `~/software/pip_venv/bin/python tools/cloud_deleted.py --out x.png` (kept / pushed under floor / deleted in red) |

`Diag.CsvLogging = true` turns on the diagnostic CSVs under `etc/` (grid_diag, floor_diag, perf_diag,
residual_cells / residual_occ_debug every `CellDumpEveryN` cycles, births, releases; previous runs rotate to
`etc/archive/`). Off by default: at `CellDumpEveryN = 25` they grow by GBs per hour. All are git-ignored.

## Known limitations / open work

- **Helios ring table disabled** (`HeliosRingElevDeg = ""`): with the manufacturer's per-ring elevations the map got
  worse; the points' frame/order has to be checked first. The uniform 32-row model is used.
- **Floor reference is z = 0**: a per-ring floor model would let the floor bands go lower.
- The bpearl driver runs without MeshFilter, single return and without footprint/skirt crop, so the agent sees the
  raw body; other consumers of the same stream see it too.
- `src/` still shares `residual_clusterer.*` (polygon helpers), `residual_occupancy_grid.*` and
  `residual_lidar_ingestor.*` with `residual_concept`.
- `generated/` keeps the `residual_concept` names until the component is regenerated from `residual_nvblox.cdsl`
  (then drop the `OUTPUT_NAME` override in `CMakeLists.txt`).
