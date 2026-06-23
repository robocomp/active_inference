# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **⚠ Working agreement: never `git commit` (or push) on your own. Always ask first and wait for explicit approval before committing.**

## What This Component Is

`kinova_controller` is a RoboComp/CORTEX agent that runs an **active-inference (EFE-gradient) controller** for a Kinova Gen3 7-DoF arm + Robotiq 2F-85 gripper. It treats arm motion as gradient descent on a Gaussian preference distribution over end-effector position (plus an orientation alignment term), and drives the arm by sending joint velocity commands through the `KinovaArm` Ice proxy (typically backed by a Webots bridge).

The agent is part of the larger `active_inference` workspace (`../`), which contains sibling agents (`robot_concept`, `room_concept`, etc.) that share a DSR graph. This component currently focuses on the arm control loop; DSR signal slots are stubbed.

## Build & Run

```bash
cmake -B build && make -C build -j12     # configure + compile
bin/kinova_controller etc/config         # run (copy etc/config to etc/yourConfig to override locally)
```

- `cbuild` is the project-preferred VS Code build wrapper (CMake Tools extension). If it's available in the editor context, use it; from a plain terminal the `cmake -B build && make -C build` form above is the working equivalent.
- The `generated/` subtree is regenerated from `kinova_controller.cdsl` by `robocompdsl` — never edit those files by hand.
- All `src/*.cpp` files must be in the `SOURCES` list in `src/CMakeLists.txt` so AUTOMOC sees their paired headers and generates `moc_*.cpp` for any class with `Q_OBJECT`. Do NOT `#include "*.cpp"` from another `.cpp` as a shortcut — that bypasses AUTOMOC and leaves Qt vtables and signals unresolved at link time.

## Architecture (the parts that need cross-file reading)

### Phase machine inside `Phase::ActiveEFE`
`SpecificWorker` has a two-level state machine:
1. **Outer phases** (`Phase` enum in `specificworker.h:118`): `SendingRestPose` → `Homing` → `ActiveEFE`. The agent commands its own homing rather than trusting the bridge, so behavior is independent of pre-homing.
2. **Inner cycle** (`CycleMode` enum, `specificworker.h:129`): `ReachingTarget` → `SendingReturn` → `Returning` → `Done`. Runs `CYCLES_MAX` random table-surface targets, returning to rest pose between each. Targets are sampled in the arm-base frame from `(x∈[0.20,0.50], y∈[−0.15,0.15], z=0.10)`.

Continuous joints (4 of the 7) accumulate revolutions across runs, so use `angular_distance()` in `specificworker.cpp` for tolerance checks — raw `|a − b|` will be wrong when the encoder reports e.g. +8.6 rad.

### EFE control (`efe_gradient.{h,cpp}`)
Two layers. The preference (EFE) layer turns the preferred tool pose into a desired EE **twist** `[v_des; ω_des]` — a precision-weighted position pull plus an orientation term (single-axis approach alignment when `gain_secondary == 0`, or a full-frame SO(3) geodesic when `> 0`; `ω_des` capped at `omega_max`). A **coordinated 6-DOF damped-least-squares resolved-rate** step (Corke RVC §8.4) then resolves that twist to joint velocities:

`q̇ = J6ᵀ (J6 J6ᵀ + λ²I₆)⁻¹ · [v_des; ω_des]`   (`dls_lambda ≈ 0.05`)

Note `J6ᵀ(J6J6ᵀ+λ²I)⁻¹` is the damped **pseudo-inverse** `J⁺_damped`, i.e. this is resolved-rate, not a Jacobian-transpose gradient. Key choices:
- **Why not Jacobian-transpose.** The earlier law was the transpose-only gradient `q̇ = −α·Jᵀ·diag(C_pos)·(f−x*)` + a separate orientation term. Solving position and orientation as two independent Jᵀ pulls made them **fight** — the orientation joints dragged the EE off the position target — which is why it needed the anisotropic-precision hack (`C_pos = {4,4,8}`) and careful gain tuning, and still chattered at 100 ms. The DLS solve descends both through **one operator** so they move together (consistent least-squares step → faster convergence in EE space), and removes the tuning hack.
- `λ` doubles as Corke's velocity-effort penalty and the singular-direction regulariser, restoring the singularity-robustness a plain inverse lacks (the one property the transpose law had for free).
- The redundant 7th DOF is resolved in the **null space of the same operator** (`N = I − J6ᵀQ6⁻¹J6`, reusing the LDLT factorization): manipulability (μ) ascent + elbow placement, which provably can't disturb the tool pose. Soft mast/table repulsions are added before the velocity clip.
- Velocity clip `max_joint_vel = 0.87` matches the Webots proto `maxVelocity = 0.8727 rad/s`.

Full derivation (EFE gradient → DLS natural gradient → the coordinated 6-DoF solve that replaced the transpose law → null-space + obstacle terms → the per-cycle law) is in **`EFE_CONTROLLER_MATH.md` §4** — keep that and this summary in sync.

### Kinematics (`kinematics.{h,cpp}`)
Pinocchio wrapper around the URDF at `gen3_robotiq_2f_85-mod.urdf`. Important: continuous joints use **two** config entries each (cos, sin), so `model.nq = 11` but `model.nv = 7`. This class converts between the 7-vector of arm angles the proxy gives us and Pinocchio's 11-vector configuration. `tool_frame` is the EE frame.

**Critical CMake constraint** (`src/CMakeLists.txt`): do NOT add `-march=native`. Pinocchio's `.so` from robotpkg is built with default Eigen alignment; `-march=native` in our TUs would promote Eigen alignment (AVX2/AVX-512) and shift `pinocchio::Data`'s layout relative to the `.so`, breaking `model.check(data)` at runtime.

### Pinocchio install (one-time)
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub noble robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
sudo apt update && sudo apt install -y robotpkg-pinocchio
```
CMake picks it up via `list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")` in `src/CMakeLists.txt`.

### Visualization
`ArmBeliefViewer3D` is a custom Qt widget (`src/arm_belief_viewer_3d.{h,cpp}`) registered with the DSR graph viewer dock at startup. Mesh root is hardcoded to `/home/pbustos/robocomp/components/webots-kinova/protos/kinova_arm_meshes`. The viewer is fed the current arm joint angles, link poses from FK, the current target, and the EE position each compute cycle.

### Shutdown safety
`~SpecificWorker` sends a zero `TJointSpeeds` to the proxy on exit — without it the bridge keeps holding the last commanded q̇ and the arm drifts in Webots after Ctrl+C. The catch is intentional: the proxy may already be unreachable on shutdown.

## RoboComp Framework Context

- Components inherit from `GenericWorker` (in `generated/`) and implement an Ice interface declared in `*.cdsl` (here: `requires KinovaArm`).
- Lifecycle = state machine: `Initialize` (once) → `Compute` (cyclic, period from `Period.Compute` in `etc/config`) → `Emergency`/`Restore` on faults.
- Configuration via `configLoader.get<T>("key")`. Periods are `getPeriod("Compute")` / `setPeriod("Compute", n)`.
- `#define HIBERNATION_ENABLED` in `specificworker.h` reduces period to 500 ms after 5 s of no method calls (currently commented out here).
- This workspace shares state across agents via the CORTEX DSR distributed graph; `options dsr` in the `.cdsl` provides graph viewers and the `G` pointer in `GenericWorker`. This agent currently doesn't read or write the graph in `compute()` — slots are stubs.

## Frames (see `FRAMES.md` and `../FRAMES.md`)

- All control math here is in the **arm-base frame**: `+X` forward of base, `+Y` left, `+Z` up. Arm base sits on the desk, so `z = 0` is the table surface and the test target `(0.4, 0, 0.1)` is 10 cm above the table, 40 cm in front of the base.
- Anything reading from `room_concept` / `robot_concept` siblings will be in the RoboComp body frame (`+X` right, `+Y` forward, `+Z` up) or the room frame — be explicit when crossing component boundaries.

## Code Conventions (workspace-wide)

- C++23, Eigen3, Qt6 (auto-moc/uic — never invoke moc/uic manually).
- `std::print` for logging.
- `and`/`or` instead of `&&`/`||`.
- Opening brace on its own line, 4-space indent, no tabs.
- `snake_case` for variables/functions, `PascalCase` for classes/structs.
- All DSR API calls return `std::optional<T>` — check `.has_value()` before `.value()`.
- Never use raw DSR data structures; always go through the `dsr_api.h` / `dsr_inner_eigen_api.h` / `dsr_camera_api.h` interfaces.
- Comments explain **why**, not what.
