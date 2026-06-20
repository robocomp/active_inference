# voxelizer
A brief introduction to the component. Describe its purpose, functionality, and any specific features here.
# voxelizer

`voxelizer` consumes RGBD and lidar data, segments semantic objects with YOLO, voxelizes them in room coordinates, and publishes the resulting scene to DSR and the local OpenGL viewer.

## Quick Start

1. Install the system dependencies.
2. Install ONNX Runtime.
3. Verify the YOLO model path.
4. Build the component.
5. Copy and edit the config file.
6. Run the agent.

## Dependencies

Typical Ubuntu packages needed for build and runtime:

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake pkg-config \
    qt6-base-dev qt6-base-dev-tools qt6-declarative-dev qt6-scxml-dev \
    libqt6opengl6-dev libopencv-dev libtbb-dev libeigen3-dev
```

This component also expects:

- RoboComp / DSR runtime and generated interfaces available in the environment
- ONNX Runtime for YOLO inference
- Optional: CUDA and TensorRT if you want GPU / TensorRT execution

## ONNX Runtime Installation

This component does not assume a fixed ONNX Runtime location. CMake searches in this order:

1. `-DONNXRUNTIME_ROOT=...`
2. `ONNXRUNTIME_ROOT` environment variable
3. Common prefixes such as `/usr/local/onnxruntime`, `/opt/onnxruntime`, `/opt/onnxruntime-gpu`, `/usr`, `/usr/local`

Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases.

Recommended layout:

```bash
sudo mkdir -p /opt
sudo tar -C /opt -xzf onnxruntime-linux-x64-gpu-<version>.tgz
sudo ln -sfn /opt/onnxruntime-linux-x64-gpu-<version> /opt/onnxruntime
```

Then configure the build with:

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

If you use a CPU-only build of ONNX Runtime, point `ONNXRUNTIME_ROOT` to that installation instead.

## YOLO Setup

The component uses ONNX-based YOLO segmentation through `YoloSegDetector`.

By default, the code expects the model path to be:

```text
yolo26l-seg.onnx
```

This file is already present in the repository root. If you want to use another model, either replace that file or set the config entry:

```toml
[Yolo]
model_path = "/absolute/or/relative/path/to/your-model.onnx"
```

Useful YOLO-related settings are:

- `Yolo.model_path`
- `Yolo.conf_thresh`
- `Yolo.iou_thresh`
- `Yolo.use_gpu`
- `Yolo.use_trt`
- `Yolo.accepted_labels`

If TensorRT inference is enabled, ONNX Runtime, CUDA, and TensorRT must be ABI-compatible. If TensorRT cannot be initialized, the detector falls back to CUDA when available.

### TensorRT version lock (and the Ubuntu-upgrade gotcha)

The TensorRT version is **pinned by the ONNX Runtime build**, not chosen freely. ORT's TensorRT
Execution Provider is compiled against one TensorRT **major** version; you cannot substitute another
(different SONAME + ABI). Per the ORT compatibility matrix
(https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html), ORT **1.22–1.23**
require **TensorRT 10.9** (`libnvinfer.so.10`, `libnvinfer_builder_resource.so.10.9.0`). **No ORT
release supports TensorRT 11** — a system `libnvinfer.so.11` is unusable here regardless.

`YoloSegDetector` (`src/yolo_processor.cpp`, `prefer_system_tensorrt_stack`) preloads the TRT-10 stack
by **absolute path** from `/usr/lib/x86_64-linux-gnu/`:
`libnvinfer.so.10`, `libnvonnxparser.so.10`, `libnvinfer_plugin.so.10` (plus
`libnvinfer_builder_resource.so.10.9.0`, found via the loader path / a probe of
`/usr/local/cuda-13.2/lib64`).

**Gotcha:** an `apt upgrade` of the system `tensorrt` packages replaces those `.so.10` libs in
`/usr/lib/x86_64-linux-gnu` with `.so.11`. YOLO then crashes inside `detect_segmentation` (a
half-initialised TRT EP corrupting memory). The CUDA toolkit's own copy of TRT 10.9 survives the apt
upgrade, so restore the `.so.10` libs from there:

```bash
# Point SRC at the CUDA toolkit dir that still has TensorRT 10.9 (adjust the CUDA version):
SRC=/usr/local/cuda-12.8/targets/x86_64-linux/lib
sudo bash -c "for f in $SRC/libnvinfer*.so.10* $SRC/libnvonnxparser*.so.10*; do ln -sfv \"\$f\" /usr/lib/x86_64-linux-gnu/; done; ldconfig"
# verify the three hardcoded preloads resolve:
for l in libnvinfer.so.10 libnvonnxparser.so.10 libnvinfer_plugin.so.10; do ls -lL /usr/lib/x86_64-linux-gnu/$l; done
```

Then hold the apt packages so the next upgrade doesn't re-break it:
`sudo apt-mark hold libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10` (names vary by distro).

**Escape hatch:** set `Yolo.use_trt = false` in `etc/config.toml`. The detector then runs on the CUDA
EP only — version-independent, identical masks, just no FP16/TensorRT speedup. Use this if the TRT
stack can't be reconciled.

## Configuration parameters
Like any other component, voxelizer requires a configuration file to start. In etc/config or etc/config.toml, you can find an example of the configuration file.

For normal use, copy one of them and edit the copy:

```bash
cp etc/config.toml etc/local_config.toml
```

```bash
cd <voxelizer's path> 
cp etc/config etc/yourConfig
```

After editing the new config file we can run the component:

```bash
cmake -B build && make -C build -j12 # Compile the component
bin/voxelizer etc/yourConfig # Execute the component
```
cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build -j$(nproc)
bin/voxelizer --Ice.Config=etc/local_config.toml
# Developer Notes

If you use the RoboComp helper in your shell environment, `cbuild` also works for rebuilding the component.

## CUDA / TensorRT Notes

- For CUDA-only inference, the normal CUDA runtime is usually enough.
- For TensorRT inference, ONNX Runtime, CUDA, and TensorRT versions must match.
- If multiple CUDA/TensorRT stacks are installed, use `LD_LIBRARY_PATH` to prioritize the intended one.

Example:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Typical startup signals to check:

- `[YOLO] effective flags: use_gpu=true use_trt=true`
- `[YoloSegDetector] TensorRT EP registered ...`
- `[YoloSegDetector] CUDA EP registered`
This section explains how to work with the generated code of voxelizer, including what can be modified and how to use key features.
## Editable Files
You can freely edit the following files:
- etc/* – Configuration files
- src/* – Component logic and implementation
- README.md – Documentation

The `generated` folder contains autogenerated files. **Do not edit these files directly**, as they will be overwritten every time the component is regenerated with RoboComp.

## ConfigLoader
The `ConfigLoader` simplifies fetching configuration parameters. Use the `get<>()` method to retrieve parameters from the configuration file.
```C++
// Syntax
type variable = this->configLoader.get<type>("ParameterName");

// Example
int computePeriod = this->configLoader.get<int>("Period.Compute");
```

## StateMachine
RoboComp components utilize a state machine to manage the main execution flow. The default states are:

1. **Initialize**:
    - Executes once after the constructor.
    - May use for parameter initialization, opening devices, and calculating constants.
2. **Compute**:
    - Executes cyclically after Initialize.
    - Place your functional logic here. If an emergency is detected, call goToEmergency() to transition to the Emergency state.
3. **Emergency**:
    - Executes cyclically during emergencies.
    - Once resolved, call goToRestore() to transition to the Restore state.
4. **Restore**:
    - Executes once to restore the component after an emergency.
    - Transitions automatically back to the Compute state.

### Setting and Getting State Periods
You can get the period of some state with de function `getPeriod` and set with `setPeriod`
```C++
int currentPeriod = getPeriod("Compute");   // Get the current Compute period
setPeriod("Compute", currentPeriod * 0.5); // Set Compute period to half
```

### Creating Custom States
To add a custom state, follow these steps in the constructor:
1. **Define Your State** Use `GRAFCETStep` to create your state. If any function is not required, use `nullptr`.

```C++
states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
                                                      std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
                                                      std::bind(&SpecificWorker::customEnter, this), // On-enter function
                                                      std::bind(&SpecificWorker::customExit, this)); // On-exit function

```
2. **Define Transitions** Add transitions between states using `addTransition`. You can trigger transitions using Qt signals such as `entered()` and `exited()` or custom signals in .h.
```C++
// Syntax
states[srcState]->addTransition(originOfSignal, signal, dstState)

// Example
states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get());

```
3. **Add State to the StateMachine** Include your state in the state machine:
```C++
statemachine.addState(states["CustomState"].get());

```

## Hibernation Flag
The `#define HIBERNATION_ENABLED` flag in `specificworker.h` activates hibernation mode. When enabled, the component reduces its state execution frequency to 500ms if no method calls are received within 5 seconds. Once a method call is received, the period is restored to its original value.

Default hibernation monitoring runs every 500ms.

## Changes Introduced in the New Code Generator
If you’re regenerating or adapting old components, here’s what has changed:

- Deprecated classes removed: `CommonBehavior`, `InnerModel`, `AGM`, `Monitors`, and `src/config.h`.
- Configuration parsing replaced with the new `ConfigLoader`, supporting both .`toml` and legacy configuration formats.
- Skeleton code split: `generated` (non-editable) and `src` (editable).
- Component period is now configurable in the configuration file.
- State machine integrated with predefined states: `Initialize`, `Compute`, `Emergency`, and `Restore`.
- With the `dsr` option, you generate `G` in the GenericWorker, making the viewer independent. If you want to use the `dsrviewer`, you will need the `Qt GUI (QMainWindow)` and the `dsr` option enabled in the **CDSL**.
- Strings in the legacy config now need to be enclosed in quotes (`""`).

## Adapting Old Components
To adapt older components to the new structure:

1. **Add** `Period.Compute` and `Period.Emergency` and swap Endpoints and Proxies with their names in the `etc/config` file.
2. **Merge** the new `src/CMakeLists.txt` and the old `CMakeListsSpecific` files.
3. **Modify** `specificworker.h`:
    - Add the `HIBERNATION_ENABLED` flag.
    - Update the constructor signature.
    - Replace `setParams` with state definitions (`Initialize`, `Compute`, etc.).
4. **Modify** `specificworker.cpp`:
    - Refactor the constructor entirely.
    - Move `setParams` logic to the `initialize` state using `ConfigLoader.get<>()`.
    - Remove the old timer and period logic and replace it with `getPeriod()` and `setPeriod()`.
    - Add the new function state `Emergency`, and `Restore`.
    - Add the following code to the implements and publish functions:
        ```C++
        #ifdef HIBERNATION_ENABLED
            hibernation = true;
        #endif
        ```
5. **Update Configuration Strings**, ensure all strings in the `config` under legacy are enclosed in quotes (`""`), as required by the new structure.
6. **Using DSR**, if you use the DSR option, note that `G` is generated in `GenericWorker`, making the viewer independent. However, to use the `dsrviewer`, you must integrate a `Qt GUI (QMainWindow)` and enable the `dsr` option in the **CDSL**. 
7. **Installing toml++**, to use the new .toml configuration format, install the toml++ library:
```bash
mkdir ~/software 2> /dev/null; git clone https://github.com/marzer/tomlplusplus.git ~/software/tomlplusplus
cd ~/software/tomlplusplus && cmake -B build && sudo make install -C build -j12 && cd -
```
8. **Installing qt6 Dependencies**
```bash
sudo apt install qt6-base-dev qt6-declarative-dev qt6-scxml-dev libqt6statemachineqml6 libqt6statemachine6

mkdir ~/software 2> /dev/null; git clone https://github.com/GillesDebunne/libQGLViewer.git ~/software/libQGLViewer
cd ~/software/libQGLViewer && qmake6 *.pro && make -j12 && sudo make install && sudo ldconfig && cd -
```
9. **Generated Code**, When the component is generated, a `generated` folder is created containing non-editable files. You can delete everything in the `src` directory except for:
- `src/specificworker.h`
- `src/specificworker.cpp`
- `src/CMakeLists.txt`
- `src/mainUI.ui`
- `README.md`
- `etc/config`
- `etc/config.toml`
- Your Clases...