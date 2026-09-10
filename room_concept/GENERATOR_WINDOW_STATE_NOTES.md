# Window State Persistence: Generator Notes

This note captures the exact code pattern that worked for `room_concept`, so it can be moved into the component generator templates.

## Scope

Add persistent `QMainWindow` geometry + dock layout save/restore support to generated C++ RoboComp components.

The working pattern has two parts:

1. `GenericWorker` owns the persistence helpers and the `QSettings` logic.
2. `SpecificWorker` triggers restore late in `initialize()` and triggers save early in its destructor.

The save must **not** happen in `GenericWorker::~GenericWorker()`, because by then derived-class UI members may already be destroyed.

## Required Generator Changes

### 1. Add helper declarations to `generated/genericworker.h`

Insert these in the protected section near `windows` and `setupViewer(...)`:

```cpp
// DSR graph viewer
std::unordered_map<std::string, std::shared_ptr<DSR::DSRViewer>> graph_viewers;
std::unordered_map<std::string, std::unique_ptr<QMainWindow>> windows;
std::shared_ptr<DSR::DSRViewer> setupViewer(std::shared_ptr<DSR::DSRGraph> graph, const std::string& prefix, QMainWindow* parent);
void restore_window_settings();
void save_window_settings() const;
```

### 2. Add helper utilities to `generated/genericworker.cpp`

Add the print header:

```cpp
#include <print>
```

Add these anonymous-namespace helpers near the top of the file:

```cpp
namespace
{
constexpr int kWindowStateVersion = 1;

QString settings_group_name(const std::string& graph_name, int agent_id)
{
    const QString graph_suffix = graph_name.empty() ? QStringLiteral("default")
                                                    : QString::fromStdString(graph_name);
    return QStringLiteral("windows/%1/%2").arg(agent_id).arg(graph_suffix);
}

const char* settings_status_to_cstr(QSettings::Status status)
{
    switch (status)
    {
        case QSettings::NoError: return "NoError";
        case QSettings::AccessError: return "AccessError";
        case QSettings::FormatError: return "FormatError";
    }
    return "Unknown";
}
}
```

### 3. Implement restore logic in `generated/genericworker.cpp`

Add this method implementation:

```cpp
void GenericWorker::restore_window_settings()
{
    QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));

    std::print("[WindowState] restore begin file={} windows={}\n",
               settings.fileName().toStdString(), windows.size());

    for (const auto& [name, window] : windows)
    {
        if (window == nullptr)
        {
            std::print("[WindowState] restore skip null window name={}\n", name);
            continue;
        }

        const QString group_name = settings_group_name(name, agent_id);
        settings.beginGroup(group_name);

        const bool has_geometry = settings.contains(QStringLiteral("geometry"));
        const bool has_state = settings.contains(QStringLiteral("state"));
        const QStringList child_keys = settings.childKeys();
        const QByteArray geometry_before = window->saveGeometry();
        const QByteArray state_before = window->saveState(kWindowStateVersion);
        const QRect rect_before = window->geometry();

        const QByteArray geometry = settings.value(QStringLiteral("geometry")).toByteArray();
        bool geometry_restored = false;
        if (!geometry.isEmpty())
            geometry_restored = window->restoreGeometry(geometry);

        const QByteArray state = settings.value(QStringLiteral("state")).toByteArray();
        bool state_restored = false;
        if (!state.isEmpty())
            state_restored = window->restoreState(state, kWindowStateVersion);

        const QByteArray geometry_after = window->saveGeometry();
        const QByteArray state_after = window->saveState(kWindowStateVersion);
        const QRect rect_after = window->geometry();
        const bool geometry_matches_loaded = !geometry.isEmpty() and geometry_after == geometry;
        const bool state_matches_loaded = !state.isEmpty() and state_after == state;
        const bool geometry_changed = geometry_after != geometry_before;
        const bool state_changed = state_after != state_before;

        std::print(
            "[WindowState] restore group={} keys={} has_geometry={} has_state={} title={} visible={} "
            "geometry_bytes={} geometry_ok={} geometry_changed={} geometry_matches_loaded={} "
            "state_bytes={} state_ok={} state_changed={} state_matches_loaded={} "
            "rect_before=({},{},{}x{}) rect_after=({},{},{}x{}) "
            "geometry_before_bytes={} geometry_after_bytes={} state_before_bytes={} state_after_bytes={}\n",
            group_name.toStdString(),
            child_keys.join(QStringLiteral(",")).toStdString(),
            has_geometry,
            has_state,
            window->windowTitle().toStdString(),
            window->isVisible(),
            geometry.size(),
            geometry_restored,
            geometry_changed,
            geometry_matches_loaded,
            state.size(),
            state_restored,
            state_changed,
            state_matches_loaded,
            rect_before.x(),
            rect_before.y(),
            rect_before.width(),
            rect_before.height(),
            rect_after.x(),
            rect_after.y(),
            rect_after.width(),
            rect_after.height(),
            geometry_before.size(),
            geometry_after.size(),
            state_before.size(),
            state_after.size());

        settings.endGroup();
    }

    std::print("[WindowState] restore end status={}\n", settings_status_to_cstr(settings.status()));
}
```

### 4. Implement save logic in `generated/genericworker.cpp`

Add this method implementation:

```cpp
void GenericWorker::save_window_settings() const
{
    QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));

    std::print("[WindowState] save begin file={} windows={}\n",
               settings.fileName().toStdString(), windows.size());

    for (const auto& [name, window] : windows)
    {
        if (window == nullptr)
        {
            std::print("[WindowState] save skip null window name={}\n", name);
            continue;
        }

        const QString group_name = settings_group_name(name, agent_id);
        const QByteArray geometry = window->saveGeometry();
        const QByteArray state = window->saveState(kWindowStateVersion);

        settings.beginGroup(group_name);
        settings.setValue(QStringLiteral("geometry"), geometry);
        settings.setValue(QStringLiteral("state"), state);
        const QStringList child_keys = settings.childKeys();
        settings.endGroup();

        std::print(
            "[WindowState] save group={} keys={} title={} visible={} geometry_bytes={} state_bytes={} "
            "pos=({},{}) size=({}x{})\n",
            group_name.toStdString(),
            child_keys.join(QStringLiteral(",")).toStdString(),
            window->windowTitle().toStdString(),
            window->isVisible(),
            geometry.size(),
            state.size(),
            window->pos().x(),
            window->pos().y(),
            window->size().width(),
            window->size().height());
    }

    settings.sync();
    std::print("[WindowState] save end status={}\n", settings_status_to_cstr(settings.status()));
}
```

### 5. Do not save from `GenericWorker::~GenericWorker()`

This is important.

Do **not** add:

```cpp
save_window_settings();
```

inside `GenericWorker::~GenericWorker()`.

That caused a shutdown crash (`pure virtual method called`) because dock/widget state was being queried after derived UI members had already started destruction.

### 6. Restore state at the end of `SpecificWorker::initialize()`

After the generated/custom UI is fully created and dock widgets have been added, call:

```cpp
std::print("[WindowState] specificworker restore request custom_frame_visible={} viewer_widget_visible={}\n",
           custom_widget.frame != nullptr ? custom_widget.frame->isVisible() : false,
           viewer_2d_ != nullptr and viewer_2d_->get_widget() != nullptr ? viewer_2d_->get_widget()->isVisible() : false);
restore_window_settings();
```

This must happen **after**:

- `GenericWorker::initialize()`
- any `add_custom_widget_to_dock(...)`
- local viewer creation
- signal connections that depend on the final UI shape

Reason: `restoreState()` only works correctly when the full dock/widget layout already exists.

### 7. Save state early in `SpecificWorker::~SpecificWorker()`

Add this near the top of the destructor, before the derived-class UI is torn down:

```cpp
save_window_settings();
```

Working order:

```cpp
SpecificWorker::~SpecificWorker()
{
    save_window_settings();
    save_robot_pose_once();
    room_concept_.stop();
    std::cout << "Destroying SpecificWorker" << std::endl;
}
```

The non-window log can stay as-is; only the persistence traces were changed to `std::print` to follow project instructions.

## Expected Runtime Behavior

### First run

At startup, there are no persisted keys yet:

```text
[WindowState] specificworker restore request custom_frame_visible=false viewer_widget_visible=false
[WindowState] restore begin file=/home/.../.config/RoboComp/room_concept.conf windows=1
[WindowState] restore group=windows/5/default keys= has_geometry=false has_state=false ...
[WindowState] restore end status=NoError
```

At shutdown, geometry and state are written:

```text
[WindowState] save begin file=/home/.../.config/RoboComp/room_concept.conf windows=1
[WindowState] save group=windows/5/default keys=geometry,state ...
[WindowState] save end status=NoError
```

### Second run

At startup, the keys should exist:

```text
[WindowState] restore group=windows/5/default keys=geometry,state has_geometry=true has_state=true ...
```

## Why this exact design matters

- `QSettings("RoboComp", agent_name)` gives a per-component settings file.
- `windows/<agent_id>/<graph_name_or_default>` avoids collisions across graphs and agent instances.
- `restoreState()` only works if the full dock layout already exists.
- saving from the base destructor is too late and can crash on shutdown.
- comparing loaded bytes against `saveGeometry()` / `saveState()` after restore made it easy to verify whether Qt really applied the loaded settings.

## Minimal Generator Checklist

1. Add protected declarations in `genericworker.h`.
2. Add helper functions and `#include <print>` in `genericworker.cpp`.
3. Generate `restore_window_settings()`.
4. Generate `save_window_settings()`.
5. Never call `save_window_settings()` from `GenericWorker::~GenericWorker()`.
6. Add `restore_window_settings()` at the end of generated `SpecificWorker::initialize()`.
7. Add `save_window_settings()` near the top of generated `SpecificWorker::~SpecificWorker()`.
8. Use `std::print` for all persistence traces.

## Files in this repo that validate the pattern

- `generated/genericworker.h`
- `generated/genericworker.cpp`
- `src/specificworker.cpp`

## Additional Template Fixes

These are separate from window-state persistence, but they should also be moved into the generator templates because they address generated-code correctness and safer reuse.

### 1. Keep subscription output handles in `generated/main.cpp`

In the `subscribe(...)` helper, do not shadow the output reference parameters.

Incorrect pattern:

```cpp
auto proxy = adapter->addWithUUID(servant)->ice_oneway();
std::shared_ptr<IceStorm::TopicPrx> topic;
```

Correct pattern:

```cpp
proxy = adapter->addWithUUID(servant)->ice_oneway();
// do not redeclare topic locally; use the reference parameter
```

Why:

- the caller must keep the real `topic` and `proxy`
- unsubscribe code after `a.exec()` depends on those handles being returned
- future code may also need to inspect or reuse the subscription handles

### 2. Add a safe viewer lookup helper in `generated/genericworker.h`

Add this protected declaration near `setupViewer(...)`:

```cpp
std::shared_ptr<DSR::DSRViewer> find_graph_viewer(const std::string& name) const;
```

Why:

- derived workers currently reach into `graph_viewers` directly
- `graph_viewers.at("")` throws if the default viewer does not exist
- the helper centralizes the lookup policy in `GenericWorker`

### 3. Implement the helper in `generated/genericworker.cpp`

Add:

```cpp
std::shared_ptr<DSR::DSRViewer> GenericWorker::find_graph_viewer(const std::string& name) const
{
    const auto it = graph_viewers.find(name);
    if (it == graph_viewers.end())
        return nullptr;

    return it->second;
}
```

### 4. Reuse the helper in `GenericWorker`-owned code

Update `trigger_graph_layout_twopi()` to use `find_graph_viewer("")` instead of reading `graph_viewers` directly.

Why:

- avoids duplicating lookup and null-check logic
- gives generated code one consistent pattern for viewer-dependent behavior

### 5. Companion usage in generated `SpecificWorker`

When a component requires the default viewer, validate that precondition explicitly before adding custom UI to it:

```cpp
auto default_viewer = find_graph_viewer("");
if (!default_viewer)
    throw std::runtime_error("SpecificWorker requires a default DSR viewer. Enable at least one Agent viewer flag for the default graph.");

default_viewer->add_custom_widget_to_dock("layout", &custom_widget);
```

Why:

- replaces an unhandled `std::out_of_range` with an explicit startup error
- makes the component's viewer dependency visible in the code

### 6. Add a generated logging module

Add a small generated logging module with shared categories and startup configuration.

Recommended files:

```cpp
generated/component_logging.h
generated/component_logging.cpp
```

Minimum declarations:

```cpp
Q_DECLARE_LOGGING_CATEGORY(logLifecycle)
Q_DECLARE_LOGGING_CATEGORY(logGraph)
Q_DECLARE_LOGGING_CATEGORY(logUi)
Q_DECLARE_LOGGING_CATEGORY(logIo)
Q_DECLARE_LOGGING_CATEGORY(logLocalizer)

void install_component_log_format();
void configure_component_logging(const ConfigLoader& config_loader);
```

Responsibilities:

- define the logging categories used by generated and handwritten code
- install one process-wide Qt log message format
- enable or disable debug logging from `Component.Debug.Verbose`

### 7. Use category-based Qt logging in generated code

Update generated `main.cpp` and `genericworker.cpp` so normal runtime logging uses Qt categories instead of raw `std::cout`, `std::cerr`, `printf`, or ad hoc ANSI-colored prefixes.

Apply these rules:

- use `logLifecycle` for startup, shutdown, topic subscription, and main-thread failures
- use `logGraph` for graph creation and graph-related initialization
- use `logUi` for window persistence and viewer/UI lifecycle
- reserve `logIo` and `logLocalizer` for handwritten runtime code

Why:

- gives generated and handwritten code one consistent logging frontend
- makes runtime severity and filtering controllable from Qt logging
- avoids mixing raw console formatting with Qt message handling

## Consolidated Generator Checklist

1. Add protected declarations in `genericworker.h` for `restore_window_settings()`, `save_window_settings()`, and `find_graph_viewer(...)`.
2. Add helper implementations in `genericworker.cpp` for window-state persistence and safe viewer lookup.
3. Keep `settings_group_name(...)` and `kWindowStateVersion` owned by `GenericWorker`, not in free helpers.
4. Never call `save_window_settings()` from `GenericWorker::~GenericWorker()`.
5. In `main.cpp`, keep subscription `topic` and `proxy` on the output reference parameters; do not shadow them.
6. Add a generated logging module with shared `QLoggingCategory` declarations and startup configuration helpers.
7. In generated `main.cpp` and `genericworker.cpp`, use category-based Qt logging instead of raw console logging.
8. In generated `SpecificWorker::initialize()`, restore window state only after the full dock/UI layout exists.
9. In generated `SpecificWorker::initialize()`, validate required default-viewer access before attaching custom widgets.
10. In generated `SpecificWorker::~SpecificWorker()`, save window state early, before derived UI members start tearing down.
