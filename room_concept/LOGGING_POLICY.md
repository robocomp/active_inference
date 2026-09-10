# Logging Policy

This policy defines one consistent logging model for active_inference components and their generated code.

## Goals

- one primary runtime logging API across generated and handwritten code
- predictable severity levels and message ownership
- no accidental mix of `std::cout`, `std::cerr`, `qDebug`, and ad hoc prefixes for the same kind of event
- debug traces and telemetry remain available without polluting normal runtime logs

## Primary Rule

Use Qt logging as the primary runtime logging frontend.

- `qCInfo` for one-time lifecycle and normal operational milestones
- `qCWarning` for recoverable problems and degraded behavior
- `qCCritical` for non-recoverable errors immediately before aborting startup, throwing, or stopping a feature
- `qCDebug` for high-volume diagnostics, disabled by default

Do not use `std::cout` or `std::cerr` for normal runtime logs.

Allowed exceptions:

- machine-readable output explicitly meant for another tool or process
- temporary migration shims in generator code while older components are being updated

## Logging Categories

Every component should define a small set of `QLoggingCategory` values and route messages through them.

Recommended base categories:

- `component.lifecycle` for startup, shutdown, subscriptions, configuration load, and state transitions
- `component.graph` for DSR graph creation, lookup, and graph/viewer initialization
- `component.ui` for window restore/save and viewer/widget setup
- `component.io` for files, sockets, SVG loading, and external bridge connections
- `component.localizer` for localization, optimization, and recovery events
- `component.debug` for temporary or high-volume diagnostics

Rule:

- generated `main.cpp` and `genericworker.cpp` should use `component.lifecycle`, `component.graph`, and `component.ui`
- handwritten algorithm code should use the domain category that owns the behavior

## Severity Policy

Use severities by effect, not by emotion.

`Info`:

- component started
- topic subscribed/unsubscribed
- graph created
- window state restored/saved
- optional feature enabled or disabled

`Warning`:

- optional config key invalid and default kept
- SVG file missing but component can still run in degraded mode
- Rerun bridge disconnected and reconnecting
- CUDA requested but unavailable, falling back to CPU

`Critical`:

- required config missing
- required viewer or graph surface missing and component cannot proceed
- mandatory topic manager or communication initialization failed

`Debug`:

- per-frame localizer details
- detailed UI state bytes or restore internals
- repeated transport or queue diagnostics

## Message Format

Do not hand-build severity tags like `[INFO]`, `[WARNING]`, or color escape sequences in the message body.

Use plain messages with enough context to stand alone:

- include the subsystem and object name when helpful
- include the key parameter that explains the state
- avoid dumping large payloads unless the category is debug-only

Good:

```cpp
qCWarning(logLocalizer) << "CUDA init failed, falling back to CPU:" << e.what();
```

Bad:

```cpp
std::cout << "[ERROR] CUDA failed" << std::endl;
```

## Configuration Policy

Separate runtime logging from debug artifacts.

Recommended config keys:

- `Component.Log.Level = info|debug|warning|critical`
- `Component.Log.Verbose = true|false`

Existing artifact/debug flags remain separate:

- `RoomConcept.DebugLog` controls CSV/file traces only
- `RoomConcept.DifferentialTest` controls test/report output only
- bridge or telemetry flags control their own output sinks only

Rule:

- a file trace flag must not silently change console log verbosity
- runtime log level must not automatically enable expensive telemetry capture

## Message Handler Policy

If the process installs a Qt message handler, it must be done once, in one place, and only to format or route Qt logs.

Rules:

- install the handler in startup code, not opportunistically in arbitrary classes
- do not use it to suppress categories globally without a documented policy
- do not mix handler-installed formatting with manual ANSI-tagged `std::cout` logs

If the project wants plain stderr output, the handler should format all Qt logs consistently and generated code should stop emitting raw `std::cout`/`std::cerr` lifecycle messages.

## Generated Code Policy

Generator templates should follow these rules by default:

- `main.cpp` logs lifecycle, topic, and Ice startup with Qt logging categories
- `genericworker.cpp` logs graph creation, period changes, and UI persistence with Qt logging categories
- no raw ANSI color codes in generated logs
- no `qDebug()` in generated code for normal lifecycle messages

Generated code should be conservative:

- `Info` for successful startup milestones
- `Warning` for recoverable subscription/topic situations
- `Critical` when startup cannot continue

## Handwritten Code Policy

Handwritten component code should follow the same frontend and severity rules.

Special cases:

- algorithm telemetry belongs in CSV, Rerun, or dedicated debug artifacts, not in normal logs
- hot loops must not emit `Info` or `Warning` repeatedly; use throttled `Debug` when needed
- helper utilities should not print directly unless they own the user-visible operation

## Migration Plan

### Phase 1

- stop adding new `std::cout` and `std::cerr` runtime logs
- add logging categories to generated and handwritten code
- keep current behavior working while replacing the most visible lifecycle logs first

### Phase 2

- convert generated `main.cpp` and `genericworker.cpp` to category-based Qt logging
- move the current Qt message handler, if still needed, to a single startup location

### Phase 3

- trim leftover `qDebug()` and raw console prints from handwritten code
- keep file-based traces and telemetry behind explicit feature flags

## Concrete Guidance For This Repo

Apply first to these files:

- `generated/main.cpp`
- `generated/genericworker.cpp`
- `src/room_concept.cpp`
- `src/svg_room_loader.cpp`
- `src/rerun_logger.cpp`

Target outcomes:

- startup and shutdown logs come from one API
- warnings and critical errors use consistent severities
- localizer CSV debug output remains separate from runtime logging
- no mixed formatting styles in the same execution path