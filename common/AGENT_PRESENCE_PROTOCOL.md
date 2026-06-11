# Agent Presence Protocol — Developer Guide

This guide explains how to use `AgentPresenceCoordinator` and `AgentPresenceMonitor`
when writing a new active-inference agent. It covers concepts, config keys, the state
machine integration, all available hooks, and step-by-step instructions.

---

## 1. Concepts

### 1.1 What the protocol solves

Every agent in the system shares a DSR (Dynamic Scene Representation) graph. An agent
may depend on other agents being alive and fully initialised before it can do useful
work. Without coordination, an agent would start processing with missing peers and
produce garbage results, or crash.

The presence protocol gives each agent:

- A **readiness handshake** — an agent advertises `active_agent = true` to the graph
  only when it is actually ready to serve peers.
- **Peer monitoring** — a background timer polls agent heartbeat nodes in the graph and
  classifies each peer as `Missing`, `Booting`, `Ready`, or `Stale`.
- A **three-state lifecycle** — `Waiting → Operating → Degraded` — driven by the
  readiness of required peers.
- **Owned-node cleanup** — when an agent exits or a peer crashes, DSR nodes that belong
  to those agents are removed automatically so stale data does not confuse other agents.

### 1.2 Two-layer design

| Class | Responsibility |
|---|---|
| `AgentPresenceMonitor` | Low-level: connects to DSR signals, runs the heartbeat timer, classifies peers, fires raw callbacks. |
| `AgentPresenceCoordinator` | High-level: wraps the monitor, owns the lifecycle state machine hooks, applies the `Policy`, and exposes a clean API for `SpecificWorker`. |

You should only interact with `AgentPresenceCoordinator`. The monitor is accessible via
`coordinator.monitor()` if you need raw peer snapshots, but that is rarely needed.

### 1.3 Peer states

```
Missing  ──heartbeat seen──►  Stale/Booting  ──active_agent=true──►  Ready
  ▲                                                                      │
  └──────────────────DSR node deleted or timeout expired────────────────┘
```

| State | Meaning |
|---|---|
| `Missing` | No agent node in the DSR graph (agent never ran or DSR node deleted). |
| `Booting` | Agent node seen, heartbeat alive, but `active_agent` attribute is `false`. |
| `Stale` | Agent node seen but heartbeat has not been updated within `heartbeat_timeout_ms`. |
| `Ready` | Agent node present, heartbeat fresh, `active_agent = true`. |

Only the `Ready` state satisfies a required-peer constraint.

---

## 2. Config file keys

All keys live in your agent's `etc/config.toml`. All keys under `[Presence]` and
`[Owns]` are optional — if absent the system starts with safe defaults.

### 2.1 `[Presence]` section

```toml
[Presence]
# Agents that must be Ready before this agent enters Operating state.
# Use names (preferred) or numeric IDs.
required_agent_names = ["robot_concept", "room_concept"]
# required_agent_ids   = [3, 5]   # alternative: numeric DSR agent IDs

# Agents that are interesting but not blocking. Lost/ready events are
# delivered to on_optional_peer_lost / on_optional_peer_ready hooks.
optional_agent_names = ["some_optional_agent"]
# optional_agent_ids   = [7]

# DSR node names that must exist in the graph before Operating is entered.
# Use this when you depend on a specific node (not an agent) being present.
node_requires = ["room", "robot"]

# Directory where peer .toml files live (for crash cleanup). Default: "etc"
config_dir = "etc"

# How often the monitor polls and evaluates peer states (milliseconds).
monitor_period_ms    = 500

# How long without a heartbeat update before a peer is considered Stale.
heartbeat_timeout_ms = 3000

# Grace window after a peer restart before it is again counted as a
# required peer. Gives it time to re-advertise active_agent=true.
rejoin_grace_ms      = 1500

# How long a required peer may stay Stale before it is declared lost.
# Set to a large value (default 120 s) to tolerate transient CPU spikes.
stale_grace_ms       = 120000

# How often AgentInfoAPI updates this agent's own heartbeat node.
agent_info_period_ms = 1000
```

### 2.2 `[Owns]` section

```toml
[Owns]
# Flat list of DSR node names this agent creates and owns.
# Deleted when this agent exits or when a crash is detected by a peer.
nodes    = ["my_agent_output", "my_agent_output*"]  # trailing * = prefix wildcard

# Subtree roots: the named node plus all children reachable via RT edges
# are deleted recursively.
subtrees = ["room"]
```

> **Important:** every node your agent inserts into DSR that should be cleaned up on
> exit must appear here. If a peer reads your config (via `config_dir`) it will clean up
> your nodes when it detects you have crashed.

---

## 3. Agent lifecycle — the three states

```
         start()
            │
            ▼
        ┌─────────┐   presenceReady signal   ┌───────────┐
        │ Waiting │ ─────────────────────────► Operating │
        └─────────┘                           └───────────┘
            ▲                                      │
            │         presenceLost signal          │
            └──────────────────────────────────────┘
                                │
                                ▼
                          ┌──────────┐
                          │ Degraded │
                          └──────────┘
```

| State | Description |
|---|---|
| **Waiting** | The agent is running but one or more required peers are not yet `Ready`. `active_agent` is set to `false` (the agent is not advertising itself as ready). The monitor automatically checks if all required peers are already ready when entering this state and fires `presenceReady` immediately if so. |
| **Operating** | All required peers are `Ready`. `active_agent` is set to `true`. `compute()` is called every period. |
| **Degraded** | A previously ready required peer has gone missing. `active_agent` is set to `false`. The agent waits for the peer to come back (which triggers `presenceReady` again, going back to Operating). |

The state transitions are driven by Qt signals. The coordinator fires
`request_presence_ready` / `request_presence_lost` and it is your job to connect those
to `Q_EMIT presenceReady()` / `Q_EMIT presenceLost()` which trigger the GRAFCET state
machine transitions (see Section 4).

---

## 4. Step-by-step integration

### Step 1 — Declare the coordinator member

In `specificworker.h`:

```cpp
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

class SpecificWorker : public GenericWorker
{
    // ...
private:
    AgentPresenceCoordinator presence_coordinator_;
    bool owned_nodes_cleaned_ = false;
    // ...
signals:
    void presenceReady();
    void presenceLost();
};
```

### Step 2 — Wire the state machine transitions

In the GRAFCET block inside the constructor (the generated template already has this
structure; just add the two `addTransition` lines):

```cpp
states["Waiting"]->addTransition(this, SIGNAL(presenceReady()),  states["Operating"].get());
states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
```

### Step 3 — Configure and start the coordinator in `initialize()`

Call this **after** `GenericWorker::initialize()` and after the DSR graph `G` is ready:

```cpp
void SpecificWorker::initialize()
{
    GenericWorker::initialize();

    // ... your own initialisation ...

    // ── Presence coordinator ───────────────────────────────────────────────
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));

    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { Q_EMIT presenceReady(); },
        .request_presence_lost  = [this]() { Q_EMIT presenceLost(); },
    });

    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted — resetting related state";
        },
        .on_optional_peer_lost  = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
    });

    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            qInfo() << "[SM] -> Waiting";
            const auto missing = presence_coordinator_.missing_required_names();
            if (!missing.empty())
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "  missing:" << m;
            }
        },
        .on_operating_enter = []()
        {
            qInfo() << "[SM] -> Operating";
        },
        .on_operating_loop = [this]()
        {
            compute();
            if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
                it->second->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = []()
        {
            qInfo() << "[SM] -> Degraded: a required peer is no longer available";
        },
    });

    presence_coordinator_.start();

    // Register cleanup on clean shutdown
    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::cleanup_owned_nodes, Qt::UniqueConnection);

    restore_window_settings();
}
```

### Step 4 — Forward state machine calls in `specificworker_presence.cpp`

Create (or extend) a `specificworker_presence.cpp` file. The coordinator owns the
state logic, so the slot bodies are one-liners:

```cpp
void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter();   }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop();     }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter();  }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop();   }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter();   }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop();    }
```

### Step 5 — Implement `cleanup_owned_nodes`

This is called both on clean exit and by the GRAFCET "stop" path:

```cpp
void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)  // guard against double-call
        return;
    owned_nodes_cleaned_ = true;

    // Delete your agent-specific DSR nodes first
    cleanup_my_dsr_nodes();

    // Then let the coordinator clean up and stop the monitor
    presence_coordinator_.cleanup_owned_nodes();
}
```

### Step 6 — Call `save_window_settings()` in the destructor

```cpp
SpecificWorker::~SpecificWorker()
{
    save_window_settings();
    // ... any other teardown ...
}
```

---

## 5. Hook reference

### 5.1 `TransitionHooks`

Fired by the monitor when the overall required-peer status changes.

| Hook | When fired |
|---|---|
| `request_presence_ready` | All required peers just became Ready. Connect to `Q_EMIT presenceReady()`. |
| `request_presence_lost` | At least one required peer just became non-Ready. Connect to `Q_EMIT presenceLost()`. |

### 5.2 `PeerHooks`

Fired for individual peer events.

| Hook | Signature | When fired |
|---|---|---|
| `on_peer_restarted` | `(uint32_t id)` | A peer node re-appeared with a different `timestamp_creation`. Use this to reset any state derived from that peer. |
| `on_optional_peer_lost` | `(string name, uint32_t id)` | An optional peer transitioned out of Ready. |
| `on_optional_peer_ready` | `(string name, uint32_t id)` | An optional peer transitioned into Ready. |

### 5.3 `LifecycleHooks`

Called by the coordinator's `waiting_enter()`, `waiting_loop()`, etc. methods, which
you forward from the GRAFCET slots (Section 4, Step 4).

| Hook | When called | Typical use |
|---|---|---|
| `on_waiting_enter` | State machine enters Waiting | Log which peers are missing. |
| `on_waiting_loop` | Every Waiting tick | Light background work (e.g. display "waiting…" in UI). |
| `on_operating_enter` | State machine enters Operating | One-time initialisation that requires peers to be ready. |
| `on_operating_loop` | Every Operating tick | **Put your main `compute()` call here.** |
| `on_degraded_enter` | State machine enters Degraded | Log the event; freeze or disable outputs. |
| `on_degraded_loop` | Every Degraded tick | Keep the UI alive while waiting for peer recovery. |

### 5.4 `Policy`

Fine-tune the coordinator's built-in behaviour. Defaults are sensible and you rarely
need to change them.

```cpp
AgentPresenceCoordinator::Policy policy;
policy.set_local_ready_false_on_waiting_enter  = true;  // default
policy.set_local_ready_true_on_operating_enter = true;  // default
policy.set_local_ready_false_on_degraded_enter = true;  // default
policy.auto_request_presence_ready_from_waiting = true; // check immediately on Waiting entry
policy.waiting_enter_order   = HookOrder::BeforeCoordinator; // default
policy.operating_enter_order = HookOrder::BeforeCoordinator;
policy.degraded_enter_order  = HookOrder::BeforeCoordinator;
presence_coordinator_.set_policy(policy);
```

`HookOrder` controls whether your `on_*_enter` hook runs **before** or **after** the
coordinator sets the `active_agent` flag. Use `AfterCoordinator` if your hook needs to
observe the updated flag.

---

## 6. Inserting DSR nodes

Whenever you insert a new node into the DSR graph, call `trigger_graph_layout_twopi()`
so the graph viewer redraws the layout to include the new node:

```cpp
if (const auto id = G->insert_node(my_node); id.has_value())
{
    qInfo() << "Node created, id=" << *id;
    // insert RT edge if needed ...
    trigger_graph_layout_twopi();
}
```

---

## 7. Peer crash recovery

When a required or optional peer crashes without a clean shutdown, its DSR agent node
eventually goes `Stale` (heartbeat stops) and then `Missing` (after `stale_grace_ms`).

The coordinator automatically looks up the peer's config file at
`<config_dir>/<peer_name>.toml` and reads its `[Owns]` section to delete the orphaned
nodes. For this to work:

1. Every agent must ship a `<agent_name>.toml` file in its `etc/` directory listing
   its owned nodes and subtrees.
2. The `config_dir` key in your config must point to the directory containing those
   peer config files (typically `"etc"` if all agents share the same `etc/` folder,
   or an absolute/relative path otherwise).

Example peer config (`etc/room_concept.toml`):

```toml
[Owns]
nodes    = ["room_concept 5", "afford_room"]
subtrees = ["room"]
```

---

## 8. Using `node_requires`

Sometimes your agent depends on a specific DSR **node** being present (not an agent).
Use `node_requires` for this:

```toml
[Presence]
node_requires = ["room", "robot"]
```

The system will not transition to `Operating` (and will return to `Degraded` if already
there) until every listed node exists in the graph. This is checked on every monitor
tick.

---

## 9. Querying presence state at runtime

```cpp
// True only when all required peers are Ready and all node_requires are met.
if (presence_coordinator_.all_required_ready())
    do_something();

// List of names/ids currently blocking the transition to Operating.
for (const auto &name : presence_coordinator_.missing_required_names())
    qInfo() << "  still waiting for:" << name.c_str();

// Raw snapshots for all known peers (for diagnostics / debug UI).
for (const auto &snap : presence_coordinator_.monitor()->snapshot())
{
    qInfo() << "peer" << snap.agent_id
            << snap.agent_name.c_str()
            << "state=" << static_cast<int>(snap.state);
}
```

---

## 10. Complete minimal `config.toml` example

```toml
[Agent]
id   = 42
name = "my_new_agent"
configFile = ""
domain = 0
graph  = true

[Period]
Compute   = 100
Emergency = 500

[Presence]
required_agent_names = ["robot_concept"]
optional_agent_names = ["some_optional_agent"]
node_requires        = ["room"]
config_dir           = "etc"
monitor_period_ms    = 500
heartbeat_timeout_ms = 3000
rejoin_grace_ms      = 1500
stale_grace_ms       = 120000
agent_info_period_ms = 1000

[Owns]
nodes    = ["my_output_node"]
subtrees = []

[Component.Debug]
Verbose = false
```

And a companion `etc/my_new_agent.toml` (used by peers for crash cleanup):

```toml
[Owns]
nodes    = ["my_output_node"]
subtrees = []
```

---

## 11. Quick checklist for a new agent

- [ ] Add `AgentPresenceCoordinator presence_coordinator_` member in `specificworker.h`
- [ ] Add `bool owned_nodes_cleaned_ = false` member
- [ ] Add `signals: presenceReady(); presenceLost();` in `specificworker.h`
- [ ] Wire `addTransition` calls in the constructor for `presenceReady`/`presenceLost`
- [ ] Call `presence_coordinator_.configure(...)` and set all hooks in `initialize()`
- [ ] Call `presence_coordinator_.start()` at the end of `initialize()`
- [ ] Connect `aboutToQuit` → `cleanup_owned_nodes` in `initialize()`
- [ ] Call `restore_window_settings()` at the very end of `initialize()`
- [ ] Forward the six GRAFCET slot calls to the coordinator in `specificworker_presence.cpp`
- [ ] Implement `cleanup_owned_nodes()` with the `owned_nodes_cleaned_` guard
- [ ] Call `save_window_settings()` in `~SpecificWorker()`
- [ ] Call `trigger_graph_layout_twopi()` after every `G->insert_node()`
- [ ] Populate `[Presence]` and `[Owns]` in `etc/config.toml`
- [ ] Create `etc/<my_agent_name>.toml` with just the `[Owns]` section for peer crash recovery
