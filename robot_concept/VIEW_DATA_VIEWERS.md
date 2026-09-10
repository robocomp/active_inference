# "View data" viewers

How right-click **View data** on a DSR node opens a viewer, and how to add a new one.

Two data sources, two recipes — but the same wiring on the agent side.

---

## The path a right-click takes

1. In cortex `dsr_gui`, `GraphNode::setType()` adds a **"View data"** action to *every* node.
2. On click, `GraphNode::request_view_data()` decides:
   - `has_inline_data()` **true** → open the built-in dsr_gui widget (legacy: data still inline in
     the graph). Only the four legacy types have one — `rgbd`, `laser`, `imu`, `person` — and only
     when their payload attribute is **present and non-empty**.
   - otherwise → `emit view_data_signal(id, type)`, re-emitted by `GraphViewer::view_data_signal`.
3. `robot_concept` connects to that signal (`wire_view_data_signal()`, `Qt::QueuedConnection`) and
   `open_stream_viewer(id, type)` maps the node to a viewer.

> **Consequence:** any node whose type is *not* one of the four legacy types (or whose legacy
> payload attribute is empty) forwards the signal — no cortex change needed. You only touch
> `dsr_gui` (and then `sudo make install libdsr_gui`) if you want to intercept one of the four
> legacy types, or add a new inline-payload probe to `has_inline_data()`.

Design rule for the widget itself: it is a **pure renderer** living in
`active_inference/common/viewers/`, is **non-`Q_OBJECT`** (the owner drives it; QTimer/signals use
lambdas → no MOC), and exposes a `set_*/push_*` API. It never reads the graph or the media plane
itself. Living in `common` (not cortex) lets it use `media_transport` without inverting the
cortex ← active_inference dependency.

---

## Recipe A — data is on the media plane (DDS stream)

For a node whose stream is published as an existing frame type (`ImageFrame`, `Image360Frame`,
`LidarFrame`, `ImuFrame`). Example viewers: `ImageStreamViewer`, `Image360Viewer`,
`LidarStreamViewer`, `ImuStreamViewer` in `src/media_stream_viewers.h`.

1. **Renderer** in `common/viewers/` (reuse `GLPointCloudViewer` / `GLImuViewer` / `ImageViewerT`,
   or add a new pure widget with a `set_*` API).
2. **Wrapper** in `src/media_stream_viewers.h` — derive from / embed the renderer; own a
   `std::unique_ptr<…Subscriber>` and a `std::jthread` **declared last**. The thread is
   **arrival-driven**: it blocks on `sub_->wait_and_poll(cb, 200)`, deep-copies the payload
   **off** the GUI thread, then pushes it in via
   `QMetaObject::invokeMethod(this, lambda, Qt::QueuedConnection)`. One frame in → one repaint.
   Destructor: `poller_.request_stop(); poller_.join();`.
3. **Wire** in `open_stream_viewer()` — one branch that builds the subscriber via the factory:
   ```cpp
   else if (node_name == "foo")
       if (auto sub = rc::media::make_image_subscriber_from_graph(*G, "foo", "rgb"))
           viewer = new rc::viewers::FooViewer(std::move(sub), "foo — … (media plane)");
   ```

Rebuild `robot_concept` only.

### A′ — the stream needs a brand-new frame type

Extra work in `common/media_transport/` before the above:
- add `foo_frame.idl`, regenerate the `*PubSubTypes.cxx` / `*TypeObjectSupport.cxx`, and list them
  in `src/CMakeLists.txt` (compiled per-agent, no shared `.so`);
- add a `FrameKind::Foo` case in `make_type_support()`;
- add `FooSubscriber` (header class + `.cpp` `init/poll/wait_and_poll/close`) and
  `make_foo_subscriber_from_graph()`, mirroring `ImuSubscriber` exactly;
- make sure the producer (robot_concept `media_ads` + `SensorMediaPublisher`) actually advertises
  and publishes the stream.

---

## Recipe B — data is a node attribute (read from the graph)

For a node type with no built-in dsr_gui widget whose data lives *in the graph*. Worked examples in
`src/graph_attr_viewers.h`:
- `RoomPolygonViewer` — `room` node's `delimiting_polygon_x/y` → `PolygonViewer`
  (`common/viewers/polygon_viewer.h`).
- `RobotMeshViewer` — `robot` node's `path` attribute (an `.obj` path) → `GLMeshViewer`
  (`common/viewers/gl_mesh_viewer.h`), loading the file with `rc::obj` (`common/obj/obj_loader.h`).
  The mesh is static, so it reloads **only when the path attribute changes** (pose updates fire the
  signal often but must not reparse the file — cache the loaded path).

Difference from Recipe A: no subscriber. Read the attribute from `G`, and drive refreshes off the
DSR update signal (which fires exactly when the producer writes — arrival-driven for free).

1. **Renderer** in `common/viewers/` — pure widget, `set_*` API, no DSR include.
2. **Wrapper** in `src/graph_attr_viewers.h` — hold `shared_ptr<DSRGraph>` + `node_id`; connect to
   `update_node_attr_signal` (and `update_node_signal`) filtered on `id == id_`, then re-read and
   push:
   ```cpp
   connect(g_.get(), &DSR::DSRGraph::update_node_attr_signal, this,
           [this](std::uint64_t id, const std::vector<std::string>&){ if(id==id_) refresh(); },
           Qt::QueuedConnection);           // NEVER DirectConnection (fires on DDS reader thread)
   // refresh(): get_node(id_) → get_attrib_by_name<foo_att>(...) → set_*()  (runs on main thread)
   ```
3. **Wire** in `open_stream_viewer()` — branch on `type` (the signal carries the node type):
   ```cpp
   else if (type == "room")
       viewer = new rc::viewers::RoomPolygonViewer(G, node_id, "room — delimiting polygon (graph)");
   ```

Rebuild `robot_concept` only.

---

## Hard rules (from CLAUDE.md)

- **`Qt::QueuedConnection`, never `Qt::DirectConnection`** on `update_node_signal` /
  `update_node_attr_signal` and on `view_data_signal` — these originate on the FastDDS reader
  threads; a Direct slot runs there and smashes the heap.
- **Deep-copy at the thread boundary.** In Recipe A the worker copies the payload
  (image→QImage, cloud→`vector<QVector3D>`, imu→scalars) *before* posting to the GUI thread —
  never share a loaned buffer / `cv::Mat` / live QImage across threads.
- **Graph reads are safe** (`get_node`/`get_attrib_by_name` are `shared_mutex`-serialized); in
  Recipe B they run on the main thread via the queued slot anyway.
- **jthread declared last** in the wrapper, so it stops/joins before the subscriber it polls is
  destroyed.

## Files

| Piece | Location |
|-------|----------|
| Reusable renderers | `common/viewers/{gl_point_cloud_viewer,gl_imu_viewer,polygon_viewer,gl_mesh_viewer}.h` |
| OBJ mesh loader | `common/obj/obj_loader.{h,cpp}` (add the `.cpp` to the agent's `src/CMakeLists.txt`) |
| Media-plane wrappers | `robot_concept/src/media_stream_viewers.h` |
| Attribute wrappers | `robot_concept/src/graph_attr_viewers.h` |
| Signal wiring + node→viewer map | `robot_concept/src/specificworker.cpp` (`wire_view_data_signal`, `open_stream_viewer`) |
| Media subscribers/factories | `common/media_transport/media_transport.{h,cpp}` |
| "View data" menu + routing | cortex `gui/.../graph_node.{h,cpp}`, `graph_viewer.{h,cpp}` |
