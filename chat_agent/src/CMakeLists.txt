set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)

# The ollama HTTP client is QNetworkAccessManager (libcurl / nlohmann are not installed here).
find_package(Qt6 REQUIRED COMPONENTS Network Widgets OpenGL OpenGLWidgets)

LIST(APPEND LIBS
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::core         # header-only helpers (room_resolve, world_query, graph_provenance)
  ai_common::media        # zero-copy media plane — the ZED / Ricoh frames the look() tool shows the model
  ai_common::obj          # OBJ loader — the canvas' 3-D page (canvas_model.cpp)
  Qt6::OpenGL Qt6::OpenGLWidgets   # common/viewers/gl_mesh_viewer.h on that page
  Qt6::Network
)

# OpenCV — look(): deep-copy, shrink and JPEG-encode a camera frame (camera_grabber.cpp, image_prep.h).
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
LIST(APPEND LIBS ${OpenCV_LIBS})

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/chat_session.cpp
  ../src/chat_window.cpp
  ../src/ollama_client.cpp
  ../src/world_digest.cpp
  ../src/intention_node.cpp
  ../src/camera_grabber.cpp
  ../src/plan_view.cpp
  ../src/canvas_panel.cpp
  ../src/canvas_runner.cpp
  ../src/canvas_snapshot.cpp
  ../src/canvas_model.cpp
  ../src/mission_watch.cpp
  ../src/system_reader.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O2 -fmax-errors=1 -fno-char8_t)
