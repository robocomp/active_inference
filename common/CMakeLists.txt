# =====================================================================================
# common/CMakeLists.txt — the shared ai_common::* libraries for the active_inference agents.
#
# CONSUMED VIA add_subdirectory(), NEVER standalone. From a component's src/CMakeLists.txt:
#
#     set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")
#     add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
#     LIST(APPEND LIBS ai_common::media ai_common::presence ai_common::core)
#
# `LIBS` is what generated/CMakeLists.txt feeds to TARGET_LINK_LIBRARIES, so appending the
# target names is the whole consumer-side change — the RoboComp framework files are untouched.
#
# DESIGN: one target per EXTERNAL-DEPENDENCY cluster (FastDDS / Pinocchio / Qt-GL / DSR-only /
# header-only). The target boundary follows the dependency boundary, because that is exactly
# what a consumer would otherwise have to know and hand-wire.
#
# ORDERING: place add_subdirectory() near the TOP of the component's src/CMakeLists.txt. It must
# run AFTER robocomp.cmake/qt.cmake (it does — src/ is textually INCLUDEd into generated/ after
# both) and BEFORE the component's add_definitions(), so the ai_common targets do NOT silently
# inherit the component's flags. Everything they need is declared explicitly below.
#
# EXCLUDE_FROM_ALL: only the targets a component actually links get built.
# =====================================================================================
cmake_minimum_required(VERSION 3.16)

option(ENABLE_ASAN "Instrument ai_common with AddressSanitizer (forwarded from the component)" OFF)
option(ENABLE_TSAN "Instrument ai_common with ThreadSanitizer (forwarded from the component)" OFF)

# -------------------------------------------------------------------------------------
# ABI guard. libdsr was built with DEFAULT Eigen alignment: sizeof(std::optional<Eigen::Transform>)
# must stay 144. -march=native raises alignof(Transform) to 32, -DEIGEN_MAX_ALIGN_BYTES=0 drops it
# to 8; either one makes get_transformation_matrix overrun or misalign the caller's stack slot
# (stack smash / SIGBUS).
#
# SCOPE — measured, not assumed. This catches the flag only when it arrives via CMAKE_CXX_FLAGS
# (command line, toolchain file, CMakeCache) or add_compile_options(). It CANNOT catch
# add_definitions(-march=native), which is the form every component here uses: verified on CMake
# 3.28 that add_definitions() with a non "-D" argument reaches the compile line but is stored in
# NO queryable property — not COMPILE_OPTIONS, not COMPILE_DEFINITIONS, not the legacy DEFINITIONS.
# Deferring the check to the consumer's directory (so it runs after their add_definitions) does not
# help for the same reason, and cmake_language(DEFER) re-evaluates its arguments in the deferred
# scope, so "${var}" arrives empty there.
#
# So this is a backstop, NOT the enforcement mechanism. The per-component comment blocks are still
# the primary defence; do not delete them on the strength of this guard.
# -------------------------------------------------------------------------------------
get_directory_property(_ai_common_dir_opts COMPILE_OPTIONS)
if("${CMAKE_CXX_FLAGS} ${_ai_common_dir_opts}" MATCHES "march=native|EIGEN_MAX_ALIGN_BYTES")
  message(FATAL_ERROR
    "ai_common: -march=native / -DEIGEN_MAX_ALIGN_BYTES change Eigen's alignment away from the "
    "prebuilt libdsr's (sizeof(optional<Transform>)==144) and cause SIGBUS/stack-smash inside "
    "get_transformation_matrix. Offending flags: ${CMAKE_CXX_FLAGS} ${_ai_common_dir_opts}")
endif()

# -------------------------------------------------------------------------------------
# External packages. These are found HERE, not inherited — and that is the design, not a
# concession: a target should declare its own usage requirements rather than pick them up from
# whatever ambient directory scope happens to be active. (It is also necessary: generated/
# CMakeLists.txt runs find_package(Eigen3) and ROBOCOMP_INITIALIZE only AFTER it INCLUDEs
# ../src/CMakeLists.txt, so neither Eigen3::Eigen nor $ROBOCOMP/classes exists yet at
# add_subdirectory() time.)
#
# The Qt6 call duplicates one qt.cmake already made; a repeat find_package on a config-mode
# package that already found its targets costs ~0 ms, and it keeps this file self-contained.
# -------------------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets StateMachine)
find_package(Qt6 QUIET COMPONENTS OpenGL OpenGLWidgets)

# FastDDS and DSR ship no usable CMake package config here — they are plain -l names, exactly as
# the framework's TARGET_LINK_LIBRARIES uses them.
set(AI_COMMON_FASTDDS_LIBS fastdds fastcdr)

# Pinocchio is only needed by ai_common::kinematics (kinova_controller, self_calibration).
# Opt in with `set(AI_COMMON_WITH_KINEMATICS ON)` BEFORE add_subdirectory().
#
# One named flag, not a magic incantation: consumers used to have to remember
# `list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")`, and forgetting it silently dropped the
# target, surfacing as an unrelated "unknown target ai_common::kinematics" at generate time. Now
# the flag says what it wants, the prefix path is OUR business, and REQUIRED fails loudly here.
#
# PATHS alone is not enough and the prefix path is not optional: pinocchioConfig.cmake does
# find_dependency(hpp-fcl), which does NOT inherit the outer call's PATHS. Setting
# CMAKE_PREFIX_PATH is what makes the transitive lookup work. It stays local — this directory has
# its own scope, so nothing leaks back to the consumer.
#
# It is opt-in rather than unconditional because pinocchio's config is expensive to probe, and 14
# of the 16 components have no use for it.
if(AI_COMMON_WITH_KINEMATICS)
  list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")
  find_package(pinocchio REQUIRED)
endif()

# DSR ships no CMake package config — dsr_core / dsr_api / dsr_gui are plain -l names, exactly
# as the framework's TARGET_LINK_LIBRARIES uses them.
set(AI_COMMON_DSR_LIBS dsr_api dsr_core)

# -------------------------------------------------------------------------------------
# Shared settings applied to every compiled ai_common target.
# -------------------------------------------------------------------------------------
function(ai_common_defaults tgt)
  # A FLOOR, not a clamp: components on C++26 (g++ 15 default here) stay on 26; human_concept
  # and room_concept clamp themselves to 23 and still satisfy this.
  target_compile_features(${tgt} PUBLIC cxx_std_23)

  # PUBLIC: -fno-char8_t changes the type of u8"" literals, so it is an ABI property, not a
  # private convenience. Every component already sets it; declaring it here makes it explicit.
  target_compile_options(${tgt} PUBLIC -fno-char8_t)
  # -O3/-fmax-errors are what the library boundary genuinely loses (the component's
  # add_definitions() does not reach this subdirectory). -Wall is NOT repeated here: qt.cmake
  # already puts it in the global CMAKE_CXX_FLAGS that every target inherits.
  target_compile_options(${tgt} PRIVATE -O3 -fmax-errors=1)

  target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(${tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Today ASan/TSan reach the common sources through the component's directory-scope
  # add_compile_options(). Behind a library boundary that is lost unless it is forwarded.
  if(ENABLE_ASAN)
    target_compile_options(${tgt} PRIVATE -fsanitize=address -fno-omit-frame-pointer -O1)
  endif()
  if(ENABLE_TSAN)
    target_compile_options(${tgt} PRIVATE -fsanitize=thread -fno-omit-frame-pointer -O1)
  endif()
endfunction()

# =====================================================================================
# ai_common::core — header-only. Zero external deps beyond Eigen.
#
#   ai_belief/{dof_spec,lidar_ray_factor,recursive_laplace}.h   existence_belief/
#   appearance_belief/  occlusion/  instance_tracker/  belief_stabilizer/  bearing_confirm/
#   motion_filter/  motion_corruption/  sample_queue/  phantom_log/  depth_projection/
#   object_anchor/  affordance_protocol/  graph3d/graph3d_scene.h
#   agent_state_publisher/{agent_status.h,agent_state_publisher.h}
#   kinematics/self_projection_capsules.h  self_tuner/self_tuner.h
#
# Some of these include <dsr/...> (depth_projection, object_anchor_contract, affordance_protocol),
# so the DSR libs ride along on the INTERFACE. Qt6::Core rides along too: dsr_api.h pulls in
# dsr_agent_info_api.h, which includes <QDebug> — every DSR consumer needs Qt Core's headers.
# =====================================================================================
add_library(ai_common_core INTERFACE)
target_include_directories(ai_common_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ai_common_core INTERFACE $ENV{ROBOCOMP}/classes)
target_compile_features(ai_common_core INTERFACE cxx_std_23)
target_compile_options(ai_common_core INTERFACE -fno-char8_t)
target_link_libraries(ai_common_core INTERFACE Eigen3::Eigen Qt6::Core ${AI_COMMON_DSR_LIBS})
add_library(ai_common::core ALIAS ai_common_core)

# =====================================================================================
# ai_common::presence — the agent presence protocol. 14 consumers, zero flag divergence.
# agent_presence_coordinator.h includes <ConfigLoader/ConfigLoader.h>, which today only resolves
# because ROBOCOMP_INITIALIZE puts $ROBOCOMP/classes on the inherited directory include path.
# Name it explicitly so the target stands on its own.
# =====================================================================================
add_library(ai_common_presence STATIC
  agent_presence_monitor/agent_presence_monitor.cpp
  agent_presence_coordinator/agent_presence_coordinator.cpp
)
ai_common_defaults(ai_common_presence)
target_include_directories(ai_common_presence PUBLIC $ENV{ROBOCOMP}/classes)
# Only what this target adds BEYOND ai_common_core (which already carries Eigen, Qt6::Core and the
# DSR libs on its INTERFACE): Qt6::StateMachine, for agent_state_publisher.h's
# <QStateMachine>/<QAbstractState>.
target_link_libraries(ai_common_presence PUBLIC ai_common_core Qt6::StateMachine)
add_library(ai_common::presence ALIAS ai_common_presence)

# =====================================================================================
# ai_common::media — the zero-copy media plane (DDS domain 7). 10 consumers, all pulling an
# IDENTICAL file set. This single target replaces a 10-line IDL splice plus an
# include_directories() line in each of them.
#
# All eight IDL .cxx are unconditional: media_transport.cpp's WriterCore::init/make_type_support
# reference every FrameKind PubSubType, so even a subscriber-only consumer needs them all.
# (With a STATIC archive the linker still only pulls the objects that are actually referenced.)
#
# The generated IDL headers include each other flat (#include "image_frameCdrAux.hpp"), which is
# why media_transport/ and media_transport/generated/idl/ are PUBLIC include dirs.
# =====================================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/media_transport/idl_sources.cmake)
add_library(ai_common_media STATIC
  media_transport/media_transport.cpp
  media_transport/lidar_plane_reader.cpp
  ${AI_COMMON_IDL_SOURCES}
)
ai_common_defaults(ai_common_media)
target_include_directories(ai_common_media PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/media_transport
  ${AI_COMMON_IDL_DIR}
)
target_link_libraries(ai_common_media PUBLIC ai_common_core ${AI_COMMON_FASTDDS_LIBS})
add_library(ai_common::media ALIAS ai_common_media)

# =====================================================================================
# ai_common::ui — TimeSeriesPlot. Qt Widgets only, no DSR gui. AUTOMOC handles it via the
# same-basename rule (timeseries_plot.h + .cpp, Q_OBJECT).
#
# Kept separate from ai_common::status_overlay below: 9 components want the plot, exactly one
# (robot_concept) wants the overlay. Merging them made 8 components compile the overlay TU for
# nothing AND pulled dsr_gui into their link line, which is what forced scene_graph_viewer to
# document "ai_common::ui deliberately NOT linked".
# =====================================================================================
add_library(ai_common_ui STATIC dashboard/timeseries_plot.cpp)
ai_common_defaults(ai_common_ui)
set_target_properties(ai_common_ui PROPERTIES AUTOMOC ON)
target_link_libraries(ai_common_ui PUBLIC ai_common_core Qt6::Gui Qt6::Widgets)
add_library(ai_common::ui ALIAS ai_common_ui)

# =====================================================================================
# ai_common::status_overlay — the observer half of the agent-health display (greys out agent
# nodes whose heartbeat went stale). robot_concept only: it is the agent that shows the complete
# graph. Separate target because this is the one thing under common/ that reaches into the DSR
# *gui* (graph_viewer/graph_node), and dsr_gui's AbstractGraphicViewer collides with the
# component-local one, so nobody else should be linking it.
# =====================================================================================
add_library(ai_common_status_overlay STATIC agent_state_publisher/agent_status_overlay.cpp)
ai_common_defaults(ai_common_status_overlay)
set_target_properties(ai_common_status_overlay PROPERTIES AUTOMOC ON)
target_link_libraries(ai_common_status_overlay PUBLIC ai_common_core Qt6::Gui Qt6::Widgets dsr_gui)
add_library(ai_common::status_overlay ALIAS ai_common_status_overlay)

# =====================================================================================
# ai_common::ui_headers — INTERFACE ONLY, and it must stay that way.
#
#   dashboard/{belief_inspector,custom_widget,evidence_monitor}.h
#   viewers/gl_{graph3d,grid_field,imu,mesh,point_cloud,skeleton}_viewer.h
#   viewers/{polygon_viewer,semantic_map_viewer,grid_surface_builder}.h
#
# These are header-only Q_OBJECT classes. They are mocced TODAY by the *consumer's* AUTOMOC,
# through its scan of the headers #included by the component's own .cpp files. That keeps
# working unchanged. Listing them as sources of a compiled target here would generate a SECOND
# moc translation unit -> duplicate staticMetaObject symbols at link. Do not.
# =====================================================================================
add_library(ai_common_ui_headers INTERFACE)
target_include_directories(ai_common_ui_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ai_common_ui_headers INTERFACE cxx_std_23)
target_compile_options(ai_common_ui_headers INTERFACE -fno-char8_t)
target_link_libraries(ai_common_ui_headers INTERFACE ai_common_core Qt6::Gui Qt6::Widgets)
if(TARGET Qt6::OpenGL AND TARGET Qt6::OpenGLWidgets)
  target_link_libraries(ai_common_ui_headers INTERFACE Qt6::OpenGL Qt6::OpenGLWidgets)
else()
  message(STATUS "ai_common: Qt6 OpenGL/OpenGLWidgets not found; common/viewers/gl_*.h consumers "
                 "must supply them (the framework's bare Qt6OpenGLWidgets name still applies)")
endif()
add_library(ai_common::ui_headers ALIAS ai_common_ui_headers)

# =====================================================================================
# ai_common::perception — mask ingestion. DSR-backed. No OpenCV: nothing under common/ actually
# includes <opencv2/...> (the one 'cv::Mat' hit is prose in a comment).
#
# These next three were one "perception" target. They are split because their consumer sets are
# DISJOINT — mask_ingestor: 6 components, graph3d: 1, obj: 2 — so bundling them made every
# consumer compile the other two for nothing. It also violated this file's own rule: obj_loader
# needs Qt6::Gui and NO DSR at all, which is a different dependency cluster.
# =====================================================================================
add_library(ai_common_perception STATIC mask_ingestor/mask_ingestor.cpp)
ai_common_defaults(ai_common_perception)
target_link_libraries(ai_common_perception PUBLIC ai_common_core)
add_library(ai_common::perception ALIAS ai_common_perception)

# =====================================================================================
# ai_common::graph3d — stratified 3D graph scene builder (scene_graph_viewer).
# Depends on dsr_api ONLY, never dsr_gui, so it stays includable from anywhere.
# =====================================================================================
add_library(ai_common_graph3d STATIC graph3d/graph3d_scene_builder.cpp)
ai_common_defaults(ai_common_graph3d)
target_link_libraries(ai_common_graph3d PUBLIC ai_common_core)
add_library(ai_common::graph3d ALIAS ai_common_graph3d)

# =====================================================================================
# ai_common::obj — Wavefront OBJ loader for the display meshes named by nodes' mesh_path attr.
# Qt6::Gui for <QVector3D>; no DSR.
# =====================================================================================
add_library(ai_common_obj STATIC obj/obj_loader.cpp)
ai_common_defaults(ai_common_obj)
target_link_libraries(ai_common_obj PUBLIC Eigen3::Eigen Qt6::Gui)
add_library(ai_common::obj ALIAS ai_common_obj)

# =====================================================================================
# ai_common::geometry — footprint + robust metrics. PURE EIGEN: no Qt, no Ice, no DSR.
#
# Deliberately does NOT link ai_common_core, which drags in Qt6::Core and the DSR libs. Keeping
# this cluster dependency-free is what lets controller/tools/ (route_bench, mppi_bench) link the
# SAME robot_footprint the component runs while still building in seconds without Qt/Ice/DSR.
# =====================================================================================
add_library(ai_common_geometry STATIC
  robot_footprint/robot_footprint.cpp
  robust_metrics/robust_metrics.cpp
)
ai_common_defaults(ai_common_geometry)
target_link_libraries(ai_common_geometry PUBLIC Eigen3::Eigen)
add_library(ai_common::geometry ALIAS ai_common_geometry)

# =====================================================================================
# ai_common::affordance — affordance management (controller, room_concept).
# affordance_manager.cpp uses QDebug and the DSR api, so this is a different dependency cluster
# from ai_common::geometry above and must stay a separate target.
# =====================================================================================
add_library(ai_common_affordance STATIC affordance_manager/affordance_manager.cpp)
ai_common_defaults(ai_common_affordance)
target_link_libraries(ai_common_affordance PUBLIC ai_common_core)
add_library(ai_common::affordance ALIAS ai_common_affordance)

# =====================================================================================
# ai_common::kinematics — Pinocchio. This target is the design in miniature: kinova_controller
# and self_calibration each hand-write the prefix path, the find_package, an
# include_directories(common/kinematics) and -DCOAL_DISABLE_HPP_FCL_WARNINGS in their base
# add_definitions() line. All four now live on the target that needs them.
# =====================================================================================
if(AI_COMMON_WITH_KINEMATICS)
  add_library(ai_common_kinematics STATIC kinematics/kinematics.cpp)
  ai_common_defaults(ai_common_kinematics)
  target_include_directories(ai_common_kinematics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kinematics)
  # PUBLIC: it suppresses warnings from Pinocchio/Coal headers, which consumers also include.
  target_compile_definitions(ai_common_kinematics PUBLIC COAL_DISABLE_HPP_FCL_WARNINGS)
  target_link_libraries(ai_common_kinematics PUBLIC ai_common_core pinocchio::pinocchio)
  add_library(ai_common::kinematics ALIAS ai_common_kinematics)
endif()
