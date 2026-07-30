/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include <fps/fps.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <QTimer>
#include <QWidget>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/agent_state_publisher/agent_status_overlay.h"
#include "sensor_media_publisher.h"

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
	~SpecificWorker();

	void FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose);
	bool is_shutting_down() const noexcept { return shutting_down_.load(); }

public slots:
	void initialize();
	void compute();
	void emergency();
	void restore();
	int  startup_check();

	// Graph-viewer housekeeping: relayout (twopi) when the graph gains/loses a NODE. update_node_signal also
	// fires on every attribute write, so we relayout only when a genuinely NEW id appears (tracked below) —
	// and DEBOUNCED (a burst of node arrivals / the per-cycle attr churn collapse into one relayout).
	void modify_node_slot(std::uint64_t id, const std::string& type);
	void modify_node_attrs_slot(std::uint64_t, const std::vector<std::string>&){}
	void modify_edge_slot(std::uint64_t, std::uint64_t, const std::string&){}
	void modify_edge_attrs_slot(std::uint64_t, std::uint64_t, const std::string&, const std::vector<std::string>&){}
	void del_edge_slot(std::uint64_t, std::uint64_t, const std::string&){}
	void del_node_slot(std::uint64_t id);

private:
	bool startup_check_flag;
	AgentPresenceCoordinator presence_coordinator_;
	bool owned_nodes_cleaned_ = false;

	struct Params
	{
		float ROBOT_WIDTH  = 0.460f;
		float ROBOT_LENGTH = 0.480f;
		float ROBOT_HEIGHT = 1.6f;

		bool  TRANSFORMS_INTERPOLATE_RT = true;
		int   LIDAR_DECIMATION_FACTOR   = 1;

		// DSR upload rates (Hz):
		//   >0: throttle to that rate
		//    0: upload every captured frame
		//   <0: disable upload
		int DSR_RGB_FPS   = 0;
		int DSR_DEPTH_FPS = 5;
		// Legacy DSR-graph lidar upload rate. <0 DISABLES it (default): lidar now flows over the media
		// plane (SensorMediaPublisher LidarFrame). Set >=0 to re-enable the graph write at that fps
		// (0 = unthrottled) for consumers still reading laser_* (e.g. controller_obstacle_tracker).
		int DSR_LIDAR_FPS = -1;

		// Media plane (zero-copy DDS) for raw sensor streams carried OUT of the graph.
		// Per-sensor gates: when false, the stream's media publisher is not created,
		// its descriptor is not advertised, and its Ice reader thread is not started.
		bool        ENABLE_ZED    = true;   // ZED RGB + depth  (rc/zed/rgb, rc/zed/depth)
		// RGB source selector (Media.zed_source): how robot_concept obtains/relays the ZED plane.
		//   "auto" (default): negotiate — bridge via Ice RPC until zed_camera is seen on DDS, then follow it.
		//   "ice" : always bridge via the CameraRGBDSimple Ice proxy (never defer to an external DDS producer).
		//   "dds" : never bridge; always treat zed_camera as the external DDS producer and only monitor.
		std::string ZED_SOURCE    = "auto";
		bool        ENABLE_RICOH  = true;   // RGBD_360 panorama (rc/ricoh/rgb)
		// 360 source selector (Media.ricoh_source): mirror of ZED_SOURCE for the Ricoh plane.
		//   "auto" (default): negotiate — bridge via Ice RPC until ricoh_omni_dds is seen on DDS, then follow it.
		//   "ice" : always bridge via the Camera360RGB Ice proxy (never defer to an external DDS producer).
		//   "dds" : never bridge; always treat ricoh_omni_dds as the external DDS producer and only monitor.
		std::string RICOH_SOURCE  = "auto";
		bool        ENABLE_LIDAR  = true;   // 3D LiDAR cloud    (rc/lidar3d/points)
		// LiDAR source selector (Media.lidar_source): mirror of the above for the two lidar3d_dds
		// producers (helios + bpearl). "auto" negotiate, "ice" always bridge, "dds" never bridge.
		std::string LIDAR_SOURCE  = "auto";
		bool        ENABLE_IMU    = true;   // IMU sample        (rc/imu/data)
		// Zero-copy SHM data-sharing QoS (Media.data_sharing). Default OFF = churn-safe
		// (one memcpy/frame over the SHM transport). Set true ONLY on a static, non-churning
		// topology to test true zero-copy loans — see rc::media::PublisherConfig::data_sharing.
		// Advertised in the descriptor so consumers (voxelizer/room_concept) adopt it.
		bool        MEDIA_DATA_SHARING = false;
		int         MEDIA_DOMAIN_ID   = 0;
		std::string MEDIA_RGB_TOPIC   = "rc/zed/rgb";
		std::string MEDIA_DEPTH_TOPIC = "rc/zed/depth";
		std::string MEDIA_RICOH_TOPIC = "rc/ricoh/rgb";
		std::string MEDIA_LIDAR_TOPIC = "rc/lidar3d/points";
		std::string MEDIA_IMU_TOPIC   = "rc/imu/data";
	};
	Params params;

	bool verbose_debug_ = false;
	std::string robot_name = "Shadow";

	// Lidar reader thread
	void read_lidar_thread();
	std::thread lidar_thread;
	std::atomic<bool> stop_lidar_thread{false};

	// IMU reader thread
	void read_imu_thread();
	std::thread imu_thread;
	std::atomic<bool> stop_imu_thread{false};

	// Per-thread frame counters (relaxed); compute() turns deltas into a 3 Hz
	// [ZEDThread]/[LidarThread]/[IMUThread] Hz readout without needing Verbose.
	std::atomic<std::uint64_t> rgbd_frames_{0};
	std::atomic<std::uint64_t> lidar_frames_{0};
	std::atomic<std::uint64_t> imu_frames_{0};
	std::atomic<std::uint64_t> ricoh_frames_{0};
	// Per-lidar counters (fed by the read_lidar_thread DDS monitor when bridging is off).
	std::atomic<std::uint64_t> helios_frames_{0};
	std::atomic<std::uint64_t> bpearl_frames_{0};

	// External-DDS RECEIVED bytes per stream, summed in the monitor/poll callbacks (nonzero only while
	// an external producer owns the plane and robot_concept is monitoring it). Combined with the LOCAL
	// publish rate to report media_bps on the sensor node regardless of who produces the stream.
	std::atomic<std::uint64_t> zed_bytes_{0};
	std::atomic<std::uint64_t> ricoh_bytes_{0};
	std::atomic<std::uint64_t> helios_bytes_{0};
	std::atomic<std::uint64_t> bpearl_bytes_{0};

	// RGBD camera reader thread
	void read_rgbd_thread();
	std::thread rgbd_thread;
	std::atomic<bool> stop_rgbd_thread{false};
	// Media-source negotiation: false once zed_camera is detected publishing the ZED
	// RGB+depth plane over DDS — robot_concept then relays its descriptor into DSR and
	// stops bridging (no pull, no publish) to avoid a double producer. True = we bridge.
	std::atomic<bool> bridge_zed_{true};

	// Ricoh 360 camera reader thread: pulls the RGBD_360 panorama over Ice
	// (Camera360RGBD) and republishes the RGB image on the media plane. Bridges
	// Ice → DDS so the hardware-facing RGBD_360 stays DDS-free.
	void read_ricoh_thread();
	std::thread ricoh_thread;
	std::atomic<bool> stop_ricoh_thread{false};
	// Media-source negotiation for the 360 plane (mirror of bridge_zed_): false once
	// ricoh_omni_dds is detected publishing the Image360Frame plane over DDS — robot_concept
	// then relays its descriptor onto the "ricoh" node and stops bridging. True = we bridge.
	// The second MediaPlaneDDS proxy (addressed at ricoh_omni_dds, port 10099) is generated
	// from the CDSL's duplicate `requires MediaPlaneDDS` as mediaplanedds1_proxy — see
	// negotiate_ricoh_media_source() where it is aliased in one place.
	std::atomic<bool> bridge_ricoh_{true};
	// false once BOTH lidar3d_dds (helios+bpearl) are detected on DDS — robot_concept relays
	// their descriptors to the 'helios'/'bpearl' nodes and stops bridging LiDAR. True = we bridge.
	std::atomic<bool> bridge_lidar_{true};

	// Media plane (zero-copy DDS); RGBD pixels leave the DSR graph here.
	SensorMediaPublisher media_;

	// ── Media-source negotiation (table-driven) ──────────────────────────────────
	// One MediaGroup per sensor plane group; negotiate() runs the same auto/ice/dds
	// state machine for each: query the producer(s) over MediaPlaneDDS, relay the live
	// descriptor(s) onto the DSR node(s), and drive the group's bridge flag. When a
	// producer is absent (auto) it resumes bridging + re-advertises our own descriptor.
	// The lidar group holds TWO planes (helios+bpearl) that share one bridge_lidar_ and
	// only stop bridging once BOTH are up. Adding a sensor = one table entry (see
	// build_media_groups()).
	struct MediaPlane
	{
		std::string node;                                                  // DSR node to relay onto
		const RoboCompMediaPlaneDDS::MediaPlaneDDSPrxPtr* proxy = nullptr;  // -> generated proxy member
		std::string last_relayed;                                          // empty = advertising our own
		std::future<std::string> pending;                                  // in-flight async getMediaDescriptor
		bool present = false;                                              // last completed query had a descriptor
	};
	struct MediaGroup
	{
		std::string tag;                                                   // log tag ("ZED","360","LiDAR")
		std::vector<MediaPlane> planes;                                    // 1, or 2 for the lidar pair
		std::atomic<bool>* bridge = nullptr;                               // shared across the group's planes
		const std::string* source = nullptr;                              // *_SOURCE selector (auto|ice|dds)
		const bool* enabled = nullptr;                                     // ENABLE_* gate
		std::string advertise_node;                                        // node re-advertised when bridging
		std::vector<std::string> advertise_streams;                        // streams re-advertised when bridging
	};
	std::vector<MediaGroup> media_groups_;
	// DSR node -> its MediaPlaneDDS ICE port (parsed from the proxy). Written onto each media node as
	// the `media_ice_port` attribute so the mind view can merge a SHM producer (zed) with its ICE
	// descriptor endpoint (mediaplanedds:12002) into one box.
	std::map<std::string, int> media_ice_ports_;
	void build_media_groups();                                             // populate media_groups_ (in initialize)
	void prime_media_groups();                                             // one bounded round so startup adoption is near-immediate
	void negotiate(MediaGroup& group);                                     // one non-blocking tick (async poll-then-relaunch)
	void relay_media_descriptor(const std::string& node_name, const std::string& descriptor_json);

	// Monitor branch shared by read_rgbd_thread + read_ricoh_thread: while an external DDS
	// producer owns a single image plane (bridge off), subscribe to it, feed the compute()
	// heartbeat counter, and print a 5 s ALIVE/SILENT report. `sub` is created on demand via
	// make_sub and reset by the caller when it resumes producing. Defined in the .cpp (both
	// instantiations live there). Frame is explicit; Sub is deduced from `sub`.
	template <class Frame, class Sub>
	void monitor_external_image_plane(std::unique_ptr<Sub>& sub,
	                                  const std::type_identity_t<std::function<std::unique_ptr<Sub>()>>& make_sub,
	                                  std::atomic<std::uint64_t>& frame_counter,
	                                  std::atomic<std::uint64_t>& byte_counter,
	                                  std::uint64_t& mon_frames,
	                                  std::chrono::steady_clock::time_point& report_at,
	                                  const char* producer, const char* stream_label);

	void waiting_enter();
	void waiting_loop();
	void operating_enter();
	void operating_loop();
	void degraded_enter();
	void degraded_loop();
	void request_shutdown();

	void cleanup_owned_nodes();
	// Diagnostic (runs once): the robot node "Shadow" is seeded from shadow.json with the fixed id
	// kCanonicalShadowId. After an unclean previous exit a stale "Shadow" can survive under a DIFFERENT
	// id (no peer reaps it without etc/robot_concept.toml), which then collides on update_node. DSR name
	// uniqueness means at most one "Shadow" exists, so we do NOT auto-delete (would drop the only robot
	// node) — we log loudly so a mismatch is visible instead of silently fatal.
	static constexpr std::uint64_t kCanonicalShadowId = 200;   // == shadow.json Shadow node id
	bool shadow_checked_ = false;
	void check_shadow_identity();
	void trigger_graph_layout_twopi();   // twopi relayout of the DSR graph viewer (startup + on node add/remove)
	void schedule_graph_relayout();      // debounced request → one twopi after a burst of structural changes

	// "View data" on a media-plane sensor node (no inline blob in the graph) is forwarded by the
	// dsr_gui GraphViewer as view_data_signal(id, type). robot_concept is the one agent with the graph
	// up AND access to every stream, so it answers by opening a live media-plane viewer: an image
	// window for zed/ricoh, a top-down point view for lidar3D/helios/bpearl. Each viewer owns its own
	// media subscriber (built via the shared factory on the main thread) and polls it on a QTimer.
	void wire_view_data_signal();                                          // connect once, main thread

	// robot_concept is the only agent that displays the COMPLETE graph, so it is where the whole
	// system's health is read at a glance: every agent paints its own node (rc::AgentStatePublisher),
	// and this overlay adds the one thing a dead agent cannot report about itself — that its
	// heartbeat went stale. See common/agent_state_publisher/agent_status_overlay.h.
	void wire_agent_status_overlay();                                      // start once, main thread
	rc::AgentStatusOverlay agent_status_overlay_;

	// The stratified 3D companion to this planar view now lives in its OWN agent,
	// active_inference/scene_graph_viewer (id 16) — it needs nothing but a DSRGraph, so tying it to
	// robot_concept's lifetime only meant it died whenever this agent restarted.
	void open_stream_viewer(std::uint64_t node_id, const std::string &type);
	std::map<std::uint64_t, QWidget*> stream_viewers_;                     // one live viewer per node id
	void on_optional_peer_lost(const std::string &name, std::uint32_t id);
	void on_optional_peer_ready(const std::string &name, std::uint32_t id);
	std::atomic<bool> shutting_down_{false};

	// DSR graph-structure tracking for the viewer relayout (main-thread only). update_node_signal fires on
	// creation AND every attribute update, so we relayout only when `id` is not yet in `known_node_ids_`.
	std::set<std::uint64_t> known_node_ids_;
	QTimer* relayout_timer_ = nullptr;   // single-shot debounce; restarted on each structural change

signals:
	void presenceReady();
	void presenceLost();
};

#endif

