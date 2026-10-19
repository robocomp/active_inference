#pragma once
/*
 * chat_session.h — one conversation between a person and the robot, grounded in the graph.
 *
 * THE FLOW OF ONE USER MESSAGE (on_user_text):
 *   1. `stop` fast path (^(stop|para|detente|alto)\b) → the `stop` command is written at once, no model involved.
 *   2. text already in the controller grammar (cross door / cross door_N / uncross / move / turn) → CommandGate → sent.
 *   3. otherwise → the model, with ONE system message rebuilt this turn (robot description
 *      from etc/system_prompt.md + interface rules + the STATE digest) and the last
 *      ~16 history messages (a tool call and its results are never split by the trim).
 *
 * TOOLS: cross_door(door ∈ current doors ∪ {nearest}), stop(), cancel_passage(), move_relative(distance_m, turn_deg),
 * find_objects(subtype?), where_is(name), look(camera?), and the episodic-memory queries episodes(since?, until?,
 * author?, kind?, target?), episode(id), outcome_stats(kind, target?), last_episode(target) (em_answers.h).
 * Each call goes through CommandGate; a rejection returns to the model as a role:"tool" message with the reason.
 * An accepted call is sent at once: no confirmation card since 2026-10-18 (command_gate.h); "stop" halts the robot.
 * Read-only queries, answered here from the digest without the controller:
 * find_objects(subtype?) and where_is(name) (object_facts.h).
 * show(name) / draw(script) put a picture on the side canvas for the PERSON (canvas_runner.h runs the script against
 * a snapshot, canvas_snapshot.h); a failed script returns its stderr and the model gets up to two re-asks WITH tools
 * to fix it. "show me the zed camera / the room / the plan" never reach the model (canvas_commands.h).
 * look(camera?) is read-only too: the newest frame of the front (ZED) or 360 (Ricoh) camera is shrunk, JPEG-encoded
 * and attached to its tool result; the follow-up request, always made then, shows it to the model (look_messages.h
 * has the message layout and the one-turn lifetime of an image; camera_grabber.h the media-plane side).
 *
 * OBJECT CHANGES: births/removals of the current room's objects are diffed (by node id) against two baselines — one
 * shown in the window at most every object_change_period_ms, one put into the NEXT request's STATE ("changes in
 * objects since the last message"). The model is never called because of them. A room change resets both silently. If a tool round produced no text, one follow-up request (without tools) lets the
 * model say what happened.
 *
 * ROBOT STATUS: the controller mirrors the command's progress into chat_intention (state / bt_state); this polls it
 * every 250 ms and prints transitions as robot status lines; completion/failure is appended to the history as a
 * role:"tool" message so the model knows how the last command ended.
 *
 * EPISODIC MEMORY: episodic_agent's domain-4 graph, joined LAZILY by SpecificWorker while that agent is present
 * (set_em_graph; nullptr = not joined). The memory tools and the STATE's "today:" line read it with
 * rc::em::load_store on demand; absent or empty ⇒ "episodic memory is not available" (em_answers.h).
 *
 * Main (GUI) thread only — graph reads use the main thread's InnerEigenAPI (CLAUDE.md ts==0 ownership rule).
 */

#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QTimer>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "camera_grabber.h"
#include "canvas_commands.h"
#include "canvas_lidar.h"
#include "canvas_runner.h"
#include "chat_log.h"
#include "chat_window.h"
#include "command_gate.h"
#include "fleet_control.h"
#include "../../common/episodic/em_answers.h"
#include "intention_node.h"
#include "mission_watch.h"
#include "system_reader.h"
#include "ollama_client.h"
#include "world_digest.h"

namespace rc::chat
{

struct SessionConfig
{
    OllamaConfig llm;
    int max_history = 16;
    int digest_period_ms = 500;
    int status_period_ms = 250;
    std::string log_dir = "logs";
    std::string system_prompt_path = "etc/system_prompt.md";   ///< robot description; empty/missing ⇒ built-in
    GateLimits limits;                 ///< move_relative ranges
    double default_step_m = 0.3;       ///< "advance a bit"
    double robot_radius_m = 0.35;      ///< approach_object: half the body width (0.70 m)
    double approach_clearance_m = 0.30;   ///< approach_object: free space kept between the body and the object
    int objects_nearest = 6;
    int object_change_period_ms = 2000;
    int look_max_side = 1280;          ///< look(): front frame shrunk to this longest side (px); see image_prep.h
    int look_max_side_360 = 1280;      ///< look(): the same for the 2:1 panorama
    int look_jpeg_quality = 80;
    std::string canvas_dir = "canvas";   ///< bootstrap.py, rcanvas.py, stock/, library/ (canvas_runner.h)
    int canvas_timeout_ms = 10000;       ///< a canvas script is killed after this long
    int canvas_live_period_ms = 100;     ///< live drawings redraw at most this often (a redraw in flight skips a tick)
    std::string canvas_memory_max = "1G";
    int gpu_period_ms = 2000;            ///< status bar's GPU memory poll (nvidia-smi takes ~50 ms; 2 s is plenty to watch)
    // agent_control (fleet_control.h): the agents the chat may start / stop, each as <fleet_dir>/<agent>/bin/<agent>
    std::string fleet_dir = "..";
    std::string fleet_agents = "room_concept,residual_concept,retina,door_concept,table_concept,chair_concept,"
                               "bottle_concept,cabinet_concept,hood_concept,human_concept,refrigerator_concept,controller,"
                               "ltsm_agent,episodic_agent";
    std::string fleet_confirm = "controller,retina";
    std::string fleet_lock;               ///< "" ⇒ $XDG_RUNTIME_DIR/ai_fleet_tour.lock (the tour scripts hold it)
    std::string fleet_log_dir = "logs/agents";
};

class ChatSession
{
public:
    ChatSession(std::shared_ptr<DSR::DSRGraph> G, SessionConfig cfg);
    ~ChatSession();

    void start();                                  ///< window up, timers running, ollama pinged + warmed
    void set_controller_present(bool present);
    /// Startup / first-Operating sweep of a crashed run's chat_intention.
    void sweep_stale() { intention_.sweep_stale(); }
    /// episodic_agent's domain-4 graph (joined by SpecificWorker while it is present); nullptr = not joined.
    void set_em_graph(std::shared_ptr<DSR::DSRGraph> em) { em_G_ = std::move(em); }

private:
    // ── the three entry points from the window ──
    void on_user_text(const QString& text);

    void ask_model(bool followup);
    /// Runs every call, pushes the tool results. Returns true if one of them attached an image (look()).
    bool handle_tool_calls(const QJsonArray& calls);
    QJsonArray tools_json() const;
    QString system_prompt() const;   ///< fixed for the whole run: description + interface rules
    QString state_message() const;   ///< "STATE:\n" + digest + this turn's extras: the LAST message of a request
    static QString load_robot_description(const std::string& path);   ///< "" on failure (logged)
    void push_history(QJsonObject m);
    QJsonArray messages_for_request() const;

    /// Run a gated call: accepted ⇒ sent now (no confirmation, 2026-10-18). Returns the tool-result text
    /// (for a role:"tool" message); `reply`, if given, gets the same outcome as a plain sentence for the person —
    /// what the history records when no model was involved (a raw tool-result string there is parroted back).
    QString dispatch(const ToolCall& call, const char* origin, QString* reply = nullptr);
    bool send_command(const std::string& command, const char* origin);

    void tick_digest();
    void track_object_changes();
    /// find_objects / where_is → the tool-result text; "" if `name` is not a query tool.
    QString answer_query(const QString& name, const QJsonObject& args);
    /// look(camera?) → the tool-result text; on success the JPEG is left in look_jpeg_ for the tool message.
    QString do_look(const QJsonObject& args);
    /// show(name, args?, live?) / draw(script, title, live?) → the tool-result text; the picture goes to the canvas.
    QString do_draw(const QString& tool, const QJsonObject& args);
    QString do_table(const QJsonObject& args);   ///< the table tool: stock/table.py with the model's rows
    /// Run `job` now and put its picture on the canvas; "" on success, else why (also shown on the canvas).
    QString draw_now(CanvasJob job, bool savable);
    void show_request(const ShowRequest& r, const QString& text);   ///< the canvas fast path
    QString show_model(const std::string& what);   ///< the 3-D page (canvas_model.h); "OK: …" or "ERROR: …"
    QString show_lidar(const std::string& which);  ///< the point-cloud page (canvas_lidar.h): helios | bpearl | both
    void canvas_stop();   ///< stop whatever the canvas keeps live: a script worker, the camera, the lidars
    std::optional<CanvasJob> library_job(const QString& name, const QStringList& args) const;
    QStringList canvas_library() const;   ///< stock + saved script names
    void write_canvas_inputs(const CanvasJob& job);
    void tick_canvas();                   ///< live re-run
    void save_canvas_script(const QString& name);
    /// approach_object: pick the object, put a standpoint short of it, send a one-step go_to mission (approach.h).
    QString do_approach(const QJsonObject& args);
    /// episodes / episode / outcome_stats / last_episode → the tool-result text; "" if `name` is not one of them.
    QString answer_memory(const QString& name, const QJsonObject& args) const;
    /// Today's records + the all-time stats from the domain-4 graph; nullopt when it is not joined.
    std::optional<rc::em::Store> em_store() const;
    void tick_status();
    void ping_llm();
    void refresh_status_bar();
    void poll_gpu();   ///< start an nvidia-smi query unless one is still running; its result lands in gpu_label_
    /// agent_control: start / stop / restart a fleet agent (fleet_control.h); the outcome is told later by tick_fleet.
    QString do_agent_control(const QJsonObject& args);
    void tick_fleet();
    bool launch_agent(const std::string& agent);   ///< `cd <agent> && setsid bin/<agent> etc/config.toml`, detached
    [[nodiscard]] QString agent_log(const std::string& agent) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    std::shared_ptr<DSR::DSRGraph> em_G_;   ///< episodic memory (domain 4), nullptr = not joined
    SessionConfig cfg_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_;
    WorldDigest digest_builder_;
    Digest digest_;
    IntentionNode intention_;
    MissionWatch mission_watch_;   ///< the controller's controller_mission node → the plan window
    rc::mission::Progress mission_progress_;   ///< its last progress (the STATE's "controller mission now" line)
    OllamaClient llm_;
    std::unique_ptr<ChatWindow> window_;
    ChatLog log_;
    CameraGrabber cameras_;
    SystemReader system_{G_, cameras_};   ///< agents / sensors / graph / inspect (after cameras_: it reads their rates)
    CanvasRunner canvas_runner_;
    CanvasJob canvas_job_;          ///< what the canvas shows now (re-run while live)
    std::optional<Camera> canvas_camera_;   ///< the stock camera drawing: frames shown directly, no script
    LidarGrabber lidars_{G_};               ///< the point-cloud page's subscribers (made only while it shows)
    bool canvas_lidar_ = false;             ///< the point-cloud page is up: the canvas timer polls lidars_
    bool lidar_logged_ = false;             ///< its first frame has been logged
    bool canvas_live_ = false;
    bool canvas_savable_ = false;
    int draw_repairs_ = 0;          ///< model re-asks after a failed draw, this user message (cap kMaxDrawRepairs)
    int lost_answers_ = 0;          ///< model answers this user message that came back empty (one is re-asked)
    bool draw_failed_ = false;      ///< a draw in the current tool round failed
    QTimer canvas_timer_;
    QByteArray look_jpeg_;   ///< set by do_look(), attached to that call's role:"tool" message
    QObject ctx_;
    QTimer digest_timer_, status_timer_, ping_timer_;

    // object change tracking (current room only; node id → fact)
    std::map<std::uint64_t, ObjectFact> objects_now_, window_baseline_, turn_baseline_;
    std::vector<std::string> turn_doors_;   ///< the current room's doors at the last request (doors_gone)
    std::uint64_t objects_room_ = 0;
    bool objects_init_ = false;
    std::int64_t window_flush_ms_ = 0;
    QString state_extra_;          ///< lines appended to the STATE for THIS request (changes, last move)
    std::string last_relative_;    ///< last move/turn command this chat sent ("" = none)

    QString robot_description_;   ///< etc/system_prompt.md, read once; empty ⇒ the built-in fallback
    std::deque<QJsonObject> history_;
    struct Turn
    {
        QString content;
        QJsonArray tool_calls;
        bool followup = false;
        bool streaming = false;
        bool carries_image = false;   ///< this request sent an image ⇒ strip it from the history when it ends
    } turn_;


    // last command written and what the controller last said about it
    std::string sent_command_;
    std::uint64_t sent_stamp_ = 0;
    std::string seen_state_, seen_step_, seen_event_;
    bool warned_unarmed_ = false, warned_unacked_ = false, reported_end_ = false;

    bool controller_present_ = false;
    bool ollama_ok_ = false;
    QString ollama_msg_ = "checking…";
    QString gpu_label_ = QStringLiteral("GPU n/a");   ///< whole-GPU memory, from gpu_memory.h
    QTimer gpu_timer_;
    QProcess gpu_proc_;   ///< the async nvidia-smi query (never waited on in the GUI thread)
    fleet::Policy fleet_policy_;
    fleet::Confirmations fleet_confirmations_;
    std::optional<fleet::Op> fleet_op_;   ///< the start / stop / restart in progress
    QTimer fleet_timer_;
    std::uint64_t user_turn_ = 0;          ///< messages typed by the person: a yes must come in a later one
};

}   // namespace rc::chat
