#pragma once
/*
 * chat_session.h — one conversation between a person and the robot, grounded in the graph.
 *
 * THE FLOW OF ONE USER MESSAGE (on_user_text):
 *   1. `stop` fast path (^(stop|para|detente|alto)\b) → the `stop` command is written at once, no model involved.
 *   2. a confirmation card is pending and the text is a yes/no → the card is confirmed/rejected in C++.
 *      The model can never confirm: confirmation exists only as a button click or this typed match.
 *   3. text already in the controller grammar (cross door / cross door_N / uncross) → CommandGate → card or reply.
 *   4. otherwise → the model, with ONE system message rebuilt this turn (robot description
 *      from etc/system_prompt.md + interface rules + the STATE digest) and the last
 *      ~16 history messages (a tool call and its results are never split by the trim).
 *
 * TOOLS: cross_door(door ∈ current doors ∪ {nearest}), stop(), cancel_passage(), move_relative(distance_m, turn_deg),
 * find_objects(subtype?), where_is(name), look(camera?).
 * Each call goes through CommandGate; a rejection returns to the model as a role:"tool" message with the reason.
 * cross_door and a LARGE move_relative only raise a card; a small move/turn is sent at once (command_gate.h).
 * Read-only queries, answered here from the digest without the controller and without a card:
 * find_objects(subtype?) and where_is(name) (object_facts.h).
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
#include <QString>
#include <QTimer>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "camera_grabber.h"
#include "chat_log.h"
#include "chat_window.h"
#include "command_gate.h"
#include "intention_node.h"
#include "mission_watch.h"
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
    GateLimits limits;                 ///< move_relative ranges + when it needs a card
    double default_step_m = 0.3;       ///< "advance a bit"
    int objects_nearest = 6;
    int object_change_period_ms = 2000;
    int look_max_side = 1280;          ///< look(): front frame shrunk to this longest side (px); see image_prep.h
    int look_max_side_360 = 1280;      ///< look(): the same for the 2:1 panorama
    int look_jpeg_quality = 80;
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

private:
    // ── the three entry points from the window ──
    void on_user_text(const QString& text);
    void confirm_card(const char* how);
    void reject_card(const char* how);

    void ask_model(bool followup);
    /// Runs every call, pushes the tool results. Returns true if one of them attached an image (look()).
    bool handle_tool_calls(const QJsonArray& calls);
    QJsonArray tools_json() const;
    QString system_prompt() const;   ///< fixed for the whole run: description + interface rules
    QString state_message() const;   ///< "STATE:\n" + digest + this turn's extras: the LAST message of a request
    static QString load_robot_description(const std::string& path);   ///< "" on failure (logged)
    void push_history(QJsonObject m);
    QJsonArray messages_for_request() const;

    /// Run a gated call: no confirmation ⇒ send now; confirmation ⇒ raise the card. Returns the tool-result text
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
    void tick_status();
    void ping_llm();
    void refresh_status_bar();

    std::shared_ptr<DSR::DSRGraph> G_;
    SessionConfig cfg_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_;
    WorldDigest digest_builder_;
    Digest digest_;
    IntentionNode intention_;
    MissionWatch mission_watch_;   ///< the controller's controller_mission node → the plan window
    OllamaClient llm_;
    std::unique_ptr<ChatWindow> window_;
    ChatLog log_;
    CameraGrabber cameras_;
    QByteArray look_jpeg_;   ///< set by do_look(), attached to that call's role:"tool" message
    QObject ctx_;
    QTimer digest_timer_, status_timer_, ping_timer_;

    // object change tracking (current room only; node id → fact)
    std::map<std::uint64_t, ObjectFact> objects_now_, window_baseline_, turn_baseline_;
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

    struct Pending
    {
        ToolCall call;
        GateResult result;
        std::string origin;
    };
    std::optional<Pending> pending_;

    // last command written and what the controller last said about it
    std::string sent_command_;
    std::uint64_t sent_stamp_ = 0;
    std::string seen_state_, seen_step_, seen_event_;
    bool warned_unarmed_ = false, warned_unacked_ = false, reported_end_ = false;

    bool controller_present_ = false;
    bool ollama_ok_ = false;
    QString ollama_msg_ = "checking…";
};

}   // namespace rc::chat
