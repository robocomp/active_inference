#include "chat_session.h"

#include <cmath>
#include <print>
#include <ranges>
#include <vector>

#include <QDateTime>
#include <QFile>
#include <cstdlib>
#include <QJsonDocument>
#include <QRegularExpression>

#include "../../common/episodic/em_query_dsr.h"            // rc::em::load_store
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::now_ms
#include "image_prep.h"
#include "look_messages.h"

namespace rc::chat
{

namespace
{
std::string norm(const QString& t) { return t.trimmed().toLower().toStdString(); }
QString q(const std::string& s) { return QString::fromStdString(s); }

std::vector<std::string> split_bar(const std::string& s)
{
    std::vector<std::string> parts;
    for (auto&& p : s | std::views::split('|')) parts.emplace_back(p.begin(), p.end());
    return parts;
}

// qwen3 may still wrap a (here empty) reasoning block around the answer; it must not reach the history.
QString strip_think(QString s)
{
    static const QRegularExpression re(QStringLiteral("<think>.*?</think>"), QRegularExpression::DotMatchesEverythingOption);
    return s.remove(re).trimmed();
}

QJsonObject tool_def(const QString& name, const QString& description, QJsonObject properties, QJsonArray required)
{
    QJsonObject params{{"type", "object"}, {"properties", properties}};
    if (not required.isEmpty()) params.insert("required", required);
    return QJsonObject{{"type", "function"},
                       {"function", QJsonObject{{"name", name}, {"description", description}, {"parameters", params}}}};
}
}   // namespace

ChatSession::ChatSession(std::shared_ptr<DSR::DSRGraph> G, SessionConfig cfg)
    : G_(std::move(G)), cfg_(std::move(cfg)), inner_(G_->get_inner_eigen_api()), digest_builder_(G_, "body", cfg_.objects_nearest),
      intention_(G_), mission_watch_(G_), llm_(cfg_.llm), cameras_(G_)
{
    robot_description_ = load_robot_description(cfg_.system_prompt_path);
    if (not log_.open(cfg_.log_dir)) std::println("[chat_agent] WARNING: cannot open a log under '{}'", cfg_.log_dir);
    else std::println("[chat_agent] logging to {}", log_.path());
    log_.write("start", QJsonObject{{"model", cfg_.llm.model}, {"url", cfg_.llm.url}, {"num_ctx", cfg_.llm.num_ctx},
                                    {"temperature", cfg_.llm.temperature},
                                    {"system_prompt", q(cfg_.system_prompt_path)},
                                    {"system_prompt_loaded", not robot_description_.isEmpty()}});
}

ChatSession::~ChatSession()
{
    llm_.cancel();
    log_.write("stop");
}

void ChatSession::start()
{
    window_ = std::make_unique<ChatWindow>(ChatWindow::Callbacks{
        .on_send = [this](const QString& t) { on_user_text(t); },
        .on_cancel = [this]()
        {
            if (llm_.busy()) { llm_.cancel(); }
        },
        .on_new = [this]()
        {
            llm_.cancel();
            history_.clear();
            look_jpeg_.clear();
            last_relative_.clear();   // "a bit more" has nothing to repeat in a new conversation
            window_->clear_transcript();
            window_->append_system("New conversation.");
            log_.write("new_conversation");
        },
    });
    window_->show();
    window_->append_system("Robot chat ready. I answer from the robot's live map only. Type \"stop\" to stop it at once.");

    QObject::connect(&digest_timer_, &QTimer::timeout, &ctx_, [this]() { tick_digest(); });
    QObject::connect(&status_timer_, &QTimer::timeout, &ctx_, [this]() { tick_status(); });
    QObject::connect(&ping_timer_, &QTimer::timeout, &ctx_, [this]() { ping_llm(); });
    digest_timer_.start(cfg_.digest_period_ms);
    status_timer_.start(cfg_.status_period_ms);
    ping_timer_.start(30000);
    tick_digest();
    ping_llm();
    refresh_status_bar();

    // TEST HOOK: CHAT_AGENT_SCRIPT=<file> types each non-empty line into the window, one every 12 s (first at 5 s),
    // so the session can be exercised end to end without a person or an X automation tool. Unset = no effect.
    if (const char* script = std::getenv("CHAT_AGENT_SCRIPT"); script != nullptr)
    {
        QFile f(QString::fromUtf8(script));
        if (f.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            int k = 0;
            for (const QString& line : QString::fromUtf8(f.readAll()).split('\n', Qt::SkipEmptyParts))
                QTimer::singleShot(5000 + 12000 * k++, &ctx_, [this, line]() { on_user_text(line.trimmed()); });
            std::println("[chat_agent] TEST SCRIPT '{}': {} lines", script, k);
        }
    }
}

void ChatSession::set_controller_present(bool present)
{
    if (present == controller_present_) return;
    controller_present_ = present;
    log_.write("controller_presence", QJsonObject{{"present", present}});
    if (window_) window_->append_system(present ? "The controller is present." : "The controller has gone away.");
    refresh_status_bar();
}

void ChatSession::ping_llm()
{
    llm_.ping([this](const OllamaClient::PingResult& r)
    {
        const bool was_ok = ollama_ok_;
        ollama_ok_ = r.server_up and r.model_present;
        if (r.message != ollama_msg_)
        {
            log_.write("ollama_ping", QJsonObject{{"up", r.server_up}, {"model_present", r.model_present},
                                                  {"version", r.version}, {"msg", r.message}});
            if (window_ and not ollama_ok_) window_->append_system(r.message);
        }
        ollama_msg_ = r.message;
        if (ollama_ok_ and not was_ok) llm_.warmup();   // load the model before the first question pays for it
        refresh_status_bar();
    });
}

void ChatSession::refresh_status_bar()
{
    if (window_)
        window_->set_status(cfg_.llm.model, ollama_ok_ ? QStringLiteral("ollama ok") : ollama_msg_, ollama_ok_,
                            controller_present_);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
void ChatSession::tick_digest()
{
    intention_.ensure();
    cameras_.discover();   // self-throttled; subscribers appear once the camera nodes + descriptors exist
    cameras_.refresh(Camera::front);   // keeps the measured frame rates current for sensors() (camera_grabber.h)
    cameras_.refresh(Camera::pano);
    digest_ = digest_builder_.build(*inner_, controller_present_, intention_.read(), today_digest_line(em_store()));
    track_object_changes();
}

// Births/removals of the current room's objects. Two baselines: the window's (flushed at most every
// object_change_period_ms, so a tracker flap inside that window cancels out) and the next model turn's (consumed in
// ask_model). Entering another room is not a change of the world: both baselines are reset without a report.
void ChatSession::track_object_changes()
{
    objects_now_.clear();
    for (const auto& o : digest_.objects)
        if (o.in_current_room) objects_now_.emplace(o.id, o);
    const std::int64_t now = static_cast<std::int64_t>(rc::provenance::now_ms());
    if (not objects_init_ or digest_.room_id != objects_room_)
    {
        if (objects_init_ and digest_.room_id != 0)
            log_.write("objects_room", QJsonObject{{"room", q(digest_.facts.room)}, {"n", static_cast<int>(objects_now_.size())}});
        objects_init_ = true;
        objects_room_ = digest_.room_id;
        window_baseline_ = turn_baseline_ = objects_now_;
        turn_doors_ = digest_.facts.doors;
        window_flush_ms_ = now;
        return;
    }
    if (now - window_flush_ms_ < cfg_.object_change_period_ms) return;
    window_flush_ms_ = now;
    const ObjectChanges c = diff_objects(window_baseline_, objects_now_);
    window_baseline_ = objects_now_;
    if (c.empty()) return;
    const std::string line = changes_line(c);
    log_.write("objects_changed", QJsonObject{{"room", q(digest_.facts.room)}, {"changes", q(line)}});
    if (window_) window_->append_system(QStringLiteral("objects — ") + q(line));
}

void ChatSession::on_user_text(const QString& text)
{
    window_->append_user(text);
    const std::string t = norm(text);
    log_.write("user", QJsonObject{{"text", text}});
    tick_digest();   // decide on the freshest facts

    // 1. stop — at once, without the model, whatever else is going on.
    if (is_stop(t))
    {
        log_.write("fast_path", QJsonObject{{"kind", "stop"}});
        dispatch({.name = "stop"}, "fast_path");
        push_history({{"role", "user"}, {"content", text}});
        push_history({{"role", "assistant"}, {"content", "Stopping the robot now."}});
        return;
    }
    // 2. already in the controller's grammar.
    if (const auto call = parse_direct(t); call.has_value())
    {
        log_.write("fast_path", QJsonObject{{"kind", "grammar"}, {"tool", q(call->name)}, {"door", q(call->door)},
                                            {"distance_m", call->distance_m}, {"turn_deg", call->turn_deg}});
        push_history({{"role", "user"}, {"content", text}});
        // The history records what happened as a plain reply, so a later question about it can be answered.
        QString reply;
        dispatch(*call, "grammar", &reply);
        push_history({{"role", "assistant"}, {"content", reply}});
        return;
    }
    // 3. the model.
    if (llm_.busy())
    {
        window_->append_system("Still answering the previous message — wait, or press Cancel.");
        return;
    }
    if (not ollama_ok_) ping_llm();
    push_history({{"role", "user"}, {"content", text}});
    ask_model(false);
}

// The robot description (etc/system_prompt.md: who Shadow is, its body, senses, abilities and the honesty/language
// rules) is read once at startup. These interface rules stay in code because they are coupled to it: the tool names
// (tools_json, dispatch) and the bearing convention (world_digest.cpp bearing_words).
// If the file is missing, a minimal built-in description takes its place.
namespace
{
const QString kFallbackDescription = QStringLiteral(
    "You are the voice of Shadow, a mobile robot in a home, talking with a person through a chat window.\n"
    "Rules:\n"
    "- Answer ONLY from the STATE (the last system message): it is the robot's live knowledge, rebuilt for this turn. "
    "If the STATE does not contain the answer, say you do not know.\n"
    "- Never invent doors, rooms, objects, distances or events.\n"
    "- Reply in the language the person writes in. Be brief: one to three sentences.\n");

const QString kInterfaceRules = QStringLiteral(
    "\n## Tools and STATE conventions\n"
    "- Use names exactly as written in the STATE (e.g. door_19).\n"
    "- cross_door and move_relative start at once when you call them; the person can stop the robot at any time. "
    "Never say the robot is moving or has crossed until the STATE or a tool result says so.\n"
    "- A shut door is no reason not to call cross_door: the crossing itself approaches the door, has it opened and "
    "plans its own path there, whatever a short move ran into. Nobody has to confirm anything: never ask the person "
    "to confirm a crossing, a move or the opening of a door. When they ask for an action, or answer yes to one you "
    "offered, call its tool now.\n"
    "- 'robot motion now' in the STATE is the MEASURED velocity: if it is not zero you ARE moving (the controller "
    "also drives on its own: exploring, a mission, a joystick), whatever your last command says.\n"
    "- stop() halts the robot at once. cancel_passage() abandons a door crossing in progress.\n"
    "- Directions: bearing 0 deg = straight ahead of the robot, positive = to its left, negative = to its right.\n"
    "- P(open) is a probability, not a fact: near 0 = probably shut, near 1 = probably open, in between = unsure.\n"
    "- move_relative: turn first (turn_deg, + left / - right), then drive (distance_m, + forward / - backward). "
    "'A bit' = %1 m unless a number is given; 'back up' = negative; 'turn around' = 180; 'turn left/right' = +90/-90. "
    "'A bit more' / 'more' REPEATS the last move/turn in the STATE with the SAME signs (after a backward move, "
    "distance_m stays negative).\n"
    "- Objects: you know only the OBJECTS lines of the STATE (current room) and the read-only tools find_objects and "
    "where_is. If none of a kind is listed say \"I have not perceived any\", never \"there is none\".\n"
    "- look(camera) shows you the current picture of a camera: 'front' (default, what is ahead) or '360' (all around, "
    "a panorama whose centre is straight ahead and whose left and right edges are behind you). The STATE has NO "
    "appearance: when the person asks what you see, to look, or about colours, materials or what something looks "
    "like, CALL look() first and answer from the picture; never say you cannot see colours. Use '360' for what is "
    "behind or around you. What you say about a picture is a visual impression, not a measurement: for how many, how "
    "far and where, use the STATE and find_objects/where_is. Lines, dots, grids and labels drawn over a picture are "
    "debug overlays, not part of the scene.\n"
    "- The robot's own software is NOT in the STATE: for any question about it CALL a tool, never guess: agents() "
    "(which programs run, their health, CPU, memory, uptime), sensors(name) (cameras, LiDAR, IMU: streams, measured "
    "frame rate, resolution, bandwidth), graph_summary() (how many nodes and edges the working memory, i.e. the "
    "graph, holds) or inspect(node) (every attribute of one node). Never tell the user to call a tool or that you "
    "lack the information: call the tool yourself and answer from its result.\n"
    "- What the robot did EARLIER TODAY is in its episodic memory; the STATE has only the one-line 'today:' summary. "
    "For missions, steps, door crossings, claims and how they ended CALL episodes(...) (filters: since/until as HH:MM, "
    "author, kind, target), episode(id) for one in detail, last_episode(target) for the latest about a door, room or "
    "object, outcome_stats(kind, target) for how often something succeeded or failed over all time. If a tool says the "
    "episodic memory is not available, say you do not remember, never guess.\n");
}   // namespace

QString ChatSession::load_robot_description(const std::string& path)
{
    if (path.empty()) return {};
    QFile f(QString::fromStdString(path));
    if (not f.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::println("[chat_agent] WARNING: cannot read the system prompt '{}' ({}); using the built-in description",
                     path, f.errorString().toStdString());
        return {};
    }
    QString text = QString::fromUtf8(f.readAll()).trimmed();
    if (text.isEmpty())
        std::println("[chat_agent] WARNING: system prompt '{}' is empty; using the built-in description", path);
    else
        std::println("[chat_agent] system prompt: {} ({} chars)", path, text.size());
    return text;
}

QString ChatSession::system_prompt() const
{
    const QString& description = robot_description_.isEmpty() ? kFallbackDescription : robot_description_;
    return description + QStringLiteral("\n") + kInterfaceRules.arg(cfg_.default_step_m, 0, 'f', 1);
}

QString ChatSession::state_message() const
{
    return QStringLiteral("STATE:\n") + q(digest_.text) + state_extra_;
}

QJsonArray ChatSession::tools_json() const
{
    QJsonArray doors;
    for (const auto& d : digest_.facts.doors) doors.append(q(d));
    doors.append(QStringLiteral("nearest"));
    QJsonArray tools;
    tools.append(tool_def("cross_door",
        "Take the robot through a door of the CURRENT room: approach it, open it if it is shut, and go through. "
        "Starts at once.",
        QJsonObject{{"door", QJsonObject{{"type", "string"}, {"enum", doors},
                                         {"description", "a door of the current room, or 'nearest'"}}}},
        QJsonArray{"door"}));
    tools.append(tool_def("stop", "Stop the robot immediately and abandon what it is doing.", QJsonObject{}, {}));
    tools.append(tool_def("cancel_passage", "Abandon the door crossing in progress (the robot stays where it is).",
                          QJsonObject{}, {}));
    tools.append(tool_def("move_relative",
        "Move the robot a short distance relative to where it is now: first turn in place by turn_deg, then drive "
        "straight distance_m. Starts at once. The robot refuses if a wall or obstacle is in the way.",
        QJsonObject{{"distance_m", QJsonObject{{"type", "number"},
                                               {"description", "metres to drive after turning: positive = forward, "
                                                               "negative = backward, 0 = do not drive"}}},
                    {"turn_deg", QJsonObject{{"type", "number"},
                                             {"description", "degrees to turn in place first: positive = left, "
                                                             "negative = right, 180 = turn around, 0 = do not turn"}}}},
        QJsonArray{"distance_m", "turn_deg"}));
    tools.append(tool_def("find_objects",
        "List the objects the robot has perceived in the CURRENT room, nearest first, with distance, bearing and size. "
        "Read-only.",
        QJsonObject{{"subtype", QJsonObject{{"type", "string"},
                                            {"description", "a class to filter by, in English singular (chair, table, "
                                                            "bottle, cabinet, refrigerator, hood, person); empty = all"}}}},
        {}));
    tools.append(tool_def("where_is",
        "Where is an object: give its name (e.g. chair_1) or its class (e.g. refrigerator). Returns its room and, if it "
        "is in the current room, its distance and bearing; or that it has not been perceived. Read-only.",
        QJsonObject{{"name", QJsonObject{{"type", "string"}, {"description", "an object name or class"}}}},
        QJsonArray{"name"}));
    tools.append(tool_def("agents",
        "The robot's software agents (programs such as controller, retina, room_concept, door_concept): which are "
        "running, their state (Operating / Degraded / Waiting), uptime, CPU and memory of each. Read-only.",
        QJsonObject{}, {}));
    tools.append(tool_def("sensors",
        "The robot's sensors (cameras, LiDAR, IMU): their streams, the frame rate MEASURED now, resolution, age of the "
        "last frame and bandwidth. Read-only.",
        QJsonObject{{"name", QJsonObject{{"type", "string"},
                                         {"description", "a sensor or camera to filter by (e.g. ricoh, 360, zed, front, "
                                                         "lidar); empty = all"}}}},
        {}));
    tools.append(tool_def("graph_summary",
        "What the robot's working memory (its graph) holds: node and edge counts by type and by the agent that created "
        "them. Read-only.", QJsonObject{}, {}));
    tools.append(tool_def("inspect",
        "Every attribute and edge of ONE node of the robot's graph, by name (e.g. door_1, Shadow, controller 7) or id. "
        "Read-only.",
        QJsonObject{{"node", QJsonObject{{"type", "string"}, {"description", "the node's name or id"}}}},
        QJsonArray{"node"}));
    const auto str = [](const char* d) { return QJsonObject{{"type", "string"}, {"description", d}}; };
    tools.append(tool_def("episodes",
        "What the robot did TODAY, from its episodic memory, with start and end time and outcome, oldest first. With no "
        "kind and no target it lists only the MISSIONS; pass kind (e.g. cross, open) for the steps and claims of that "
        "kind, target for everything aimed at one node, or call episode(id) for one mission's steps and claims. All "
        "filters optional. Read-only.",
        QJsonObject{{"since", str("only episodes that started at or after this time: HH:MM (today, local time) or "
                                  "epoch milliseconds")},
                    {"until", str("only episodes that started before this time: HH:MM (today, local time) or epoch "
                                  "milliseconds")},
                    {"author", QJsonObject{{"type", "string"}, {"enum", QJsonArray{"human", "self"}},
                                           {"description", "human = asked by a person, self = on the robot's own"}}},
                    {"kind", str("mission, or a step/claim kind such as cross, open, approach")},
                    {"target", str("a door, room or object name, e.g. door_27")}},
        {}));
    tools.append(tool_def("episode",
        "One episode from today's episodic memory in detail, by id (as listed by episodes): its steps and claims, the "
        "beliefs read before and after (e.g. P(open) of a door) and what happened meanwhile. Read-only.",
        QJsonObject{{"id", str("the episode id, e.g. 20261017-3")}}, QJsonArray{"id"}));
    tools.append(tool_def("outcome_stats",
        "How often something succeeded, failed or was lost over ALL TIME (today included), its mean duration and "
        "information gain, from the episodic memory. Steps and claims of the same kind are separate rows; missions are "
        "split into asked-by-a-person and on-its-own. Read-only.",
        QJsonObject{{"kind", str("mission, or a step/claim kind such as cross, open, approach")},
                    {"target", str("optional: a door/room/object name, e.g. door_27; empty = all of that kind")}},
        QJsonArray{"kind"}));
    tools.append(tool_def("last_episode",
        "The most recent episode today about one door, room or object (aimed at it or that read its beliefs). "
        "Read-only.",
        QJsonObject{{"target", str("a door, room or object name, e.g. door_27")}}, QJsonArray{"target"}));
    tools.append(tool_def("look",
        "Look through a camera now: the current picture is shown to you in the next message. Read-only.",
        QJsonObject{{"camera", QJsonObject{{"type", "string"}, {"enum", QJsonArray{"front", "360"}},
                                           {"description", "'front' = the camera facing ahead (default); "
                                                           "'360' = the all-round panorama"}}}},
        {}));
    return tools;
}

void ChatSession::push_history(QJsonObject m)
{
    history_.push_back(std::move(m));
    // Trim from the front, never leaving a tool result whose call was cut (a leading role:"tool" message).
    while (static_cast<int>(history_.size()) > cfg_.max_history) history_.pop_front();
    while (not history_.empty() and history_.front().value("role").toString() == "tool") history_.pop_front();
}

// ★ THE STATE GOES LAST. ollama reuses the processed prompt up to the first token that differs from the previous
//   request. The STATE changes every turn, so inside the leading system message it forced the whole prompt —
//   description, rules, tools, history — to be processed again on every request: 1.8 s at 11k tokens on the RTX
//   5070 Ti, against 44 ms with the STATE as a trailing system message (measured 2026-10-17). The model reads a
//   trailing system message (qwen3-vl renderer; checked the same day: it answers from it, and invents without it).
QJsonArray ChatSession::messages_for_request() const
{
    QJsonArray msgs;
    msgs.append(QJsonObject{{"role", "system"}, {"content", system_prompt()}});
    for (const auto& m : history_) msgs.append(m);
    msgs.append(QJsonObject{{"role", "system"}, {"content", state_message()}});
    return msgs;
}

void ChatSession::ask_model(bool followup)
{
    turn_ = Turn{.followup = followup};
    // STATE additions for this request: what changed among the objects and the doors since the last request, and the last
    // relative move (so "a bit more" knows its direction even after the history has been trimmed).
    state_extra_.clear();
    if (const ObjectChanges c = diff_objects(turn_baseline_, objects_now_); not c.empty())
        state_extra_ += QStringLiteral("changes in objects since the last message: ") + q(changes_line(c)) + "\n";
    turn_baseline_ = objects_now_;
    if (const auto gone = doors_gone(turn_doors_, digest_.facts.doors); not gone.empty())
    {
        std::string names;
        for (const auto& g : gone) names += (names.empty() ? "" : ", ") + g;
        state_extra_ += QStringLiteral("doors gone since the last message: %1 (deleted from the map: it is not there "
                                       "any more, whatever earlier messages said)\n").arg(q(names));
    }
    turn_doors_ = digest_.facts.doors;
    state_extra_ += QStringLiteral("last move/turn sent from this chat: ") +
                    (last_relative_.empty() ? QStringLiteral("none") : q(explain_relative_command(last_relative_))) + "\n";
    const QJsonArray msgs = messages_for_request();
    const QJsonArray tools = followup ? QJsonArray{} : tools_json();
    QJsonArray hist;
    for (const auto& m : history_)
    {
        hist.append(m);
        if (has_image(m)) turn_.carries_image = true;
    }
    log_.write("request", QJsonObject{{"followup", followup}, {"state", q(digest_.text)}, {"n_messages", msgs.size()},
                                      {"tools", tools}, {"history", redact_images(hist)}});
    const bool ok = llm_.chat(msgs, tools, OllamaClient::Callbacks{
        .on_token = [this](const QString& tok)
        {
            if (not turn_.streaming) { window_->begin_assistant(); turn_.streaming = true; }
            turn_.content += tok;
            window_->append_token(tok);
        },
        .on_tool_calls = [this](const QJsonArray& calls) { turn_.tool_calls = calls; },
        .on_done = [this](const ChatStats& st)
        {
            window_->end_assistant();
            window_->set_busy(false);
            if (turn_.carries_image) strip_images(history_);   // an image lives for the one request that used it
            log_.write("response", QJsonObject{{"content", turn_.content}, {"tool_calls", turn_.tool_calls},
                                               {"eval_count", static_cast<qint64>(st.eval_count)}, {"prompt_eval_count", static_cast<qint64>(st.prompt_eval_count)},
                                               {"first_token_ms", st.first_token_ms}, {"wall_ms", st.wall_ms},
                                               {"server_ms", st.total_ms}, {"done_reason", st.done_reason}});
            const QString content = strip_think(turn_.content);
            if (turn_.tool_calls.isEmpty())
            {
                if (not content.isEmpty()) push_history({{"role", "assistant"}, {"content", content}});
                return;
            }
            push_history({{"role", "assistant"}, {"content", content}, {"tool_calls", turn_.tool_calls}});
            const bool followup = turn_.followup;
            const bool image = handle_tool_calls(turn_.tool_calls);
            // A tool round with no words: one follow-up (no tools) so the model can tell the person what happened.
            // A picture was asked for: the follow-up is where the model gets to see it, words or not.
            if ((content.isEmpty() or image) and not followup) ask_model(true);
        },
        .on_error = [this](const QString& err)
        {
            window_->end_assistant();
            window_->set_busy(false);
            if (turn_.carries_image) strip_images(history_);
            log_.write("error", QJsonObject{{"where", "llm"}, {"msg", err}});
            window_->append_system(err == "cancelled" ? QStringLiteral("(answer cancelled)") : QStringLiteral("Model error: ") + err);
            if (err != "cancelled") ping_llm();
        },
    });
    if (not ok) { window_->append_system("The model is busy."); return; }
    window_->set_busy(true);
}

bool ChatSession::handle_tool_calls(const QJsonArray& calls)
{
    bool image = false;
    for (const auto& c : calls)
    {
        const QJsonObject fn = c.toObject().value("function").toObject();
        const QString name = fn.value("name").toString();
        QJsonObject args = fn.value("arguments").toObject();
        if (fn.value("arguments").isString())   // some templates send the arguments JSON-encoded
            args = QJsonDocument::fromJson(fn.value("arguments").toString().toUtf8()).object();
        QString result = answer_query(name, args);
        if (result.isEmpty())
        {
            // A number may arrive as a JSON number or as a string ("0.5"); a missing one is 0.
            const auto num = [&args](const char* key) -> double
            {
                const QJsonValue v = args.value(key);
                if (v.isDouble()) return v.toDouble();
                if (v.isString()) return parse_number(v.toString().trimmed().toStdString()).value_or(std::nan(""));
                return 0.0;
            };
            const ToolCall call{.name = name.toStdString(), .door = args.value("door").toString().toStdString(),
                                .distance_m = num("distance_m"), .turn_deg = num("turn_deg")};
            result = dispatch(call, "llm");
        }
        if (not look_jpeg_.isEmpty())
        {
            push_history(make_look_tool_message(result, look_jpeg_));
            look_jpeg_.clear();
            image = true;
        }
        else push_history({{"role", "tool"}, {"tool_name", name}, {"content", result}});
    }
    return image;
}

QString ChatSession::answer_query(const QString& name, const QJsonObject& args)
{
    if (name == "look") return do_look(args);
    QString out;
    if (QString m = answer_memory(name, args); not m.isEmpty()) out = std::move(m);
    else if (name == "find_objects")
        out = q(answer_find_objects(digest_.objects, args.value("subtype").toString().toStdString(), digest_.facts.room));
    else if (name == "where_is")
        out = q(answer_where_is(digest_.objects, args.value("name").toString().toStdString(), digest_.facts.room));
    else if (name == "agents")
        out = q(answer_agents(system_.agents()));
    else if (name == "sensors")
        out = q(answer_sensors(system_.sensors(), args.value("name").toString().toStdString()));
    else if (name == "graph_summary")
        out = q(answer_graph(system_.graph(digest_.facts.room)));
    else if (name == "inspect")
    {
        const std::string node = args.value("node").toString().toStdString();
        const auto v = system_.inspect(node);
        out = v.has_value() ? q(answer_inspect(*v)) : QStringLiteral("no node named '%1' in the graph").arg(q(node));
    }
    else return {};
    log_.write("query", QJsonObject{{"tool", name}, {"args", args}, {"result", out}});
    return out.trimmed();
}

std::optional<rc::em::Store> ChatSession::em_store() const
{
    if (not em_G_) return std::nullopt;
    return rc::em::load_store(*em_G_);
}

QString ChatSession::answer_memory(const QString& name, const QJsonObject& args) const
{
    if (name != "episodes" and name != "episode" and name != "outcome_stats" and name != "last_episode") return {};
    const auto arg = [&](const char* k) { return args.value(k).toString().toStdString(); };
    const auto st = em_store();
    if (name == "episode") return q(answer_episode(st, arg("id")));
    if (name == "outcome_stats") return q(answer_outcome_stats(st, arg("kind"), arg("target")));
    if (name == "last_episode") return q(answer_last_episode(st, arg("target")));
    // episodes: since/until as "HH:MM" or epoch ms (a string, or a JSON number the model may send instead)
    std::optional<std::uint64_t> t[2];
    const char* keys[2] = {"since", "until"};
    for (int i = 0; i < 2; ++i)
    {
        const QJsonValue v = args.value(keys[i]);
        if (v.isDouble() and v.toDouble() >= 0) t[i] = std::uint64_t(v.toDouble());
        else if (const std::string s = v.toString().trimmed().toStdString(); not s.empty())
        {
            t[i] = parse_em_time(s);
            if (not t[i])
                return QStringLiteral("ERROR: %1 '%2' not understood — use HH:MM (today, local time) or epoch milliseconds")
                    .arg(keys[i], q(s));
        }
    }
    return q(answer_episodes(st, t[0], t[1], arg("author"), arg("kind"), arg("target")));
}

QString ChatSession::do_look(const QJsonObject& args)
{
    const QString arg = args.value("camera").toString();
    const auto front = look_camera_is_front(arg);
    if (not front.has_value())
    {
        log_.write("look", QJsonObject{{"camera", arg}, {"ok", false}, {"reason", "unknown camera"}});
        return QStringLiteral("ERROR: unknown camera '%1' — use 'front' or '360'.").arg(arg);
    }
    const Camera cam = *front ? Camera::front : Camera::pano;
    auto frame = cameras_.grab(cam);
    if (not frame.has_value())
    {
        log_.write("look", QJsonObject{{"camera", arg}, {"ok", false}, {"reason", q(frame.error())}});
        window_->append_system(QStringLiteral("look: ") + q(frame.error()));
        return QStringLiteral("ERROR: ") + q(frame.error());
    }
    const int side = cam == Camera::front ? cfg_.look_max_side : cfg_.look_max_side_360;
    const auto img = prepare_image(frame->bgr, side, cfg_.look_jpeg_quality);
    if (not img.has_value())
    {
        log_.write("look", QJsonObject{{"camera", arg}, {"ok", false}, {"reason", "encode failed"}});
        return QStringLiteral("ERROR: camera not available: the frame could not be encoded.");
    }
    const QByteArray jpeg(reinterpret_cast<const char*>(img->jpeg.data()), static_cast<qsizetype>(img->jpeg.size()));
    const QString label = q(frame->label);
    look_jpeg_ = jpeg;
    log_.write("look", QJsonObject{{"camera", arg}, {"ok", true}, {"label", label}, {"age_ms", static_cast<qint64>(frame->age_ms)},
                                   {"src_w", frame->bgr.cols}, {"src_h", frame->bgr.rows}, {"w", img->width},
                                   {"h", img->height}, {"jpeg_bytes", static_cast<qint64>(img->jpeg.size())}});
    window_->append_system(QStringLiteral("looking through the %1 (%2x%3, %4 ms old)")
                               .arg(label).arg(img->width).arg(img->height).arg(frame->age_ms));
    return look_tool_result(label, frame->age_ms, img->width, img->height);
}

QString ChatSession::dispatch(const ToolCall& call, const char* origin, QString* reply)
{
    QString sink;
    QString& say = reply != nullptr ? *reply : sink;
    const GateResult r = validate(call, digest_.facts, cfg_.limits);
    log_.write("gate", QJsonObject{{"origin", origin}, {"tool", q(call.name)}, {"door", q(call.door)},
                                   {"distance_m", call.distance_m}, {"turn_deg", call.turn_deg}, {"ok", r.ok},
                                   {"command", q(r.command)}, {"reason", q(r.reason)}});
    if (not r.ok)
    {
        window_->append_system(QStringLiteral("Not possible: ") + q(r.reason));
        say = QStringLiteral("I can't do that: %1.").arg(q(r.reason));
        return QStringLiteral("REJECTED: ") + q(r.reason);
    }
    if (not r.reason.empty()) window_->append_system(q(r.reason));
    const bool sent = send_command(r.command, origin);
    say = sent ? QStringLiteral("Sent '%1' to the controller.").arg(q(r.command))
               : QStringLiteral("I could not send '%1'.").arg(q(r.command));
    return sent ? QStringLiteral("OK: '%1' sent to the controller.").arg(q(r.command))
                : QStringLiteral("FAILED: could not write the command to the graph.");
}

bool ChatSession::send_command(const std::string& command, const char* origin)
{
    const auto stamp = intention_.send(command);
    log_.write("intention_write", QJsonObject{{"command", q(command)}, {"origin", origin}, {"ok", stamp.has_value()},
                                              {"stamp", static_cast<qint64>(stamp.value_or(0))}});
    if (not stamp.has_value())
    {
        window_->append_system(QStringLiteral("Could not write the command '%1' to the graph.").arg(q(command)));
        return false;
    }
    sent_command_ = command;
    sent_stamp_ = *stamp;
    if (command.starts_with("move ") or command.starts_with("turn ")) last_relative_ = command;
    seen_state_ = "waiting";
    seen_step_.clear();
    seen_event_.clear();
    warned_unarmed_ = warned_unacked_ = reported_end_ = false;
    window_->append_system(QStringLiteral("→ sent to the controller: %1").arg(q(command)));
    std::println("[chat_agent] command '{}' sent ({})", command, origin);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// The controller's report on the last command, every 250 ms.
void ChatSession::tick_status()
{
    if (window_)   // whatever the mission's source: chat, remote_cmd.txt, the panel
    {
        auto u = mission_watch_.poll();
        if (u.mission.has_value()) window_->set_plan(u.mission->first, QString::fromStdString(u.mission->second));
        if (u.progress.has_value()) window_->set_plan_progress(*u.progress);
    }
    if (sent_command_.empty()) return;
    const IntentionStatus s = intention_.read();
    if (s.command != sent_command_ or s.stamp != sent_stamp_) return;   // our node was re-created
    // Until the controller reports on THIS stamp, s.state is "" (intention_node.cpp read()).
    const std::string state = s.state.empty() ? std::string("waiting") : s.state;
    const auto parts = split_bar(s.bt_state);
    const std::string step = parts.size() > 0 ? parts[0] : "";
    const std::string event = parts.size() > 1 ? parts[1] : "";
    const bool unarmed = parts.size() > 2 and parts[2] == "driving=off";

    if (state == "waiting" and not warned_unacked_ and rc::provenance::now_ms() - sent_stamp_ > 3000)
    {
        warned_unacked_ = true;
        window_->append_robot(controller_present_ ? "the controller has not picked the command up yet"
                                                  : "no controller is running to pick the command up");
        log_.write("intention_unacked", QJsonObject{{"command", q(sent_command_)}});
    }
    if (state == seen_state_ and step == seen_step_ and event == seen_event_) return;
    log_.write("intention_state", QJsonObject{{"command", q(sent_command_)}, {"state", q(state)}, {"bt_state", q(s.bt_state)}});

    if (state == "in_progress")
    {
        if (step != seen_step_ or event != seen_event_)
        {
            QString line = q(step);
            if (not event.empty() and event != seen_event_) line += " — " + q(event);
            window_->append_robot(line);
        }
        if (unarmed and not warned_unarmed_)
        {
            warned_unarmed_ = true;
            window_->append_robot("the base is not armed: press Run in the controller window to let it drive");
        }
    }
    else if (not reported_end_ and (state == "completed" or state == "aborted" or state.starts_with("failed")))
    {
        reported_end_ = true;
        // stop / uncross END in "aborted" by design — that is their success, not a failure.
        const bool halt = sent_command_ == "stop" or sent_command_ == "uncross";
        QString line = halt and state == "aborted" ? QStringLiteral("'%1' done").arg(q(sent_command_))
                                                     : QStringLiteral("'%1' %2").arg(q(sent_command_), q(state));
        if (not event.empty()) line += " — " + q(event);
        window_->append_robot(line);
        push_history({{"role", "tool"}, {"tool_name", "robot_status"},
                      {"content", q(robot_status_text(sent_command_, state, event))}});
    }
    seen_state_ = state;
    seen_step_ = step;
    seen_event_ = event;
}

}   // namespace rc::chat
