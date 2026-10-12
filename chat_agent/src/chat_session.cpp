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

#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::now_ms

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
      intention_(G_), llm_(cfg_.llm)
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
            last_relative_.clear();   // "a bit more" has nothing to repeat in a new conversation
            pending_.reset();
            window_->hide_card();
            window_->clear_transcript();
            window_->append_system("New conversation.");
            log_.write("new_conversation");
        },
        .on_confirm = [this]() { confirm_card("button"); },
        .on_reject = [this]() { reject_card("button"); },
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
    digest_ = digest_builder_.build(*inner_, controller_present_, intention_.read());
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
        if (pending_.has_value()) { pending_.reset(); window_->hide_card(); }
        log_.write("fast_path", QJsonObject{{"kind", "stop"}});
        dispatch({.name = "stop"}, "fast_path");
        push_history({{"role", "user"}, {"content", text}});
        push_history({{"role", "assistant"}, {"content", "Stopping the robot now."}});
        return;
    }
    // 2. a typed answer to the card.
    if (pending_.has_value() and is_yes(t)) { push_history({{"role", "user"}, {"content", text}}); confirm_card("typed"); return; }
    if (pending_.has_value() and is_no(t)) { push_history({{"role", "user"}, {"content", text}}); reject_card("typed"); return; }
    // 3. already in the controller's grammar.
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
    // 4. the model.
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
// and the confirmation card (tools_json, dispatch) and the bearing convention (world_digest.cpp bearing_words).
// If the file is missing, a minimal built-in description takes its place.
namespace
{
const QString kFallbackDescription = QStringLiteral(
    "You are the voice of Shadow, a mobile robot in a home, talking with a person through a chat window.\n"
    "Rules:\n"
    "- Answer ONLY from the STATE block below: it is the robot's live knowledge, rebuilt for this turn. "
    "If the STATE does not contain the answer, say you do not know.\n"
    "- Never invent doors, rooms, objects, distances or events.\n"
    "- Reply in the language the person writes in. Be brief: one to three sentences.\n");

const QString kInterfaceRules = QStringLiteral(
    "\n## Tools and STATE conventions\n"
    "- Use names exactly as written in the STATE (e.g. door_19).\n"
    "- cross_door moves the robot and is carried out only after the person confirms on a card in the window: "
    "never say the robot is moving or has crossed until the STATE or a tool result says so. You cannot confirm "
    "anything yourself.\n"
    "- stop() halts the robot at once. cancel_passage() abandons a door crossing in progress.\n"
    "- Directions: bearing 0 deg = straight ahead of the robot, positive = to its left, negative = to its right.\n"
    "- P(open) is a probability, not a fact: near 0 = probably shut, near 1 = probably open, in between = unsure.\n"
    "- move_relative: turn first (turn_deg, + left / - right), then drive (distance_m, + forward / - backward). "
    "'A bit' = %1 m unless a number is given; 'back up' = negative; 'turn around' = 180; 'turn left/right' = +90/-90. "
    "'A bit more' / 'more' REPEATS the last move/turn in the STATE with the SAME signs (after a backward move, "
    "distance_m stays negative).\n"
    "- Objects: you know only the OBJECTS lines of the STATE (current room) and the read-only tools find_objects and "
    "where_is. If none of a kind is listed say \"I have not perceived any\", never \"there is none\".\n");
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
    return description + QStringLiteral("\n") + kInterfaceRules.arg(cfg_.default_step_m, 0, 'f', 1) +
           QStringLiteral("\nSTATE:\n") + q(digest_.text) + state_extra_;
}

QJsonArray ChatSession::tools_json() const
{
    QJsonArray doors;
    for (const auto& d : digest_.facts.doors) doors.append(q(d));
    doors.append(QStringLiteral("nearest"));
    QJsonArray tools;
    tools.append(tool_def("cross_door",
        "Take the robot through a door of the CURRENT room: approach it, open it if it is shut, and go through. "
        "Shows the person a confirmation card; nothing moves until they confirm.",
        QJsonObject{{"door", QJsonObject{{"type", "string"}, {"enum", doors},
                                         {"description", "a door of the current room, or 'nearest'"}}}},
        QJsonArray{"door"}));
    tools.append(tool_def("stop", "Stop the robot immediately and abandon what it is doing.", QJsonObject{}, {}));
    tools.append(tool_def("cancel_passage", "Abandon the door crossing in progress (the robot stays where it is).",
                          QJsonObject{}, {}));
    tools.append(tool_def("move_relative",
        "Move the robot a short distance relative to where it is now: first turn in place by turn_deg, then drive "
        "straight distance_m. Small moves start at once; long ones ask the person to confirm. The robot refuses if a "
        "wall or obstacle is in the way.",
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
    return tools;
}

void ChatSession::push_history(QJsonObject m)
{
    history_.push_back(std::move(m));
    // Trim from the front, never leaving a tool result whose call was cut (a leading role:"tool" message).
    while (static_cast<int>(history_.size()) > cfg_.max_history) history_.pop_front();
    while (not history_.empty() and history_.front().value("role").toString() == "tool") history_.pop_front();
}

QJsonArray ChatSession::messages_for_request() const
{
    QJsonArray msgs;
    msgs.append(QJsonObject{{"role", "system"}, {"content", system_prompt()}});
    for (const auto& m : history_) msgs.append(m);
    return msgs;
}

void ChatSession::ask_model(bool followup)
{
    turn_ = Turn{.followup = followup};
    // STATE additions for this request: what changed among the objects since the last request, and the last
    // relative move (so "a bit more" knows its direction even after the history has been trimmed).
    state_extra_.clear();
    if (const ObjectChanges c = diff_objects(turn_baseline_, objects_now_); not c.empty())
        state_extra_ += QStringLiteral("changes in objects since the last message: ") + q(changes_line(c)) + "\n";
    turn_baseline_ = objects_now_;
    state_extra_ += QStringLiteral("last move/turn sent from this chat: ") +
                    (last_relative_.empty() ? QStringLiteral("none") : q(explain_relative_command(last_relative_))) + "\n";
    const QJsonArray msgs = messages_for_request();
    const QJsonArray tools = followup ? QJsonArray{} : tools_json();
    QJsonArray hist;
    for (const auto& m : history_) hist.append(m);
    log_.write("request", QJsonObject{{"followup", followup}, {"state", q(digest_.text)}, {"n_messages", msgs.size()},
                                      {"tools", tools}, {"history", hist}});
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
            handle_tool_calls(turn_.tool_calls);
            // A tool round with no words: one follow-up (no tools) so the model can tell the person what happened.
            if (content.isEmpty() and not followup) ask_model(true);
        },
        .on_error = [this](const QString& err)
        {
            window_->end_assistant();
            window_->set_busy(false);
            log_.write("error", QJsonObject{{"where", "llm"}, {"msg", err}});
            window_->append_system(err == "cancelled" ? QStringLiteral("(answer cancelled)") : QStringLiteral("Model error: ") + err);
            if (err != "cancelled") ping_llm();
        },
    });
    if (not ok) { window_->append_system("The model is busy."); return; }
    window_->set_busy(true);
}

void ChatSession::handle_tool_calls(const QJsonArray& calls)
{
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
        push_history({{"role", "tool"}, {"tool_name", name}, {"content", result}});
    }
}

QString ChatSession::answer_query(const QString& name, const QJsonObject& args)
{
    QString out;
    if (name == "find_objects")
        out = q(answer_find_objects(digest_.objects, args.value("subtype").toString().toStdString(), digest_.facts.room));
    else if (name == "where_is")
        out = q(answer_where_is(digest_.objects, args.value("name").toString().toStdString(), digest_.facts.room));
    else return {};
    log_.write("query", QJsonObject{{"tool", name}, {"args", args}, {"result", out}});
    return out.trimmed();
}

QString ChatSession::dispatch(const ToolCall& call, const char* origin, QString* reply)
{
    QString sink;
    QString& say = reply != nullptr ? *reply : sink;
    const GateResult r = validate(call, digest_.facts, cfg_.limits);
    log_.write("gate", QJsonObject{{"origin", origin}, {"tool", q(call.name)}, {"door", q(call.door)},
                                   {"distance_m", call.distance_m}, {"turn_deg", call.turn_deg}, {"ok", r.ok},
                                   {"confirm", r.needs_confirmation}, {"command", q(r.command)}, {"reason", q(r.reason)}});
    if (not r.ok)
    {
        window_->append_system(QStringLiteral("Not possible: ") + q(r.reason));
        say = QStringLiteral("I can't do that: %1.").arg(q(r.reason));
        return QStringLiteral("REJECTED: ") + q(r.reason);
    }
    if (r.needs_confirmation)
    {
        pending_ = Pending{.call = call, .result = r, .origin = origin};
        window_->show_card(QStringLiteral("<b>Confirm:</b> %1<br><small>Click Confirm, or type yes / sí. "
                                          "Cancel, no, or stop to drop it.</small>").arg(q(r.summary).toHtmlEscaped()));
        log_.write("card_shown", QJsonObject{{"command", q(r.command)}});
        say = QStringLiteral("Please confirm on the card: %1.").arg(q(r.summary));
        return QStringLiteral("PENDING CONFIRMATION: a card asking the person to confirm '%1' is shown. "
                              "The robot will not move unless they confirm.").arg(q(r.command));
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

void ChatSession::confirm_card(const char* how)
{
    if (not pending_.has_value()) return;
    const Pending p = *pending_;
    pending_.reset();
    window_->hide_card();
    tick_digest();
    // Re-validate: the door may have gone, or the controller, since the card was raised.
    const GateResult r = validate(p.call, digest_.facts, cfg_.limits);
    log_.write("confirm", QJsonObject{{"how", how}, {"command", q(p.result.command)}, {"still_valid", r.ok},
                                      {"reason", q(r.reason)}});
    if (not r.ok or r.command != p.result.command)
    {
        const QString why = r.ok ? QStringLiteral("the nearest door changed to %1 — ask again").arg(q(r.command))
                                 : q(r.reason);
        window_->append_system(QStringLiteral("Confirmed, but no longer possible: ") + why);
        push_history({{"role", "tool"}, {"tool_name", "confirmation"}, {"content", "Confirmed by the person but NOT sent: " + why}});
        return;
    }
    const bool sent = send_command(r.command, how);
    push_history({{"role", "tool"}, {"tool_name", "confirmation"},
                  {"content", sent ? QStringLiteral("The person CONFIRMED; '%1' was sent to the controller.").arg(q(r.command))
                                   : QStringLiteral("The person confirmed but the command could not be written.")}});
}

void ChatSession::reject_card(const char* how)
{
    if (not pending_.has_value()) return;
    log_.write("reject", QJsonObject{{"how", how}, {"command", q(pending_->result.command)}});
    window_->append_system(QStringLiteral("Cancelled: %1").arg(q(pending_->result.command)));
    push_history({{"role", "tool"}, {"tool_name", "confirmation"},
                  {"content", QStringLiteral("The person DECLINED '%1'. Nothing was sent.").arg(q(pending_->result.command))}});
    pending_.reset();
    window_->hide_card();
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// The controller's report on the last command, every 250 ms.
void ChatSession::tick_status()
{
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
                      {"content", QStringLiteral("The command '%1' ended: %2. %3").arg(q(sent_command_), q(state), q(event))}});
    }
    seen_state_ = state;
    seen_step_ = step;
    seen_event_ = event;
}

}   // namespace rc::chat
