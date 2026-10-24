#include "chat_session.h"

#include <csignal>
#include <cmath>
#include <print>
#include <ranges>
#include <vector>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <cstdlib>
#include <QJsonDocument>
#include <QRegularExpression>

#include "../../common/episodic/em_query_dsr.h"            // rc::em::load_store
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::now_ms
#include "../../common/room_resolve/room_resolve.h"
#include <opencv2/imgcodecs.hpp>

#include "approach.h"
#include "canvas_model.h"
#include "canvas_snapshot.h"
#include "controller_facts.h"
#include "gpu_memory.h"
#include "vlm_status.h"
#include <QHostAddress>
#include <QHostInfo>
#include <QNetworkInterface>
#include <QNetworkRequest>
#include <QUrl>
#include "image_prep.h"
#include "look_messages.h"
#include "mission_plan.h"
#include "sim_door.h"
#include "topology.h"

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
      intention_(G_), mission_watch_(G_), llm_(cfg_.llm), cameras_(G_),
      canvas_runner_(CanvasRunConfig{.canvas_dir = q(cfg_.canvas_dir), .timeout_ms = cfg_.canvas_timeout_ms,
                                     .memory_max = q(cfg_.canvas_memory_max)})
{
    robot_description_ = load_robot_description(cfg_.system_prompt_path);
    if (not cfg_.skills_dir.empty()) skills_ = rc::chat::load_skills(cfg_.skills_dir);
    fleet_policy_ = fleet::Policy{.agents = fleet::split_names(cfg_.fleet_agents),
                                  .confirm = fleet::split_names(cfg_.fleet_confirm)};
    if (not log_.open(cfg_.log_dir)) std::println("[chat_agent] WARNING: cannot open a log under '{}'", cfg_.log_dir);
    else std::println("[chat_agent] logging to {}", log_.path());
    log_.write("start", QJsonObject{{"model", cfg_.llm.model}, {"url", cfg_.llm.url}, {"num_ctx", cfg_.llm.num_ctx},
                                    {"temperature", cfg_.llm.temperature},
                                    {"system_prompt", q(cfg_.system_prompt_path)},
                                    {"system_prompt_loaded", not robot_description_.isEmpty()}});
}

ChatSession::~ChatSession()
{
    QObject::disconnect(&gpu_proc_, nullptr, nullptr, nullptr);   // its destructor may still finish a query: no callback then
    llm_.cancel();
    log_.write("stop");
}

void ChatSession::start()
{
    start_ms_ = rc::provenance::now_ms();
    window_ = std::make_unique<ChatWindow>(ChatWindow::Callbacks{
        .on_send = [this](const QString& t) { on_user_text(t); },
        .on_cancel = [this]()
        {
            if (llm_.busy()) { llm_.cancel(); }   // the planner's own request ends in its on_error ("cancelled")
            else if (planner_.has_value())        // Esc in the gap before the planner request starts
            {
                window_->append_system(QStringLiteral("(planning cancelled)"));
                drop_planner(QStringLiteral("REJECTED: the person cancelled the planning"));
            }
        },
        .on_new = [this]()
        {
            planner_.reset();            // before cancel(): its error callback then has no job to answer
            pending_mission_.reset();
            fix_key_.clear();            // a failure's fix, and the robot events still queued, belong to the
            wakes_.clear();              // conversation that was cleared: never delivered into the new one
            llm_.cancel();
            window_->set_busy(false);
            history_.clear();
            look_jpeg_.clear();
            last_relative_.clear();   // "a bit more" has nothing to repeat in a new conversation
            window_->clear_transcript();
            window_->append_system("New conversation.");
            log_.write("new_conversation");
        },
        .canvas = CanvasPanel::Callbacks{
            .on_close = [this]()
            {
                canvas_stop();
                window_->hide_canvas();
            },
            .on_save = [this](const QString& name) { save_canvas_script(name); },
        },
    });
    window_->show();
    // The canvas is open from the start (× closes it; any drawing reopens it), with a hint until the first drawing.
    window_->show_canvas();
    window_->canvas()->show_message(QStringLiteral("Canvas"),
        QStringLiteral("Ask me to show or draw something here:\n\"show me the zed camera\", \"show me the room\", "
                       "\"show me the plan\", or anything else, e.g. \"plot how far each object is\"."));
    window_->append_system("Robot chat ready. I answer from the robot's live map only. Type \"stop\" to stop it at once.");

    QObject::connect(&digest_timer_, &QTimer::timeout, &ctx_, [this]() { tick_digest(); });
    QObject::connect(&status_timer_, &QTimer::timeout, &ctx_, [this]() { tick_status(); });
    QObject::connect(&ping_timer_, &QTimer::timeout, &ctx_, [this]() { ping_llm(); });
    QObject::connect(&canvas_timer_, &QTimer::timeout, &ctx_, [this]() { tick_canvas(); });
    QObject::connect(&gpu_timer_, &QTimer::timeout, &ctx_, [this]() { poll_gpu(); });
    QObject::connect(&fleet_timer_, &QTimer::timeout, &ctx_, [this]()
    {
        // ★ The tool answered "starting" before the outcome was known: the model hears how it ended (live 2026-10-24:
        //   only the window heard "no executable", and the model told the person the agent was starting).
        if (auto end = tick_fleet(); end.has_value())
        {
            log_.write("wake_queued", QJsonObject{{"kind", "agent_control"}, {"note", q(end->note)}});
            wakes_.push(std::move(end->note));
        }
    });
    QObject::connect(&gpu_proc_, &QProcess::finished, &ctx_, [this](int code, QProcess::ExitStatus status)
    {
        const auto m = (status == QProcess::NormalExit and code == 0)
                           ? parse_gpu_memory(gpu_proc_.readAllStandardOutput().toStdString()) : std::nullopt;
        gpu_label_ = q(gpu_memory_label(m));
        refresh_status_bar();
    });
    QObject::connect(&sim_door_proc_, &QProcess::finished, &ctx_,
                     [this](int code, QProcess::ExitStatus status) { on_sim_door_done(status == QProcess::NormalExit ? code : -1); });
    QObject::connect(&sim_door_proc_, &QProcess::errorOccurred, &ctx_, [this](QProcess::ProcessError e)
    {
        if (e == QProcess::FailedToStart) on_sim_door_done(-1);   // the other errors end in finished() too
    });
    QObject::connect(&gpu_proc_, &QProcess::errorOccurred, &ctx_, [this](QProcess::ProcessError e)
    {
        if (e != QProcess::FailedToStart) return;   // the other errors end in finished() too
        gpu_label_ = q(gpu_memory_label(std::nullopt));   // no nvidia-smi on this machine
        refresh_status_bar();
    });
    canvas_timer_.start(cfg_.canvas_live_period_ms);
    setup_vlm_status();
    gpu_timer_.start(cfg_.gpu_period_ms);
    poll_gpu();
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
                            controller_present_, gpu_label_);
}

void ChatSession::poll_gpu()
{
    if (vlm_remote_) { poll_vlm(); return; }   // the model is not on this GPU: ask the server what it holds
    if (gpu_proc_.state() != QProcess::NotRunning) return;   // a slow query is not stacked on
    gpu_proc_.start(QStringLiteral("nvidia-smi"), {QStringLiteral("--query-gpu=memory.used,memory.total"),
                                                   QStringLiteral("--format=csv,noheader,nounits")});
}

void ChatSession::setup_vlm_status()
{
    const QUrl url(cfg_.llm.url);
    const QString host = url.host();
    std::vector<std::string> mine;
    for (const auto& a : QNetworkInterface::allAddresses()) mine.push_back(a.toString().toStdString());
    vlm_remote_ = not is_local_host(host.toStdString(), mine);
    if (not vlm_remote_) return;
    vlm_ip_ = host.toStdString();   // until the lookups answer (an ip literal is already final)
    // QHostInfo::lookupHost is asynchronous; a name resolves forward, an address resolves in reverse (/etc/hosts counts).
    const auto reverse = [this](const QString& ip)
    {
        QHostInfo::lookupHost(ip, &ctx_, [this, ip](const QHostInfo& info)
        {
            vlm_host_ = (info.error() == QHostInfo::NoError and info.hostName() != ip) ? info.hostName().toStdString() : "";
        });
    };
    if (QHostAddress(host).isNull())
        QHostInfo::lookupHost(host, &ctx_, [this, host, reverse](const QHostInfo& info)
        {
            if (info.error() != QHostInfo::NoError or info.addresses().isEmpty()) return;
            vlm_ip_ = info.addresses().first().toString().toStdString();
            reverse(QString::fromStdString(vlm_ip_));
        });
    else
        reverse(host);
}

void ChatSession::poll_vlm()
{
    if (vlm_reply_) return;   // a slow request is not stacked on
    QNetworkRequest req(QUrl(cfg_.llm.url + QStringLiteral("/api/ps")));
    req.setTransferTimeout(1500);
    vlm_reply_ = vlm_nam_.get(req);
    QObject::connect(vlm_reply_, &QNetworkReply::finished, &ctx_, [this, r = vlm_reply_]()
    {
        std::optional<VlmMemory> mem;
        if (r->error() == QNetworkReply::NoError)
            mem = parse_ollama_ps(r->readAll().toStdString(), cfg_.llm.model.toStdString());
        r->deleteLater();
        gpu_label_ = q(vlm_status_label(cfg_.llm.model.toStdString(), vlm_ip_, vlm_host_, mem));
        refresh_status_bar();
    });
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
    ++user_turn_;
    last_user_text_ = text;
    wake_turn_ = false;
    // ★ The person spoke while a plan was being written: the plan is for a conversation that has moved on. Resolved
    //   FIRST, so its tool result lands right after its call in the history (a call and its results are never split
    //   by this message's turns) and a late reply can never become a card (stop, then a card, is an anti-goal).
    if (planner_.has_value() and not plan_still_wanted(planner_->turn, user_turn_))
        drop_planner(QStringLiteral("REJECTED: the person spoke again before the plan was ready"));
    // A mission card lives for exactly one message: taken here, so whatever this message is, the card is gone after
    // it. It survives only as this message's yes (1a, an agent_control yes, goes first and takes the yes).
    std::optional<MissionCard> card = std::exchange(pending_mission_, std::nullopt);
    if (card.has_value())
    {
        const bool fleet_yes = fleet_confirmations_.pending().has_value() and fleet::is_yes(text.toStdString());
        if (not card_still_valid(card->turn, user_turn_) or not fleet::is_yes(text.toStdString()) or fleet_yes)
        {
            log_.write("mission_card_dropped", QJsonObject{{"card", static_cast<qint64>(card->serial)}});
            window_->append_system(QStringLiteral("(the planned mission was not started)"));
            card.reset();
        }
    }
    tick_digest();   // decide on the freshest facts
    draw_repairs_ = 0;
    lost_answers_ = 0;

    // 1. stop — at once, without the model, whatever else is going on.
    if (is_stop(t))
    {
        log_.write("fast_path", QJsonObject{{"kind", "stop"}});
        dispatch({.name = "stop"}, "fast_path");
        push_history({{"role", "user"}, {"content", text}});
        push_history({{"role", "assistant"}, {"content", "Stopping the robot now."}});
        return;
    }
    // 1d. pause / continue / cancel the running mission: the person's alone, never the model's (spec §5.6) — and only
    //     where the words mean the mission: otherwise ("sigue recto", "espera, ¿qué ves?") on to the normal path.
    if (const auto act = mission_control_action(t, mission_progress_.status, running_op(mission_progress_)); act.has_value())
    {
        mission_control(*act, text);
        return;
    }
    // 1a. a plain yes to agent_control's question: carried out here, not by the model (which answered "yes" by saying
    //     the agent was stopped without calling the tool, eval 2026-10-19).
    if (const auto& pend = fleet_confirmations_.pending(); pend.has_value() and fleet::is_yes(text.toStdString()))
    {
        const QJsonObject args{{"action", q(std::string(fleet::to_string(pend->action)))}, {"agent", q(pend->agent)}};
        log_.write("fast_path", QJsonObject{{"kind", "agent_control_yes"}, {"args", args}});
        const QString result = do_agent_control(args);
        if (not result.startsWith(QStringLiteral("OK"))) window_->append_system(QStringLiteral("agents — ") + result);
        push_history({{"role", "user"}, {"content", text}});
        push_history({{"role", "assistant"}, {"content", result.startsWith(QStringLiteral("OK"))
                                                             ? QStringLiteral("Doing it: ") + result.mid(4)
                                                             : result}});
        return;
    }
    // 1c. a plain yes to a mission card: sent here, never by the model (anti-goal: a mission without the person's yes).
    if (card.has_value())
    {
        const std::string title = card->mission.title.empty() ? std::string("the mission") : card->mission.title;
        log_.write("fast_path", QJsonObject{{"kind", "mission_yes"}, {"card", static_cast<qint64>(card->serial)},
                                            {"title", q(title)}});
        const bool sent = send_command(rc::mission::encode_json(card->mission), "mission_card");
        const QString reply = sent ? QStringLiteral("Starting: ") + q(title)
                                   : QStringLiteral("I could not send the mission '%1' to the controller.").arg(q(title));
        window_->begin_assistant();
        window_->append_token(reply);
        window_->end_assistant();
        push_history({{"role", "user"}, {"content", text}});
        push_history({{"role", "assistant"}, {"content", reply}});
        return;
    }
    // 1b. "show me the zed camera / the room / the plan", "close the canvas": a stock drawing, without the model.
    if (const auto show = parse_show(t); show.has_value())
    {
        show_request(*show, text);
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
        window_->append_system("Still answering the previous message — wait, or press Esc.");
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
constexpr int kMaxDrawRepairs = 2;   ///< re-asks after a failed draw script, per user message

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
    "offered, call its tool now. The one question you do ask is the one agent_control returns as NOT DONE.\n"
    "- plan_mission answers OK: tell the person in one sentence what the plan does and that it starts when they say "
    "yes. ASK: put its question to the person. REJECTED: say why it cannot be done.\n"
    "- What the STATE says a door or object \"offers\" (affordances, approach sides) is the controller's current opinion, "
    "not a reason to refuse: when the person asks to go through a door or to approach an object (from whichever side "
    "they name), call cross_door / approach_object with what they said; if it cannot, the tool replies and says why. "
    "Never answer \"I cannot\" to such a request without calling the tool.\n"
    "- Pass the argument the person named: when they name a door, an object, a class, a node or a room, put exactly "
    "that in the call (the node to zoom a graph on, the object whose model to show); never leave it empty or default.\n"
    "- \"Open door_1\" / \"close the door\" ALONE is sim_door (when you have it): the simulator moves the door and the "
    "robot stays where it is. cross_door only when the person wants to go THROUGH the door.\n"
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
    "- show(...) and draw(...) are for the PERSON: they put a picture on the canvas beside the chat. 'Show me', "
    "'display', 'put on the screen', 'draw', 'plot', 'map' ⇒ show or draw, never look (look shows a picture only to "
    "YOU). show = a ready-made picture (the camera, the room map, the mission plan, the working-memory graph, a 3-D "
    "model of the robot or an object) — also when they say 'draw' it; table = a table or list of words or numbers, also "
    "when they say 'draw' it (it needs no room or sensor data; 'show it/that/the list' = your previous answer); draw = anything else (a "
    "chart, a plot, a marked map), as a Python script. After it, say in one short sentence what is on the canvas; do not "
    "describe a picture you have not seen.\n"
    "- Of the robot's own software the STATE has only the names of the agents alive ('agents alive now'); for "
    "anything more about it CALL a tool, never guess: agents() "
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
    QString prompt = description + QStringLiteral("\n") + kInterfaceRules.arg(cfg_.default_step_m, 0, 'f', 1);
    if (not skills_.empty())
        prompt += QStringLiteral("\nSkills: notes you can read when a request needs them. Call use_skill(name) BEFORE "
                                 "acting on such a request.\n") + q(rc::chat::skills_index(skills_));
    return prompt;
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
        QStringLiteral("Take the robot through a door of the CURRENT room: approach it, open it if it is shut, and go "
                       "through. Starts at once.") +
            (sim_door_available() ? QStringLiteral(" Only when the person wants to GO THROUGH it; to just open or "
                                                   "close a door, sim_door.") : QString{}),
        QJsonObject{{"door", QJsonObject{{"type", "string"}, {"enum", doors},
                                         {"description", "a door of the current room, or 'nearest'"}}}},
        QJsonArray{"door"}));
    if (sim_door_available())
        tools.append(tool_def("sim_door",
            "Have the SIMULATOR open or close a door of the CURRENT room at once (the robot does not move and does not "
            "touch it). When the person asks to open or close a door without going through it. Not to go through a "
            "door: cross_door opens it on the way.",
            QJsonObject{{"action", QJsonObject{{"type", "string"}, {"enum", QJsonArray{"open", "close"}}}},
                        {"door", QJsonObject{{"type", "string"}, {"enum", doors},
                                             {"description", "a door of the current room, or 'nearest'"}}}},
            QJsonArray{"action", "door"}));
    tools.append(tool_def("stop", "Stop the robot immediately and abandon what it is doing.", QJsonObject{}, {}));
    tools.append(tool_def("cancel_passage", "Abandon the door crossing in progress (the robot stays where it is).",
                          QJsonObject{}, {}));
    tools.append(tool_def("move_relative",
        "Move the robot a short distance relative to where it is now: first turn in place by turn_deg, then drive "
        "straight distance_m. Starts at once. The robot refuses if a wall or obstacle is in the way. Not for going to "
        "an object (approach_object) or through a door (cross_door).",
        QJsonObject{{"distance_m", QJsonObject{{"type", "number"},
                                               {"description", "metres to drive after turning: positive = forward, "
                                                               "negative = backward, 0 = do not drive"}}},
                    {"turn_deg", QJsonObject{{"type", "number"},
                                             {"description", "degrees to turn in place first: positive = left, "
                                                             "negative = right, 180 = turn around, 0 = do not turn"}}}},
        QJsonArray{"distance_m", "turn_deg"}));
    tools.append(tool_def("approach_object",
        "Drive to an object of the CURRENT room and stop in front of it, facing it: the object itself says where to "
        "stand and from which sides it can be approached (the STATE lists them); the controller plans the path around "
        "obstacles. Give its name (chair_1) or its class (table: the nearest one), and optionally the side in the "
        "person's own words. Starts at once.",
        QJsonObject{{"object", QJsonObject{{"type", "string"}, {"description", "an object name or class"}}},
                    {"side", QJsonObject{{"type", "string"},
                                         {"description", "ONLY when the person names a side: the side to approach from, as they said it "
                                                         "(right, the far side, behind it, por la izquierda) or a side "
                                                         "name from the STATE; empty = the object's default"}}}},
        QJsonArray{"object"}));
    tools.append(tool_def("go_around_object",
        "Drive a full loop around an object of the CURRENT room, stopping on each side facing it, and end back where "
        "the loop began; sides the robot cannot reach (against a wall) are skipped. Give its name or class, and "
        "optionally the direction the person says. Starts at once. To stop at ONE side use approach_object.",
        QJsonObject{{"object", QJsonObject{{"type", "string"}, {"description", "an object name or class"}}},
                    {"direction", QJsonObject{{"type", "string"},
                                              {"description", "ONLY when the person says one: left or right (por la "
                                                              "izquierda / derecha); empty = left"}}}},
        QJsonArray{"object"}));
    tools.append(tool_def("plan_mission",
        "Plan a MISSION: any request with more than one step, a condition or a return (cross a door and explore the "
        "next room, do this then that, come back). Pass the request in your own words with the names from the STATE; "
        "the plan is shown to the person and starts only when they say yes. For ONE action use its own tool "
        "(cross_door, approach_object, go_around_object, move_relative).",
        QJsonObject{{"request", QJsonObject{{"type", "string"},
                                            {"description", "what the person asked for, every step, with the names "
                                                            "from the STATE (door_1, room_2, chair_1)"}}}},
        QJsonArray{"request"}));
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
    tools.append(tool_def("known_rooms",
        "The rooms the robot remembers (long-term memory), which are live now, and the passages (doorways) joining them: "
        "their two rooms, live door, how often crossed and how reliably passable. Read-only.",
        QJsonObject{}, {}));
    tools.append(tool_def("route_to",
        "The remembered route from the CURRENT room to a remembered room: the passages to cross, in order. Read-only; "
        "it does not move the robot (to go, plan a mission with the passages it lists). Call it for any question about "
        "getting to a remembered room: the STATE names those rooms but not the passages.",
        QJsonObject{{"room", QJsonObject{{"type", "string"}, {"description", "a remembered room name, e.g. room_3"}}}},
        QJsonArray{"room"}));
    tools.append(tool_def("agents",
        "The robot's software agents: which are running, their state (Operating / Degraded / Waiting), uptime, CPU "
        "and memory of each. Read-only.",
        QJsonObject{}, {}));
    {
        QString names;
        for (const auto& a : fleet_policy_.agents) names += (names.isEmpty() ? "" : ", ") + q(a);
        tools.append(tool_def("agent_control",
            "Start, stop or restart one of the robot's software agents (" + names + "). When the person asks to "
            "start, stop or restart an agent, call it with the name they give, also one not listed: the tool says "
            "why not. Never to answer a question about them (that is agents); a general question whether you "
            "can start agents is answered yes, in words. Stopping or restarting the controller or retina needs the person's yes: call it first; the tool says so, you "
            "ask, and you call it again with the same arguments only after they answer yes.",
            QJsonObject{{"action", QJsonObject{{"type", "string"}, {"enum", QJsonArray{"start", "stop", "restart"}}}},
                        {"agent", QJsonObject{{"type", "string"},
                                              {"description", "the agent's name, e.g. chair_concept"}}}},
            QJsonArray{"action", "agent"}));
    }
    tools.append(tool_def("sensors",
        "The robot's sensors (cameras, LiDAR, IMU): their streams, the frame rate MEASURED now, resolution, age of the "
        "last frame and bandwidth: numbers about the streams, not what they measure. To let the person SEE what the "
        "LiDARs or cameras measure use show (lidar / camera). Read-only.",
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
    if (not skills_.empty())
    {
        QJsonArray names;
        for (const auto& s : skills_) names.append(q(s.name));
        tools.append(tool_def("use_skill",
            "Read one of the skills listed in the system prompt (how to write a mission, door and exploring tips). "
            "Read-only; returns the skill's text.",
            QJsonObject{{"name", QJsonObject{{"type", "string"}, {"enum", names}, {"description", "the skill's name"}}}},
            QJsonArray{"name"}));
    }
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
    {
        QJsonArray names;
        for (const auto& n : canvas_library()) names.append(n);
        tools.append(tool_def("show",
            "Show the PERSON a ready-made picture on the canvas beside the chat (you do not see it: look is for you). "
            "camera = the live camera picture (args 'zed' = front, 'ricoh' = 360); layout = the current room from "
            "above with the robot, doors and objects; plan = the mission the robot runs; graph = the robot's working "
            "memory as a graph of nodes and edges, the DSR graph: what it knows and how it is connected — 'the graph', 'your "
            "memory graph', also when they say draw (args: a node name to zoom in on it, e.g. 'room_1'; empty = all); "
            "model = the 3-D model of the robot or of an object, which the person can turn with the mouse (args: "
            "'robot', an object name like chair_1, or a class like table); lidar = the live 3-D LiDAR point clouds "
            "in the robot's frame (args: leave EMPTY for both, the usual answer; 'helios' or 'bpearl' only when the person names one); "
            "other names are drawings "
            "saved earlier. Only when one of these is exactly what was asked; any other chart, plot or picture: draw.",
            QJsonObject{{"name", QJsonObject{{"type", "string"}, {"enum", names}, {"description", "which drawing"}}},
                        {"args", QJsonObject{{"type", "string"},
                                             {"description", "camera: 'zed' (front, default) or 'ricoh' (360); graph: the node the person "
                                                             "named (e.g. room_1), empty = all; model: the object the "
                                                             "person named (chair_1, or its class) or 'robot', never empty"}}},
                        {"live", QJsonObject{{"type", "boolean"},
                                             {"description", "true = keep redrawing it (default for camera "
                                                             "and layout); false = draw once"}}}},
            QJsonArray{"name"}));
        tools.append(tool_def("draw",
            "Draw a NEW picture for the PERSON on the canvas beside the chat by writing a Python script (NOT for the "
            "camera, the room map, the plan, the working-memory graph or a 3-D model: show has those, even when the "
            "person says 'draw'; NOT for a table: table; check show first): a chart, a plot, a map with something marked. Python 3 with matplotlib/numpy/PIL, no network. "
            "`from rcanvas import *` gives: w = world() (w.room.name, w.room.polygon [[x,y],...]; w.robot.x/.y/"
            ".heading (rad, the direction it faces)/.radius; w.doors [d.name, d.a, d.b (ends [x,y]), d.p_open]; "
            "w.objects [o.name, o.type, o.x, o.y, o.yaw, o.width, o.depth, o.height, o.dist]; w.target.x/.y; any of "
            "these may be None; doors, objects, nodes and edges are objects with ATTRIBUTES, o.name, o.dist: never o[0]), image('zed'|'ricoh') -> RGB numpy array or None, fig = figure(), ax = "
            "layout_axes(fig) (equal-aspect map axes), draw_layout(ax, w) (room+doors+objects+robot+target), "
            "draw_room/draw_doors/draw_objects/draw_robot/draw_target(ax, w); g = graph() (the working-memory graph: "
            "g.nodes [n.id, n.name, n.type, n.agent], g.edges [e.src, e.dst, e.type] by node id), draw_graph(focus=None) "
            "-> a PIL image of it. Metres, room frame. The script MUST end "
            "with save(fig) (or save_image(img)). If it fails you get the error back: fix it and call draw again.",
            QJsonObject{{"script", QJsonObject{{"type", "string"}, {"description", "the Python source"}}},
                        {"title", QJsonObject{{"type", "string"}, {"description", "a short caption, e.g. 'distance to "
                                                                                  "each object'"}}},
                        {"live", QJsonObject{{"type", "boolean"},
                                             {"description", "true = keep redrawing it with fresh data; "
                                                             "false (default) = draw once"}}}},
            QJsonArray{"script", "title"}));
        tools.append(tool_def("table",
            "Show the PERSON a table on the canvas beside the chat, from rows you already have: your previous answer, "
            "the STATE or a tool result (a list of agents, of objects and distances, of what you can do). No script; "
            "a table is never a draw. Short rows, as many as the content has.",
            QJsonObject{{"title", QJsonObject{{"type", "string"}, {"description", "a short caption"}}},
                        {"header", QJsonObject{{"type", "array"}, {"items", QJsonObject{{"type", "string"}}},
                                               {"description", "the column names"}}},
                        {"rows", QJsonObject{{"type", "array"},
                                             {"items", QJsonObject{{"type", "array"}, {"items", QJsonObject{{"type", "string"}}}}},
                                             {"description", "one list of cells per row, as many cells as the header"}}}},
            QJsonArray{"title", "rows"}));
        tools.append(tool_def("hide_canvas", "Close the drawing canvas beside the chat.", QJsonObject{}, {}));
    }
    tools.append(tool_def("look",
        "Look through a camera now: the current picture is shown to YOU (not to the person) in the next message, so you "
        "can answer about it. To show a picture to the person use show or draw. Read-only.",
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

void ChatSession::ask_model(bool followup, bool offer_tools)
{
    turn_ = Turn{.followup = followup, .offer_tools = offer_tools};
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
    if (const auto m = mission_now_line(mission_progress_); m.has_value()) state_extra_ += q(*m) + "\n";
    if (const auto topo = remembered_topology(); topo.has_value())
        if (const auto line = rc::chat::remembered_rooms_line(*topo, digest_.facts.room); not line.empty())
            state_extra_ += q(line) + "\n";
    state_extra_ += q(alive_line(system_.agents())) + "\n";   // the names from the graph, so none are made up
    state_extra_ += QStringLiteral("last move/turn sent from this chat: ") +
                    (last_relative_.empty() ? QStringLiteral("none") : q(explain_relative_command(last_relative_))) + "\n";
    const QJsonArray msgs = messages_for_request();
    // ★ A follow-up carries the SAME tools as the turn before it. ollama reuses the processed prompt only up to the first
    //   token that differs, and the template writes the tools right after the system prompt: dropping them on the
    //   follow-up and restoring them on the next message reprocessed the whole history twice per tool round (873 ms of
    //   a 1086 ms cold prompt on the 5070 Ti, measured 2026-10-22; several seconds on a Jetson). With the tools on offer
    //   the 8b still answered 35/35 follow-ups in words; a call it makes anyway is never run (on_done).
    const QJsonArray tools = offer_tools ? tools_json() : QJsonArray{};
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
                // What the model says after a tool result is shown, not kept: the call and its result already hold what
                // happened, and the 8b copies its own follow-ups ("I am approaching table_7 … I will stop in front of
                // it") as the answer to the next request, with no call (live 2026-10-19: 0/5 calls replayed with
                // them in the history, 5/5 without).
                if (not content.isEmpty())
                {
                    if (not turn_.followup) push_history({{"role", "assistant"}, {"content", content}});
                }
                // Tokens came but neither words nor a call: ollama dropped a tool call it could not parse (a draw script
                // with 'Today\'s' in it is not JSON, eval 2026-10-19). Ask once more; then say so instead of silence.
                else if (st.eval_count > 0 and lost_answers_++ == 0) ask_model(turn_.followup);
                else if (not turn_.followup) window_->append_system(QStringLiteral("the model's answer could not be read; please ask again"));
                return;
            }
            // A follow-up is for words: a call in it is not run (it would be a second command the person never gave).
            // Its words are shown; with none, ask once more without the tools, as before (a cache miss, rare).
            if (turn_.followup)
            {
                log_.write("followup_call_dropped", QJsonObject{{"tool_calls", turn_.tool_calls}});
                if (content.isEmpty() and turn_.offer_tools) ask_model(true, false);
                return;
            }
            push_history({{"role", "assistant"}, {"content", content}, {"tool_calls", turn_.tool_calls}});
            draw_failed_ = false;
            const bool image = handle_tool_calls(turn_.tool_calls);
            // plan_mission queued the planner: it runs once THIS callback has returned (the client is the same, one
            // request at a time); its result is the round's last tool message, and the follow-up waits for it.
            if (planner_.has_value())
            {
                planner_->image = image;
                window_->set_busy(true);
                QTimer::singleShot(0, &ctx_, [this]() { run_planner(); });
                return;
            }
            after_tool_round(not content.isEmpty(), image);
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

void ChatSession::after_tool_round(bool had_words, bool image)
{
    // A drawing script failed: the model gets its error back WITH the tools, to fix it (at most twice).
    if (draw_failed_ and draw_repairs_ < kMaxDrawRepairs)
    {
        ++draw_repairs_;
        ask_model(false);
        return;
    }
    // A tool round with no words: one follow-up so the model can tell the person what happened.
    // A picture was asked for: the follow-up is where the model gets to see it, words or not.
    if (not had_words or image) ask_model(true);
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
        if (name == "plan_mission")
        {
            const QString refused = queue_planner(args);
            if (refused.isEmpty()) continue;   // queued: its tool message comes when the planner has answered
            push_history({{"role", "tool"}, {"tool_name", name}, {"content", refused}});
            continue;
        }
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
    if (name == "draw" or name == "show") return do_draw(name, args);
    if (name == "table") return do_table(args);
    if (name == "hide_canvas")
    {
        canvas_stop();
        window_->hide_canvas();
        return QStringLiteral("OK: the canvas is closed.");
    }
    if (name == "approach_object") return do_approach(args);
    if (name == "agent_control") return do_agent_control(args);
    if (name == "go_around_object") return do_go_around(args);
    if (name == "sim_door") return do_sim_door(args);
    QString out;
    if (name == "use_skill")
    {
        const std::string want = args.value("name").toString().toStdString();
        const auto it = std::ranges::find(skills_, want, &rc::chat::Skill::name);
        out = it != skills_.end() ? q(it->body) : QStringLiteral("REJECTED: no skill '%1'").arg(q(want));
    }
    else if (QString m = answer_memory(name, args); not m.isEmpty()) out = std::move(m);
    else if (name == "find_objects")
        out = q(answer_find_objects(digest_.objects, args.value("subtype").toString().toStdString(), digest_.facts.room));
    else if (name == "where_is")
        out = q(answer_where_is(digest_.objects, args.value("name").toString().toStdString(), digest_.facts.room));
    else if (name == "known_rooms" or name == "route_to")
    {
        const auto topo = remembered_topology();
        if (not topo.has_value()) out = QStringLiteral("No remembered topology is published (is ltsm_agent running?).");
        else if (name == "known_rooms") out = q(rc::chat::answer_known_rooms(*topo, digest_.facts.room));
        else out = q(rc::chat::answer_route_to(*topo, digest_.facts.room, args.value("room").toString().trimmed().toStdString()));
    }
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

std::optional<rc::chat::Topology> ChatSession::remembered_topology() const
{
    // ltsm_agent's summary of its own (domain 2) memory, published on the live graph (ltsm_agent/src/topology_publish.h).
    if (not G_) return std::nullopt;
    const auto node = G_->get_node("ltsm_topology");
    if (not node.has_value()) return std::nullopt;
    const auto json = G_->get_attrib_by_name<ltsm_topology_att>(node.value());
    if (not json.has_value()) return std::nullopt;
    auto t = rc::chat::decode_topology(json.value());
    if (not t.has_value()) return std::nullopt;
    return std::move(t.value());
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

QString ChatSession::do_approach(const QJsonObject& args)
{
    const std::string what = args.value("object").toString().trimmed().toStdString();
    const std::string room = digest_.facts.room;
    const auto refuse = [&](const std::string& why)
    {
        log_.write("approach", QJsonObject{{"object", q(what)}, {"ok", false}, {"reason", q(why)}});
        window_->append_system(QStringLiteral("Not possible: ") + q(why));
        return QStringLiteral("REJECTED: ") + q(why);
    };
    if (not digest_.facts.controller_present) return refuse("the controller is not running, so the robot cannot move");
    if (room.empty()) return refuse("the current room is not known yet");
    const auto o = pick_object(digest_.objects, what);
    if (not o.has_value()) return refuse(o.error());
    // Both in the room frame: the object's options and the go_to pose are room poses. Optionals checked (CLAUDE.md).
    const auto To = inner_->get_transformation_matrix(room, o->name);
    const auto Tr = inner_->get_transformation_matrix(room, "body");
    if (not To.has_value() or not Tr.has_value())
        return refuse("where '" + o->name + "' or the robot is in " + room + " cannot be resolved right now");
    const std::string side_words = args.value("side").toString().trimmed().toStdString();
    // ★ THE OBJECT PROPOSES (2026-10-18): an object that offers its approach affordance says where to stand and from
    //   which sides; the chat only resolves the person's words to one of those and names it to the controller.
    if (const auto aff = G_->get_node("aff_" + o->name + "_approach"); aff.has_value())
    {
        const auto names = G_->get_attrib_by_name<aff_option_names_att>(*aff);
        const auto xs = G_->get_attrib_by_name<aff_option_x_m_att>(*aff);
        const auto ys = G_->get_attrib_by_name<aff_option_y_m_att>(*aff);
        const auto ws = G_->get_attrib_by_name<aff_option_yaw_rad_att>(*aff);
        std::vector<ApproachSide> sides;
        if (names.has_value() and xs.has_value() and ys.has_value() and ws.has_value())
            if (auto p = parse_sides(*names, xs->get(), ys->get(), ws->get()); p.has_value()) sides = std::move(*p);
        label_sides(sides, {To->translation().x(), To->translation().y()}, {Tr->translation().x(), Tr->translation().y()});
        const auto side = resolve_side(sides, side_words);
        if (not side.has_value()) return refuse(o->name + ": " + side.error());
        const std::string cmd = "approach " + o->name + (side->empty() ? "" : " from " + *side);
        std::string where;
        for (const auto& s : sides)
            if (s.name == (side->empty() ? sides.front().name : *side)) where = s.view + " side as seen from here";
        log_.write("approach", QJsonObject{{"object", q(o->name)}, {"ok", true}, {"side", q(*side)}, {"command", q(cmd)}});
        if (not send_command(cmd, "llm")) return QStringLiteral("FAILED: could not write the command to the graph.");
        return QStringLiteral("OK: '%1' sent: %2 chose where to stand%3; the controller plans the path around obstacles.")
            .arg(q(cmd), q(o->name), where.empty() ? QString{} : QStringLiteral(" (%1)").arg(q(where)));
    }
    // An object whose agent does not offer `approach` yet (only chair does, so far): the old pose path, without sides.
    if (not side_words.empty())
        return refuse(o->name + " does not offer approach sides yet; ask without a side to go to the side the robot is on");
    const double standoff = cfg_.robot_radius_m + cfg_.approach_clearance_m;
    const auto p = approach_point({To->translation().x(), To->translation().y()},
                                  {Tr->translation().x(), Tr->translation().y()}, footprint_radius(*o), standoff);
    if (not p.has_value())
        return QStringLiteral("OK: the robot is already next to %1; nothing to drive.").arg(q(o->name));
    auto am = approach_mission(o->name, room, *p);
    am.request = last_user_text_.toStdString();
    const std::string json = rc::mission::encode_json(am);
    log_.write("approach", QJsonObject{{"object", q(o->name)}, {"ok", true}, {"x", p->x}, {"y", p->y}, {"yaw", p->yaw}});
    if (not send_command(json, "llm")) return QStringLiteral("FAILED: could not write the mission to the graph.");
    return QStringLiteral("OK: mission 'approach %1' sent: the controller drives to a spot %2 m from its centre, facing "
                          "it, planning its path around obstacles.")
        .arg(q(o->name)).arg(footprint_radius(*o) + standoff, 0, 'f', 2);
}

QString ChatSession::do_go_around(const QJsonObject& args)
{
    const std::string what = args.value("object").toString().trimmed().toStdString();
    const std::string room = digest_.facts.room;
    const auto refuse = [&](const std::string& why)
    {
        log_.write("go_around", QJsonObject{{"object", q(what)}, {"ok", false}, {"reason", q(why)}});
        window_->append_system(QStringLiteral("Not possible: ") + q(why));
        return QStringLiteral("REJECTED: ") + q(why);
    };
    if (not digest_.facts.controller_present) return refuse("the controller is not running, so the robot cannot move");
    if (room.empty()) return refuse("the current room is not known yet");
    const auto o = pick_object(digest_.objects, what);
    if (not o.has_value()) return refuse(o.error());
    const auto left = parse_around_direction(args.value("direction").toString().toStdString());
    if (not left.has_value())
        return refuse("'" + args.value("direction").toString().toStdString() + "' is not a direction: left or right");
    const auto To = inner_->get_transformation_matrix(room, o->name);
    const auto Tr = inner_->get_transformation_matrix(room, "body");
    if (not To.has_value() or not Tr.has_value())
        return refuse("where '" + o->name + "' or the robot is in " + room + " cannot be resolved right now");
    std::vector<ApproachSide> sides;   // the object's own standpoints, when it offers them (do_approach)
    if (const auto aff = G_->get_node("aff_" + o->name + "_approach"); aff.has_value())
    {
        const auto names = G_->get_attrib_by_name<aff_option_names_att>(*aff);
        const auto xs = G_->get_attrib_by_name<aff_option_x_m_att>(*aff);
        const auto ys = G_->get_attrib_by_name<aff_option_y_m_att>(*aff);
        const auto ws = G_->get_attrib_by_name<aff_option_yaw_rad_att>(*aff);
        if (names.has_value() and xs.has_value() and ys.has_value() and ws.has_value())
            if (auto p = parse_sides(*names, xs->get(), ys->get(), ws->get()); p.has_value()) sides = std::move(*p);
    }
    const double ring = footprint_radius(*o) + cfg_.robot_radius_m + cfg_.approach_clearance_m;
    auto m = go_around_mission(o->name, room, {To->translation().x(), To->translation().y()},
                               {Tr->translation().x(), Tr->translation().y()}, sides, ring, *left);
    m.request = last_user_text_.toStdString();
    const bool own = sides.size() >= 3;
    log_.write("go_around", QJsonObject{{"object", q(o->name)}, {"ok", true}, {"stops", int(m.steps.size())},
                                        {"own_sides", own}, {"left", *left}});
    if (not send_command(rc::mission::encode_json(m), "llm")) return QStringLiteral("FAILED: could not write the mission to the graph.");
    return QStringLiteral("OK: mission 'go around %1' sent: %2 stops %3, setting off to the robot's %4 and ending back "
                          "at the first; a stop it cannot reach is skipped. The chat reports progress; do not say it is done.")
        .arg(q(o->name)).arg(m.steps.size() - 1)
        .arg(own ? QStringLiteral("at the sides it offers") : QStringLiteral("%1 m from its centre").arg(ring, 0, 'f', 2),
             *left ? QStringLiteral("left") : QStringLiteral("right"));
}

// ── plan_mission: the planner request, its fix rounds and the card (mission_plan.h) ──────────────────────────
namespace
{
constexpr int kMaxPlanFixes = 2;   ///< planner re-asks with the checker's error, per plan_mission call
}

bool ChatSession::name_exists(std::string_view name) const
{
    if (G_ and G_->get_node(std::string(name)).has_value()) return true;
    // a remembered room (named after the legs that lead into it) or a passage of the long-term memory (topology.h)
    if (const auto topo = remembered_topology(); topo.has_value()) return rc::chat::remembered_name(*topo, name);
    return false;
}

QString ChatSession::queue_planner(const QJsonObject& args)
{
    const QString request = args.value("request").toString().trimmed();
    const auto refuse = [&](const QString& why)
    {
        log_.write("plan_mission", QJsonObject{{"request", request}, {"ok", false}, {"reason", why}});
        return QStringLiteral("REJECTED: ") + why;
    };
    if (planner_.has_value()) return refuse(QStringLiteral("one mission at a time: a plan is already being written"));
    if (request.isEmpty()) return refuse(QStringLiteral("the request is empty: pass what the person asked for"));
    if (not digest_.facts.controller_present) return refuse(QStringLiteral("the controller is not running, so no mission can run"));
    planner_ = PlannerJob{.request = request, .request_text = last_user_text_, .turn = user_turn_, .from_wake = wake_turn_};
    log_.write("plan_mission", QJsonObject{{"request", request}, {"ok", true}});
    return {};
}

// ★ The planner sees ONLY the mission skills, the STATE and the request (the eval's layout, check_llm_missions.py):
//   none of the chat's history, rules or tools, so what it writes is what tests/check_llm_missions.py measures.
void ChatSession::run_planner()
{
    if (not planner_.has_value())   // dropped meanwhile (the person spoke, Esc, new conversation)
    {
        if (not llm_.busy()) window_->set_busy(false);   // a message that went to the model keeps its own indicator
        return;
    }
    QString skills;
    for (const char* name : {"missions", "doors", "exploring"})
        if (const auto it = std::ranges::find(skills_, std::string(name), &rc::chat::Skill::name); it != skills_.end())
            skills += (skills.isEmpty() ? QString{} : QStringLiteral("\n\n")) + q(it->body).trimmed();
    if (skills.isEmpty())
    {
        finish_planner(QStringLiteral("REJECTED: the mission skills are not loaded (etc/skills/missions.md)"));
        return;
    }
    QJsonArray msgs{QJsonObject{{"role", "system"}, {"content", skills}},
                    QJsonObject{{"role", "system"}, {"content", planner_state()}},
                    QJsonObject{{"role", "user"}, {"content", q(planner_request(planner_->request_text.toStdString(),
                                                                                planner_->request.toStdString(),
                                                                                planner_->from_wake))}}};
    for (const auto& m : planner_->rounds) msgs.append(m);
    planner_->reply.clear();
    const QString model = cfg_.llm.planner_model.isEmpty() ? cfg_.llm.model : cfg_.llm.planner_model;
    log_.write("planner_request", QJsonObject{{"model", model}, {"request", planner_->request},
                                              {"said", planner_->from_wake ? QString{} : planner_->request_text},
                                              {"fixes", planner_->fixes}});
    const bool ok = llm_.chat(msgs, {}, OllamaClient::Callbacks{
        .on_token = [this](const QString& tok) { if (planner_.has_value()) planner_->reply += tok; },
        .on_tool_calls = {},
        .on_done = [this](const ChatStats& st)
        {
            if (not planner_.has_value()) return;   // resolved meanwhile (drop_planner)
            log_.write("planner_response", QJsonObject{{"content", planner_->reply}, {"wall_ms", st.wall_ms},
                                                       {"eval_count", static_cast<qint64>(st.eval_count)},
                                                       {"done_reason", st.done_reason}});
            on_planned(strip_think(planner_->reply));
        },
        .on_error = [this](const QString& err)
        {
            if (not planner_.has_value()) return;
            log_.write("error", QJsonObject{{"where", "planner"}, {"msg", err}});
            if (err == "cancelled")
            {
                window_->append_system(QStringLiteral("(planning cancelled)"));
                finish_planner(QStringLiteral("REJECTED: the person cancelled the planning"), false);
                return;
            }
            finish_planner(QStringLiteral("REJECTED: the planner failed: ") + err);
        },
    }, RequestOptions{.model = model, .temperature = 0.0, .stream = false});
    if (not ok) finish_planner(QStringLiteral("REJECTED: the model is busy"));
}

void ChatSession::on_planned(const QString& reply)
{
    if (not planner_.has_value() or not plan_still_wanted(planner_->turn, user_turn_)) return;   // a late reply: no card
    const std::string text = reply.toStdString();
    if (const auto question = planned_ask(text); question.has_value())
    {
        finish_planner(QStringLiteral("ASK: ") + q(*question));
        return;
    }
    auto m = check_planned(text, [this](std::string_view n) { return name_exists(n); });
    if (not m.has_value())
    {
        log_.write("planner_refused", QJsonObject{{"error", q(m.error())}, {"fixes", planner_->fixes}});
        if (planner_->fixes >= kMaxPlanFixes)
        {
            finish_planner(QStringLiteral("REJECTED: ") + q(m.error()));
            return;
        }
        ++planner_->fixes;
        planner_->rounds.append(QJsonObject{{"role", "assistant"}, {"content", reply}});
        planner_->rounds.append(QJsonObject{{"role", "user"}, {"content",
            QStringLiteral("That mission was refused: %1. Write the whole mission again, fixed. Output ONLY the mission "
                           "text, or one ask: line.").arg(q(m.error()))}});
        QTimer::singleShot(0, &ctx_, [this]() { run_planner(); });   // after this callback, like the first request
        return;
    }
    m->request = planner_->request_text.toStdString();
    // A plan written in answer to a failure's wake (its own turn, or the person's answer to an ASK) repairs that
    // mission: the card carries its key, and EM links the two. Any later plan is a new subject.
    if (not fix_key_.empty() and fix_still_open(fix_turn_, user_turn_)) m->fixes = fix_key_;
    fix_key_.clear();
    if (pending_mission_.has_value())   // a repair round planned again while the first card waits
    {
        log_.write("mission_card_replaced", QJsonObject{{"card", static_cast<qint64>(pending_mission_->serial)}});
        window_->append_system(QStringLiteral("(replaced the previous plan)"));
    }
    pending_mission_ = MissionCard{.mission = *m, .serial = ++card_serial_, .turn = user_turn_};
    // ★ The card shows the mission AS IT WILL RUN (printed back from the checked IR), never the model's raw reply.
    const std::string shown = rc::mission::print_text(*m);
    log_.write("mission_card", QJsonObject{{"card", static_cast<qint64>(card_serial_)}, {"text", q(shown)},
                                           {"fixes", planner_->fixes}});
    window_->append_card(planner_->request, q(shown).trimmed());
    finish_planner(QStringLiteral("OK: card shown; wait for the person's yes. It shows:\n") + q(shown).trimmed());
}

QString ChatSession::planner_state() const
{
    // ★ The planner has no route_to: the way to a remembered room is given to it, legs in order (topology.h ways_line).
    QString st = state_message();
    if (const auto topo = remembered_topology(); topo.has_value())
        if (const auto line = rc::chat::ways_line(*topo, digest_.facts.room); not line.empty()) st += q(line) + "\n";
    return st;
}

void ChatSession::drop_planner(const QString& result)
{
    if (not planner_.has_value()) return;
    const QString request = planner_->request;
    planner_.reset();   // before cancel(): the request's on_error then finds no job
    if (llm_.busy()) llm_.cancel();
    window_->set_busy(false);
    log_.write("query", QJsonObject{{"tool", "plan_mission"}, {"args", QJsonObject{{"request", request}}}, {"result", result}});
    push_history({{"role", "tool"}, {"tool_name", "plan_mission"}, {"content", result}});
}

void ChatSession::finish_planner(const QString& result, bool speak)
{
    if (not planner_.has_value()) return;
    const PlannerJob job = *std::move(planner_);
    planner_.reset();
    window_->set_busy(false);
    log_.write("query", QJsonObject{{"tool", "plan_mission"}, {"args", QJsonObject{{"request", job.request}}}, {"result", result}});
    push_history({{"role", "tool"}, {"tool_name", "plan_mission"}, {"content", result}});
    // Always a follow-up: whatever the model said before its call cannot tell the person an ASK or a REJECTED.
    if (speak) after_tool_round(false, job.image);
}

// ── sim_door: the simulator opens / closes a door (sim_door.h, tools/sim_door.py) ────────────────────────────────

bool ChatSession::sim_door_available() const
{
    if (cfg_.sim_door_endpoint.empty()) return false;
    const auto robots = G_->get_nodes_by_type("robot");
    return not robots.empty() and G_->get_attrib_by_name<robot_gt_x_att>(robots.front()).has_value();
}

QString ChatSession::do_sim_door(const QJsonObject& args)
{
    const QString action = args.value("action").toString().trimmed().toLower();
    std::string door = args.value("door").toString().trimmed().toLower().toStdString();
    const std::string room = digest_.facts.room;
    const auto refuse = [&](const std::string& why)
    {
        log_.write("sim_door", QJsonObject{{"args", args}, {"ok", false}, {"reason", q(why)}});
        return QStringLiteral("REJECTED: ") + q(why);
    };
    if (not sim_door_available()) return refuse("there is no simulator to move doors (this is the real robot)");
    if (sim_door_proc_.state() != QProcess::NotRunning)
        return refuse("still waiting for the simulator about " + sim_door_what_.toStdString());
    if (action != "open" and action != "close") return refuse("action must be open or close");
    if (room.empty()) return refuse("the current room is not known yet");
    if (door.empty() or door == "nearest" or door == "the door" or door == "door")
    {
        if (not digest_.facts.nearest_door.has_value()) return refuse("no door of " + room + " is known");
        door = *digest_.facts.nearest_door;
    }
    if (std::ranges::find(digest_.facts.doors, door) == digest_.facts.doors.end())
        return refuse("'" + door + "' is not a door of " + room + " (the current room)");
    const auto robots = G_->get_nodes_by_type("robot");
    const auto gx = G_->get_attrib_by_name<robot_gt_x_att>(robots.front());
    const auto gy = G_->get_attrib_by_name<robot_gt_y_att>(robots.front());
    const auto ga = G_->get_attrib_by_name<robot_gt_angle_att>(robots.front());
    const auto Td = inner_->get_transformation_matrix(room, door);
    const auto Tr = inner_->get_transformation_matrix(room, "body");
    if (not gx.has_value() or not gy.has_value() or not ga.has_value() or not Td.has_value() or not Tr.has_value())
        return refuse("where " + door + " or the robot is cannot be resolved right now");
    const Eigen::Vector2d w = room_point_in_world({Td->translation().x(), Td->translation().y()}, Tr->matrix(),
                                                  gx.value(), gy.value(), ga.value());
    sim_door_what_ = action + " " + q(door);
    sim_door_proc_.setProgram(QStringLiteral("python3"));
    sim_door_proc_.setArguments({q(cfg_.sim_door_script), action, QString::number(w.x(), 'f', 3),
                                 QString::number(w.y(), 'f', 3), QString::number(cfg_.sim_door_match_max_m, 'f', 2),
                                 q(cfg_.sim_door_endpoint)});
    sim_door_proc_.start();
    log_.write("sim_door", QJsonObject{{"args", args}, {"ok", true}, {"door", q(door)}, {"world_x", w.x()}, {"world_y", w.y()}});
    window_->append_system(QStringLiteral("simulator — %1").arg(sim_door_what_));
    return QStringLiteral("OK: asked the simulator to %1 %2. The chat tells the person when it answers; do not say it "
                          "is done.").arg(action, q(door));
}

void ChatSession::on_sim_door_done(int code)
{
    const QJsonObject r = QJsonDocument::fromJson(sim_door_proc_.readAllStandardOutput().trimmed()).object();
    QString msg;
    if (r.value("ok").toBool())
        msg = QStringLiteral("the simulator is moving it (%1, %2 m from where the map has it); it takes a few seconds")
                  .arg(r.value("id").toString()).arg(r.value("dist_m").toDouble(), 0, 'f', 2);
    else if (r.contains("refusal"))
        msg = QStringLiteral("the simulator refused (%1)").arg(r.value("refusal").toString());
    else if (r.contains("error"))
        msg = QStringLiteral("not done: ") + r.value("error").toString();
    else
        msg = QStringLiteral("not done: the helper failed (exit %1): %2")
                  .arg(code).arg(QString::fromUtf8(sim_door_proc_.readAllStandardError()).trimmed().right(200));
    log_.write("sim_door_done", QJsonObject{{"what", sim_door_what_}, {"result", r}, {"exit", code}});
    window_->append_system(QStringLiteral("simulator — %1: %2").arg(sim_door_what_, msg));
    push_history({{"role", "tool"}, {"tool_name", "sim_door"}, {"content", sim_door_what_ + ": " + msg}});
}

// ─── the side canvas (canvas_runner.h / canvas_snapshot.h / canvas_panel.h) ──────────────────────────────────────

QStringList ChatSession::canvas_library() const
{
    QStringList names;
    for (const char* sub : {"/stock", "/library"})
        for (const QFileInfo& f : QDir(canvas_runner_.canvas_dir() + sub).entryInfoList({"*.py"}, QDir::Files, QDir::Name))
            if (not names.contains(f.completeBaseName()) and f.completeBaseName() != "table")   // the table tool's
                names.append(f.completeBaseName());
    if (not names.contains("plan")) names.append("plan");     // not a script: the plan page of the canvas
    if (not names.contains("model")) names.append("model");   // not a script: the 3-D page (canvas_model.h)
    if (not names.contains("lidar")) names.append("lidar");   // not a script: the point-cloud page (canvas_lidar.h)
    return names;
}

std::optional<CanvasJob> ChatSession::library_job(const QString& name, const QStringList& args) const
{
    for (const char* sub : {"/stock/", "/library/"})
    {
        const QString path = canvas_runner_.canvas_dir() + sub + name + ".py";
        QFile f(path);
        if (not f.open(QIODevice::ReadOnly | QIODevice::Text)) continue;
        CanvasJob job{.title = name, .script_path = path, .args = args, .source = QString::fromUtf8(f.readAll())};
        job.wants_images = job.source.contains("image(");
        job.wants_graph = job.source.contains("graph(");
        return job;
    }
    return std::nullopt;
}

void ChatSession::write_canvas_inputs(const CanvasJob& job)
{
    const QSize sz = window_->canvas()->drawing_size();
    const QJsonObject snap = build_canvas_snapshot(*G_, *inner_, digest_, mission_progress_, cfg_.robot_radius_m,
                                                   SnapshotSize{std::max(sz.width(), 320), std::max(sz.height(), 240)});
    if (not write_canvas_snapshot(canvas_runner_.run_dir(), snap))
        std::println("[chat_agent] canvas: cannot write the snapshot in {}", canvas_runner_.run_dir().toStdString());
    // Camera frames only for a script that reads them (encoding a 360 frame every second is not free).
    for (const auto& [cam, file] : {std::pair{Camera::front, "/zed.jpg"}, std::pair{Camera::pano, "/ricoh.jpg"}})
    {
        const std::string path = (canvas_runner_.run_dir() + file).toStdString();
        if (const auto fr = job.wants_images ? cameras_.grab(cam) : std::expected<GrabbedFrame, std::string>(
                                                                         std::unexpected(std::string())); fr.has_value())
            cv::imwrite(path, fr->bgr, {cv::IMWRITE_JPEG_QUALITY, 90});
        else QFile::remove(QString::fromStdString(path));   // no stale frame from an earlier run
    }
    // The whole graph only for a script that reads it (graph() in rcanvas.py).
    if (job.wants_graph)
    {
        std::map<std::uint32_t, std::string> agent_names;
        for (const auto& a : system_.agents()) agent_names[a.id] = a.name;
        write_canvas_json(canvas_runner_.run_dir(), QStringLiteral("graph.json"), build_canvas_graph(*G_, agent_names));
    }
    else QFile::remove(canvas_runner_.run_dir() + QStringLiteral("/graph.json"));
}

namespace
{
// The stock camera drawing is the frame itself: shown straight from CameraGrabber's deep copy, no Python, so a live
// camera runs at the canvas period instead of at a script's start-up cost.
std::optional<Camera> stock_camera(const CanvasJob& job)
{
    if (not job.script_path.endsWith(QStringLiteral("/stock/camera.py"))) return std::nullopt;
    const QString a = job.args.isEmpty() ? QString() : job.args.front().toLower();
    return a == "ricoh" or a == "360" or a == "pano" ? Camera::pano : Camera::front;
}

QImage to_qimage(const cv::Mat& bgr)
{
    return QImage(bgr.data, bgr.cols, bgr.rows, static_cast<qsizetype>(bgr.step), QImage::Format_BGR888).copy();
}
}   // namespace

QString ChatSession::draw_now(CanvasJob job, bool savable)
{
    canvas_stop();            // the previous drawing's live worker, camera or lidars
    window_->show_canvas();   // first, so the snapshot carries the panel's real size
    if (const auto cam = stock_camera(job); cam.has_value())
    {
        auto frame = cameras_.grab(*cam);
        log_.write("draw", QJsonObject{{"title", job.title}, {"camera", int(*cam)}, {"live", job.live},
                                       {"ok", frame.has_value()}, {"error", frame.has_value() ? QString() : q(frame.error())}});
        if (not frame.has_value())
        {
            window_->canvas()->show_message(job.title, q(frame.error()));
            return q(frame.error());
        }
        window_->canvas()->show_drawing(job.title, to_qimage(frame->bgr), job.live, job.source, false);
        canvas_camera_ = cam;
        canvas_job_ = std::move(job);
        canvas_savable_ = false;
        canvas_live_ = canvas_job_.live;
        return {};
    }
    canvas_camera_.reset();
    write_canvas_inputs(job);
    const CanvasRunResult r = canvas_runner_.run_blocking(job);
    log_.write("draw", QJsonObject{{"title", job.title}, {"script", job.script_path}, {"args", job.args.join(' ')},
                                   {"live", job.live}, {"ok", r.ok}, {"ms", r.ms}, {"error", r.error}});
    if (not r.ok)
    {
        window_->canvas()->show_message(job.title, QStringLiteral("Could not draw it: ") + r.error);
        return r.error;
    }
    window_->canvas()->show_drawing(job.title, r.image, job.live, job.source, savable);
    canvas_job_ = std::move(job);
    canvas_savable_ = savable;
    canvas_live_ = canvas_job_.live;
    return {};
}

void ChatSession::tick_canvas()
{
    if (not window_ or not window_->canvas_visible()) return;
    if (not canvas_live_ and not canvas_lidar_) return;
    if (canvas_lidar_)
    {
        lidars_.retry();   // a subscriber that is not up yet (self-throttled); keeps the chosen sensors
        if (lidars_.poll(*inner_))
        {
            window_->canvas()->update_lidar(lidars_.cloud(0), lidars_.cloud(1), lidars_.stamp_ms(), q(lidars_.status()));
            if (not lidar_logged_)
            {
                lidar_logged_ = true;
                log_.write("lidar_first_frame", QJsonObject{{"status", q(lidars_.status())}});
                std::println("[chat_agent] canvas: first LiDAR frame — {}", lidars_.status());
            }
        }
        return;
    }
    if (canvas_camera_.has_value())
    {
        if (auto frame = cameras_.grab(*canvas_camera_); frame.has_value())
            window_->canvas()->update_image(to_qimage(frame->bgr));
        return;
    }
    if (canvas_runner_.running()) return;   // the previous redraw is still drawing: skip this tick
    write_canvas_inputs(canvas_job_);
    canvas_runner_.redraw(canvas_job_, [this](CanvasRunResult r)
    {
        if (not canvas_live_) return;
        if (r.ok) { window_->canvas()->update_image(r.image); return; }
        // A live drawing that starts failing (a camera gone, a room with no polygon yet…): stop, say why, don't spin.
        canvas_live_ = false;
        window_->canvas()->set_live(false);
        window_->append_system(QStringLiteral("canvas: the live drawing stopped — ") + r.error.section('\n', 0, 0));
        log_.write("draw_live_failed", QJsonObject{{"title", canvas_job_.title}, {"error", r.error}});
    });
}

void ChatSession::canvas_stop()
{
    canvas_live_ = false;
    canvas_runner_.kill();
    canvas_camera_.reset();
    if (canvas_lidar_) lidars_.release();
    canvas_lidar_ = false;
}

QString ChatSession::show_lidar(const std::string& which_raw)
{
    std::string which = which_raw;
    std::ranges::transform(which, which.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::vector<std::string> sensors;
    if (which.find("helios") != std::string::npos or which == "top" or which == "upper") sensors = {"helios"};
    else if (which.find("bpearl") != std::string::npos or which == "dome" or which == "lower") sensors = {"bpearl"};
    canvas_stop();
    lidars_.want(sensors);
    const bool any = G_->get_node("helios").has_value() or G_->get_node("bpearl").has_value();
    log_.write("draw_lidar", QJsonObject{{"which", q(which)}, {"sensor_nodes", any}});
    if (not any) return QStringLiteral("ERROR: the robot has no 3-D LiDAR nodes (helios / bpearl) in the graph");
    canvas_lidar_ = true;
    lidar_logged_ = false;
    window_->show_canvas();
    window_->canvas()->show_lidar(sensors.empty() ? QStringLiteral("3-D LiDARs: helios + bpearl")
                                                  : QStringLiteral("3-D LiDAR: %1").arg(q(sensors.front())),
                                  QStringLiteral("robot frame · helios and bpearl in two colours · waiting for points…"));
    return QStringLiteral("OK: the %1 point cloud is on the canvas, live, in the robot's frame; the person can turn it "
                          "with the mouse.").arg(sensors.empty() ? QStringLiteral("3-D LiDARs'") : q(sensors.front()));
}

QString ChatSession::show_model(const std::string& what)
{
    const auto m = find_model(*G_, digest_.objects, what);
    log_.write("draw_model", QJsonObject{{"what", q(what)}, {"ok", m.has_value()},
                                         {"result", q(m.has_value() ? m->description : m.error())}});
    if (not m.has_value()) return QStringLiteral("ERROR: ") + q(m.error());
    canvas_stop();
    window_->show_canvas();
    window_->canvas()->show_model(QStringLiteral("3-D model: %1").arg(q(m->node)), m->triangles, q(m->description));
    return QStringLiteral("OK: the 3-D model of %1 is on the canvas (%2 triangles; the person can turn it with the mouse).")
        .arg(q(m->node)).arg(m->triangles.size() / 3);
}

void ChatSession::show_request(const ShowRequest& r, const QString& text)
{
    log_.write("fast_path", QJsonObject{{"kind", "show"}, {"what", int(r.kind)}, {"arg", q(r.arg)}});
    push_history({{"role", "user"}, {"content", text}});
    QString reply;
    if (r.kind == ShowRequest::Kind::hide)
    {
        canvas_stop();
        window_->hide_canvas();
        reply = QStringLiteral("Canvas closed.");
    }
    else if (r.kind == ShowRequest::Kind::plan)
    {
        canvas_stop();
        window_->show_canvas();
        window_->canvas()->show_plan();
        reply = window_->canvas()->plan()->has_mission() ? QStringLiteral("The mission plan is on the canvas.")
                                                          : QStringLiteral("No mission yet: the canvas shows an empty plan.");
    }
    else if (r.kind == ShowRequest::Kind::lidar)
    {
        const QString res = show_lidar(r.arg);
        reply = res.startsWith("OK") ? QStringLiteral("Showing the %1 on the canvas, live (drag to turn it).")
                                           .arg(r.arg.empty() ? QStringLiteral("3-D LiDARs") : q(r.arg))
                                     : QStringLiteral("I cannot show the LiDARs: ") + res.mid(7);
    }
    else if (r.kind == ShowRequest::Kind::model)
    {
        const QString res = show_model(r.arg);
        reply = res.startsWith("OK") ? QStringLiteral("Showing the 3-D model of %1 on the canvas (drag to turn it).")
                                           .arg(r.arg == "robot" ? QStringLiteral("the robot") : q(r.arg))
                                     : QStringLiteral("I cannot show that model: ") + res.mid(7);
    }
    else if (r.kind == ShowRequest::Kind::graph)
    {
        auto job = library_job("graph", r.arg.empty() ? QStringList{} : QStringList{q(r.arg)});
        if (not job.has_value()) reply = QStringLiteral("I cannot draw: the stock canvas scripts are missing.");
        else
        {
            job->title = r.arg.empty() ? QStringLiteral("working memory graph")
                                       : QStringLiteral("working memory around %1").arg(q(r.arg));
            const QString title = job->title;
            const QString err = draw_now(std::move(*job), false);
            reply = err.isEmpty() ? QStringLiteral("Showing the %1 on the canvas.").arg(title)
                                  : QStringLiteral("I could not draw the %1: %2").arg(title, err.trimmed().section('\n', -1));
        }
    }
    else
    {
        const bool cam = r.kind == ShowRequest::Kind::camera;
        auto job = library_job(cam ? "camera" : "layout", cam ? QStringList{q(r.arg)} : QStringList{});
        if (not job.has_value()) reply = QStringLiteral("I cannot draw: the stock canvas scripts are missing.");
        else
        {
            job->title = cam ? (r.arg == "zed" ? QStringLiteral("front camera (ZED)") : QStringLiteral("360 camera (Ricoh)"))
                             : QStringLiteral("room layout");
            job->live = true;
            const QString title = job->title;
            const QString err = draw_now(std::move(*job), false);
            reply = err.isEmpty() ? QStringLiteral("Showing the %1 on the canvas.").arg(title)
                                  : QStringLiteral("I could not draw the %1: %2").arg(title, err.section('\n', 0, 0));
        }
    }
    window_->append_system(reply);
    push_history({{"role", "assistant"}, {"content", reply}});
}

QString ChatSession::do_draw(const QString& tool, const QJsonObject& args)
{
    const QString stock = tool == "show" ? args.value("name").toString().trimmed() : QString();
    const QString script = tool == "draw" ? args.value("script").toString() : QString();
    QString title = args.value("title").toString().trimmed();
    // live: as asked; unasked, a stock camera/layout is live (it is a view of now), a script draws once.
    const QJsonValue lv = args.value("live");
    const bool live = lv.isBool() ? lv.toBool()
                      : lv.isString() ? lv.toString() == QStringLiteral("true")
                                      : (stock == "camera" or stock == "layout");
    const QStringList extra = args.value("args").toString().split(' ', Qt::SkipEmptyParts);
    const auto fail = [this](const QString& why)
    {
        draw_failed_ = true;
        return QStringLiteral("ERROR: ") + why;
    };
    if (stock == "lidar")
    {
        // One sensor only when the PERSON named it: qwen3-vl passes 'helios' for "let me see the lidar points" 5/5,
        // whatever the tool description says, and the person then sees half the robot's LiDAR without asking.
        QString which = args.value("args").toString().trimmed().toLower();
        QString asked;
        for (auto it = history_.rbegin(); it != history_.rend(); ++it)
            if (it->value("role").toString() == "user") { asked = it->value("content").toString().toLower(); break; }
        if (not which.isEmpty() and not asked.contains(which)) which.clear();
        return show_lidar(which.toStdString());
    }
    if (stock == "model")
    {
        const QString res = show_model(args.value("args").toString().trimmed().toStdString());
        if (res.startsWith("ERROR")) draw_failed_ = false;   // not a script to fix: the model just reports it
        return res;
    }
    if (stock == "plan")
    {
        canvas_stop();
        window_->show_canvas();
        window_->canvas()->show_plan();
        return QStringLiteral("OK: the mission plan is on the canvas.");
    }
    CanvasJob job;
    bool savable = false;
    if (not script.trimmed().isEmpty())
    {
        const QString path = canvas_runner_.run_dir() + QStringLiteral("/script.py");
        QFile f(path);
        if (not f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
            return fail(QStringLiteral("cannot write the script: ") + f.errorString());
        f.write(script.toUtf8());
        f.close();
        job = CanvasJob{.title = title.isEmpty() ? QStringLiteral("drawing") : title, .script_path = path, .args = extra,
                        .source = script, .live = live, .wants_images = script.contains("image("),
                        .wants_graph = script.contains("graph(")};
        savable = true;
    }
    else if (not stock.isEmpty())
    {
        auto j = library_job(stock, extra);
        if (not j.has_value())
            return fail(QStringLiteral("no ready-made drawing '%1'; known: %2").arg(stock, canvas_library().join(", ")));
        job = std::move(*j);
        job.live = live;
        if (not title.isEmpty()) job.title = title;
    }
    else return fail(tool == "show" ? QStringLiteral("give the name of a ready-made drawing")
                                    : QStringLiteral("give the Python source in script"));
    const QString shown = job.title;
    const QString err = draw_now(std::move(job), savable);
    if (not err.isEmpty())
    {
        window_->append_system(QStringLiteral("canvas: the drawing failed"));
        return fail(err);
    }
    window_->append_system(QStringLiteral("canvas: %1%2").arg(shown, live ? QStringLiteral(" (live)") : QString()));
    return QStringLiteral("OK: '%1' is drawn on the canvas beside the chat%2.")
        .arg(shown, live ? QStringLiteral(", redrawn continuously") : QString());
}

QString ChatSession::do_table(const QJsonObject& args)
{
    // stock/table.py draws {title, header, rows} with rcanvas.draw_table: the model gives the words, never a script
    const QString title = args.value("title").toString().trimmed();
    const QJsonObject spec{{"title", title}, {"header", args.value("header")}, {"rows", args.value("rows")}};
    auto job = library_job("table", {QString::fromUtf8(QJsonDocument(spec).toJson(QJsonDocument::Compact))});
    if (not job.has_value()) return QStringLiteral("ERROR: the stock canvas scripts are missing.");
    job->title = title.isEmpty() ? QStringLiteral("table") : title;
    const QString shown = job->title;
    if (const QString err = draw_now(std::move(*job), false); not err.isEmpty())
    {
        window_->append_system(QStringLiteral("canvas: the table failed"));
        draw_failed_ = true;
        return QStringLiteral("ERROR: ") + err.trimmed().section('\n', -1);
    }
    window_->append_system(QStringLiteral("canvas: ") + shown);
    return QStringLiteral("OK: the table '%1' is on the canvas beside the chat.").arg(shown);
}

void ChatSession::save_canvas_script(const QString& raw)
{
    const QString name = q(canvas_script_name(raw.toStdString()));
    if (name.isEmpty() or canvas_job_.source.isEmpty())
    {
        window_->append_system(QStringLiteral("canvas: nothing to save under that name"));
        return;
    }
    const QString dir = canvas_runner_.canvas_dir() + QStringLiteral("/library");
    QDir().mkpath(dir);
    QFile f(dir + "/" + name + ".py");
    if (not f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        window_->append_system(QStringLiteral("canvas: cannot save — ") + f.errorString());
        return;
    }
    f.write(QStringLiteral("# %1 — saved from the chat canvas %2\n")
                .arg(canvas_job_.title, QDateTime::currentDateTime().toString(Qt::ISODate)).toUtf8());
    f.write(canvas_job_.source.toUtf8());
    log_.write("draw_saved", QJsonObject{{"name", name}, {"title", canvas_job_.title}});
    window_->append_system(QStringLiteral("canvas: saved as '%1' — ask for it by name next time").arg(name));
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
    window_->append_system(QStringLiteral("→ sent to the controller: %1").arg(q(display_command(command))));
    std::println("[chat_agent] command '{}' sent ({})", command, origin);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// The controller's report on the last command, every 250 ms.
void ChatSession::tick_status()
{
    // Wakes queued on an EARLIER tick: by now the command's own end (robot_status, below) is in the history before the
    // wake, never after the model has been asked.
    drain_wakes();
    {   // whatever the mission's source: chat, remote_cmd.txt, the panel
        auto u = mission_watch_.poll();
        if (u.mission.has_value())
        {
            mission_progress_ = {};   // a new mission: the old one's progress is not its
            mission_ir_ = u.mission->first;
        }
        if (window_ and u.mission.has_value()) window_->set_plan(u.mission->first, QString::fromStdString(u.mission->second));
        if (u.progress.has_value()) on_mission_progress(*u.progress);   // + the STATE's mission line (ask_model)
        if (window_ and u.progress.has_value()) window_->set_plan_progress(*u.progress);
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
        log_.write("intention_unacked", QJsonObject{{"command", q(display_command(sent_command_))}});
    }
    if (state == seen_state_ and step == seen_step_ and event == seen_event_) return;
    log_.write("intention_state", QJsonObject{{"command", q(display_command(sent_command_))}, {"state", q(state)}, {"bt_state", q(s.bt_state)}});

    // A mission (the chat's JSON) and the words about one (pause / unpause / cancel / resume) are told by the
    // progress lines of controller_mission (on_mission_progress); this node only repeats them, under the word's name.
    const bool about_mission = sent_command_.starts_with('{');
    const bool mission_word = sent_command_ == "pause" or sent_command_ == "unpause" or sent_command_ == "cancel"
                              or sent_command_ == "resume";
    if (mission_word)
    {
        // Only a refusal is news: the word was not taken (the mission ended meanwhile), the mission's own end is not.
        if (not reported_end_ and state.starts_with("failed") and not mission_progress_.status.starts_with("failed"))
        {
            reported_end_ = true;
            QString line = QStringLiteral("'%1' refused").arg(q(sent_command_));
            if (const auto bar = state.find('|'); bar != std::string::npos) line += " — " + q(state.substr(bar + 1));
            window_->append_robot(line);
        }
    }
    else if (state == "in_progress")
    {
        if (not about_mission and (step != seen_step_ or event != seen_event_))
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
        const std::string shown = display_command(sent_command_);
        QString line = halt and state == "aborted" ? QStringLiteral("'%1' done").arg(q(shown))
                                                     : QStringLiteral("'%1' %2").arg(q(shown), q(state));
        if (not event.empty()) line += " — " + q(event);
        // A mission's end is already in the window as its progress lines (✓ / ✗ / ■): only the model is told here.
        if (not about_mission) window_->append_robot(line);
        push_history({{"role", "tool"}, {"tool_name", "robot_status"},
                      {"content", q(robot_status_text(shown, state, event))}});
    }
    seen_state_ = state;
    seen_step_ = step;
    seen_event_ = event;
}

void ChatSession::on_mission_progress(rc::mission::Progress now)
{
    rc::mission::Progress before = std::exchange(mission_progress_, now);
    if (before.serial != now.serial) before = {};   // another mission's snapshot is no baseline for this one
    // A mission already under way (or over) when this chat started is a baseline, not news: no lines, no wakes.
    if (not mission_primed_)
    {
        mission_primed_ = true;
        if (now.status.empty() or (now.t0 != 0 and now.t0 < start_ms_)) return;   // t0 = 0: started, not ticked yet
    }
    if (not window_) return;
    for (const auto& line : progress_lines(before, now, mission_ir_)) window_->append_robot(q(line));
    // ★ Only the chat's own missions wake the model: one from remote_cmd.txt or the panel is shown, never talked about.
    if (now.source != "chat") return;
    const auto queue = [this, &now](std::string kind, std::string note, std::string fixes = {})
    {
        log_.write("wake_queued", QJsonObject{{"kind", q(kind)}, {"mission", static_cast<qint64>(now.serial)}, {"note", q(note)}});
        wakes_.push(std::move(note), std::move(fixes));
    };
    if (const auto topics = new_reports(before, now, mission_ir_); not topics.empty())
    {
        // ★ The snapshot is taken NOW, as the step runs: the robot goes on moving while the model writes (spec §5.3).
        tick_digest();
        const std::string room = physical_room();
        auto objects = digest_.objects;
        if (room != digest_.facts.room)   // the first visit of a proto-room: `current` still names the room left
            for (auto& o : objects) o.in_current_room = not room.empty() and o.room == room;
        const std::string seen = answer_find_objects(std::move(objects), "", room);
        for (const auto& topic : topics) queue("report", report_wake(topic, room, seen));
    }
    if (now.status.starts_with("failed|") and not before.status.starts_with("failed|"))
        if (const auto note = failure_wake(now, mission_ir_); note.has_value())
            queue("failure", *note, std::format("m{}", now.serial));
    if (now.status == "completed" and before.status != "completed") queue("summary", summary_wake(now));
}

void ChatSession::drain_wakes()
{
    // ★ Never while a model request or a planner job is in flight: a wake is not the person speaking, so it must not
    //   resolve a plan (on_user_text's rule) nor interleave with a tool round.
    const auto note = wakes_.pop_if_idle(llm_.busy() or planner_.has_value());
    if (not note.has_value()) return;
    if (not wakes_.last_fixes().empty())
    {
        fix_key_ = wakes_.last_fixes();
        fix_turn_ = user_turn_;
    }
    log_.write("wake", QJsonObject{{"note", q(*note)}, {"fixes", q(wakes_.last_fixes())}});
    draw_repairs_ = 0;
    lost_answers_ = 0;
    if (not ollama_ok_) ping_llm();
    wake_turn_ = true;
    push_history({{"role", "user"}, {"content", q(wake_message(*note))}});
    ask_model(false);
}

std::string ChatSession::physical_room() const
{
    // room_resolve.h rule 2a, without rule 1 in front of it: the room whose robot RT is still LIVE is where the pose is
    // being published, i.e. where the robot is, even while ltsm's `current` names the room it has just left.
    std::string live;
    int n = 0;
    for (const auto& e : G_->get_edges_by_type("RT"))
    {
        const auto a = G_->get_node(e.from());
        const auto b = G_->get_node(e.to());
        if (not a.has_value() or not b.has_value()) continue;
        const DSR::Node* room = a->type() == "room" and b->type() == "robot"   ? &*a
                                : b->type() == "room" and a->type() == "robot" ? &*b
                                                                               : nullptr;
        if (room == nullptr or not G_->get_attrib_by_name<valid_att>(e).value_or(true)) continue;
        live = room->name();
        ++n;
    }
    return n == 1 ? live : digest_.facts.room;
}

void ChatSession::mission_control(const MissionControl& act, const QString& text)
{
    QString reply = q(act.reply);
    log_.write("fast_path", QJsonObject{{"kind", "mission_control"}, {"status", q(mission_progress_.status)},
                                        {"word", q(act.word)}});
    if (not act.word.empty() and not send_command(act.word, "fast_path"))
        reply = QStringLiteral("I could not send '%1' to the controller.").arg(q(act.word));
    window_->begin_assistant();
    window_->append_token(reply);
    window_->end_assistant();
    push_history({{"role", "user"}, {"content", text}});
    push_history({{"role", "assistant"}, {"content", reply}});
}

// ── agent_control: start / stop / restart a fleet agent (fleet_control.h) ────────────────────────────────────────

namespace
{
const std::filesystem::path kProc{"/proc"};
std::int64_t fleet_now() { return static_cast<std::int64_t>(rc::provenance::now_ms()); }
}   // namespace

QString ChatSession::agent_log(const std::string& agent) const
{
    return QDir(q(cfg_.fleet_log_dir)).absoluteFilePath(q(agent) + QStringLiteral(".log"));
}

QString ChatSession::do_agent_control(const QJsonObject& args)
{
    const auto action = fleet::parse_action(args.value("action").toString().toStdString());
    const auto reply = [&](const QString& out)
    {
        log_.write("agent_control", QJsonObject{{"args", args}, {"result", out}});
        return out;
    };
    if (not action.has_value()) return reply(QStringLiteral("refused: action must be start, stop or restart"));
    if (fleet_op_.has_value())
        return reply(QStringLiteral("refused: still busy with %1 %2; wait until the chat tells it is done")
                         .arg(q(std::string(fleet::to_string(fleet_op_->action()))), q(fleet_op_->agent())));
    const auto agent = fleet::resolve_agent(args.value("agent").toString().toStdString(), fleet_policy_.agents);
    if (not agent.has_value()) return reply(QStringLiteral("refused: ") + q(agent.error()));
    const std::filesystem::path lock =
        not cfg_.fleet_lock.empty() ? std::filesystem::path(cfg_.fleet_lock)
            : std::filesystem::path(qEnvironmentVariable("XDG_RUNTIME_DIR", QStringLiteral("/tmp")).toStdString())
                  / "ai_fleet_tour.lock";
    const auto pids = fleet::find_pids(kProc, *agent);
    const std::int64_t now = fleet_now();
    const bool confirmed = fleet_confirmations_.confirmed(*action, *agent, user_turn_, now);
    const auto d = fleet::decide(*action, *agent, fleet_policy_, fleet::tour_running(lock, kProc), not pids.empty(),
                                 confirmed);
    if (d.kind == fleet::Decision::Kind::refuse) return reply(QStringLiteral("refused: ") + q(d.text));
    // ★ No OK for a launch that cannot happen (live 2026-10-24): checked before the person is asked to confirm a
    //   restart, and before its SIGTERM.
    if (*action != fleet::Action::stop)
        if (const auto missing = fleet::missing_executable(fleet::agent_executable(cfg_.fleet_dir, *agent)))
            return reply(QStringLiteral("NOT DONE: ") + q(*missing) +
                         QStringLiteral(". Tell the person it cannot be started from the chat; do not say it is starting."));
    if (d.kind == fleet::Decision::Kind::ask)
    {
        fleet_confirmations_.ask(*action, *agent, user_turn_, now);
        return reply(QStringLiteral("NOT DONE: ask the person \"%1\" and call agent_control again with the same "
                                    "arguments only if they answer yes.").arg(q(d.text)));
    }
    if (*action != fleet::Action::start)
        for (int pid : pids) ::kill(pid, SIGTERM);   // never SIGKILL (fleet_control.h)
    fleet_op_.emplace(*action, *agent, now);
    window_->append_system(QStringLiteral("agents — %1 %2").arg(q(std::string(fleet::to_string(*action))), q(*agent)));
    if (const auto end = tick_fleet(); end.has_value())   // over at once (a failed launch, a stop already done)
        return reply((end->ok ? QStringLiteral("OK: ") : QStringLiteral("NOT DONE: ")) + q(end->note));
    fleet_timer_.start(250);   // the outcome takes seconds: a quarter-second look is enough
    const QString what = *action == fleet::Action::stop    ? QStringLiteral("SIGTERM sent to %1")
                         : *action == fleet::Action::start ? QStringLiteral("starting %1")
                                                           : QStringLiteral("restarting %1");
    return reply(QStringLiteral("OK: ") + what.arg(q(*agent)) +
                 QStringLiteral(". You will be told how it ended (a robot event); until then do not say it is running or "
                                "has stopped."));
}

std::optional<ChatSession::FleetEnd> ChatSession::tick_fleet()
{
    if (not fleet_op_.has_value())
    {
        fleet_timer_.stop();
        return std::nullopt;
    }
    const std::int64_t now = fleet_now();
    const std::string agent = fleet_op_->agent();
    const bool alive = not fleet::find_pids(kProc, agent).empty();
    bool up = false;
    if (fleet_op_->launched_ms() > 0)
        for (const auto& a : system_.agents())
            up = up or (a.name == agent and fleet::started_since(a, now - fleet_op_->launched_ms()));
    auto step = fleet_op_->tick(now, alive, up);
    bool ok = step.done and (step.report.find(" is up") != std::string::npos or
                             step.report.find("has stopped") != std::string::npos);
    if (step.launch and not launch_agent(agent))
    {
        step.done = true;
        ok = false;
        step.report = agent + " could not be launched from " + fleet::agent_executable(cfg_.fleet_dir, agent).string() + ".";
    }
    if (not step.done) return std::nullopt;
    QString msg = QStringLiteral("agents — ") + q(step.report);
    const QString log = ok ? QString{} : agent_log(agent);
    if (not ok) msg += QStringLiteral(" Its log: ") + log;
    log_.write("agent_control_done", QJsonObject{{"agent", q(agent)}, {"ok", ok}, {"report", msg}});
    window_->append_system(msg);
    const fleet::Action action = fleet_op_->action();
    fleet_op_.reset();
    fleet_timer_.stop();
    return FleetEnd{.ok = ok, .note = fleet::control_done_note(action, agent, ok, step.report, log.toStdString())};
}

bool ChatSession::launch_agent(const std::string& agent)
{
    const auto path = fleet::agent_executable(cfg_.fleet_dir, agent);
    const QString exe = q(path.string());
    const QDir dir(q(path.parent_path().parent_path().string()));   // <agent>/, where etc/config.toml is
    if (not QFileInfo(exe).isExecutable()) return false;
    QDir().mkpath(q(cfg_.fleet_log_dir));
    QProcess p;
    p.setProgram(QStringLiteral("setsid"));   // its own session, as the tour scripts start it: it outlives the chat
    p.setArguments({exe, QStringLiteral("etc/config.toml")});
    p.setWorkingDirectory(dir.absolutePath());
    p.setStandardInputFile(QProcess::nullDevice());
    p.setStandardOutputFile(agent_log(agent), QIODevice::Append);
    p.setStandardErrorFile(agent_log(agent), QIODevice::Append);
    return p.startDetached();
}

}   // namespace rc::chat
