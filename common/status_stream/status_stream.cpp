// rc::StatusStream — see status_stream.h for the contract (threading, replay, commands).
#include "status_stream.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonValue>
#include <QLocalServer>
#include <QLocalSocket>
#include <QThread>
#include <QTimer>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <format>
#include <locale>
#include <ranges>

#include <sys/stat.h>
#include <unistd.h>

namespace rc
{
namespace
{
std::atomic<StatusStream*> g_instance{nullptr};
QtMessageHandler g_prev_handler = nullptr;
thread_local bool t_in_handler = false;

long long now_ms() { return QDateTime::currentMSecsSinceEpoch(); }

status::Level level_of(QtMsgType t)
{
    switch (t)
    {
        case QtDebugMsg:    return status::Level::Debug;
        case QtInfoMsg:     return status::Level::Info;
        case QtWarningMsg:  return status::Level::Warning;
        case QtCriticalMsg: return status::Level::Critical;
        case QtFatalMsg:    return status::Level::Fatal;
    }
    return status::Level::Info;
}

std::string readlink_fd(int fd)
{
    std::error_code ec;
    const auto p = std::filesystem::read_symlink(std::format("/proc/self/fd/{}", fd), ec);
    return ec ? std::string{} : p.string();
}

QString socket_dir()
{
    const QByteArray xdg = qgetenv("XDG_RUNTIME_DIR");
    const QString dir = xdg.isEmpty() ? QString("/tmp/rc_status_%1").arg(::getuid())
                                      : QString::fromLocal8Bit(xdg) + "/rc_status";
    QDir().mkpath(dir);
    ::chmod(dir.toLocal8Bit().constData(), 0700);
    return dir;
}

// The Qt half of the logging capture. Chained: the previous handler still prints whatever the
// routing sends to the terminal, so the console looks as it did for every line that stays there.
void qt_handler(QtMsgType type, const QMessageLogContext& ctx, const QString& msg)
{
    bool terminal = true;
    if (auto* s = g_instance.load(std::memory_order_acquire); s != nullptr and not t_in_handler)
    {
        t_in_handler = true;
        const auto line = msg.toStdString();
        terminal = s->route_and_post(level_of(type), line, "qt");
        t_in_handler = false;
    }
    if (terminal or type == QtFatalMsg)
    {
        if (g_prev_handler) g_prev_handler(type, ctx, msg);
        else
        {
            const auto b = msg.toLocal8Bit();
            std::fprintf(stderr, "%s\n", b.constData());
        }
    }
}
}  // namespace

StatusStream* StatusStream::instance() { return g_instance.load(std::memory_order_acquire); }

std::string_view StatusStream::tag_of(std::string_view line)
{
    std::size_t i = 0;
    while (i < line.size() and (line[i] == ' ' or line[i] == '"' or line[i] == '\t')) ++i;
    if (i >= line.size() or line[i] != '[') return {};
    const auto end = line.find(']', i + 1);
    if (end == std::string_view::npos or end - i - 1 > 40 or end == i + 1) return {};
    return line.substr(i + 1, end - i - 1);
}

StatusStream::StatusStream(Options opts, QObject* parent)
    : QObject(parent), opts_(std::move(opts))
{
    // ── the per-run events file ──
    const auto start = QDateTime::currentDateTime();
    if (not opts_.events_dir.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(opts_.events_dir, ec);
        events_path_ = opts_.events_dir + "/agent_events_"
                       + start.toString("yyyy-MM-dd_HH-mm-ss").toStdString() + ".jsonl";
        events_.open(events_path_, std::ios::out | std::ios::app);
        if (events_.is_open())
            events_.imbue(std::locale::classic());
        else
            events_path_.clear();
    }

    // ── the socket ──
    socket_path_ = socket_dir() + QString("/%1_%2.sock").arg(QString::fromStdString(opts_.agent)).arg(opts_.id);
    server_ = new QLocalServer(this);
    server_->setSocketOptions(QLocalServer::UserAccessOption);
    QLocalServer::removeServer(socket_path_);   // a stale socket from a crashed run
    if (not server_->listen(socket_path_))
        std::fprintf(stderr, "[status] cannot listen on %s: %s — the viewer cannot attach; events still go to %s\n",
                     socket_path_.toLocal8Bit().constData(), server_->errorString().toLocal8Bit().constData(),
                     events_path_.empty() ? "(no file)" : events_path_.c_str());
    connect(server_, &QLocalServer::newConnection, this, &StatusStream::on_new_connection);

    drain_timer_ = new QTimer(this);
    connect(drain_timer_, &QTimer::timeout, this, &StatusStream::drain);
    drain_timer_->start(opts_.drain_ms);

    g_instance.store(this, std::memory_order_release);
    status::event_sink().store(&StatusStream::sink_event, std::memory_order_release);
    status::log_sink().store(&StatusStream::sink_log, std::memory_order_release);
    status::flush_sink().store(&StatusStream::sink_flush, std::memory_order_release);
    g_prev_handler = qInstallMessageHandler(&qt_handler);

    QStringList args = QCoreApplication::instance() ? QCoreApplication::arguments() : QStringList{};
    status::Arr argv;
    for (const auto& a : args) argv.s(a.toStdString());
    post("hello", status::Obj{}
                      .s("agent", opts_.agent)
                      .i("id", opts_.id)
                      .i("pid", ::getpid())
                      .i("start_ms", start.toMSecsSinceEpoch())
                      .s("cwd", std::filesystem::current_path().string())
                      .s("stdout", readlink_fd(1))
                      .s("stderr", readlink_fd(2))
                      .s("events", events_path_.empty() ? std::string{} : std::filesystem::absolute(events_path_).string())
                      .s("socket", socket_path_.toStdString())
                      .raw("argv", argv.str())
                      .body());
}

StatusStream::~StatusStream()
{
    flush();
    qInstallMessageHandler(g_prev_handler);
    status::event_sink().store(nullptr, std::memory_order_release);
    status::log_sink().store(nullptr, std::memory_order_release);
    status::flush_sink().store(nullptr, std::memory_order_release);
    g_instance.store(nullptr, std::memory_order_release);
    if (server_) server_->close();
    QLocalServer::removeServer(socket_path_);
}

bool StatusStream::listening() const { return server_ and server_->isListening(); }

// ── posting (any thread) ────────────────────────────────────────────────────────────────────────────
std::string StatusStream::make_line(std::uint64_t seq, long long t_ms, std::string_view kind,
                                    std::string_view body) const
{
    std::string line = std::format(R"({{"v":1,"seq":{},"t":{},"kind":)", seq, t_ms);
    line += status::Obj::quoted(kind);
    if (not body.empty()) { line += ','; line += body; }
    line += "}\n";
    return line;
}

void StatusStream::post(std::string_view kind, std::string_view body)
{
    using Cls = Pending::Cls;
    Cls cls = Cls::Sticky;
    if (kind == "log") cls = Cls::Log;
    else if (kind == "state") cls = Cls::State;
    else if (kind == "reply" or kind == "cmd") cls = Cls::Transient;

    const long long t = now_ms();
    std::lock_guard lk(mtx_);
    Pending p;
    p.seq = next_seq_++;
    p.cls = cls;
    p.line = make_line(p.seq, t, kind, body);
    if (queue_.size() >= opts_.queue_cap)
    {
        queue_.pop_front();
        ++dropped_;
    }
    queue_.push_back(std::move(p));
}

bool StatusStream::route_and_post(status::Level level, std::string_view line, std::string_view origin)
{
    const auto tag = tag_of(line);
    const bool terminal = level >= status::Level::Critical
                          or not opts_.to_terminal
                          or opts_.to_terminal(level, tag);
    // Qt lines below the capture level are left exactly as they were: printed, not recorded.
    if (origin == "qt" and level < opts_.capture_min)
        return true;
    post("log", status::Obj{}
                    .s("level", status::to_string(level))
                    .s("tag", tag)
                    .s("msg", line)
                    .b("term", terminal)
                    .s("src", origin)
                    .body());
    return terminal;
}

void StatusStream::sink_event(std::string_view kind, std::string_view body)
{
    if (auto* s = instance()) s->post(kind, body);
}

bool StatusStream::sink_log(status::Level level, std::string_view line, bool to_stderr)
{
    auto* s = instance();
    if (s == nullptr) return true;
    return s->route_and_post(level, line, to_stderr ? "stderr" : "stdout");
}

void StatusStream::sink_flush()
{
    if (auto* s = instance(); s != nullptr and QThread::currentThread() == s->thread())
        s->flush();
}

// ── main thread ─────────────────────────────────────────────────────────────────────────────────────
void StatusStream::drain()
{
    std::deque<Pending> batch;
    std::uint64_t dropped = 0;
    {
        std::lock_guard lk(mtx_);
        batch.swap(queue_);
        dropped = dropped_;
    }
    if (dropped != dropped_reported_)
    {
        const auto n = dropped - dropped_reported_;
        dropped_reported_ = dropped;
        post("log", status::Obj{}
                        .s("level", "warning")
                        .s("tag", "status")
                        .s("msg", std::format("[status] {} event(s) dropped: the queue was full between drains", n))
                        .b("term", false)
                        .s("src", "status")
                        .body());
    }
    if (batch.empty()) return;

    for (auto& p : batch)
    {
        if (events_.is_open()) events_ << p.line;
        write_to_clients(p.line);
        switch (p.cls)
        {
            case Pending::Cls::Sticky:
                replay_sticky_.push_back(std::move(p));
                if (replay_sticky_.size() > opts_.replay_sticky_cap) replay_sticky_.pop_front();
                break;
            case Pending::Cls::Log:
                replay_logs_.push_back(std::move(p));
                if (replay_logs_.size() > opts_.replay_logs) replay_logs_.pop_front();
                break;
            case Pending::Cls::State:
                replay_state_ = std::move(p);
                have_state_ = true;
                break;
            case Pending::Cls::Transient:
                break;
        }
    }
    if (events_.is_open()) events_.flush();
}

void StatusStream::write_to_clients(const std::string& line)
{
    for (auto* c : clients_)
    {
        if (c->state() != QLocalSocket::ConnectedState) continue;
        c->write(line.data(), static_cast<qint64>(line.size()));
        if (c->bytesToWrite() > opts_.client_cap_bytes)
        {
            // A viewer that stopped reading must not grow this process: drop it. It can reconnect
            // and the replay brings it back up to date.
            c->abort();
        }
    }
}

void StatusStream::flush()
{
    if (QThread::currentThread() != thread()) return;
    drain();
    if (events_.is_open()) events_.flush();
    for (auto* c : clients_)
        if (c->state() == QLocalSocket::ConnectedState)
        {
            c->flush();
            c->waitForBytesWritten(100);
        }
}

void StatusStream::stopping(std::string_view reason)
{
    post("lifecycle", status::Obj{}.s("state", "stopping").s("reason", reason).body());
    flush();
}

void StatusStream::on_new_connection()
{
    while (auto* s = server_->nextPendingConnection())
    {
        drain();          // anything already queued goes into the replay first, in order
        clients_.push_back(s);
        n_clients_ = static_cast<int>(clients_.size());
        connect(s, &QLocalSocket::readyRead, this, [this, s] { on_ready_read(s); });
        connect(s, &QLocalSocket::disconnected, this, [this, s]
        {
            std::erase(clients_, s);
            n_clients_ = static_cast<int>(clients_.size());
            s->deleteLater();
        });
        send_replay(s);
    }
}

void StatusStream::send_replay(QLocalSocket* s)
{
    std::vector<const Pending*> all;
    all.reserve(replay_sticky_.size() + replay_logs_.size() + 1);
    for (const auto& p : replay_sticky_) all.push_back(&p);
    for (const auto& p : replay_logs_) all.push_back(&p);
    if (have_state_) all.push_back(&replay_state_);
    std::ranges::sort(all, {}, &Pending::seq);
    std::string blob;
    for (const auto* p : all) blob += p->line;
    // Marks the end of the replay, so a viewer knows what follows is live.
    blob += make_line(0, now_ms(), "replay_end", {});
    s->write(blob.data(), static_cast<qint64>(blob.size()));
}

void StatusStream::register_command(const std::string& name, const std::string& description,
                                    const std::string& confirm, CommandHandler handler)
{
    commands_[name] = Command{description, confirm, std::move(handler)};
    post_commands();
}

void StatusStream::post_commands()
{
    status::Arr list;
    for (const auto& [name, c] : commands_)
        list.raw(status::Obj{}.s("name", name).s("description", c.description).s("confirm", c.confirm).str());
    post("commands", status::Obj{}.raw("list", list.str()).body());
}

void StatusStream::reply(QLocalSocket* s, long long id, const std::string& cmd, const Reply& r)
{
    status::Obj body;
    body.i("id", id).s("cmd", cmd).b("ok", r.ok).s("msg", r.msg);
    std::string b = body.body();
    if (not r.extra.empty()) { b += ','; b += r.extra; }
    std::uint64_t seq;
    {
        std::lock_guard lk(mtx_);
        seq = next_seq_++;
    }
    const auto line = make_line(seq, now_ms(), "reply", b);
    if (s->state() == QLocalSocket::ConnectedState)
        s->write(line.data(), static_cast<qint64>(line.size()));
    // The record: who ran what and what came of it, for every viewer and the file.
    post("cmd", status::Obj{}.s("cmd", cmd).b("ok", r.ok).s("msg", r.msg).body());
}

void StatusStream::on_ready_read(QLocalSocket* s)
{
    while (s->canReadLine())
    {
        const QByteArray raw = s->readLine(64 * 1024).trimmed();
        if (raw.isEmpty()) continue;
        QJsonParseError err{};
        const auto doc = QJsonDocument::fromJson(raw, &err);
        if (err.error != QJsonParseError::NoError or not doc.isObject())
        {
            reply(s, -1, "", Reply{false, "not a JSON object: " + err.errorString().toStdString(), {}});
            continue;
        }
        const auto obj = doc.object();
        const auto id = static_cast<long long>(obj.value("id").toDouble(-1));
        const auto cmd = obj.value("cmd").toString().toStdString();
        if (cmd == "ping")
        {
            reply(s, id, cmd, Reply{true, "pong", {}});
            continue;
        }
        const auto it = commands_.find(cmd);
        if (it == commands_.end())
        {
            reply(s, id, cmd, Reply{false, "'" + cmd + "' is not a registered command", {}});
            continue;
        }
        Reply r;
        try { r = it->second.handler(obj.value("args").toObject()); }
        catch (const std::exception& e) { r = Reply{false, std::string("handler threw: ") + e.what(), {}}; }
        reply(s, id, cmd, r);
    }
    // A client must not grow an unbounded read buffer either.
    if (s->bytesAvailable() > 64 * 1024) s->abort();
}

}  // namespace rc
