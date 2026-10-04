/*
 * common/status_stream/status_stream.h — rc::StatusStream: an agent's live status, as newline-delimited
 * JSON on a local socket, for a terminal viewer (room_concept/tools/room_tui.py) to attach to.
 *
 * GENERIC. Nothing here knows which agent it serves; the agent posts events and registers commands.
 *
 * TRANSPORT. A QLocalServer at  $XDG_RUNTIME_DIR/rc_status/<agent>_<id>.sock  (falls back to
 * /tmp/rc_status_<uid>/ when XDG_RUNTIME_DIR is unset). One JSON object per line:
 *     {"v":1,"seq":N,"t":<wall ms>,"kind":"<kind>", ...fields}
 * Every event is ALSO appended to  <events_dir>/agent_events_<YYYY-mm-dd_HH-MM-SS>.jsonl , so nothing
 * is lost when no viewer is attached.
 *
 * THREADING — the rules this class is built around:
 *   · It is OWNED BY THE MAIN THREAD. Accept, read, write and the file all happen there, on a drain
 *     timer, and every socket write is non-blocking. A client whose unsent buffer passes ~1 MB is
 *     dropped rather than allowed to grow the agent's memory or stall it.
 *   · ANY thread may post (rc::status::event / rc::status::log, or post() directly). Posting holds the
 *     mutex only for the push into a bounded queue; when the queue is full the OLDEST event is dropped
 *     and the drop is counted and reported.
 *   · It NEVER touches DSR.
 *
 * REPLAY. A late viewer must still see how the agent started, so the stream keeps: hello, commands,
 * config, and every phase/loaded/peer/sm/stall/fatal/lifecycle event; the last 200 log events; and
 * the latest state snapshot. A new client receives all of it, in seq order, before the live stream.
 *
 * LOGGING. A chained qInstallMessageHandler turns Qt messages at or above Options::capture_min into
 * `log` events. Whether the line ALSO reaches the terminal is the agent's routing decision
 * (Options::to_terminal); with no routing function every line reaches the terminal exactly as before.
 *
 * COMMANDS. The same socket accepts {"cmd":"<name>","id":N,"args":{...}} lines. Only names the agent
 * REGISTERED are accepted (a whitelist); anything else is refused with a reply. Handlers run on the
 * main thread from the Qt event loop.
 *
 * FATAL PATHS call flush() before std::exit: it drains synchronously and pushes the bytes out.
 */
#pragma once

#include "status_sink.h"

#include <QJsonObject>
#include <QObject>
#include <QString>

#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class QLocalServer;
class QLocalSocket;
class QTimer;

namespace rc
{

class StatusStream : public QObject
{
    Q_OBJECT
public:
    struct Options
    {
        std::string agent = "agent";
        int id = 0;
        std::string events_dir = "tmp";          ///< where agent_events_<start>.jsonl goes; "" = no file
        std::size_t queue_cap = 8192;            ///< pending events between drains (drop-oldest)
        std::size_t replay_logs = 200;           ///< log events kept for a late viewer
        std::size_t replay_sticky_cap = 4000;    ///< sticky events kept (phases, loaded, sm, ...)
        long long client_cap_bytes = 1 << 20;    ///< unsent bytes before a slow client is dropped
        int drain_ms = 50;
        /// Qt messages at or above this level become `log` events.
        status::Level capture_min = status::Level::Warning;
        /// Routing: TRUE ⇒ the line also reaches the terminal. Called on the posting thread, so it
        /// must be thread-safe (a pure function of its arguments is). Empty ⇒ everything reaches it.
        std::function<bool(status::Level, std::string_view tag, std::string_view line)> to_terminal;
    };

    /// Reply: ok + a human message, plus optional extra fields (a JSON object body, no braces).
    struct Reply
    {
        bool ok = true;
        std::string msg;
        std::string extra;
    };
    using CommandHandler = std::function<Reply(const QJsonObject& args)>;

    explicit StatusStream(Options opts, QObject* parent = nullptr);
    ~StatusStream() override;

    StatusStream(const StatusStream&) = delete;
    StatusStream& operator=(const StatusStream&) = delete;

    /// The live instance, or nullptr. Set by the constructor, cleared by the destructor.
    static StatusStream* instance();

    /// MAIN THREAD. `confirm` is the question a viewer should ask before sending it ("" = no question).
    void register_command(const std::string& name, const std::string& description,
                          const std::string& confirm, CommandHandler handler);

    /// ANY THREAD. `body` is a JSON object body WITHOUT braces (rc::status::Obj::body()).
    void post(std::string_view kind, std::string_view body);

    /// MAIN THREAD. Drain synchronously and push the bytes out to the file and every client.
    void flush();

    /// MAIN THREAD. Announce the stop (a sticky `lifecycle` event) and flush.
    void stopping(std::string_view reason);

    [[nodiscard]] bool listening() const;
    [[nodiscard]] bool has_clients() const { return n_clients_ > 0; }
    [[nodiscard]] QString socket_path() const { return socket_path_; }
    [[nodiscard]] const std::string& events_path() const { return events_path_; }

    /// The `[tag]` a line opens with ("" when it has none). Leading spaces/quotes are skipped.
    static std::string_view tag_of(std::string_view line);

    /// ANY THREAD. Record one log line as a `log` event and return whether it ALSO goes to the
    /// terminal. `origin` is "qt", "stdout" or "stderr". Used by the sinks and the Qt handler.
    bool route_and_post(status::Level level, std::string_view line, std::string_view origin);

private:
    struct Pending
    {
        std::uint64_t seq = 0;
        std::string line;            ///< the full JSON line, newline included
        enum class Cls : std::uint8_t { Sticky, Log, State, Transient } cls = Cls::Transient;
    };

    void on_new_connection();
    void on_ready_read(QLocalSocket* s);
    void drain();
    void write_to_clients(const std::string& line);
    void send_replay(QLocalSocket* s);
    void reply(QLocalSocket* s, long long id, const std::string& cmd, const Reply& r);
    void post_commands();
    std::string make_line(std::uint64_t seq, long long t_ms, std::string_view kind, std::string_view body) const;

    // sink trampolines (installed into status_sink.h's atomics)
    static void sink_event(std::string_view kind, std::string_view body);
    static bool sink_log(status::Level level, std::string_view line, bool to_stderr);
    static void sink_flush();

    Options opts_;
    QLocalServer* server_ = nullptr;
    QTimer* drain_timer_ = nullptr;
    QString socket_path_;
    std::string events_path_;
    std::ofstream events_;
    std::vector<QLocalSocket*> clients_;
    int n_clients_ = 0;

    // ── shared with posting threads ──
    std::mutex mtx_;
    std::deque<Pending> queue_;
    std::uint64_t next_seq_ = 1;
    std::uint64_t dropped_ = 0;

    // ── main thread only ──
    std::deque<Pending> replay_sticky_;
    std::deque<Pending> replay_logs_;
    Pending replay_state_;
    bool have_state_ = false;
    std::uint64_t dropped_reported_ = 0;

    struct Command
    {
        std::string description, confirm;
        CommandHandler handler;
    };
    std::map<std::string, Command> commands_;
};

}  // namespace rc
