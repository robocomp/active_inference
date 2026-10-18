#pragma once
/*
 * canvas_runner.h — runs one canvas script (canvas/bootstrap.py → the script) in a confined child process and
 * returns the picture it drew (out.png) or why it drew none.
 *
 *   [systemd-run --user --scope -p MemoryMax=… -p CPUQuota=100%]  [unshare -rn]  python3 -I bootstrap.py script args…
 *    cwd = run_dir (snapshot.json + camera frames written by the session just before)
 *
 * Confinement, probed once at construction (what this machine allows is logged):
 *   · memory/CPU cap through a transient systemd user scope (same mechanism as `cbuild`);
 *   · no network: a user+net namespace when unprivileged user namespaces are allowed (AppArmor may forbid them),
 *     and ALWAYS bootstrap.py's socket-level block — the guard is against an accidental script, not an adversary;
 *   · a wall-clock timeout, after which the child is killed;
 *   · -I: no PYTHON* env, no user site, the script's directory is not importable.
 *
 * Two ways to run:
 *   run_blocking()  one fresh process, for a tool call or the first picture of a drawing (the result goes back to
 *                   the model in the same round; the GUI waits at most timeout_ms; ~350 ms for stock/layout.py).
 *   redraw()        a LIVE drawing's next picture, from a persistent worker (bootstrap.py --serve) that keeps the
 *                   interpreter and matplotlib loaded: ~45 ms for the same layout. One worker per drawing: asking
 *                   for a different script/args restarts it, so nothing a script did outlives its drawing.
 *                   Asynchronous; a tick that finds a redraw in flight is skipped (the drawing sets its own pace).
 * No Q_OBJECT; GUI thread only.
 */

#include <functional>
#include <memory>

#include <QImage>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

namespace rc::chat
{

struct CanvasRunConfig
{
    QString canvas_dir = QStringLiteral("canvas");   ///< bootstrap.py, rcanvas.py, stock/, library/
    QString run_dir;                                 ///< "" ⇒ $XDG_RUNTIME_DIR (or /tmp)/chat_agent_canvas
    QString python = QStringLiteral("python3");
    int timeout_ms = 10000;
    QString memory_max = QStringLiteral("1G");
};

struct CanvasJob
{
    QString title;
    QString script_path;   ///< absolute path of the script to run
    QStringList args;
    QString source;        ///< the script's text (shown under "</>" and saved to the library)
    bool live = false;
    bool wants_images = false;   ///< the script mentions image(): the session writes the camera frames
    bool wants_graph = false;    ///< the script mentions graph(): the session writes graph.json
};

struct CanvasRunResult
{
    bool ok = false;
    QImage image;
    QString error;   ///< the reason or the stderr tail when !ok
    qint64 ms = 0;
};

class CanvasRunner
{
public:
    explicit CanvasRunner(CanvasRunConfig cfg);
    ~CanvasRunner();

    [[nodiscard]] const QString& run_dir() const { return run_dir_; }
    [[nodiscard]] const QString& canvas_dir() const { return canvas_dir_; }
    [[nodiscard]] QString confinement() const;   ///< "scope+netns", "scope, socket-block only", …
    [[nodiscard]] bool running() const;

    CanvasRunResult run_blocking(const CanvasJob& job);
    void redraw(const CanvasJob& job, std::function<void(CanvasRunResult)> done);
    void kill();          ///< the one-shot run and the live worker

private:
    [[nodiscard]] QStringList command(const QStringList& bootstrap_args) const;   ///< program first
    void launch(const CanvasJob& job);
    CanvasRunResult collect(int exit_code, bool timed_out);
    CanvasRunResult picture(qint64 started_ms) const;   ///< out.png, or why there is none
    void stop_worker();
    void worker_reply(const QByteArray& line);
    void worker_fail(const QString& why);

    CanvasRunConfig cfg_;
    QString canvas_dir_, run_dir_;
    bool use_scope_ = false, use_netns_ = false;
    std::unique_ptr<QProcess> proc_;
    QObject ctx_;
    qint64 started_ms_ = 0;
    int generation_ = 0;
    // the live worker
    std::unique_ptr<QProcess> worker_;
    QString worker_key_;                                ///< script + args it serves
    QByteArray worker_out_, worker_err_;
    std::function<void(CanvasRunResult)> worker_done_;  ///< set while a redraw is in flight
    qint64 worker_started_ms_ = 0;
    int worker_gen_ = 0;
    int worker_req_ = 0;   ///< which redraw a timeout belongs to
};

}   // namespace rc::chat
