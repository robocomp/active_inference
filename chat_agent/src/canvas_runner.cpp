#include "canvas_runner.h"

#include <print>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTimer>

namespace rc::chat
{

namespace
{
bool exits_zero(const QString& program, const QStringList& args)
{
    QProcess p;
    p.start(program, args);
    if (not p.waitForFinished(3000)) { p.kill(); p.waitForFinished(500); return false; }
    return p.exitStatus() == QProcess::NormalExit and p.exitCode() == 0;
}

QString tail_lines(const QString& text, int n)
{
    QStringList lines = text.split('\n', Qt::SkipEmptyParts);
    if (lines.size() > n) lines = lines.mid(lines.size() - n);
    return lines.join('\n');
}
}   // namespace

CanvasRunner::CanvasRunner(CanvasRunConfig cfg) : cfg_(std::move(cfg))
{
    canvas_dir_ = QFileInfo(cfg_.canvas_dir).absoluteFilePath();
    run_dir_ = cfg_.run_dir;
    if (run_dir_.isEmpty())
    {
        const QString base = qEnvironmentVariable("XDG_RUNTIME_DIR", QDir::tempPath());
        run_dir_ = base + QStringLiteral("/chat_agent_canvas_%1").arg(QCoreApplication::applicationPid());
    }
    QDir().mkpath(run_dir_);
    use_scope_ = exits_zero(QStringLiteral("systemd-run"), {"--user", "--scope", "--quiet", "true"});
    use_netns_ = exits_zero(QStringLiteral("unshare"), {"-rn", "true"});
    std::println("[chat_agent] canvas: scripts in {}, runs in {}, confinement: {}", canvas_dir_.toStdString(),
                 run_dir_.toStdString(), confinement().toStdString());
    if (not QFileInfo::exists(canvas_dir_ + QStringLiteral("/bootstrap.py")))
        std::println("[chat_agent] WARNING: canvas: {}/bootstrap.py not found — drawing will fail", canvas_dir_.toStdString());
}

CanvasRunner::~CanvasRunner() { kill(); }

QString CanvasRunner::confinement() const
{
    QStringList parts;
    parts << (use_scope_ ? QStringLiteral("systemd scope MemoryMax=%1").arg(cfg_.memory_max)
                         : QStringLiteral("NO memory cap (systemd-run --user unavailable)"));
    parts << (use_netns_ ? QStringLiteral("network namespace") : QStringLiteral("socket-level network block only"));
    parts << QStringLiteral("timeout %1 s").arg(cfg_.timeout_ms / 1000.0, 0, 'f', 1);
    return parts.join(", ");
}

bool CanvasRunner::running() const
{
    return (proc_ and proc_->state() != QProcess::NotRunning) or static_cast<bool>(worker_done_);
}

QStringList CanvasRunner::command(const QStringList& bootstrap_args) const
{
    QStringList argv;
    if (use_scope_)
        argv << "systemd-run" << "--user" << "--scope" << "--quiet" << "-p" << ("MemoryMax=" + cfg_.memory_max)
             << "-p" << "MemorySwapMax=0" << "-p" << "CPUQuota=100%" << "--";
    if (use_netns_) argv << "unshare" << "-rn";
    argv << cfg_.python << "-I" << (canvas_dir_ + QStringLiteral("/bootstrap.py")) << bootstrap_args;
    return argv;
}

void CanvasRunner::kill()
{
    stop_worker();
    ++generation_;   // a pending async completion of the killed run is ignored
    if (not proc_) return;
    if (proc_->state() != QProcess::NotRunning)
    {
        proc_->kill();
        proc_->waitForFinished(1000);
    }
    proc_.reset();
}

void CanvasRunner::launch(const CanvasJob& job)
{
    kill();
    QFile::remove(run_dir_ + QStringLiteral("/out.png"));
    proc_ = std::make_unique<QProcess>();
    proc_->setWorkingDirectory(run_dir_);
    QStringList argv = command(QStringList{job.script_path} + job.args);
    const QString program = argv.takeFirst();
    started_ms_ = QDateTime::currentMSecsSinceEpoch();
    proc_->start(program, argv);
}

CanvasRunResult CanvasRunner::collect(int exit_code, bool timed_out)
{
    CanvasRunResult r;
    r.ms = QDateTime::currentMSecsSinceEpoch() - started_ms_;
    const QString err = proc_ ? QString::fromUtf8(proc_->readAllStandardError()) : QString();
    if (timed_out)
        r.error = QStringLiteral("the script was stopped after %1 s (too slow, or an endless loop)").arg(cfg_.timeout_ms / 1000);
    else if (exit_code != 0)
        r.error = QStringLiteral("the script failed (exit %1):\n%2").arg(exit_code).arg(tail_lines(err, 20));
    else return picture(started_ms_);
    return r;
}

CanvasRunResult CanvasRunner::picture(qint64 started_ms) const
{
    CanvasRunResult r{.ms = QDateTime::currentMSecsSinceEpoch() - started_ms};
    if (QImage img(run_dir_ + QStringLiteral("/out.png")); img.isNull())
        r.error = QStringLiteral("the script ended without drawing: it must finish with save(fig) or save_image(img)");
    else
    {
        r.ok = true;
        r.image = std::move(img);
    }
    return r;
}

CanvasRunResult CanvasRunner::run_blocking(const CanvasJob& job)
{
    launch(job);
    if (not proc_->waitForStarted(3000))
    {
        CanvasRunResult r{.error = QStringLiteral("could not start %1: %2").arg(cfg_.python, proc_->errorString())};
        kill();
        return r;
    }
    const bool finished = proc_->waitForFinished(cfg_.timeout_ms);
    if (not finished) { proc_->kill(); proc_->waitForFinished(1000); }
    auto r = collect(proc_->exitStatus() == QProcess::NormalExit ? proc_->exitCode() : -1, not finished);
    proc_.reset();
    return r;
}

void CanvasRunner::stop_worker()
{
    ++worker_gen_;   // queued signals of the old worker are ignored
    worker_done_ = nullptr;
    worker_key_.clear();
    if (not worker_) return;
    if (worker_->state() != QProcess::NotRunning)
    {
        worker_->kill();
        worker_->waitForFinished(1000);
    }
    worker_.release()->deleteLater();   // may be inside one of its own signals
}

void CanvasRunner::worker_fail(const QString& why)
{
    auto done = std::move(worker_done_);
    const qint64 t0 = worker_started_ms_;
    stop_worker();   // a worker that timed out or died is not trusted with the next picture
    if (done) done(CanvasRunResult{.error = why, .ms = QDateTime::currentMSecsSinceEpoch() - t0});
}

void CanvasRunner::worker_reply(const QByteArray& line)
{
    if (not worker_done_) return;
    auto done = std::move(worker_done_);
    worker_done_ = nullptr;
    if (line == "ok") { done(picture(worker_started_ms_)); return; }
    // "err <traceback>", newlines escaped by bootstrap.py
    QString tb = QString::fromUtf8(line.mid(4));
    tb.replace(QStringLiteral("\\n"), QStringLiteral("\n")).replace(QStringLiteral("\\\\"), QStringLiteral("\\"));
    done(CanvasRunResult{.error = QStringLiteral("the script failed:\n%1").arg(tail_lines(tb, 20)),
                         .ms = QDateTime::currentMSecsSinceEpoch() - worker_started_ms_});
}

void CanvasRunner::redraw(const CanvasJob& job, std::function<void(CanvasRunResult)> done)
{
    if (worker_done_) return;   // one in flight: the caller skips this tick
    const QString key = job.script_path + '\n' + job.args.join('\n');
    if (not worker_ or worker_->state() == QProcess::NotRunning or key != worker_key_)
    {
        stop_worker();
        worker_ = std::make_unique<QProcess>();
        worker_->setWorkingDirectory(run_dir_);
        QStringList argv = command(QStringList{"--serve", job.script_path} + job.args);
        const QString program = argv.takeFirst();
        worker_key_ = key;
        worker_out_.clear();
        worker_err_.clear();
        const int gen = worker_gen_;
        QObject::connect(worker_.get(), &QProcess::readyReadStandardOutput, &ctx_, [this, gen]()
        {
            if (gen != worker_gen_) return;
            worker_out_ += worker_->readAllStandardOutput();
            for (qsizetype nl; gen == worker_gen_ and (nl = worker_out_.indexOf('\n')) >= 0;)
            {
                const QByteArray line = worker_out_.left(nl);
                worker_out_.remove(0, nl + 1);
                worker_reply(line);
            }
        });
        QObject::connect(worker_.get(), &QProcess::readyReadStandardError, &ctx_, [this, gen]()
        {
            if (gen != worker_gen_) return;
            worker_err_ += worker_->readAllStandardError();
            if (worker_err_.size() > 16384) worker_err_ = worker_err_.right(8192);
        });
        QObject::connect(worker_.get(), &QProcess::finished, &ctx_, [this, gen](int code, QProcess::ExitStatus)
        {
            if (gen != worker_gen_) return;
            worker_fail(QStringLiteral("the drawing worker ended (exit %1):\n%2")
                            .arg(code).arg(tail_lines(QString::fromUtf8(worker_err_), 20)));
        });
        worker_->start(program, argv);
    }
    worker_done_ = std::move(done);
    worker_started_ms_ = QDateTime::currentMSecsSinceEpoch();
    QFile::remove(run_dir_ + QStringLiteral("/out.png"));
    worker_->write("\n");
    const int gen = worker_gen_, req = ++worker_req_;
    QTimer::singleShot(cfg_.timeout_ms, &ctx_, [this, gen, req]()
    {
        if (gen != worker_gen_ or not worker_done_ or req != worker_req_) return;
        worker_fail(QStringLiteral("the script was stopped after %1 s (too slow, or an endless loop)")
                        .arg(cfg_.timeout_ms / 1000));
    });
}

}   // namespace rc::chat
