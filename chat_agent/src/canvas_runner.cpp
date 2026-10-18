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

bool CanvasRunner::running() const { return proc_ and proc_->state() != QProcess::NotRunning; }

void CanvasRunner::kill()
{
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
    QStringList argv;
    if (use_scope_)
        argv << "systemd-run" << "--user" << "--scope" << "--quiet" << "-p" << ("MemoryMax=" + cfg_.memory_max)
             << "-p" << "MemorySwapMax=0" << "-p" << "CPUQuota=100%" << "--";
    if (use_netns_) argv << "unshare" << "-rn";
    argv << cfg_.python << "-I" << (canvas_dir_ + QStringLiteral("/bootstrap.py")) << job.script_path << job.args;
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
    else if (QImage img(run_dir_ + QStringLiteral("/out.png")); img.isNull())
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

void CanvasRunner::start(const CanvasJob& job, std::function<void(CanvasRunResult)> done)
{
    launch(job);
    const int gen = generation_;
    QObject::connect(proc_.get(), &QProcess::finished, &ctx_, [this, gen, done](int code, QProcess::ExitStatus st)
    {
        if (gen != generation_) return;
        auto r = collect(st == QProcess::NormalExit ? code : -1, false);
        proc_.release()->deleteLater();   // inside its own signal: do not delete it here
        done(std::move(r));
    });
    QObject::connect(proc_.get(), &QProcess::errorOccurred, &ctx_, [this, gen, done](QProcess::ProcessError e)
    {
        if (gen != generation_ or e != QProcess::FailedToStart) return;
        ++generation_;
        proc_.release()->deleteLater();
        done(CanvasRunResult{.error = QStringLiteral("could not start %1").arg(cfg_.python)});
    });
    QTimer::singleShot(cfg_.timeout_ms, &ctx_, [this, gen, done]()
    {
        if (gen != generation_ or not running()) return;
        ++generation_;
        proc_->kill();
        proc_->waitForFinished(1000);
        auto r = collect(-1, true);
        proc_.reset();
        done(std::move(r));
    });
}

}   // namespace rc::chat
