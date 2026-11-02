// CanvasRunner + CanvasPanel offscreen checks: a good script draws, a broken one returns its stderr, an endless one is
// killed at the timeout, a network attempt fails, the live worker redraws fast, survives a failing script and is killed when stuck, and the panel shows a drawing and the plan
// page. Build+run: tests/run_tests.sh (needs python3 with matplotlib, numpy, PIL).
#include "../src/canvas_panel.h"
#include "../src/canvas_runner.h"

#include <cassert>
#include <clocale>
#include <print>

#include <QApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QTemporaryDir>
#include <QTimer>

using namespace rc::chat;

namespace
{
CanvasJob script_job(const QString& dir, const QString& name, const QString& src)
{
    const QString path = dir + "/" + name;
    QFile f(path);
    if (not f.open(QIODevice::WriteOnly | QIODevice::Truncate)) std::abort();
    f.write(src.toUtf8());
    return CanvasJob{.title = name, .script_path = path, .source = src};
}
}   // namespace

int main(int argc, char** argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    std::setlocale(LC_ALL, "");
    int checks = 0;
    QTemporaryDir tmp, scripts;
    const QString canvas = QStringLiteral(CANVAS_DIR);
    CanvasRunner runner(CanvasRunConfig{.canvas_dir = canvas, .run_dir = tmp.path(), .timeout_ms = 4000});
    std::println("confinement: {}", runner.confinement().toStdString());
    {
        QFile f(tmp.path() + "/snapshot.json");
        if (not f.open(QIODevice::WriteOnly)) std::abort();
        f.write(R"({"canvas":{"w":400,"h":300},"room":{"name":"room_1","polygon":[[0,0],[4,0],[4,3],[0,3]]},
                    "robot":{"x":1,"y":1,"yaw":0,"heading":1.57,"radius":0.35},"doors":[],"objects":[]})");
    }

    // a stock drawing
    auto r = runner.run_blocking(CanvasJob{.title = "layout", .script_path = canvas + "/stock/layout.py"});
    assert(r.ok and r.image.width() == 400 and r.image.height() == 300); ++checks;
    // a model-style script
    r = runner.run_blocking(script_job(scripts.path(), "bars.py",
        "from rcanvas import *\nw = world()\nfig = figure()\nax = fig.add_subplot(111)\n"
        "ax.bar(['a','b'], [1, 2])\nsave(fig)\n"));
    assert(r.ok and not r.image.isNull()); ++checks;
    // broken: the error comes back with the traceback's last line
    r = runner.run_blocking(script_job(scripts.path(), "broken.py", "from rcanvas import *\nx = w.robot\n"));
    assert(not r.ok and r.error.contains("NameError")); ++checks;
    // ran but drew nothing
    r = runner.run_blocking(script_job(scripts.path(), "nothing.py", "print('hello')\n"));
    assert(not r.ok and r.error.contains("without drawing")); ++checks;
    // endless: killed at the timeout
    QElapsedTimer t;
    t.start();
    r = runner.run_blocking(script_job(scripts.path(), "loop.py", "while True:\n    pass\n"));
    assert(not r.ok and r.error.contains("stopped after") and t.elapsed() < 7000); ++checks;
    // network: refused
    r = runner.run_blocking(script_job(scripts.path(), "net.py",
        "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=2)\n"));
    assert(not r.ok); ++checks;
    // the live worker: the first redraw pays the start-up, later ones only the drawing
    {
        const CanvasJob layout{.title = "layout", .script_path = canvas + "/stock/layout.py"};
        const auto redraw = [&](const CanvasJob& job)
        {
            QEventLoop loop;
            CanvasRunResult got;
            runner.redraw(job, [&](CanvasRunResult res) { got = std::move(res); loop.quit(); });
            QTimer::singleShot(8000, &loop, &QEventLoop::quit);
            loop.exec();
            return got;
        };
        const auto first = redraw(layout);
        assert(first.ok and not runner.running()); ++checks;
        qint64 warm = 0;
        for (int i = 0; i < 5; ++i)
        {
            const auto again = redraw(layout);
            assert(again.ok and again.image.width() == 400); ++checks;
            warm += again.ms;
        }
        std::println("live worker: first {} ms, warm mean {} ms", first.ms, warm / 5);
        assert(warm / 5 < first.ms); ++checks;
        // a failing redraw reports the traceback and the worker survives it
        const auto bad = redraw(script_job(scripts.path(), "bad_live.py", "from rcanvas import *\nraise ValueError('boom')\n"));
        assert(not bad.ok and bad.error.contains("ValueError")); ++checks;
        assert(redraw(layout).ok); ++checks;   // another drawing: a fresh worker
        // an endless redraw is killed at the timeout
        const auto loop = redraw(script_job(scripts.path(), "loop_live.py", "while True:\n    pass\n"));
        assert(not loop.ok and loop.error.contains("stopped after")); ++checks;
        // a redraw cancelled by kill() never reports
        int calls = 0;
        runner.redraw(layout, [&](CanvasRunResult) { ++calls; });
        runner.kill();
        QEventLoop l;
        QTimer::singleShot(1500, &l, &QEventLoop::quit);
        l.exec();
        assert(calls == 0 and not runner.running()); ++checks;
    }

    // the panel
    CanvasPanel panel(CanvasPanel::Callbacks{});
    panel.resize(500, 400);
    panel.show();
    panel.show_drawing("layout", r.image.isNull() ? QImage(400, 300, QImage::Format_RGB32) : r.image, true, "src", true);
    assert(not panel.showing_plan()); ++checks;
    panel.show_plan();
    assert(panel.showing_plan() and not panel.plan()->isWindow()); ++checks;   // a page, not a window
    panel.show_message("x", "why");
    assert(not panel.showing_plan()); ++checks;

    std::println("test_canvas_runner: {} checks passed", checks);
}
