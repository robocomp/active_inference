// The chat window comes back at the size the person left it, with or without the canvas open (chat_window.cpp).
// Never mapped on screen (WA_DontShowOnScreen; the offscreen platform crashes flushing the canvas's GL widget), settings in a private XDG_CONFIG_HOME. Build+run: tests/run_tests.sh
#include "../src/chat_window.h"

#include <cassert>
#include <cstdlib>
#include <print>

#include <QApplication>
#include <QDir>
#include <QTemporaryDir>

namespace
{
int closed_width(QWidget& w)
{
    w.close();
    QApplication::processEvents();
    return w.width();
}
}   // namespace

int main(int argc, char** argv)
{
    QTemporaryDir cfg;
    setenv("XDG_CONFIG_HOME", cfg.path().toUtf8().constData(), 1);
    QApplication app(argc, argv);
    int checks = 0;

    // ── closed with the canvas open: it reopens at that size, the canvas inside it, not added once more ─────────
    int left_w = 0, left_h = 0;
    {
        rc::chat::ChatWindow w({});
        w.setAttribute(Qt::WA_DontShowOnScreen);
        w.show();
        w.show_canvas();
        w.resize(1500, 700);   // the person's size, canvas included
        QApplication::processEvents();
        left_w = w.width();
        left_h = w.height();
        // closing does not resize the window: a resize there reaches the window manager after the geometry is saved
        assert(closed_width(w) == left_w); ++checks;
    }
    {
        rc::chat::ChatWindow w({});
        w.setAttribute(Qt::WA_DontShowOnScreen);
        w.show();
        w.show_canvas();       // the session opens it at every start
        QApplication::processEvents();
        assert(w.width() == left_w and w.height() == left_h); ++checks;
        assert(w.canvas_visible()); ++checks;
        w.hide_canvas();       // closing the canvas still gives its width back
        QApplication::processEvents();
        assert(w.width() < left_w); ++checks;
        w.resize(800, 600);    // the person's size, no canvas
        QApplication::processEvents();
        left_w = w.width();
        left_h = w.height();
        assert(closed_width(w) == left_w); ++checks;
    }
    // ── closed without the canvas: it reopens at that size, and opening the canvas widens it ─────────────────────
    {
        rc::chat::ChatWindow w({});
        w.setAttribute(Qt::WA_DontShowOnScreen);
        w.show();
        QApplication::processEvents();
        assert(w.width() == left_w and w.height() == left_h); ++checks;
        w.show_canvas();
        QApplication::processEvents();
        assert(w.width() > left_w); ++checks;
    }
    std::println("test_chat_window: {} checks passed", checks);
}
