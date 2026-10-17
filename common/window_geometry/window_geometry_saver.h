#pragma once

#include <functional>
#include <utility>

#include <QEvent>
#include <QObject>
#include <QTimer>
#include <QWidget>

namespace rc::ui
{
// Saves a window's geometry shortly after every resize/move/close, not only on a clean quit. The quit-time
// save (aboutToQuit / the worker destructor) never runs when the agent dies some other way: the terminal tab
// closing (SIGHUP is not trapped), a crash, or kill -9. Those were exactly the restarts that came back at the
// old size. Debounced so that a drag writes the file once, when it ends, not on every intermediate event.
// Parented to the window, so it dies with it:   new rc::ui::GeometrySaver(window, [this] { save(); });
class GeometrySaver : public QObject
{
public:
    GeometrySaver(QWidget *window, std::function<void()> save) : QObject(window), save_(std::move(save))
    {
        timer_.setSingleShot(true);
        timer_.setInterval(500);
        QObject::connect(&timer_, &QTimer::timeout, this, [this] { save_(); });
        window->installEventFilter(this);
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        switch (event->type())
        {
            case QEvent::Resize:
            case QEvent::Move:
                timer_.start();
                break;
            case QEvent::Close:   // closing the window: save now, the process may be about to go
                timer_.stop();
                save_();
                break;
            default:
                break;
        }
        return QObject::eventFilter(watched, event);
    }

private:
    std::function<void()> save_;
    QTimer timer_;
};
}  // namespace rc::ui
