#include "component_logging.h"

#include <QByteArray>

#include <atomic>
#include <cstdio>

Q_LOGGING_CATEGORY(logLifecycle, "component.lifecycle")
Q_LOGGING_CATEGORY(logGraph, "component.graph")
Q_LOGGING_CATEGORY(logUi, "component.ui")
Q_LOGGING_CATEGORY(logIo, "component.io")
Q_LOGGING_CATEGORY(logLocalizer, "component.localizer")

class ComponentLogState
{
public:
    static void install_handler()
    {
        QtMessageHandler old_handler = qInstallMessageHandler(&ComponentLogState::message_handler);
        if (old_handler != &ComponentLogState::message_handler)
            previous_handler_ = old_handler;
    }

    static void set_verbose(bool verbose)
    {
        verbose_.store(verbose, std::memory_order_relaxed);
    }

private:
    static void message_handler(QtMsgType type, const QMessageLogContext& context, const QString& message)
    {
        if (!verbose_.load(std::memory_order_relaxed) && type == QtDebugMsg)
            return;

        if (previous_handler_)
        {
            previous_handler_(type, context, message);
            return;
        }

        const QByteArray formatted = qFormatLogMessage(type, context, message).toLocal8Bit();
        std::fprintf(stderr, "%s\n", formatted.constData());
        std::fflush(stderr);
    }

    static std::atomic_bool verbose_;
    static QtMessageHandler previous_handler_;
};

std::atomic_bool ComponentLogState::verbose_{false};
QtMessageHandler ComponentLogState::previous_handler_ = nullptr;

void install_component_log_format()
{
    qSetMessagePattern(QStringLiteral("%{if-category}%{category}: %{endif}%{message}"));
    ComponentLogState::install_handler();
}

void configure_component_logging(const ConfigLoader& config_loader)
{
    const bool verbose = config_loader.exists("Component.Debug.Verbose")
                             ? config_loader.get<bool>("Component.Debug.Verbose")
                             : false;

    ComponentLogState::set_verbose(verbose);

    QLoggingCategory::setFilterRules(
        verbose
            ? QStringLiteral("*.debug=true\n"
                             "component.lifecycle.debug=true\n"
                             "component.graph.debug=true\n"
                             "component.ui.debug=true\n"
                             "component.io.debug=true\n"
                             "component.localizer.debug=true\n")
            : QStringLiteral("*.debug=false\n"
                             "component.lifecycle.debug=false\n"
                             "component.graph.debug=false\n"
                             "component.ui.debug=false\n"
                             "component.io.debug=false\n"
                             "component.localizer.debug=false\n"));

    qCInfo(logLifecycle) << (verbose ? "Verbose logging enabled" : "Verbose logging disabled");
}