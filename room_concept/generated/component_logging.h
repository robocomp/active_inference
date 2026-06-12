#pragma once

#include <ConfigLoader/ConfigLoader.h>
#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(logLifecycle)
Q_DECLARE_LOGGING_CATEGORY(logGraph)
Q_DECLARE_LOGGING_CATEGORY(logUi)
Q_DECLARE_LOGGING_CATEGORY(logIo)
Q_DECLARE_LOGGING_CATEGORY(logLocalizer)

void install_component_log_format();
void configure_component_logging(const ConfigLoader& config_loader);