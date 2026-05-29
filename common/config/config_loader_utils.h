#pragma once

#include <ConfigLoader/ConfigLoader.h>
#include <QDebug>

#include <utility>

namespace rc
{
class ConfigLoaderUtils
{
public:
    template <typename TargetType, typename LoadType = TargetType>
    static void load_required(const ConfigLoader& config_loader, const char* key, TargetType& target)
    {
        try
        {
            target = static_cast<TargetType>(config_loader.get<LoadType>(key));
        }
        catch (const std::exception& e)
        {
            qCritical() << "Required config key" << key << "is missing or invalid:" << e.what();
            throw;
        }
    }

    template <typename TargetType, typename LoadType = TargetType>
    static void load_optional(const ConfigLoader& config_loader, const char* key, TargetType& target)
    {
        if (!config_loader.exists(key))
            return;

        try
        {
            target = static_cast<TargetType>(config_loader.get<LoadType>(key));
        }
        catch (const std::exception& e)
        {
            qWarning() << "Ignoring invalid config key" << key << ':' << e.what();
        }
    }

    template <typename LoadType, typename AssignFn>
    static void load_optional_apply(const ConfigLoader& config_loader, const char* key, AssignFn&& assign)
    {
        if (!config_loader.exists(key))
            return;

        try
        {
            std::forward<AssignFn>(assign)(config_loader.get<LoadType>(key));
        }
        catch (const std::exception& e)
        {
            qWarning() << "Ignoring invalid config key" << key << ':' << e.what();
        }
    }
};
}  // namespace rc