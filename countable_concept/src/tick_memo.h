/*
 * tick_memo.h — per-tick memo of expensive, pure-within-a-tick lookups (no graph dependency; unit-testable).
 *
 * A tick works on ONE consistent pose: a transform asked for twice with the same arguments inside one compute()
 * returns the first answer. The owner calls clear() at the start of every tick (and the memo is only reachable
 * through a thread_local pointer set for the duration of compute(), so code outside a tick, or on a worker
 * thread, bypasses it). Failures (nullopt) are never stored, so a transform that appears mid-tick is still found.
 */
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace rc
{
template <class V>
class TickMemo
{
public:
    // Returns the stored value for `key`, or runs compute() (→ std::optional<V>) once and stores a success.
    // `hit` reports whether the answer came from the memo.
    template <class F>
    std::optional<V> get(const std::string& key, F&& compute, bool& hit)
    {
        if (const auto it = map_.find(key); it != map_.end())
        {
            hit = true;
            return it->second;
        }
        hit = false;
        std::optional<V> v = std::forward<F>(compute)();
        if (v.has_value())
            map_.emplace(key, *v);
        return v;
    }
    void clear() { map_.clear(); }
    [[nodiscard]] std::size_t size() const { return map_.size(); }
private:
    std::unordered_map<std::string, V> map_;
};
}  // namespace rc
