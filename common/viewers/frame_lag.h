/*
 *  Frame-lag readout for stream viewers (active_inference/common).
 *
 *  ONE quantity: how old the frame you are looking at is — wall-clock now minus the frame's SOURCE
 *  capture stamp, both in epoch ms. Every media-plane producer stamps the acquisition instant (the
 *  webots bridge even back-dates it by the sample's in-simulator age, see acquisition_wall_ms), so
 *  this is end-to-end latency: capture → transport → display, not an intra-viewer timing.
 *
 *  WHY IT IS SMOOTHED. The raw per-frame value jitters over roughly one producer poll period —
 *  measured 2026-09-10 on the live fleet: rc/lidar3d/points p10 17 ms / p90 57 ms around a 37 ms
 *  median, rc/zed/rgb 18/51 around 38. Printed raw at 20-30 Hz that is an unreadable blur, and the
 *  blur carries no information a human wants: the phase between a producer's sampling and its
 *  consumer's poll is not something you can act on. The EMA (~10 frames) shows the level, which is.
 *
 *  Both the sample rate and the display rate matter and they are different: sample() is called per
 *  ARRIVAL (that is what makes the mean a transport latency rather than a paint-timing artefact),
 *  while text() is called per REPAINT, which for a viewer with a refresh timer keeps running when
 *  the stream stops. That is why staleness is reported from the data itself (below) instead of the
 *  display simply freezing on its last good value — a frozen lag readout is the exact failure the
 *  readout exists to expose.
 */
#ifndef RC_COMMON_FRAME_LAG_H
#define RC_COMMON_FRAME_LAG_H

#include <QString>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>

namespace rc::viewers
{

// Age of a frame in ms: wall-clock now minus its source capture stamp (epoch ms). Signed on purpose
// — negative means the producer's clock runs ahead of ours (or a simulated clock is being published
// as wall time), which is worth seeing rather than hiding. nullopt for a stampless frame.
inline std::optional<double> frame_lag_ms(std::uint64_t src_stamp_ms)
{
	if(src_stamp_ms == 0)
		return std::nullopt;
	const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                        std::chrono::system_clock::now().time_since_epoch()).count();
	return static_cast<double>(now_ms) - static_cast<double>(src_stamp_ms);
}

// Smoothed lag readout for a status overlay. Fed one sample per arrival; rendered per repaint.
class LagMeter
{
public:
	// One arrival. Both EMAs are driven by SOURCE stamps only: the lag from this frame's age, the
	// inter-frame period from consecutive stamps (never wall-clock deltas between our own wake-ups,
	// which collapse toward zero when a drain-to-newest reader returns an already-queued frame).
	void sample(std::uint64_t src_stamp_ms)
	{
		const auto lag = frame_lag_ms(src_stamp_ms);
		if(not lag)
			return;
		lag_ms_ = have_lag_ ? (kAlpha * lag_ms_ + (1.0 - kAlpha) * *lag) : *lag;
		have_lag_ = true;
		if(last_stamp_ != 0 and src_stamp_ms > last_stamp_)
		{
			const double dt = static_cast<double>(src_stamp_ms - last_stamp_);
			if(dt > 0.5 and dt < 10000.0)
				period_ms_ = (period_ms_ > 0.0) ? (kAlpha * period_ms_ + (1.0 - kAlpha) * dt) : dt;
		}
		last_stamp_ = src_stamp_ms;
	}

	// Label text for the overlay. Normally the smoothed lag. If the newest frame in hand is older
	// than the smoothed lag by several of its OWN periods, the stream has stopped delivering and the
	// raw, growing age is appended — the comparison is between two measured quantities of this very
	// stream (its latency and its period), so it needs no fixed millisecond cutoff and it adapts to a
	// 2 Hz stream and a 30 Hz one alike.
	QString text() const
	{
		if(not have_lag_)
			return QStringLiteral("lag: n/a");
		QString s = QString::asprintf("lag: %.0f ms", lag_ms_);
		if(const auto age = frame_lag_ms(last_stamp_);
		   age and period_ms_ > 0.0 and *age > lag_ms_ + kStalePeriods * period_ms_)
			s += QString::asprintf("  (stale %.1f s)", *age / 1000.0);
		return s;
	}

private:
	static constexpr double kAlpha = 0.9;      // ~10-frame time constant: settles in ~0.4 s at 25 Hz
	static constexpr double kStalePeriods = 3.0;   // missed deliveries, in units of the stream's own period
	double lag_ms_ = 0.0, period_ms_ = 0.0;
	bool have_lag_ = false;
	std::uint64_t last_stamp_ = 0;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_FRAME_LAG_H
