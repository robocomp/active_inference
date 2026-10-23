/*
 * proposal_publisher.h -- the PROPOSAL stage of the sensor_calibration agent (plan Task 2.3). Plain data in, an
 * optional wire payload out: no DDS, no DSR, no Qt (the graph write is sensor_calibration/src/proposal_writer.h).
 *
 * WIRE FORMAT (cortex attributes `mount_proposal`, `mount_proposal_version`, `mount_proposal_ready`, written on the
 * agent's OWN node only; robot_concept is the only writer of the body->sensor RT edges and commits in Phase 6):
 *     mount_proposal  std::vector<float>, 4 sensors in the FIXED order helios, bpearl, zed, ricoh; each 42 floats =
 *                     6 deltas (x y z roll pitch yaw) then the 6x6 covariance row-major  => exactly 168 floats.
 *     mount_proposal_version  int, +1 on every new proposal.
 *     mount_proposal_ready    bool, false until the Phase 6 commit protocol exists.
 *
 * CONVENTION (same as robot_concept/src/mount_overlay.h). The deltas are a CORRECTION in BODY-ALIGNED axes at the
 * sensor origin:  t = t_nominal + t_corr,  R = Rx(roll) Ry(pitch) Rz(yaw) * R_nominal  (left-multiplied, so a yaw
 * reads the same for all four sensors). Units: metres and radians.
 *
 * A DoF that nothing has measured stays at the PRIOR: delta 0 and its prior covariance. `informed` says how much the
 * data has to say about each DoF (data precision against prior precision); it is carried for the consumer and for
 * logs and is NOT a cut-off -- the covariance already encodes it.
 *
 * RATE LIMITER (this is a limiter on GRAPH WRITES, not on the estimate -- the estimate is never gated). A proposal
 * is emitted when (a) at least min_interval_ms passed since the last write (<= 1 Hz) and (b) some DoF's posterior
 * mean has moved, since the last PUBLISHED proposal, by more than move_sigma (0.1) of its own posterior sigma.
 * Both exist only to keep the DDS write rate and the consumers' wakeups sane; (b) is relative to the
 * posterior's own sigma, so a tight posterior publishes small moves and a loose one waits. The very first
 * proposal is always emitted.
 */
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace sc::calib
{

inline constexpr std::size_t kMountSensors = 4;                       // helios, bpearl, zed, ricoh (this order)
inline constexpr std::size_t kMountDoF = 6;                           // x y z roll pitch yaw
inline constexpr std::size_t kFloatsPerSensor = kMountDoF + kMountDoF * kMountDoF;   // 42
inline constexpr std::size_t kProposalFloats = kMountSensors * kFloatsPerSensor;     // 168
inline constexpr std::array<const char*, kMountSensors> kMountSensorNames{"helios", "bpearl", "zed", "ricoh"};

/// One sensor's mount posterior: a correction (see CONVENTION) with its covariance.
struct MountPosterior
{
    std::array<double, kMountDoF> mean{};                                  ///< (x y z roll pitch yaw) deltas
    std::array<double, kMountDoF * kMountDoF> cov{};                       ///< row-major 6x6
    std::array<double, kMountDoF> data_precision{};                        ///< information the data contributed per DoF
    std::array<double, kMountDoF> prior_precision{};                       ///< information of the prior per DoF
    /// The prior: zero correction, diagonal covariance (sigma_xyz [m], sigma_rpy [rad]).
    [[nodiscard]] static MountPosterior prior(double sigma_xyz, double sigma_rpy);
    [[nodiscard]] double sigma(std::size_t i) const;                       ///< sqrt of the diagonal
    /// data precision / prior precision for a DoF (0 = prior only); informative, never a gate.
    [[nodiscard]] double informed(std::size_t i) const;
};

using MountPosteriors = std::array<MountPosterior, kMountSensors>;

/// What goes on the wire.
struct ProposalPayload
{
    std::vector<float> mount_proposal;   ///< kProposalFloats
    int  version = 0;
    bool ready = false;
};

[[nodiscard]] std::vector<float> pack_proposal(const MountPosteriors& p);
/// Inverse of pack_proposal (what robot_concept uses in Phase 6): nullopt unless exactly kProposalFloats floats.
/// data/prior precision are not on the wire and come back zero.
[[nodiscard]] std::optional<MountPosteriors> unpack_proposal(const std::vector<float>& v);

struct PublisherConfig
{
    std::int64_t min_interval_ms = 1000;   ///< write-rate limiter, not an estimator parameter
    double       move_sigma = 0.1;         ///< write-rate limiter: fraction of a DoF's own sigma
};

class ProposalPublisher
{
public:
    explicit ProposalPublisher(PublisherConfig cfg = {}) : cfg_(cfg) {}
    /// Offer the current posteriors. Returns the payload to write (version already incremented) or nullopt.
    /// `ready` stays false until the Phase 6 commit protocol exists.
    [[nodiscard]] std::optional<ProposalPayload> offer(const MountPosteriors& p, std::int64_t now_ms);
    [[nodiscard]] int version() const { return version_; }
private:
    PublisherConfig cfg_;
    std::optional<MountPosteriors> last_;
    std::int64_t last_ms_ = 0;
    int version_ = 0;
};

}   // namespace sc::calib
