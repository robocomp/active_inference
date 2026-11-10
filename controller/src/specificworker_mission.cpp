// GROUNDED MISSIONS — the controller's side of language_agent/docs/mission_language_2026-10-13.md (2026-10-17).
//
// rc::mission::Interpreter (common/mission/mission_interpreter.h) runs the step tree; MissionHost maps each leaf to
// what the controller already does (§3.1):
//   cross / open   DoorPassage through the door (open: stop_after = Open)        done: passage Done
//   go_to pose     the manual target (Target mode), the same as a click           done: arrival · no route · cleared
//   explore room   Command mode with no door required: the selector is free       ended by the interpreter (until/budget)
//   wait           the base held, armed (Target mode, no target)                  ended by the interpreter (s/until/resume)
//   move           RelativeMotion ("turn <deg> move <m>")                          done: its Done / Failed
// A leaf fails on the FIRST failed completion of what it started (a passage step ending timeout/unreachable); the
// mission's `retry=N` is what tries again, so the number of attempts is the user's, not the passage's.
//
// Phase-1 limits, stated where they bite:
//   · go_to takes a pose in the CURRENT room; cross-room routes are Phase 1(c) (resolver). The pose's yaw is not
//     driven to yet (a manual target has no final facing).
//   · a passage ref resolves through the LIVE passage_<n> (ltsm passage_live.h), which exists only while the room
//     beyond is proto; otherwise it is refused (passage_unresolved).
//   · explored(room) is room_concept's `room_explored_p` on the room node (mission_explored.h, Task 5): absent =
//     never assessed = unknown, which never satisfies an `until`.
// Everything here runs on the control thread.
#include "specificworker.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto: a cross into a provisional room is surveyed
#include "../../common/room_resolve/room_resolve.h"   // physical_room: what `bind` latches

#include <chrono>
#include <cmath>
#include <format>
#include <print>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "../../common/mission/mission_json.h"
#include "../../common/world_query/door_query.h"
#include "mission_explored.h"
#include "mission_room_hold.h"
#include "mission_approach_hold.h"
#include "mission_door_relation.h"
#include "../../common/ltsm_topology/topology_decode.h"   // crossed_doors: the exploration drive's remembered crossings

using rc::mission::LeafState;
using rc::mission::LeafStatus;
using rc::mission::Ref;

// ── Host ─────────────────────────────────────────────────────────────────────────────────────────────────────────

namespace
{
std::uint64_t hold_clock_ms()
{
	return std::uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
}   // namespace

// A room-scoped step with no resolved room yet (the first room is still being born) is NOT YET: hold it (RoomHold, its own
// budget) and start it once current_room resolves. Only a resolved DIFFERENT room is wrong_room (start_now).
std::optional<std::string> SpecificWorker::MissionHost::start(const rc::mission::Step &leaf)
{
	hold_.clear();
	relation_hold_.clear();
	approach_hold_.reset();
	if (rc::RoomHold::scoped(leaf) and w_.world_model_.graph_state().room_name.empty())
	{
		std::println("[mission] step {} ({}) waits: the current room is not resolved yet", leaf.id, leaf.op);
		hold_.hold(leaf, hold_clock_ms());
		return std::nullopt;
	}
	return start_now(leaf);
}

std::optional<std::string> SpecificWorker::MissionHost::start_now(const rc::mission::Step &leaf)
{
	const std::string &op = leaf.op;
	const std::string here = w_.world_model_.graph_state().room_name;
	{
		std::lock_guard lock(w_.journal_mx_);
		w_.journal_grounding_.erase(leaf.id);   // this attempt's grounding, or none if it is refused
	}
	if (op == "close")
	{
		// `close <door>` (2026-11-01, user: the PLAN triggers door_concept's close affordance). A relation may resolve late —
		// `close room_1` waits for ltsm's recall_room_1 under the face just crossed — and the robot HOLDS meanwhile (it is at
		// the door it is about to close: exploring would walk it out of reach).
		const Ref ref = *leaf.ref("door");
		const auto door = w_.resolve_door(ref);
		if (not door.has_value() and rc::relation_of(ref) != rc::DoorRelationKind::None)
		{
			std::println("[mission] step {} (close {}) waits: no door in '{}' resolves yet — holding", leaf.id, ref.name, here);
			relation_hold_.hold(leaf, hold_clock_ms());
			w_.hold_base();
			return std::nullopt;
		}
		if (not door.has_value()) return door.error();
		const auto next = next_goal_of(leaf);
		const auto r = w_.start_door_close(*door, "mission", next);
		if (not r.accepted) return r.reason;
		door_ = *door;
		chain_target_ = next.has_value();
		ground(leaf, *door, QJsonObject{{"door", QString::fromStdString(*door)}});
		return std::nullopt;
	}
	if (op == "cross" or op == "open")
	{
		const Ref ref = *leaf.ref("door");   // ref() returns by value: a copy, not a reference to a temporary
		const auto door = w_.resolve_door(ref);
		if (not door.has_value() and rc::relation_of(ref) != rc::DoorRelationKind::None)
		{
			// `cross <room>` / `cross new`: the recall (or the new door) may appear late — hold and re-resolve each poll.
			std::println("[mission] step {} ({} {}) waits: no door in '{}' resolves yet — exploring the room meanwhile", leaf.id, op,
			             ref.name, here);
			relation_hold_.hold(leaf, hold_clock_ms());
			explore_while_waiting(here);
			return std::nullopt;
		}
		if (not door.has_value()) return door.error();
		const auto r = w_.start_door_passage(*door, op == "open" ? rc::DoorPassage::Step::Open : rc::DoorPassage::Step::Cross,
		                                     "mission");
		if (not r.accepted) return r.reason;
		door_ = *door;
		w_.session_.door_passage().set_next_goal(next_goal_of(leaf));   // one step of look-ahead (re-tried each poll while unknown)
		QJsonObject g{{"door", QString::fromStdString(*door)}};
		if (ref.kind == Ref::Kind::Passage)
		{
			g["passage"] = QString::fromStdString(ref.name);
			g["from"] = QString::fromStdString(ref.from);
		}
		ground(leaf, *door, g);
		return std::nullopt;
	}
	if (op == "go_to")
	{
		const auto to = leaf.ref("to");
		if (not to.has_value() or to->kind != Ref::Kind::Pose) return std::string("no_target");
		if (not w_.robot_in_named_room(to->pose.room))
		{
			std::println("[mission] go_to: the target is in '{}' and the robot in '{}' (routes between rooms: Phase 1c)",
			             to->pose.room, here);
			return std::string("wrong_room");
		}
		w_.cancel_relative_motion("superseded by a mission go_to");
		w_.session_.door_passage().stop();
		w_.session_.approach_order().stop();
		w_.session_.mission().set_mode(rc::DriveMode::Target);   // mode first: set_mode drops a click target
		const Eigen::Vector2f p{static_cast<float>(to->pose.x), static_cast<float>(to->pose.y)};
		w_.session_.mission().set_click_target(p);
		w_.set_manual_target(QPointF(p.x(), p.y()));
		target_epoch_ = w_.session_.manual_target_status().epoch;
		// The pose the step drives to — an anchor already resolved by the interpreter.
		ground(leaf, "", {{"pose", QJsonObject{{"room", QString::fromStdString(to->pose.room)}, {"x", to->pose.x},
		                                        {"y", to->pose.y}, {"yaw", to->pose.yaw}}}});
		return std::nullopt;
	}
	if (op == "explore")
	{
		const auto room = leaf.ref("room");
		// ★ `here` is the localisation's room; current_room() is where the robot physically is. Accept either.
		const auto physical = current_room();
		if (not w_.robot_in_named_room(room->name) and not (physical.has_value() and room->name == *physical))
		{
			std::println("[mission] explore: '{}' is not the room the robot is in ('{}')", room->name, here);
			return std::string("wrong_room");
		}
		w_.cancel_relative_motion("superseded by a mission explore");
		w_.session_.door_passage().stop();
		w_.session_.approach_order().stop();
		w_.session_.mission().set_click_target(std::nullopt);
		w_.clear_manual_target();
		w_.session_.mission().set_mode(rc::DriveMode::Command);   // no door required: the selector chooses freely
		w_.session_.set_command_explore(true, room->name);         // …opened on purpose by this order, for its room only
		room_ = room->name;
		room_id_ = w_.world_model_.graph_state().room_id;   // the room by ID: a rename in place is not leaving it
		ground(leaf, room_, {{"room", QString::fromStdString(room_)}});
		return std::nullopt;
	}
	if (op == "approach")
	{
		const Ref ref = *leaf.ref("object");
		const std::string side = leaf.string("side").value_or(std::string{});
		w_.session_.mission().set_click_target(std::nullopt);
		w_.clear_manual_target();
		const auto r = w_.start_approach(ref.name, side, "mission", rc::ApproachOrder::After::Return);
		if (not r.accepted) return r.reason;
		approach_id_ = r.approach_id;
		QJsonObject g{{"object", QString::fromStdString(ref.name)}};
		if (not side.empty()) g["side"] = QString::fromStdString(side);
		ground(leaf, ref.name, g);
		return std::nullopt;
	}
	if (op == "wait")
	{
		w_.hold_base();
		return std::nullopt;
	}
	if (op == "move")
	{
		std::string text;
		if (const auto deg = leaf.number("turn_deg"); deg.has_value()) text += std::format("turn {} ", *deg);
		if (const auto m = leaf.number("distance_m"); m.has_value()) text += std::format("move {}", *m);
		const auto r = w_.start_relative_motion(text, "mission");
		if (not r.accepted) return r.reason;
		motion_id_ = r.motion_id;
		return std::nullopt;
	}
	return "unsupported:" + op;
}

void SpecificWorker::MissionHost::ground(const rc::mission::Step &leaf, const std::string &target, const QJsonObject &grounding)
{
	QJsonObject text{{"grounding", grounding}};
	if (not target.empty()) text["target"] = QString::fromStdString(target);
	std::lock_guard lock(w_.journal_mx_);
	w_.journal_grounding_[leaf.id] = QJsonDocument(text).toJson(QJsonDocument::Compact).toStdString();
}

// A relation that resolves to no door yet (`cross new` before the door is seen) must not stand still: a door is only born
// from a view of it (recog_try29 leg 4: the explore step was already satisfied, the hold stood the robot still for 360 s and
// DOOR_2 was never seen). Explore the room until the door passage takes over (start_door_passage closes the explore).
void SpecificWorker::MissionHost::explore_while_waiting(const std::string &room)
{
	if (room.empty()) return;
	w_.session_.mission().set_click_target(std::nullopt);
	w_.clear_manual_target();
	w_.session_.mission().set_mode(rc::DriveMode::Command);
	w_.session_.set_command_explore(true, room);
}

// ── LOOK-AHEAD OF ONE STEP (chained_action.h) ─────────────────────────────────────────────────────────────────────────────
// The first standpoint of the step after `leaf` (a `close` is skipped: it runs on the way), in the CURRENT room's frame:
//   open/cross <door>  -> door_concept's approach vantage on that door's axis (aff_<door>_approach), carried over from the door's
//                         own room through the graph's RT chain when it is another room;
//   go_to <pose>       -> the pose, when it is in this room.
// Anything else, or anything that does not resolve now, is nullopt: the step then behaves exactly as before.
std::optional<rc::chain::NextGoal> SpecificWorker::MissionHost::next_goal_of(const rc::mission::Step &leaf)
{
	const rc::mission::Step *nx = rc::chain::next_step(plan_, leaf.id);
	if (nx == nullptr or w_.G == nullptr) return std::nullopt;
	const std::string label = std::format("{} -> {}", leaf.id, nx->id);
	if (nx->op == "go_to")
	{
		const auto to = nx->ref("to");
		if (not to.has_value() or to->kind != Ref::Kind::Pose or not w_.robot_in_named_room(to->pose.room)) return std::nullopt;
		return rc::chain::NextGoal{"", {static_cast<float>(to->pose.x), static_cast<float>(to->pose.y)}, static_cast<float>(to->pose.yaw), label};
	}
	if (nx->op != "open" and nx->op != "cross") return std::nullopt;
	const auto ref = nx->ref("door");
	if (not ref.has_value()) return std::nullopt;
	const auto door = w_.resolve_door(*ref);
	if (not door.has_value()) return std::nullopt;
	const auto dn = w_.G->get_node(*door);
	const auto an = w_.G->get_node("aff_" + *door + "_approach");
	if (not dn.has_value() or not an.has_value()) return std::nullopt;
	const auto x = w_.G->get_attrib_by_name<epistemic_target_x_m_att>(*an);
	const auto y = w_.G->get_attrib_by_name<epistemic_target_y_m_att>(*an);
	const auto yaw = w_.G->get_attrib_by_name<epistemic_target_yaw_rad_att>(*an);
	if (not x.has_value() or not y.has_value() or not std::isfinite(*x) or not std::isfinite(*y)) return std::nullopt;
	Eigen::Vector2f p(*x, *y);
	float heading = yaw.has_value() and std::isfinite(*yaw) ? *yaw : 0.f;
	const auto &gs = w_.world_model_.graph_state();
	if (const auto room_id = rc::world::room_of(*w_.G, dn->id()); room_id != 0 and room_id != gs.room_id)
	{
		const auto room = w_.G->get_node(room_id);
		auto *ie = w_.world_model_.inner_eigen();
		if (not room.has_value() or ie == nullptr) return std::nullopt;
		const auto T = ie->get_transformation_matrix(gs.room_name, room->name());
		if (not T.has_value()) return std::nullopt;
		const Eigen::Vector3d q = T->matrix().block<3, 3>(0, 0).cast<double>() * Eigen::Vector3d(*x, *y, 0.0) + T->matrix().block<3, 1>(0, 3).cast<double>();
		p = Eigen::Vector2f(static_cast<float>(q.x()), static_cast<float>(q.y()));
		heading += static_cast<float>(std::atan2(T->matrix()(1, 0), T->matrix()(0, 0)));
	}
	return rc::chain::NextGoal{*door, p, heading, label};
}

LeafStatus SpecificWorker::MissionHost::poll(const rc::mission::Step &leaf)
{
	const std::string &op = leaf.op;
	if (relation_hold_.holding())
	{
		const auto door = w_.resolve_door(*leaf.ref("door"));
		switch (relation_hold_.check(door.has_value() ? std::optional<std::string>(*door) : std::nullopt, hold_clock_ms()))
		{
			case rc::RelationHold::Verdict::Waiting: return {};
			case rc::RelationHold::Verdict::TimedOut: return {.state = LeafState::Failed, .detail = "door_unresolved"};
			case rc::RelationHold::Verdict::Ready:
				if (auto refused = start_now(leaf); refused.has_value()) return {.state = LeafState::Failed, .detail = *refused};
				return {};   // started: the passage is polled from the next cycle
		}
	}
	if (hold_.holding())
	{
		switch (hold_.check(w_.world_model_.graph_state().room_name, hold_clock_ms()))
		{
			case rc::RoomHold::Verdict::Waiting: return {};
			case rc::RoomHold::Verdict::TimedOut: return {.state = LeafState::Failed, .detail = "room_unresolved"};
			case rc::RoomHold::Verdict::Ready:
				if (auto refused = start_now(leaf); refused.has_value()) return {.state = LeafState::Failed, .detail = *refused};
				break;   // started: this very poll reports it as any started leaf
		}
	}
	if (op == "close")
	{
		const auto v = w_.session_.door_passage().view();
		const auto end_chain = [&]
		{
			if (not std::exchange(chain_target_, false)) return;
			w_.session_.mission().set_click_target(std::nullopt);   // the next step owns the base
			w_.clear_manual_target();
		};
		if (v.door != door_ or v.step == rc::DoorPassage::Step::Idle) { end_chain(); return {.state = LeafState::Failed, .detail = "abandoned"}; }
		if (v.step != rc::DoorPassage::Step::Done) return {};
		end_chain();
		if (v.failed) return {.state = LeafState::Failed, .detail = v.last_outcome.empty() ? std::string("door_left_open") : v.last_outcome};
		return {.state = LeafState::Succeeded};
	}
	if (op == "cross" or op == "open")
	{
		// the look-ahead may resolve late (the next door is in a room not yet known): keep trying until it does
		if (not door_.empty() and w_.session_.door_passage().view().door == door_ and not w_.session_.door_passage().next_goal().has_value())
			if (const auto ng = next_goal_of(leaf); ng.has_value()) w_.session_.door_passage().set_next_goal(ng);
		// `cross new` / `cross <room>` name a RELATION, not a node: if the door it resolved to leaves the graph before the
		// crossing (door_concept removed it for absence — recog_try26 leg 3: phantom door_12 removed, the passage then waited
		// 600 s for a door that no longer existed), drop the passage and resolve the relation again.
		if (const Ref ref = *leaf.ref("door"); rc::relation_of(ref) != rc::DoorRelationKind::None and not door_.empty()
		    and w_.G != nullptr and not w_.G->get_node(door_).has_value()
		    and (w_.session_.door_passage().view().step == rc::DoorPassage::Step::Approach
		         or w_.session_.door_passage().view().step == rc::DoorPassage::Step::Open))
		    // only while still on the NEAR side: once the passage drives THROUGH, door_concept re-makes the door on the far
		    // side — possibly before room_concept switches the room (recog_try31 leg 2), so the room is no witness
		{
			std::println("[mission] step {} ({} {}): '{}' left the graph before the crossing — resolving '{}' again", leaf.id, op,
			             ref.name, door_, ref.name);
			if (w_.session_.door_passage().view().door == door_) w_.session_.door_passage().stop();
			door_.clear();
			relation_hold_.hold(leaf, hold_clock_ms());
			explore_while_waiting(w_.world_model_.graph_state().room_name);
			return {};
		}
		const auto v = w_.session_.door_passage().view();
		if (v.door != door_ or v.step == rc::DoorPassage::Step::Idle) return {.state = LeafState::Failed, .detail = "abandoned"};
		// a FAILED passage ends Done too (finish()): it is a failure, not a success (Live fix 3: a ruled-out look)
		if (v.failed) return {.state = LeafState::Failed, .detail = v.last_outcome.empty() ? std::string("failed") : v.last_outcome};
		if (v.step == rc::DoorPassage::Step::Done)
		{
			// ★ A NEW ROOM IS SURVEYED BEFORE IT IS LEFT (user 2026-10-31: "until the room is established the robot must follow
			//   the affordances drawn by the epistemic planner"). pingpong13 leg 1 stood 18 s in the provisional room_2 and
			//   left it unpromoted; ltsm archived that half-seen layout and every re-entry relocalised against it. So a cross
			//   that lands in a PROVISIONAL room ends only when room_concept promotes it; meanwhile the room is explored —
			//   the selector free to take room_concept's epistemic affordances (afford_room), as `explore <room>`.
			const auto &gs = w_.world_model_.graph_state();
			if (op == "cross" and w_.G != nullptr and gs.room_id != 0 and rc::room::is_proto(*w_.G, gs.room_id))
			{
				if (survey_room_id_ != gs.room_id)
				{
					survey_room_id_ = gs.room_id;
					std::println("[mission] step {}: through into '{}', still provisional — surveying it on the epistemic "
					             "affordances until room_concept establishes it", leaf.id, gs.room_name);
					w_.session_.mission().set_mode(rc::DriveMode::Command);
					w_.session_.set_command_explore(true, gs.room_name);
				}
				return {};
			}
			if (survey_room_id_ != 0)
				std::println("[mission] step {}: '{}' is established — the cross is complete", leaf.id, gs.room_name);
			survey_room_id_ = 0;
			return {.state = LeafState::Succeeded};
		}
		// A hand-back of the APPROACH is not a failure: door_concept re-sites it; wait for the re-offer (ApproachHold).
		const bool on_approach = v.step == rc::DoorPassage::Step::Approach;
		const bool present = on_approach and w_.G != nullptr and w_.G->get_node("aff_" + door_ + "_approach").has_value();
		switch (approach_hold_.check(on_approach, v.attempts, present, v.step_state == "executing", hold_clock_ms(),
		                                  std::uint64_t(std::max(0.0, leaf.number("budget_s").value_or(0.0)) * 1000.0)))
		{
			case rc::ApproachHold::Verdict::Waiting: return {};
			case rc::ApproachHold::Verdict::Failed:
				return {.state = LeafState::Failed, .detail = std::format("approach_{}", v.last_outcome)};
			case rc::ApproachHold::Verdict::NotHeld: break;
		}
		if (v.attempts > 0)
			return {.state = LeafState::Failed,
			        .detail = v.step == rc::DoorPassage::Step::Open
			                      ? std::string("door_wont_open")
			                      : std::format("{}_{}", rc::DoorPassage::step_name(v.step), v.last_outcome)};
		return {};
	}
	if (op == "go_to")
	{
		using E = ControllerSession::ManualTargetEnd;
		const auto st = w_.session_.manual_target_status();
		if (st.epoch != target_epoch_) return {.state = LeafState::Failed, .detail = "abandoned"};
		switch (st.end)
		{
			case E::Pending: return {};
			case E::Reached: return {.state = LeafState::Succeeded};
			case E::NoRoute: return {.state = LeafState::Failed, .detail = "no_route"};
			case E::Cleared: return {.state = LeafState::Failed, .detail = "abandoned"};
		}
		return {};
	}
	if (op == "explore")
	{
		// Only a resolved DIFFERENT room ID ends the step; the name is no witness (room_concept renames a recognised room in place).
		const auto &gs = w_.world_model_.graph_state();
		switch (rc::explore_room_verdict(room_id_, gs.room_id, room_, gs.room_name))
		{
			case rc::ExploreRoom::Outside:
				std::println("[mission] explore {}: the robot is now in room id {} ('{}'), not id {} — outside_room", room_, gs.room_id,
				             gs.room_name, room_id_);
				return {.state = LeafState::Failed, .detail = "outside_room"};
			case rc::ExploreRoom::Renamed:
				// the contest is scoped by NAME: swap the name only — an executing claim, lock-on and plan survive
				std::println("[mission] explore {}: the room was renamed in place ({} -> {}, node id {}) — the step follows the id", room_,
				             room_, gs.room_name, room_id_);
				room_ = gs.room_name;
				w_.session_.rename_command_explore_room(room_);
				break;
			case rc::ExploreRoom::Inside: break;
		}
		return {};
	}
	if (op == "approach")
	{
		const auto v = w_.session_.approach_order().view();
		if (v.id != approach_id_) return {.state = LeafState::Failed, .detail = "abandoned"};
		if (v.active) return {};
		if (v.outcome == "satisfied") return {.state = LeafState::Succeeded};
		return {.state = LeafState::Failed, .detail = v.outcome.empty() ? std::string("abandoned") : v.outcome};
	}
	if (op == "move")
	{
		using P = rc::RelativeMotion::Phase;
		const auto v = w_.relative_motion_.view();
		if (v.id != motion_id_) return {.state = LeafState::Failed, .detail = "abandoned"};
		if (v.phase == P::Done) return {.state = LeafState::Succeeded};
		if (v.phase == P::Failed) return {.state = LeafState::Failed, .detail = v.last_event};
		if (v.phase == P::Cancelled) return {.state = LeafState::Failed, .detail = "abandoned"};
		return {};
	}
	return {};   // wait: the interpreter ends it
}

void SpecificWorker::MissionHost::cancel(const rc::mission::Step &leaf)
{
	const std::string &op = leaf.op;
	if (hold_.holding()) { hold_.clear(); return; }   // never started: nothing to stop
	if (relation_hold_.holding()) { relation_hold_.clear(); return; }
	if (op == "cross" or op == "open" or op == "close")
	{
		if (w_.session_.door_passage().view().door == door_) w_.session_.door_passage().stop();
		if (std::exchange(chain_target_, false))
		{
			w_.session_.mission().set_click_target(std::nullopt);
			w_.clear_manual_target();
		}
	}
	else if (op == "go_to")
	{
		const auto st = w_.session_.manual_target_status();
		if (st.epoch == target_epoch_ and st.end == ControllerSession::ManualTargetEnd::Pending)
		{
			w_.session_.mission().set_click_target(std::nullopt);
			w_.clear_manual_target();
		}
	}
	else if (op == "move")
	{
		if (w_.relative_motion_.view().id == motion_id_) w_.cancel_relative_motion("the mission step ended");
	}
	else if (op == "approach")
	{
		if (w_.session_.approach_order().view().id == approach_id_) w_.session_.approach_order().stop();
	}
	// explore / wait: what the next step (or the end of the mission) starts replaces them
}

void SpecificWorker::MissionHost::say(const std::string &text)
{
	std::println("[mission] says: {}", text);
}

void SpecificWorker::MissionHost::halt()
{
	w_.hold_base();
}

std::optional<rc::mission::Pose> SpecificWorker::MissionHost::pose()
{
	const auto &gs = w_.world_model_.graph_state();
	const auto p = w_.world_model_.read_robot_pose_latest(w_.current_time_ms());
	if (not p.has_value() or gs.room_name.empty()) return std::nullopt;
	// yaw: the controller's own theta, so an anchor captured here and driven to here mean the same heading.
	return rc::mission::Pose{.room = gs.room_name, .x = p->pos.x(), .y = p->pos.y(), .yaw = p->theta};
}

// ★ WHERE THE ROBOT PHYSICALLY IS, not the identity resolver's room (final review I1): right after a crossing ltsm's
//   `current` may still name the room just left while the robot stands in the new proto-room. `bind` latching the old
//   room would send a later `explore $next` back through the door. Same rule as the chat's report (room_resolve.h).
std::optional<std::string> SpecificWorker::MissionHost::current_room()
{
	const std::string here = rc::room::physical_room_name(*w_.G);
	return here.empty() ? std::nullopt : std::optional<std::string>(here);
}

void SpecificWorker::MissionHost::on_bind(const std::string &name, const std::string &room)
{
	std::println("[mission] bind {} = {} (localisation's room: '{}')", name, room, w_.world_model_.graph_state().room_name);
}

std::optional<double> SpecificWorker::MissionHost::eval(const rc::mission::Pred &p)
{
	if (p.name == "in_room")
	{
		if (w_.world_model_.graph_state().room_name.empty()) return std::nullopt;
		return w_.robot_in_named_room(p.refs.at("room").name) ? 1.0 : 0.0;
	}
	if (p.name == "door_open")
	{
		const auto door = w_.resolve_door(p.refs.at("door"));
		if (not door.has_value()) return std::nullopt;
		const auto node = w_.G->get_node(*door);
		if (not node.has_value()) return std::nullopt;
		const auto prob = w_.G->get_attrib_by_name<door_open_prob_att>(node.value());
		return prob.has_value() ? std::optional<double>(prob.value()) : std::nullopt;
	}
	if (p.name == "explored") return rc::mission::explored_belief(*w_.G, w_.world_model_.graph_state().room_aliases, p.refs.at("room").name);
	return std::nullopt;
}

// ── Worker side ──────────────────────────────────────────────────────────────────────────────────────────────────

std::optional<rc::mission::Error> SpecificWorker::start_mission(rc::mission::Mission m, const char *source)
{
	const std::string title = m.title;
	const std::vector<rc::mission::Step> plan_steps = m.steps;
	std::string ir = rc::mission::encode_json(m);
	if (auto e = interpreter_.start(std::move(m), current_time_ms()); e.has_value())
	{
		std::println("[{}] mission '{}' refused: {}", source, title, e->message());
		return e;
	}
	mission_host_.set_plan(plan_steps);   // the look-ahead reads the step tree (chained_action.h)
	journal_mission_started(source, ir);   // before the first tick: the superseded mission's steps close first
	mission_source_ = source;
	mission_ir_json_ = std::move(ir);
	// ★ A MISSION IS A "GO", like a click or a relative move: it arms the base output (Stop disarmed it). It reached
	//   here through the chat's confirmation card or a script someone dropped on purpose.
	paused_ = false;
	motion_commander_.set_output_enabled(true);
	driving_enabled_ = true;
	hibernationTick();
	mission_poll_ms_ = 0;   // the first step starts now, not in 100 ms
	for (const auto &e : interpreter_.take_events()) std::println("[mission] {}", e);
	service_mission();
	return std::nullopt;
}

void SpecificWorker::cancel_mission(const std::string &outcome, bool hold)
{
	if (interpreter_.state() != rc::mission::Interpreter::State::Running) return;
	if (interpreter_.paused()) paused_ = false;   // ★ a cancelled frozen mission must not leave the base held
	interpreter_.cancel(outcome, current_time_ms());
	session_.set_command_explore(false);   // a cancelled mission's explore no longer holds the contest open
	{
		std::lock_guard lock(journal_mx_);
		journal_steps();   // now, so a command that replaces the mission is journaled after its end
	}
	for (const auto &e : interpreter_.take_events()) std::println("[mission] {}", e);
	mission_node_.publish(*G, interpreter_, mission_source_, mission_ir_json_);
	if (hold) hold_base();
}

void SpecificWorker::service_mission()
{
	using S = rc::mission::Interpreter::State;
	if (interpreter_.state() != S::Running) return;
	const auto now = current_time_ms();
	if (now - mission_poll_ms_ < 100) return;
	mission_poll_ms_ = now;
	interpreter_.tick(now);
	for (const auto &e : interpreter_.take_events()) std::println("[mission] {}", e);
	mission_node_.publish(*G, interpreter_, mission_source_, mission_ir_json_);   // written only on change
	if (interpreter_.state() != S::Running) hold_base();   // completed or failed: the robot stays where it is
}

void SpecificWorker::hold_base()
{
	cancel_relative_motion("the mission holds the base");
	session_.door_passage().stop();
	session_.approach_order().stop();
	session_.mission().set_mode(rc::DriveMode::Target);
	session_.mission().set_click_target(std::nullopt);
	clear_manual_target();
}

// The live doors of the current room (any depth) with the relations `cross <room>` / `cross new` select by.
std::vector<rc::DoorFacts> SpecificWorker::door_facts()
{
	std::vector<rc::DoorFacts> out;
	const auto &gs = world_model_.graph_state();
	if (G == nullptr or gs.room_id == 0) return out;
	const auto robot_T = inner_eigen_api_ and not gs.robot_name.empty()
	                         ? inner_eigen_api_->get_transformation_matrix(gs.room_name, gs.robot_name) : std::nullopt;
	for (const auto &name : rc::world::doors_of_room(*G, gs.room_id))
	{
		const auto door = G->get_node(name);
		if (not door.has_value()) continue;
		rc::DoorFacts f{.name = name};
		for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(door.value(), "RT"))
			if (const auto child = G->get_node(e.to()); child.has_value() and child->name().starts_with("recall_"))
				f.recall_rooms.insert(child->name().substr(7));
		for (const auto &e : G->get_edges_to_id(door->id()))
			if (e.type() == "match") f.has_match = true;
		f.crossed = rc::exit_fact::is_crossed_face(*G, door->id());
		if (G->get_node(rc::DoorPassage::look_name(name)).has_value()) door_looked_.insert(name);
		f.looked = door_looked_.contains(name);
		f.open = G->get_attrib_by_name<door_open_prob_att>(door.value()).value_or(-1.f) >= 0.5f;
		if (robot_T.has_value() and inner_eigen_api_)
			if (const auto T = inner_eigen_api_->get_transformation_matrix(gs.room_name, name); T.has_value())
				f.dist = (T->translation() - robot_T->translation()).head<2>().norm();
		out.push_back(std::move(f));
	}
	return out;
}

// Is the robot in the room a mission calls `name`? By node id, so a room renamed in place is still itself.
bool SpecificWorker::robot_in_named_room(const std::string &name)
{
	const auto &gs = world_model_.graph_state();
	std::optional<std::uint64_t> live;
	if (G != nullptr)
		if (const auto n = G->get_node(name); n.has_value()) live = n->id();
	return rc::robot_in_room(gs.room_id, gs.room_name, gs.room_aliases, live, name);
}

std::expected<std::string, std::string> SpecificWorker::resolve_door(const rc::mission::Ref &r)
{
	if (rc::relation_of(r) != rc::DoorRelationKind::None)
	{
		const auto facts = door_facts();
		auto expanded = facts;
		if (rc::relation_of(r) == rc::DoorRelationKind::Room)
		{
			rc::RoomAliases ids = world_model_.graph_state().room_aliases;
			for (const auto &n : G->get_nodes_by_type("room")) ids[n.name()] = n.id();
			rc::expand_recalls(expanded, ids, r.name);   // recall_<old name> or recall_<new name> of the same room
		}
		if (const auto door = rc::pick_door(r, expanded); door.has_value()) return *door;
		return std::unexpected(std::format("no door of '{}' resolves '{}' yet", world_model_.graph_state().room_name, r.name));
	}
	if (r.kind == Ref::Kind::Node)
	{
		// Live fix 3: a remembered door (door_concept offers aff_door_look_<door>) resolves — the passage starts with the look
		if (not G->get_node(r.name).has_value() and not rc::DoorPassage::look_available(*G, r.name))
			return std::unexpected(std::format("door '{}' is not in the graph", r.name));
		return r.name;
	}
	if (r.kind != Ref::Kind::Passage) return std::unexpected(std::string("not a door"));
	const auto &gs = world_model_.graph_state();
	if (not robot_in_named_room(r.from)) return std::unexpected(std::string("wrong_room"));
	// The live passage_<n> (ltsm passage_live.h) holds a `match` to each of its two faces: take the one in this room.
	const auto passage = G->get_node(r.name);
	if (not passage.has_value()) return std::unexpected(std::string("passage_unresolved"));
	for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(passage.value(), "match"))
		if (const auto door = G->get_node(e.to()); door.has_value() and rc::world::room_of(*G, door->id()) == gs.room_id)
			return door->name();
	return std::unexpected(std::string("passage_unresolved"));
}

// ── The episodic journal (specificworker.h) ──────────────────────────────────────────────────────────────────────

namespace
{
using JK = rc::em::JournalEntry::Kind;

std::string compact(const QJsonObject &o) { return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString(); }

/// A StepEnd's outcome: the interpreter's outcome word ("satisfied", "until", "unreachable", "superseded", …); a skipped
/// step keeps the failure it skipped, as "skipped|<word>".
std::string step_outcome(const rc::mission::Interpreter::StepTransition &t)
{
	return t.state == rc::mission::Interpreter::StepState::Skipped ? "skipped|" + t.outcome : t.outcome;
}

/// A single command's StepEnd outcome from its MissionEnd state (ChatInbox::status_of's words): completed →
/// satisfied, failed|x → x, aborted → cancelled, aborted|x → x — the interpreter's step vocabulary.
std::string command_step_outcome(const std::string &state)
{
	if (state == "completed") return "satisfied";
	if (state == "aborted") return "cancelled";
	if (const auto bar = state.find('|'); bar != std::string::npos) return state.substr(bar + 1);
	return state;
}

std::string journal_source(const char *source) { return std::string_view(source) == "gui" ? "panel" : source; }
}   // namespace

void SpecificWorker::journal_push(rc::em::JournalEntry e)
{
	if (e.t_ms == 0) e.t_ms = current_time_ms();
	journal_.push(std::move(e));
}

void SpecificWorker::journal_steps()
{
	for (const auto &t : interpreter_.take_transitions())
	{
		if (journal_mission_.empty()) continue;   // no mission open in the journal: nothing to attach them to
		rc::em::JournalEntry e{.t_ms = t.t, .mission = journal_mission_, .step_id = t.id, .op = t.op,
		                       .attempt = t.attempt, .iteration = t.iteration};
		if (t.state == rc::mission::Interpreter::StepState::Running)
		{
			e.kind = JK::StepStart;
			const auto g = journal_grounding_.find(t.id);
			e.text = g != journal_grounding_.end() ? g->second : compact(QJsonObject{{"grounding", QJsonObject{}}});
		}
		else
		{
			e.kind = JK::StepEnd;
			e.outcome = step_outcome(t);
		}
		journal_push(std::move(e));
	}
	if (not journal_mission_.empty() and interpreter_.state() != rc::mission::Interpreter::State::Running)
	{
		journal_push({.kind = JK::MissionEnd, .mission = journal_mission_, .outcome = interpreter_.status()});
		journal_mission_.clear();
	}
}

void SpecificWorker::journal_check_command(bool superseded)
{
	if (not journal_command_.has_value()) return;
	const bool driving = driving_enabled_ and not paused_;
	const auto &tr = journal_command_->tracked;
	const bool motion = tr.kind == rc::ChatInbox::Tracked::Kind::Motion;
	const bool approach = tr.kind == rc::ChatInbox::Tracked::Kind::Approach;
	const auto passage = session_.door_passage().view();
	std::string state = motion ? rc::ChatInbox::status_of_motion(tr, relative_motion_.view(), driving).first
	                  : approach ? rc::ChatInbox::status_of_approach(tr, session_.approach_order().view(), driving).first
	                           : rc::ChatInbox::status_of(tr, passage, driving).first;
	// A passage never fails by itself — it waits for its step to be offered again — so its failure is SEEN while it
	// runs (the same word MissionHost::poll reports) and becomes the outcome if it then ends without succeeding.
	if (not motion and not approach and passage.door == tr.door and passage.attempts > 0 and passage.step != rc::DoorPassage::Step::Idle
	    and passage.step != rc::DoorPassage::Step::Done)
		journal_command_->failure = passage.step == rc::DoorPassage::Step::Open
		                                ? std::string("door_wont_open")
		                                : std::format("{}_{}", rc::DoorPassage::step_name(passage.step), passage.last_outcome);
	if (state == "in_progress" or state.empty())
	{
		if (not superseded) return;
		state = "aborted|superseded";
	}
	if (state.starts_with("aborted") and not journal_command_->failure.empty()) state = "failed|" + journal_command_->failure;
	journal_push({.kind = JK::StepEnd, .mission = journal_command_->id, .step_id = "1", .op = motion ? "move" : approach ? "approach" : "cross",
	              .outcome = command_step_outcome(state), .attempt = 1});
	journal_push({.kind = JK::MissionEnd, .mission = journal_command_->id, .outcome = state});
	journal_command_.reset();
}

// After run_mission_command accepted a command that starts something: it is a one-step mission "c<n>". cancel / resume
// are about the interpreter's mission (journaled by journal_steps); a refused command started nothing.
void SpecificWorker::journal_command(const MissionResult &r, const std::string &text, const char *source)
{
	if (not r.accepted or r.kind == "cancel" or r.kind == "resume"
	    or r.kind == "pause" or r.kind == "unpause") return;
	std::lock_guard lock(journal_mx_);
	journal_steps();                 // a mission it replaced (cancel_mission already drained it; cheap if so)
	journal_check_command(true);     // the command it replaced: its own end if it had one, else superseded
	const std::string id = std::format("c{}", ++journal_commands_);
	journal_push({.kind = JK::MissionStart, .mission = id, .op = r.kind, .source = journal_source(source), .text = text});
	QJsonObject step{{"grounding", not r.door.empty() ? QJsonObject{{"door", QString::fromStdString(r.door)}}
	                               : not r.object.empty() ? QJsonObject{{"object", QString::fromStdString(r.object)}}
	                                                      : QJsonObject{}}};
	if (not r.door.empty()) step["target"] = QString::fromStdString(r.door);
	if (not r.object.empty()) step["target"] = QString::fromStdString(r.object);
	journal_push({.kind = JK::StepStart, .mission = id, .step_id = "1", .op = r.kind, .text = compact(step), .attempt = 1});
	using TK = rc::ChatInbox::Tracked::Kind;
	if (r.kind == "cross") journal_command_ = JournalCommand{.id = id, .tracked = {.kind = TK::Cross, .door = r.door}};
	else if (r.kind == "move") journal_command_ = JournalCommand{.id = id, .tracked = {.kind = TK::Motion, .motion_id = r.motion_id}};
	else if (r.kind == "approach")
		journal_command_ = JournalCommand{.id = id, .tracked = {.kind = TK::Approach, .approach_id = r.approach_id}};
	else
	{
		// stop / uncross: done the moment they are accepted.
		journal_push({.kind = JK::StepEnd, .mission = id, .step_id = "1", .op = r.kind, .outcome = "satisfied", .attempt = 1});
		journal_push({.kind = JK::MissionEnd, .mission = id, .outcome = "completed"});
	}
}

void SpecificWorker::journal_mission_started(const char *source, const std::string &ir)
{
	std::lock_guard lock(journal_mx_);
	// A superseding start() cancelled the old mission's steps inside the interpreter: they are queued now, and the
	// interpreter already reports the NEW mission as running, so the old one is closed here by hand.
	const std::string old = std::exchange(journal_mission_, {});
	for (const auto &t : interpreter_.take_transitions())
		if (not old.empty())
			journal_push({.t_ms = t.t, .kind = t.state == rc::mission::Interpreter::StepState::Running ? JK::StepStart : JK::StepEnd,
			              .mission = old, .step_id = t.id, .op = t.op, .outcome = step_outcome(t), .attempt = t.attempt,
			              .iteration = t.iteration});
	if (not old.empty()) journal_push({.kind = JK::MissionEnd, .mission = old, .outcome = "aborted|superseded"});
	journal_check_command(true);
	journal_grounding_.clear();
	journal_mission_ = std::format("m{}", interpreter_.serial());
	journal_push({.kind = JK::MissionStart, .mission = journal_mission_, .source = journal_source(source), .text = ir});
}

// The AffordanceManager's claim observer (control thread, inside publish_executing / mark_reached / a release).
void SpecificWorker::journal_claim(const rc::AffordanceManager::ClaimEvent &c)
{
	std::lock_guard lock(journal_mx_);
	rc::em::JournalEntry e{.mission = not journal_mission_.empty() ? journal_mission_
	                                  : journal_command_.has_value() ? journal_command_->id : std::string{},
	                       .op = c.kind_label};
	if (c.kind == rc::AffordanceManager::ClaimEvent::Kind::End)
	{
		e.kind = JK::ClaimEnd;
		e.outcome = c.outcome;
		journal_push(std::move(e));
		return;
	}
	e.kind = JK::ClaimStart;
	const QJsonDocument contract = QJsonDocument::fromJson(QByteArray::fromStdString(c.contract));
	e.text = compact(QJsonObject{{"affordance", QString::fromStdString(c.affordance)},
	                             {"producer", QString::fromStdString(c.producer)},
	                             {"target", QString::fromStdString(c.target)},
	                             {"target_type", QString::fromStdString(c.target_type)},
	                             {"contract", contract.isObject() ? QJsonValue(contract.object()) : QJsonValue(QString::fromStdString(c.contract))}});
	if (std::isfinite(c.gain)) e.promised = c.gain;   // absent is not zero
	// The candidates that COMPETED: an ineligible one was never in the selection (not a tuning cutoff), and listing
	// every offered affordance made each ClaimStart — and so the whole wire — grow with the scene.
	QJsonArray candidates;
	for (const auto &k : affordance_manager_.last_candidates())
		if (k.eligible)
			candidates.append(QJsonObject{{"node_name", QString::fromStdString(k.node_name)},
			                              {"parent_type", QString::fromStdString(k.parent_type)},
			                              {"gain", k.gain}, {"efe_score", k.efe_score},
			                              {"state", QString::fromStdString(k.state)}});
	e.candidates = QJsonDocument(candidates).toJson(QJsonDocument::Compact).toStdString();
	journal_push(std::move(e));
}

void SpecificWorker::service_journal()
{
	if (not G) return;
	std::lock_guard lock(journal_mx_);
	journal_steps();
	journal_check_command(false);
	// The drive mode, read once per pass rather than hooked at its ~20 set_mode call sites (MissionRunner::set_mode is
	// the only writer, and every change persists at least one pass): one choke point, nothing to forget.
	if (const auto mode = session_.mission().mode(); mode != journal_drive_mode_)
	{
		journal_drive_mode_ = mode;
		journal_push({.kind = JK::DriveMode, .op = rc::to_string(mode)});
	}
	// Panel tours, read at the same choke point (tour_journal.h): the runner's state, not its ~5 stop() call sites.
	{
		const auto &runner = session_.mission();
		const auto *tour = runner.running_mission();
		for (auto &e : journal_tours_.observe({.running = tour != nullptr, .name = tour ? tour->name : std::string{},
		                                       .loops = runner.laps_remaining(), .mode = rc::to_string(runner.mode()),
		                                       .stop_reason = runner.stop_reason()}))
			journal_push(std::move(e));
	}
	// The wire goes on our own em_journal node (journal_node.h), written only when it changed: nobody else writes
	// that node, so there is nothing to re-assert.
	if (journal_.dirty() and journal_node_.publish(*G, journal_.to_wire())) journal_.clear_dirty();
}

// ════ EXPLORATION TOUR (exploration_drive.h; spec docs/superpowers/specs/2026-11-05-exploration-tour-design.md) ════

void SpecificWorker::set_explore_tour(bool on)
{
	if (not on)
	{
		explore_drive_.reset();
		explore_in_flight_.clear();
		if (explore_csv_.is_open()) explore_csv_.close();
		std::println("[explore-tour] OFF");
		return;
	}
	explore_drive_.emplace(params.explore_max_room_s);
	explore_in_flight_.clear();
	explore_csv_.open("etc/exploration_drive.csv", std::ios::trunc);
	explore_csv_.imbue(std::locale::classic());
	explore_csv_ << "t_ms,room,promoted,cause,door,is_return,candidates\n";
	session_.mission().set_mode(rc::DriveMode::AffordancesOnly);
	std::println("[explore-tour] ON (MaxRoomS {:.0f} s): explore each room until promoted, then cross the nearest found, "
	             "uncrossed door; else return through the room's first entry door", params.explore_max_room_s);
}

std::string SpecificWorker::entry_mirror_of(const std::string &crossed)
{
	if (G == nullptr) return {};
	const auto d = G->get_node(crossed);
	if (not d.has_value()) return {};
	for (const auto &n : G->get_nodes_by_type("door"))
		if (n.name() != crossed and rc::exit_fact::twin_of(*G, n.id()) == d->id()) return n.name();
	for (const auto &e : G->get_edges_to_id(d->id()))
		if (e.type() == "match")
			if (const auto p = G->get_node(e.from()); p.has_value() and p->name().starts_with("passage_"))
				for (const auto &m : DSR::DSRGraph::get_node_edges_by_type(p.value(), "match"))
					if (const auto other = G->get_node(m.to()); other.has_value() and other->name() != crossed) return other->name();
	return {};
}

std::vector<rc::explore::DoorCand> SpecificWorker::explore_doors(const std::string &topo_json)
{
	const std::set<std::string> remembered = rc::topo::crossed_doors(topo_json);
	std::vector<rc::explore::DoorCand> doors;
	for (const auto &f : door_facts())
		doors.push_back({.door = f.name, .found = G->get_node("aff_" + f.name + "_approach").has_value(),
		                 .crossed = f.crossed or remembered.contains(f.name), .dist_m = f.dist,
		                 .believed = [&] { const auto n = G->get_node(f.name); return n.has_value() and rc::explore::door_believed(G->get_attrib_by_name<obj_exist_prob_att>(n.value())); }()});
	return doors;
}

void SpecificWorker::service_explore_tour()
{
	if (not explore_drive_.has_value() or G == nullptr) return;
	const auto now_ms = static_cast<std::uint64_t>(
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
	if (now_ms - explore_poll_ms_ < 200) return;   // 5 Hz: the drive decides on room-scale events
	explore_poll_ms_ = now_ms;
	const double now_s = static_cast<double>(now_ms) / 1000.0;
	const auto &gs = world_model_.graph_state();
	if (gs.room_id == 0 or gs.room_name.empty()) return;
	auto &passage = session_.door_passage();
	std::string topo_json;
	if (const auto t = G->get_node("ltsm_topology"); t.has_value())
		if (const auto j = G->get_attrib_by_name<ltsm_topology_att>(t.value()); j.has_value()) topo_json = j.value();
	// A CROSSING WITHDRAWN (clean16, 2026-11-09): the room this cycle is the room the last crossing went into (same id) under the name of the
	// room it came from: recognition concluded the robot never left. The door is failed there, the false room forgotten, and the drive
	// goes on: nothing here waits for a room or a polygon.
	if (explore_last_cross_.has_value())
	{
		if (gs.room_id != explore_last_cross_->into_id) explore_last_cross_.reset();
		else if (gs.room_name == explore_last_cross_->from_room and explore_last_cross_->from_room != explore_last_cross_->into_room)
		{
			const auto c = *std::exchange(explore_last_cross_, std::nullopt);
			explore_drive_->note_crossing_withdrawn(c.door, c.entry, c.into_room, c.from_room);
			explore_pending_crossed_.clear();
			if (explore_in_flight_ == c.door) { passage.stop(); explore_in_flight_.clear(); explore_mirror_.clear(); }
			session_.mission().set_mode(rc::DriveMode::AffordancesOnly);
			std::println("[explore-tour] the crossing through '{}' was withdrawn: still in '{}', door failed, the drive goes on", c.door, c.from_room);
		}
	}
	// the passage the drive started: crossed (into a new room), failed, or abandoned
	if (not explore_in_flight_.empty() and explore_mirror_.empty())
		explore_mirror_ = entry_mirror_of(explore_in_flight_);   // while the old face still lives (final review I2)
	// the in-flight door's node VANISHED before the crossing (explore_tour8: door_concept's existence removed door_3 under a
	// misassociated mask; the passage then waited for ever for affordances of a node that no longer existed). A door that
	// is gone cannot be crossed: retarget the passage to the live door now standing for the same passage, else end it as a
	// failure so the drive decides again (a failed Return ends in a logged Stuck, never a loop).
	// ... or when the door is still there but no longer believed (clean14: phantom door_3 disproved by failed opens to P 0.31, its `open` withdrawn, the passage waiting for ever)
	const auto in_flight_node = explore_in_flight_.empty() ? std::nullopt : G->get_node(explore_in_flight_);
	const bool in_flight_disbelieved = in_flight_node.has_value()
	                                   and not rc::explore::door_believed(G->get_attrib_by_name<obj_exist_prob_att>(in_flight_node.value()));
	if (not explore_in_flight_.empty() and gs.room_name == explore_from_room_ and (not in_flight_node.has_value() or in_flight_disbelieved))
	{
		std::string topo_now;
		if (const auto t = G->get_node("ltsm_topology"); t.has_value())
			if (const auto j = G->get_attrib_by_name<ltsm_topology_att>(t.value()); j.has_value()) topo_now = j.value();
		const auto live = rc::topo::live_door_for(topo_now, explore_in_flight_);
		const std::string gone = explore_in_flight_;
		passage.stop();
		explore_in_flight_.clear();
		explore_mirror_.clear();
		if (not in_flight_disbelieved and live.has_value() and G->get_node(*live).has_value()
		    and start_door_passage(*live, rc::DoorPassage::Step::Cross, "explore", /*then_affordances=*/true).accepted)
		{
			explore_in_flight_ = *live;
			std::println("[explore-tour] '{}' vanished before the crossing — retargeted to its passage's live door '{}'", gone, *live);
		}
		else
		{
			explore_drive_->note_failed(gone);
			session_.mission().set_mode(rc::DriveMode::AffordancesOnly);
			std::println("[explore-tour] '{}' {} — failed", gone, in_flight_disbelieved ? "is no longer believed to be a door" : "vanished before the crossing and no live door stands for its passage");
		}
	}
	// a door found AFTER a Return was decided supersedes it while the Return has not committed to its doorway (explore_tour11:
	// room_3 promoted, the drive returned 0.2 s later, DOOR_0 born 2.6 s after that while the robot drove to the entry door)
	if (not explore_in_flight_.empty() and explore_in_flight_return_ and gs.room_name == explore_from_room_)
		if (const auto st = passage.view().step; st == rc::DoorPassage::Step::Look or st == rc::DoorPassage::Step::Approach)
			if (const auto late = explore_drive_->reconsider_return(gs.room_name, explore_doors(topo_json)); late.has_value())
			{
				const std::string was = explore_in_flight_;
				passage.stop();
				explore_in_flight_.clear();
				explore_mirror_.clear();
				explore_in_flight_return_ = false;
				if (start_door_passage(*late->door, rc::DoorPassage::Step::Cross, "explore", /*then_affordances=*/true).accepted)
				{
					explore_in_flight_ = *late->door;
					std::println("[explore-tour] {}: '{}' found after the return through '{}' was decided — it supersedes the return",
					             gs.room_name, *late->door, was);
				}
				else
				{
					explore_drive_->note_failed(*late->door);
					session_.mission().set_mode(rc::DriveMode::AffordancesOnly);
					std::println("[explore-tour] {}: '{}' found after the return was decided, but its passage was refused — the drive decides again",
					             gs.room_name, *late->door);
				}
			}
	if (not explore_in_flight_.empty())
	{
		const auto v = passage.view();
		const bool ended = v.door != explore_in_flight_ or v.step == rc::DoorPassage::Step::Idle or v.step == rc::DoorPassage::Step::Done;
		if (ended and not passage.active())
		{
			// the CROSSING is the room switch: explore_tour4 crossed into room_2 and the passage still ended `cross_unreachable`
			const bool crossed = gs.room_name != explore_from_room_ or (v.door == explore_in_flight_ and v.step == rc::DoorPassage::Step::Done and not v.failed);
			if (crossed)
			{
				explore_drive_->note_crossed(explore_in_flight_);
				std::string entry = explore_mirror_.empty() ? entry_mirror_of(explore_in_flight_) : explore_mirror_;
				if (entry.empty()) entry = rc::topo::live_door_for(topo_json, explore_in_flight_).value_or("");   // the passage's face in view
				if (entry == explore_in_flight_) entry.clear();   // the door just crossed is never its own entry face
				if (entry.empty()) { explore_pending_crossed_ = explore_in_flight_; explore_pending_room_ = gs.room_name; explore_pending_since_s_ = now_s; }
				explore_drive_->note_entered(gs.room_name, entry, now_s);
				if (gs.room_name != explore_from_room_)   // a crossing into a room with another name: if recognition concludes the name was false, it is withdrawn (above)
					explore_last_cross_ = ExploreLastCross{.door = explore_in_flight_, .entry = entry, .from_room = explore_from_room_, .into_room = gs.room_name, .into_id = gs.room_id};
				std::println("[explore-tour] entered '{}' through '{}' (entry door '{}')", gs.room_name, explore_in_flight_,
				             entry.empty() ? "not resolved yet — no guess" : entry);
			}
			else
			{
				explore_drive_->note_failed(explore_in_flight_);
				std::println("[explore-tour] the passage through '{}' did not cross ({}) — skipped for this visit", explore_in_flight_,
				             v.last_outcome.empty() ? "abandoned" : v.last_outcome);
			}
			explore_in_flight_.clear();
			explore_mirror_.clear();
			session_.mission().set_mode(rc::DriveMode::AffordancesOnly);
		}
	}
	// an entry face not resolved at the crossing: the LTSM summary names the passage's face in view once it catches up
	if (explore_in_flight_.empty() and not explore_pending_crossed_.empty())
	{
		if (explore_pending_room_ != gs.room_name) explore_pending_crossed_.clear();
		else if (const auto e = rc::topo::live_door_for(topo_json, explore_pending_crossed_); e.has_value() and G->get_node(*e).has_value())
		{
			explore_drive_->note_entered(gs.room_name, *e, now_s);
			std::println("[explore-tour] entry door of '{}' resolved from the LTSM passage: '{}'", gs.room_name, *e);
			explore_pending_crossed_.clear();
		}
	}
	// no decision until the face just come in by is known (explore_tour5: back in room_1 the drive decided in the cycle it entered
	// and chose door_4, the mirror of the doorway just crossed — the twin edge's old face was gone and the LTSM not caught up).
	// Bounded by the same flagged room limit: an entry never resolved stops holding the drive after MaxRoomS.
	if (not explore_pending_crossed_.empty() and explore_pending_room_ == gs.room_name
	    and now_s - explore_pending_since_s_ < params.explore_max_room_s)
		return;
	// a recognition proposal is being judged: this room may be about to become a remembered one (explore_tour6 decided a Return
	// under the provisional name room_3 while ltsm proposed it was room_1) — no decision until the verdict
	// Only while it is PENDING: ltsm keeps an accepted proposal node until the robot leaves the room (explore_tour7 held for
	// 12 min after "ACCEPTED: room_4 adopted as r2_room_2"); room_concept writes state accepted | rejected (room_proposal.h).
	bool proposal_settled = false;
	for (const auto &n : G->get_nodes_by_type("metaconcept"))
		if (G->get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "room_proposal")
		{
			if (G->get_attrib_by_name<state_att>(n).value_or(std::string("pending")) == "pending") return;
			proposal_settled = true;
		}
	// ... and BEFORE there is a proposal (clean13): back in R1, still the provisional room_5, the drive decided a Return 1 s before ltsm proposed room_1, and the
	// robot left R1 for the bedroom while the graph said R1. While recognition's best explanation of this room is a remembered room, the room's identity is
	// open: no decision. A room recognition says is new is not held. Bounded by the same flagged room limit.
	if (not proposal_settled and explore_drive_->within_room_time(gs.room_name, now_s))
		if (const auto rec = G->get_node("recognition"); rec.has_value())
			if (const auto post = G->get_attrib_by_name<recog_posterior_att>(rec.value());
			    post.has_value() and rc::explore::recognition_leans_known(post->get()))
				return;
	// the operator's and missions' orders come first (final review I1): no decision while either holds the base
	if (interpreter_.state() == rc::mission::Interpreter::State::Running or session_.approach_order().active()) return;
	// the drive's inputs
	const auto doors = explore_doors(topo_json);
	const bool promoted = not rc::room::is_proto(*G, gs.room_id);
	std::optional<rc::explore::DoorQuestions> questions;
	if (const auto q = G->get_node("door_questions"); q.has_value())
		questions = rc::explore::DoorQuestions{.room_id = G->get_attrib_by_name<parent_att>(q.value()).value_or(0),
		                                       .state = G->get_attrib_by_name<state_att>(q.value()).value_or(std::string{})};
	const auto d = explore_drive_->step(gs.room_name, promoted, doors, passage.active() or not explore_in_flight_.empty(), now_s,
	                                    rc::topo::remembered_uncrossed(topo_json, gs.room_name), rc::explore::doors_settled(questions, gs.room_id));
	if (d.cause == rc::explore::Cause::None) return;
	// the record (one row per decision) + one line
	std::string cands;
	for (const auto &c : doors)
		cands += std::format("{}{}:{}:{}:{:.2f}", cands.empty() ? "" : ";", c.door, c.found ? 1 : 0, c.crossed ? 1 : 0, c.dist_m);
	const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	explore_csv_ << wall_ms << ',' << gs.room_name << ',' << (promoted ? 1 : 0) << ',' << rc::explore::to_string(d.cause) << ','
	             << d.door.value_or("") << ',' << (d.is_return ? 1 : 0) << ',' << cands << '\n';
	explore_csv_.flush();
	std::println("[explore-tour] {}: {} -> {}", gs.room_name, rc::explore::to_string(d.cause), d.door.value_or("-"));
	if (d.cause == rc::explore::Cause::Stuck)
	{
		std::println("[explore-tour] STUCK in '{}': the way back failed and nothing else is left — the drive stops", gs.room_name);
		set_explore_tour(false);
		return;
	}
	if (d.cause == rc::explore::Cause::Done)
	{
		std::println("[explore-tour] DONE — no reachable room has a found, uncrossed door");
		set_explore_tour(false);
		return;
	}
	if (d.door.has_value())
	{
		// a door name goes stale when its room is left and re-entered (final review C1): the passage's live door now
		std::string door = *d.door;
		if (not G->get_node(door).has_value())
			if (const auto live = rc::topo::live_door_for(topo_json, door); live.has_value())
			{
				std::println("[explore-tour] '{}' is gone; its passage's live door is '{}'", door, *live);
				door = *live;
			}
		const auto r = start_door_passage(door, rc::DoorPassage::Step::Cross, "explore", /*then_affordances=*/true);
		if (r.accepted) { explore_in_flight_ = door; explore_from_room_ = gs.room_name; explore_pending_crossed_.clear(); explore_in_flight_return_ = d.is_return; }
		else explore_drive_->note_failed(*d.door);
	}
}
