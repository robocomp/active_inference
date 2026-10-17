// GROUNDED MISSIONS — the controller's side of chat_agent/docs/mission_language_2026-10-13.md (2026-10-17).
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
//   · explored(room) has no graph attribute yet (Phase 0): it is unknown, so it never satisfies an `until`.
// Everything here runs on the control thread.
#include "specificworker.h"

#include <format>
#include <print>

#include "../../common/world_query/door_query.h"

using rc::mission::LeafState;
using rc::mission::LeafStatus;
using rc::mission::Ref;

// ── Host ─────────────────────────────────────────────────────────────────────────────────────────────────────────

std::optional<std::string> SpecificWorker::MissionHost::start(const rc::mission::Step &leaf)
{
	const std::string &op = leaf.op;
	const std::string here = w_.world_model_.graph_state().room_name;
	if (op == "cross" or op == "open")
	{
		const auto door = w_.resolve_door(*leaf.ref("door"));
		if (not door.has_value()) return door.error();
		const auto r = w_.start_door_passage(*door, op == "open" ? rc::DoorPassage::Step::Open : rc::DoorPassage::Step::Cross,
		                                     "mission");
		if (not r.accepted) return r.reason;
		door_ = *door;
		return std::nullopt;
	}
	if (op == "go_to")
	{
		const auto to = leaf.ref("to");
		if (not to.has_value() or to->kind != Ref::Kind::Pose) return std::string("no_target");
		if (to->pose.room != here)
		{
			std::println("[mission] go_to: the target is in '{}' and the robot in '{}' (routes between rooms: Phase 1c)",
			             to->pose.room, here);
			return std::string("wrong_room");
		}
		w_.cancel_relative_motion("superseded by a mission go_to");
		w_.session_.door_passage().stop();
		w_.session_.mission().set_mode(rc::DriveMode::Target);   // mode first: set_mode drops a click target
		const Eigen::Vector2f p{static_cast<float>(to->pose.x), static_cast<float>(to->pose.y)};
		w_.session_.mission().set_click_target(p);
		w_.set_manual_target(QPointF(p.x(), p.y()));
		target_epoch_ = w_.session_.manual_target_status().epoch;
		return std::nullopt;
	}
	if (op == "explore")
	{
		const auto room = leaf.ref("room");
		if (room->name != here)
		{
			std::println("[mission] explore: '{}' is not the room the robot is in ('{}')", room->name, here);
			return std::string("wrong_room");
		}
		w_.cancel_relative_motion("superseded by a mission explore");
		w_.session_.door_passage().stop();
		w_.session_.mission().set_click_target(std::nullopt);
		w_.clear_manual_target();
		w_.session_.mission().set_mode(rc::DriveMode::Command);   // no door required: the selector chooses freely
		room_ = room->name;
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

LeafStatus SpecificWorker::MissionHost::poll(const rc::mission::Step &leaf)
{
	const std::string &op = leaf.op;
	if (op == "cross" or op == "open")
	{
		const auto v = w_.session_.door_passage().view();
		if (v.door != door_ or v.step == rc::DoorPassage::Step::Idle) return {.state = LeafState::Failed, .detail = "abandoned"};
		if (v.step == rc::DoorPassage::Step::Done) return {.state = LeafState::Succeeded};
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
		if (w_.world_model_.graph_state().room_name != room_) return {.state = LeafState::Failed, .detail = "outside_room"};
		return {};
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
	if (op == "cross" or op == "open")
	{
		if (w_.session_.door_passage().view().door == door_) w_.session_.door_passage().stop();
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

std::optional<double> SpecificWorker::MissionHost::eval(const rc::mission::Pred &p)
{
	if (p.name == "in_room")
	{
		const std::string &here = w_.world_model_.graph_state().room_name;
		if (here.empty()) return std::nullopt;
		return p.refs.at("room").name == here ? 1.0 : 0.0;
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
	return std::nullopt;   // explored: no graph attribute yet (Phase 0)
}

// ── Worker side ──────────────────────────────────────────────────────────────────────────────────────────────────

std::optional<rc::mission::Error> SpecificWorker::start_mission(rc::mission::Mission m, const char *source)
{
	const std::string title = m.title;
	if (auto e = interpreter_.start(std::move(m), current_time_ms()); e.has_value())
	{
		std::println("[{}] mission '{}' refused: {}", source, title, e->message());
		return e;
	}
	mission_source_ = source;
	++mission_serial_;
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
	interpreter_.cancel(outcome, current_time_ms());
	for (const auto &e : interpreter_.take_events()) std::println("[mission] {}", e);
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
	if (interpreter_.state() != S::Running) hold_base();   // completed or failed: the robot stays where it is
}

void SpecificWorker::hold_base()
{
	cancel_relative_motion("the mission holds the base");
	session_.door_passage().stop();
	session_.mission().set_mode(rc::DriveMode::Target);
	session_.mission().set_click_target(std::nullopt);
	clear_manual_target();
}

std::expected<std::string, std::string> SpecificWorker::resolve_door(const rc::mission::Ref &r)
{
	if (r.kind == Ref::Kind::Node)
	{
		if (not G->get_node(r.name).has_value()) return std::unexpected(std::format("door '{}' is not in the graph", r.name));
		return r.name;
	}
	if (r.kind != Ref::Kind::Passage) return std::unexpected(std::string("not a door"));
	const auto &gs = world_model_.graph_state();
	if (r.from != gs.room_name) return std::unexpected(std::string("wrong_room"));
	// The live passage_<n> (ltsm passage_live.h) holds a `match` to each of its two faces: take the one in this room.
	const auto passage = G->get_node(r.name);
	if (not passage.has_value()) return std::unexpected(std::string("passage_unresolved"));
	for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(passage.value(), "match"))
		if (const auto door = G->get_node(e.to()); door.has_value() and rc::world::room_of(*G, door->id()) == gs.room_id)
			return door->name();
	return std::unexpected(std::string("passage_unresolved"));
}
