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

#include <cmath>
#include <format>
#include <print>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "../../common/mission/mission_json.h"
#include "../../common/world_query/door_query.h"

using rc::mission::LeafState;
using rc::mission::LeafStatus;
using rc::mission::Ref;

// ── Host ─────────────────────────────────────────────────────────────────────────────────────────────────────────

std::optional<std::string> SpecificWorker::MissionHost::start(const rc::mission::Step &leaf)
{
	const std::string &op = leaf.op;
	const std::string here = w_.world_model_.graph_state().room_name;
	{
		std::lock_guard lock(w_.journal_mx_);
		w_.journal_grounding_.erase(leaf.id);   // this attempt's grounding, or none if it is refused
	}
	if (op == "cross" or op == "open")
	{
		const Ref ref = *leaf.ref("door");   // ref() returns by value: a copy, not a reference to a temporary
		const auto door = w_.resolve_door(ref);
		if (not door.has_value()) return door.error();
		const auto r = w_.start_door_passage(*door, op == "open" ? rc::DoorPassage::Step::Open : rc::DoorPassage::Step::Cross,
		                                     "mission");
		if (not r.accepted) return r.reason;
		door_ = *door;
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
		// The pose the step drives to — an anchor already resolved by the interpreter.
		ground(leaf, "", {{"pose", QJsonObject{{"room", QString::fromStdString(to->pose.room)}, {"x", to->pose.x},
		                                        {"y", to->pose.y}, {"yaw", to->pose.yaw}}}});
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
		ground(leaf, room_, {{"room", QString::fromStdString(room_)}});
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
	std::string ir = rc::mission::encode_json(m);
	if (auto e = interpreter_.start(std::move(m), current_time_ms()); e.has_value())
	{
		std::println("[{}] mission '{}' refused: {}", source, title, e->message());
		return e;
	}
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
	interpreter_.cancel(outcome, current_time_ms());
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
	const auto passage = session_.door_passage().view();
	std::string state = motion ? rc::ChatInbox::status_of_motion(tr, relative_motion_.view(), driving).first
	                           : rc::ChatInbox::status_of(tr, passage, driving).first;
	// A passage never fails by itself — it waits for its step to be offered again — so its failure is SEEN while it
	// runs (the same word MissionHost::poll reports) and becomes the outcome if it then ends without succeeding.
	if (not motion and passage.door == tr.door and passage.attempts > 0 and passage.step != rc::DoorPassage::Step::Idle
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
	journal_push({.kind = JK::StepEnd, .mission = journal_command_->id, .step_id = "1", .op = tr.kind == rc::ChatInbox::Tracked::Kind::Motion ? "move" : "cross",
	              .outcome = command_step_outcome(state), .attempt = 1});
	journal_push({.kind = JK::MissionEnd, .mission = journal_command_->id, .outcome = state});
	journal_command_.reset();
}

// After run_mission_command accepted a command that starts something: it is a one-step mission "c<n>". cancel / resume
// are about the interpreter's mission (journaled by journal_steps); a refused command started nothing.
void SpecificWorker::journal_command(const MissionResult &r, const std::string &text, const char *source)
{
	if (not r.accepted or r.kind == "cancel" or r.kind == "resume") return;
	std::lock_guard lock(journal_mx_);
	journal_steps();                 // a mission it replaced (cancel_mission already drained it; cheap if so)
	journal_check_command(true);     // the command it replaced: its own end if it had one, else superseded
	const std::string id = std::format("c{}", ++journal_commands_);
	journal_push({.kind = JK::MissionStart, .mission = id, .op = r.kind, .source = journal_source(source), .text = text});
	QJsonObject step{{"grounding", r.door.empty() ? QJsonObject{} : QJsonObject{{"door", QString::fromStdString(r.door)}}}};
	if (not r.door.empty()) step["target"] = QString::fromStdString(r.door);
	journal_push({.kind = JK::StepStart, .mission = id, .step_id = "1", .op = r.kind, .text = compact(step), .attempt = 1});
	using TK = rc::ChatInbox::Tracked::Kind;
	if (r.kind == "cross") journal_command_ = JournalCommand{.id = id, .tracked = {.kind = TK::Cross, .door = r.door}};
	else if (r.kind == "move") journal_command_ = JournalCommand{.id = id, .tracked = {.kind = TK::Motion, .motion_id = r.motion_id}};
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
