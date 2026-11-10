# A robot that acts to know

**Active inference in a working cognitive architecture.** Twenty hours over five days,
Montevideo, 30 November – 4 December, 9:00–13:00. Each day is a four-hour session, half theory and
half hands-on lab.

## The course

The course follows one claim through a complete robotic system: that perception and action
minimise a single quantity, and that a robot representing its own uncertainty can decide what to
do next by asking which action would reduce it most. The material comes from a doctoral thesis in
which this is not a framework but the thing actually running: a mobile robot, Shadow, building a
semantic model of an apartment.

On the first day we pose the whole problem as a challenge. A robot is put down in an unknown
apartment. It must:

1. estimate the room it stands in;
2. localise itself in that room, and learn its own body (wheels, gyro, camera mounts) from the
   corrections the room supplies;
3. let the objects of the room — tables, chairs, bottles — be born, tracked and removed as beliefs;
4. find the door and cross it;
5. estimate the new room, and remember how the two connect.

It is a large problem, but it can be understood as a whole. The course builds the tools first, then
the machine that runs them, then follows beliefs through time and into action. Each day's lab works on
a small two-dimensional version of the problem, written in Python.

Theory stays in the foreground. Each session grounds it in code, and in the measurements that
revealed when the equations were wrong. Some of the most instructive results are recent and not
yet published; they are presented as work in progress.

## Schedule

The thread that runs through all five days: **averaging removes independent error, not shared
error.** It appears in many guises — the common-mode ceiling on how much a mask can teach, a prior
re-centred on its own estimate, a fused precision better than its best member, a metric that
scores what the robot never saw.

### Day 1 — The challenge, probability, and free energy
We watch the full cycle run in simulation, and name what each stage must infer. Then we set up the
mathematics, starting from the beginning:
- joint distributions and the three operations everything is built from: marginalisation,
  conditioning and factorisation (the product rule, and Bayes' rule as a consequence of it);
- the same operations on Gaussians, in covariance and in precision form;
- the chain rule: a joint of N variables factorises into N factors, in N! equivalent orders, and
  why we write the causal one (sparse, physical, modular, and it can generate predictions); where
  a plain controller or a learnt policy, p(a|o), sits in that picture;
- generative models, variational inference and the evidence lower bound;
- free energy as complexity minus accuracy;
- free energy differentiated with respect to the belief and with respect to the action, and a first
  look at expected free energy, with its pragmatic and epistemic terms;
- perception as inference under a model rather than reconstruction from data;
- the Laplace approximation, which gives Gaussian beliefs whose covariance is the inverse
  curvature, the form everything later takes.

*Lab:* a warm-up on a door/LiDAR probability table and a Gaussian; then meet the toy world, fit a
wall from a LiDAR scan as a Laplace belief, test whether its covariance is honest, and choose
between two models of it by free energy.

### Day 2 — The CORTEX architecture: concepts and memories
- Concepts as the fabric of the architecture: one agent per kind of thing, a concept's dimensions
  (geometry, aspect, life cycle, function), the class held by the agent and the instances on the graph.
- A distributed working memory held as a shared graph: geometry and meaning on the same edges, no
  messages between agents, and what a replicated store obliges everyone to assume.
- The memories as the same concepts kept for longer: episodic, long-term spatial, long-term object;
  and what is still to build, including amortisation.
- One global free energy divided among processes: each boundary is a conditional-independence
  assumption, and it has a price in nats.
- The methodological commitment of the course, no thresholds: a gate is what a model looks like before
  the modelling has been done, and its replacement is almost always a precision. And the one case
  where it is not.

*Lab:* agents that coordinate only through a graph with an ownership contract; what happens when the
published covariance is ignored; when the sum of the agents' objectives is the global one and what the
difference costs; a range gate against a precision; a bias that no precision can bound.

### Day 3 — Concepts: geometry, life cycle, aspect
What a concept agent is: a residual and a Jacobian on a shared engine, what it publishes, a common contract.
Then three of a concept's dimensions:
- Geometry: the room as the first belief born, with description length as its price; localisation as one
  more belief, with symmetry as a discrete search; data association; and what changes when errors are
  shared rather than independent, the common-mode ceiling.
- Life cycle: birth as a comparison between two models rather than a count of detections; absence as
  evidence, weighted by the probability of having detected the object, with walls in the line of sight;
  forgetting as inferred volatility rather than a constant process noise.
- Aspect: what a chair looks like at two levels, the class that can be imagined and the instance that
  can be recognised; a second sight as a comparison in which "new" is one of the candidates.
  Work in progress, marked as such.

*Lab:* an object is born by a free-energy comparison, removed by evidence of absence and held when it
is occluded; a constant process noise is replaced by an inferred one. Then estimate the toy room's
walls and localise in it, meet the room's symmetry, and watch a thousand points with a shared offset
refuse to shrink the uncertainty. Finally, decide whether a chair is one you remember or a new one,
by position and by look, and see what a wrong variance does.

### Day 4 — Action
Why action needs a second functional: expected free energy, its pragmatic and epistemic terms, and the identity
of information gain, mutual information and D-optimal design.
- Affordances as published, priced offers: what an object can do for the robot's uncertainty, measured in nats.
- Selection by expected free energy, with hand-offs that follow from the values rather than from a rule.
- The next best view as optimal experiment design, and the trap of predicting through what is already known.
- Control: routes as optimisation, path tracking as pole assignment, and why a well-posed geometry lets the
  follower do without an obstacle term.
- The robot turns the same objective on itself: calibration as inference, the gauges no data can fix, which
  motions make which parameters observable, drift and why a prior must never be re-centred on its own running
  estimate, and standing still as an observation.

*Lab:* choose viewpoints by expected information; let agents publish priced offers on the toy graph and watch
the controller hand over between them; explore a room with an occupancy grid that knows when to stop; follow a
route with a double pole and deform it around an obstacle; recover a hidden wheel-scale error and gyro bias from
the localiser's corrections.

### Day 5 — Structure, evidence and honesty (to be confirmed)
Relations between objects as latent variables, and the double counting that appears whenever an
estimate is returned to those who produced it. Then how such a system is evaluated when no recorded
dataset can answer the question: policy against policy in one closed loop, and an honest balance of
which mechanisms earned their place, nulls included.

## Throughout

The emphasis falls on active inference as an engineering discipline rather than a vocabulary:
- what it costs to take the free-energy principle literally;
- which design questions it settles, which it leaves open, and how to tell the difference.

Participants should leave able to read the literature critically, and to judge which of it a real
system can carry.

## Assessment

An individual final project, compulsory.

## Prerequisites

- Probability: Bayes' rule, Gaussians, expectations.
- Linear algebra: inverses, determinants, eigenvalues.
- Basic optimisation: least squares, Gauss–Newton.
- Python with numpy.

No prior knowledge of active inference, SLAM or ROS is assumed.

## Repository layout

```
slides/   Beamer sources, one file per day (make → dayN.pdf)
labs/     toyworld/ (the shared Python package), dayN.ipynb, solutions/, tests/
instructor_notes.md   timing, measurement stories with exact numbers and their sources
```

Set up the labs with:

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r labs/requirements.txt
cd labs && jupyter lab
```
