# A robot that acts to know

**Active inference in a working cognitive architecture.** Fifteen hours over five days,
Montevideo, 30 November – 4 December. Each day is a three-hour session, with about 1 h 30 of
theory, a break, and about 1 h 15 of hands-on lab.

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

It is a large problem, but it can be understood as a whole. The course takes it in five steps, one
stage a day. Each day's lab adds that stage to a small two-dimensional version of the problem
written in Python, and by Friday participants run the complete cycle on their own laptops.

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

### Day 2 — Estimating the room, and finding oneself in it
- The room layout as a belief with two regimes: acquiring an unknown room, and operating in a
  known one.
- An inlier band derived from the model rather than chosen.
- The price of a shape in nats.
- Localisation as one more belief, with the room's geometry supplying the likelihood.
- Symmetry as a discrete search.
- What changes when errors are shared rather than independent: the common-mode ceiling.

*Lab:* estimate the toy room's walls and the robot's pose together. Meet the room's symmetry, and
watch a thousand points with a shared offset refuse to shrink the uncertainty.

### Day 3 — Acting to know: the body and the room
Why action needs a second functional: expected free energy, its pragmatic and epistemic terms, and
the identity of information gain, mutual information and D-optimal design; where to look next,
and travel priced in the same nats. Then the robot turns the same objective on itself:
- calibration as inference;
- the gauges no data can fix;
- which motions make which parameters observable (parked, straight, turning in place);
- drift, and why a prior must never be re-centred on its own running estimate;
- standing still as an observation;
- why ordinary driving makes a dedicated calibration manoeuvre unnecessary.

*Lab:* the toy robot has a hidden wheel-scale error and gyro bias. Recover them from the
localiser's corrections, and check the recovery against the identity
recovery = 1 − (σ_post/σ_prior)². Then choose viewpoints by expected information: first on one
wall, then with an occupancy-grid explorer that knows when to stop.

### Day 4 — Concepts fill the room
- A distributed working memory as a shared graph, and the decomposition of one global free energy
  into terms owned by separate processes.
- Concept agents: each holds a belief about one class of object, publishes what only it can
  compute, and answers to a common contract.
- Where beliefs come from and how they end: birth as model comparison rather than counting,
  absence as evidence rather than a missed detection, forgetting as inferred volatility.
- The methodological commitment that gives the course its spine — no thresholds: a gate is what a
  model looks like before the modelling has been done, and the right replacement is almost always
  a precision.

*Lab:* object agents write beliefs into the toy graph. An object is born by a free-energy
comparison, removed by evidence of absence, and held — not removed — when it is merely occluded.

### Day 5 — Acting to know, and how to tell whether it worked
- Affordances as published, priced offers: what an object can do for the robot's uncertainty,
  measured in nats.
- Selection by expected free energy.
- The door as an offer, and the new room as the next problem.
- A spatial memory that records how rooms connect (the open end of the challenge).

Then the harder half: how such a system is evaluated when no recorded dataset can answer the
question; why comparison must set policy against policy inside one closed loop; and the case
studies in which a benchmark scored code the robot never ran, a planner never drove its own plan,
and a metric rewarded extrapolation over observation.

*Lab:* close the loop. An explorer finishes room 1, selects the door, crosses, estimates room 2,
and stores both. Then grade the run two ways and watch one of the metrics lie.

## Throughout

The emphasis falls on active inference as an engineering discipline rather than a vocabulary:
- what it costs to take the free-energy principle literally;
- which design questions it settles, which it leaves open, and how to tell the difference.

Participants should leave able to read the literature critically, and to judge which of it a real
system can carry.

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
