# A robot that acts to know

**Active inference in a working cognitive architecture.** Fifteen hours over five days,
Montevideo, 30 November – 4 December. Each day is a three-hour session, half theory and
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

### Day 1 — The challenge and probability
We watch the full cycle run in simulation, and name what each stage must infer. Then we set up the
mathematics, starting from the beginning:
- joint distributions and the three operations everything is built from: marginalisation,
  conditioning and factorisation (the product rule, and Bayes' rule as a consequence of it);
- the same operations on Gaussians, in covariance and in precision form;
- the chain rule: a joint of N variables factorises into N factors, in N! equivalent orders, and
  why we write the causal one (sparse, physical, modular, and it can generate predictions); where
  a plain controller or a learnt policy, p(a|o), sits in that picture.

*Lab:* the three operations by hand on a door/LiDAR probability table and on a Gaussian wall; one
joint in two orders; a robot whose action depends on the world only through what it has seen.

### Day 2 — Free energy: perception and action
- Why exact Bayes cannot be computed, and the belief that replaces the posterior: generative models,
  variational inference and the evidence lower bound;
- free energy as complexity minus accuracy;
- the Laplace approximation, which gives Gaussian beliefs whose covariance is the inverse
  curvature, the form everything later takes; Gauss–Newton to find it;
- free energy as the price of a model;
- free energy differentiated with respect to the belief and with respect to the action:
  perception and action as two ways to lower the same quantity;
- expected free energy: how to choose an action before its reading exists, as what a look would
  teach plus how far it leaves the goal.

*Lab:* meet the toy world, fit a wall from a LiDAR scan as a Laplace belief, test whether its
covariance is right, and choose between two models of it by free energy.

### Day 3 — The CORTEX architecture: concepts and memories
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

### Day 4 — One concept, from birth to action: the chair
What a concept agent is: a residual and a Jacobian on a shared engine, what it publishes, a common contract.
Then one concept, the chair, through all it does:
- The room it stands in, briefly: the room as a probability model, its outline chosen by free energy with a
  price per corner, and the robot localised against it on a distance field.
- Geometry: the chair's pose as a Laplace belief from its masks; data association; the common-mode ceiling.
- Life cycle: birth as a comparison between two models rather than a count of detections; absence as
  evidence, weighted by the probability of having detected the object, with walls in the line of sight;
  forgetting as inferred volatility rather than a constant process noise.
- Aspect: how a chair's look is captured and stored, at two levels: the class that can be imagined
  (sampled from the prior and rendered) and the instance, measured one viewing direction at a time.
  Work in progress.
- Function: expected free energy, recalled from Day 2, and the identity of information gain, mutual
  information and D-optimal design. The chair publishes two offers, a look priced in nats and an approach;
  the controller is a selector that compares every offer in one currency, then follows a route to the
  winner. Offers from different agents compete fairly only when each agent's covariance is calibrated
  and no point is counted twice; how to put preferences on the same scale is still open.

*Lab:* an object is born by a free-energy comparison, removed by evidence of absence and held when it is
occluded; a constant process noise is replaced by an inferred one. Localise in the toy room, meet its
symmetry, and watch a thousand points with a shared offset refuse to shrink the uncertainty. Then choose
viewpoints by expected information, and let agents publish priced offers on the toy graph and watch the
controller hand over between them.

### Day 5 — Talking to the robot, and what comes next
- Which chair do you mean? Recognition from the look stored on Day 4: a second sight as a comparison in
  which "new" is one of the candidates. Work in progress.
- The language agent: a language model that sees a digest of the graph and acts only through tools; it
  proposes plans and the controller runs them; a question is a query on a belief the robot already holds;
  imagining is the same model run on a changed copy of the world; every claim must come from a tool that
  read a belief.
- The inner voice and open directions, as a discussion: the chat turned inward on the robot's own
  surprises (proposed, not built), procedural memory, amortisation, a model of who is speaking, deciding
  when to act by expected free energy, and common sense from a vision-language model as a prior.

*Lab:* decide whether a chair is one you remember or a new one, by position and by look, and see what a
wrong variance does. Then build a toy language agent: tools that read beliefs, a check that refuses every
claim no tool returned, and a plan whose condition reads a belief when it runs.

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
