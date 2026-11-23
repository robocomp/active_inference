# %% [markdown]
# # Day 5 — Which chair, and a language agent
#
# - **A. Second sight.** Is this the chair I saw, or a new one? Position against appearance.
# - **B. A toy language agent.** Tools that read beliefs, a check that refuses any claim no tool
#   returned, and a plan whose condition reads a belief when it runs.
#
# About two hours. If time runs short, do A, then B up to the claim check (exercise 6).

# %%
import sys
from pathlib import Path

LABS = Path.cwd() if (Path.cwd() / "toyworld").exists() else Path.cwd().parent
sys.path.insert(0, str(LABS))

import numpy as np

# %% [markdown]
# ## A. Second sight: is this the chair I saw?
#
# The robot left a room with three chairs in it, and has come back. A chair is detected. Long-term
# memory holds what it knew about the three, and the question is the same kind of question as a
# birth: which of several worlds explains this detection in fewer nats? The worlds are *it is
# chair 0*, *chair 1*, *chair 2*, or **it is a chair I have never met**. "New" has to be on the
# list, or every detection is forced onto some remembered chair.
#
# **Where it is.** A remembered chair either stayed where it was, or was moved and could be anywhere
# in the room, with probability $\pi$. A new chair could be anywhere:
#
# $$ p(x\mid k) = (1-\pi)\,\mathcal N(x;\mu_k, 2\sigma_p^2 I) + \frac{\pi}{A},
#    \qquad p(x\mid\text{new}) = \frac1A. $$
#
# The covariance is $2\sigma_p^2$ because both the memory and the new detection carry the position
# error $\sigma_p$. This is the form the robot uses, in `common/recall_match/recall_match.h`.
#
# **What it looks like.** A view is summarised as a vector $a$ of 16 numbers. Chairs differ: each
# has its own look $m$, drawn from the class, $m\sim\mathcal N(c,\tau^2 I)$. A view of a chair
# scatters around its look, $a\sim\mathcal N(m,\sigma_a^2 I)$. So the instance and the class make
# different predictions, and their ratio is the evidence:
#
# $$ r_k = \ln\mathcal N(a; m_k,\sigma_a^2 I) - \ln\mathcal N(a; c,(\tau^2+\sigma_a^2) I). $$
#
# The score of chair $k$ is $\ln p(x\mid k) + r_k$, the score of "new" is $\ln(1/A)$, and the highest
# wins. A candidate must be strictly better than "new"; a tie goes to "new". This toy is cleaner than
# the robot: its look vectors are made up, and the robot's acting recall today uses **position only**.
#
# **Exercise 1.** Write `log_location` and `decide`. **Exercise 2.** Write `log_app_ratio`.

# %%
ROOM_W, ROOM_H, ROOM_A = 6.0, 4.0, 24.0
DIM, TAU, SIG_A, SIG_P, PI_MOVE = 16, 1.0, 0.5, 0.15, 0.5
CLASS_LOOK = np.zeros(DIM)
LOW, HIGH = np.array([0.5, 0.5]), np.array([ROOM_W - 0.5, ROOM_H - 0.5])


def log_gauss_iso(x, mu, var):
    return -0.5 * (np.sum((x - mu) ** 2) / var + len(x) * np.log(2 * np.pi * var))


def log_location(x, mu):
    """ln p(x | k): stayed near mu, or moved and uniform over the room."""
    # BEGIN SOLUTION
    stayed = np.log(1 - PI_MOVE) + log_gauss_iso(x, mu, 2 * SIG_P ** 2)
    moved = np.log(PI_MOVE) - np.log(ROOM_A)
    return np.logaddexp(stayed, moved)
    # END SOLUTION


def log_app_ratio(a, m):
    """r_k: nats by which this chair's look explains the view better than the class does."""
    # BEGIN SOLUTION
    return log_gauss_iso(a, m, SIG_A ** 2) - log_gauss_iso(a, CLASS_LOOK, TAU ** 2 + SIG_A ** 2)
    # END SOLUTION


def decide(x, a, mus, looks, use_look=True, allow_new=True):
    """Index of the winning chair, or len(mus) for 'new'."""
    # BEGIN SOLUTION
    score = np.array([log_location(x, mu) + (log_app_ratio(a, m) if use_look else 0.0) for mu, m in zip(mus, looks)])
    best = int(np.argmax(score))
    if not allow_new or score[best] > -np.log(ROOM_A):
        return best
    return len(mus)
    # END SOLUTION


# %% [markdown]
# **Exercise 3.** Run the table below and predict each number before you read it. There are three
# situations (the chair stayed, the chair was moved, the chair is new), two kinds of room (every
# chair looks different, or all three are the same model) and three rules (always take the nearest
# candidate; position only; position and look).

# %%
def trial(r_, case, twins, use_look, allow_new, inflate=1.0):
    mus = r_.uniform(LOW, HIGH, (3, 2))
    looks = (np.tile(CLASS_LOOK + TAU * r_.standard_normal(DIM), (3, 1)) if twins
             else CLASS_LOOK + TAU * r_.standard_normal((3, DIM)))
    k = int(r_.integers(3))
    if case == "new":
        where, look, truth = r_.uniform(LOW, HIGH), (looks[0] if twins else CLASS_LOOK + TAU * r_.standard_normal(DIM)), 3
    else:
        where, look, truth = (mus[k] if case == "stayed" else r_.uniform(LOW, HIGH)), looks[k], k
    x = where + SIG_P * r_.standard_normal(2)
    a = look + inflate * SIG_A * r_.standard_normal(DIM)         # inflate: this view is noisier than the model assumes
    return decide(x, a, mus, looks, use_look, allow_new) == truth


def rate(case, twins, use_look, allow_new, inflate=1.0, n=4000):
    r_ = np.random.default_rng(0)
    return np.mean([trial(r_, case, twins, use_look, allow_new, inflate) for _ in range(n)])


for twins in (False, True):
    print("three chairs of the same model" if twins else "three chairs that look different")
    for case in ("stayed", "moved", "new"):
        print(f"  {case:7s} nearest {rate(case, twins, False, False):5.1%}   position {rate(case, twins, False, True):5.1%}"
              f"   position + look {rate(case, twins, True, True):5.1%}")

# %% [markdown]
# Read the table against the equations. On position alone, a chair that was moved scores
# $\ln(\pi/A)$ and "new" scores $\ln(1/A)$, so with $\pi<1$ it **cannot** win unless it happens to land
# near a remembered place: it is born again under a new name. A new chair set down near a remembered
# place takes that chair's identity. Only the look can tell those apart, and only if chairs differ:
# three chairs of one model give it nothing to work with, and a new chair of that model is confidently
# mistaken for an old one.
#
# **Exercise 4.** A chair seen from the other side does not look like the view in memory. Model that
# crudely: the true scatter of the view is `inflate` times what the model assumes. Run `inflate` from
# 1 to 3, in the room where the chairs look different, and compare with the position-only row above.

# %%
for f in (1.0, 1.5, 2.0, 3.0):
    print(f"view noise x{f:.1f}:  stayed {rate('stayed', False, True, True, f):5.1%}   moved {rate('moved', False, True, True, f):5.1%}"
          f"   new {rate('new', False, True, True, f):5.1%}")

# %% [markdown]
# **Lesson.** Evidence that is too sure of itself vetoes the right answer. At twice the assumed scatter
# the look throws away chairs that never moved, and position alone would have kept them. This is why
# an instance's look has to be stored **per viewing direction**, compared only with views from a
# similar direction, and given a variance that matches its real scatter before it is allowed to count. The class and the
# instance are both in the ratio $r_k$: the instance predicts the view, and the class is what a
# stranger would have looked like.

# %% [markdown]
# ## B. A toy language agent
#
# On the robot a language model talks to the person, but it never holds a belief of its own. It sees a
# short text digest of the graph and acts through **tools**: each tool reads one belief, or posts one
# intention. Then a check runs on its reply: every claim must match something a tool returned in this
# turn. If it does not, the claim is not kept.
#
# Here the model is a script, so the lab needs no network and gives the same answer every time.
# The rules being tested are the same ones the robot uses.
#
# ### B1. The graph the agent reads
#
# Two rooms joined by two doors, a robot, and three chairs. Every number on a node is a belief published
# by its owner: a door agent's $P(\text{open})$, a chair agent's pose and its standard deviation.

# %%
from toyworld import Graph

def make_graph():
    g = Graph()
    living = g.add_node("living", "room", parent=g.root, pose=np.array([0.0, 0.0, 0.0]))
    kitchen = g.add_node("kitchen", "room", parent=g.root, pose=np.array([6.0, 0.0, 0.0]))
    g.add_node("door_1", "door", parent=living, pose=np.array([6.0, 1.0, 0.0]), owner="door_concept", p_open=0.92)
    g.add_node("door_2", "door", parent=living, pose=np.array([6.0, 3.2, 0.0]), owner="door_concept", p_open=0.15)
    g.add_node("robot", "robot", parent=living, pose=np.array([1.0, 0.8, 0.0]), owner="room_concept")
    g.add_node("chair_22", "chair", parent=living, pose=np.array([2.5, 2.0, 0.3]), owner="chair_concept", sd=0.04, p_exists=0.97)
    g.add_node("chair_23", "chair", parent=kitchen, pose=np.array([1.5, 1.2, 1.6]), owner="chair_concept", sd=0.09, p_exists=0.88)
    g.add_node("chair_34", "chair", parent=kitchen, pose=np.array([3.1, 2.4, 0.0]), owner="chair_concept", sd=0.30, p_exists=0.41)
    return g

graph = make_graph()
print(graph)

# %% [markdown]
# ### B2. Tools
#
# A tool returns a small record: which node it read, which attribute, and the value. The record is what
# the claim check will compare against, so the source travels with the number.
#
# **Exercise 5.** Write the three tools:
# - `where_is(name)`: the room a node hangs from and its pose in that room, attribute `"room"`;
# - `door_open(name)`: the door agent's $P(\text{open})$, attribute `"p_open"`;
# - `find(cls)`: the names of every node of that type whose existence belief is above one half,
#   attribute `"found"`. A chair at $P(\text{exists}) = 0.41$ is not reported as seen.

# %%
def record(node, attr, value):
    return {"node": node, "attr": attr, "value": value}


def where_is(name):
    # BEGIN SOLUTION
    parent = graph.parent(name)
    pose = graph.rt[graph.node(name).id][1].pose
    return record(name, "room", (graph.node(parent).name, np.round(pose[:2], 2).tolist()))
    # END SOLUTION


def door_open(name):
    # BEGIN SOLUTION
    return record(name, "p_open", graph.node(name).attrs["p_open"])
    # END SOLUTION


def find(cls):
    # BEGIN SOLUTION
    seen = [n.name for n in graph.by_type(cls) if n.attrs.get("p_exists", 1.0) > 0.5]
    return record(cls, "found", seen)
    # END SOLUTION


TOOLS = {"where_is": where_is, "door_open": door_open, "find": find}
print(where_is("chair_23"), door_open("door_2"), find("chair"), sep="\n")

# %% [markdown]
# ### B3. Two scripted models, and the check
#
# A model's reply here is a sentence plus the **claims** in it, as data: `(node, attribute, value)`.
# A real system has to pull the claims out of free text; the robot does it with the tool records it
# already holds. The careful model calls a tool before each claim. The sloppy one sometimes answers
# from what it "remembers", which is exactly what a language model does when no tool ran.
#
# **Exercise 6.** Write `unbacked(claims, transcript)`: the claims that no tool record in this turn's
# transcript supports. A record supports a claim when node and attribute match and the value is the
# same (numbers to within 0.005).

# %%
def careful_model(request, call):
    request = request.lower()
    if "open" in request:
        door = request.split()[1].rstrip("?")
        r = call("door_open", door)
        word = "probably open" if r["value"] > 0.5 else "probably closed"
        return f"{door} is {word} (P = {r['value']:.2f}).", [(door, "p_open", r["value"])]
    if "where" in request:
        name = request.split()[-1].rstrip("?")
        r = call("where_is", name)
        return f"{name} is in the {r['value'][0]}.", [(name, "room", r["value"])]
    if "chairs" in request:
        r = call("find", "chair")
        return f"I see {len(r['value'])} chairs: {', '.join(r['value'])}.", [("chair", "found", r["value"])]
    return "I do not know.", []


def sloppy_model(request, call):
    request = request.lower()
    if "open" in request:                                   # answers without calling the door agent
        door = request.split()[1].rstrip("?")
        return f"Yes, {door} is open.", [(door, "p_open", 1.0)]
    if "chairs" in request:                                  # counts every chair node, believed or not
        names = [n.name for n in graph.by_type("chair")]
        return f"I see {len(names)} chairs.", [("chair", "found", names)]
    return careful_model(request, call)


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 0.005
    return a == b


def unbacked(claims, transcript):
    # BEGIN SOLUTION
    return [c for c in claims
            if not any(r["node"] == c[0] and r["attr"] == c[1] and same(r["value"], c[2]) for r in transcript)]
    # END SOLUTION


def turn(model, request):
    transcript = []
    def call(tool, arg):
        r = TOOLS[tool](arg)
        transcript.append(r)
        return r
    text, claims = model(request, call)
    bad = unbacked(claims, transcript)
    if bad:
        text = f"[not kept: {len(bad)} claim(s) no tool supports] " + text
    return text, bad


for request in ("Is door_2 open?", "Where is chair_23?", "How many chairs do you see?"):
    for model in (careful_model, sloppy_model):
        text, bad = turn(model, request)
        print(f"{model.__name__:14s} {request:28s} -> {text}")

# %% [markdown]
# **Questions.**
# - The sloppy model's door answer is wrong, and its chair answer is "right" in a sense: three chair
#   nodes exist. Why is the second one also refused?
# - What should the agent keep in its history after refusing a claim, and why does it matter for the
#   next turn?

# %% [markdown] solution
# *Answers.*
# - Because no tool said it. `find` reports chairs the robot believes in; chair_34 is at
#   $P(\text{exists}) = 0.41$. The node is on the graph, but the belief says it is probably not there.
#   Counting nodes is reading the graph's bookkeeping, not the robot's belief.
# - A sentence saying that nothing was checked, not the false sentence. A model copies its own history:
#   a false claim left in it comes back in the next answer. The robot keeps "nothing was planned or sent,
#   and the robot did not move" in its place.

# %% [markdown]
# ### B4. A plan whose condition reads a belief
#
# The model writes plans; it does not drive. A plan is data the controller runs, and a condition in it
# reads a belief **when the step runs**, not when the plan was written.
#
# **Exercise 7.** Write `run(plan)`. Steps are `("go", target)`, or
# `("if", (door, "<", level), then_steps, else_steps)`, where the condition compares the door's current
# $P(\text{open})$ with `level`. Return the list of targets the robot drives to. Run the plan, let the
# door agent revise door_2 to 0.85, and run the same plan again.

# %%
plan = [("if", ("door_2", "<", 0.5),
         [("go", "door_1"), ("go", "kitchen")],      # door_2 probably closed: go round by door_1
         [("go", "door_2"), ("go", "kitchen")])]     # door_2 probably open: the short way


def run(plan):
    # BEGIN SOLUTION
    route = []
    for step in plan:
        if step[0] == "go":
            route.append(step[1])
        else:
            door, _, level = step[1]
            branch = step[2] if door_open(door)["value"] < level else step[3]
            route += run(branch)
    return route
    # END SOLUTION


print("door_2 at 0.15:", run(plan))
graph.update("door_2", p_open=0.85)              # the door agent saw the leaf swing open
print("door_2 at 0.85:", run(plan))
graph = make_graph()

# %% [markdown]
# **Lesson.** The language model chose which beliefs to read and wrote the plan. Every number came from
# a belief, and the check refused what no tool returned. The rule "say only what a tool returned" is
# something you can test.
#
# ### B5. Optional: a real local model
#
# If a local model server is running (for example `ollama serve` with a small model pulled), the cell
# below asks it to answer one question using only a JSON tool request, runs the tool, and applies the
# same check. Without a server it prints a note and does nothing.

# %%
import json
import urllib.request

def local_model(request, call, model="qwen2.5:3b", url="http://localhost:11434/api/chat"):
    system = ('You answer questions about a robot. Reply ONLY with JSON: {"tool": name, "arg": string}. '
              f"Tools: {list(TOOLS)}.")
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": request}]})
    req = urllib.request.Request(url, body.encode(), {"Content-Type": "application/json"})
    ask = json.loads(json.loads(urllib.request.urlopen(req, timeout=60).read())["message"]["content"])
    r = call(ask["tool"], ask["arg"])
    return f"{r['node']}: {r['attr']} = {r['value']}", [(r["node"], r["attr"], r["value"])]

try:
    print(turn(local_model, "Is door_1 open?"))
except Exception as e:  # no server, no model, or a reply that is not the JSON asked for
    print("no local model available:", type(e).__name__)
