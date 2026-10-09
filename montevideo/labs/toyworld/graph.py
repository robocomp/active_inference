"""A toy shared working memory: a graph of nodes holding beliefs, linked by RT (pose) edges.

It mirrors the shape of the robot's real graph (CORTEX/DSR) without the distribution:
every node is a belief, the RT edges form a tree rooted at 'world', and agents read and write
only through this object.
"""
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from . import se2


@dataclass
class Node:
    id: int
    name: str
    type: str
    owner: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class RT:
    pose: np.ndarray                                        # child pose in the parent frame
    cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


class Graph:
    def __init__(self):
        self._ids = count()
        self.nodes = {}
        self.rt = {}        # child id -> (parent id, RT)
        self.edges = {}     # (a, b, type) -> attrs, for non-geometric relations
        self.root = self.add_node("world", "root")

    # --- nodes -------------------------------------------------------------------------------
    def add_node(self, name, type, parent=None, pose=None, cov=None, owner="", **attrs):
        node = Node(next(self._ids), name, type, owner, dict(attrs))
        self.nodes[node.id] = node
        if parent is not None:
            self.set_rt(parent, node.id, pose if pose is not None else np.zeros(3), cov)
        return node.id

    def node(self, key):
        if isinstance(key, (int, np.integer)):
            return self.nodes[key]
        return next(n for n in self.nodes.values() if n.name == key)

    def by_type(self, type):
        return [n for n in self.nodes.values() if n.type == type]

    def update(self, key, **attrs):
        self.node(key).attrs.update(attrs)

    def remove(self, key):
        """Remove a node, its subtree, and every edge touching them."""
        nid = self.node(key).id
        for child in [c for c, (p, _) in self.rt.items() if p == nid]:
            self.remove(child)
        self.rt.pop(nid, None)
        self.edges = {k: v for k, v in self.edges.items() if nid not in k[:2]}
        del self.nodes[nid]

    # --- geometry ----------------------------------------------------------------------------
    def set_rt(self, parent, child, pose, cov=None):
        p, c = self.node(parent).id, self.node(child).id
        self.rt[c] = (p, RT(np.asarray(pose, float), np.zeros((3, 3)) if cov is None else np.asarray(cov, float)))

    def parent(self, key):
        nid = self.node(key).id
        return self.rt[nid][0] if nid in self.rt else None

    def world_pose(self, key):
        """Compose RT edges from the root down to `key`."""
        nid, pose = self.node(key).id, np.zeros(3)
        chain = []
        while nid in self.rt:
            p, rt = self.rt[nid]
            chain.append(rt.pose)
            nid = p
        for link in reversed(chain):
            pose = se2.compose(pose, link)
        return pose

    def transform(self, dst, src):
        """Pose of `src` expressed in the frame of `dst`."""
        return se2.ominus(self.world_pose(src), self.world_pose(dst))

    # --- relations ---------------------------------------------------------------------------
    def add_edge(self, a, b, type, **attrs):
        self.edges[(self.node(a).id, self.node(b).id, type)] = dict(attrs)

    def __repr__(self):
        lines = []

        def walk(nid, depth):
            n = self.nodes[nid]
            lines.append("  " * depth + f"{n.name} [{n.type}]" + (f" ({n.owner})" if n.owner else ""))
            for c in sorted(c for c, (p, _) in self.rt.items() if p == nid):
                walk(c, depth + 1)

        walk(self.root, 0)
        for (a, b, t) in self.edges:
            lines.append(f"{self.nodes[a].name} --{t}--> {self.nodes[b].name}")
        return "\n".join(lines)
