"""The toy apartment: rectilinear rooms joined by doors, with a few objects.

This is the ground truth. The robot never reads it directly; it only sees it through
`lidar.Lidar.scan` and its own odometry.
"""
from dataclasses import dataclass, field

import numpy as np

from . import se2
from .geometry import point_in_polygon, segments_of


@dataclass
class Room:
    name: str
    polygon: np.ndarray  # (N, 2), counter-clockwise

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, float)


@dataclass
class Door:
    name: str
    rooms: tuple  # (room_a, room_b)
    p0: np.ndarray  # endpoints of the gap, on the shared wall
    p1: np.ndarray

    def __post_init__(self):
        self.p0, self.p1 = np.asarray(self.p0, float), np.asarray(self.p1, float)

    @property
    def centre(self):
        return 0.5 * (self.p0 + self.p1)

    @property
    def width(self):
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class Thing:
    """An object in a room. kind 'box' has size (w, d); kind 'disc' has size (r,)."""
    name: str
    kind: str
    pose: np.ndarray  # (x, y, yaw) in the world frame
    size: tuple

    def __post_init__(self):
        self.pose = np.asarray(self.pose, float)

    def outline(self):
        if self.kind != "box":
            raise ValueError("only boxes have an outline")
        w, d = self.size
        corners = np.array([[-w, -d], [w, -d], [w, d], [-w, d]]) / 2
        return se2.transform_points(self.pose, corners)


def _cut(seg, gap_p0, gap_p1, tol=1e-9):
    """Remove the gap [gap_p0, gap_p1] from segment seg if the gap lies on it."""
    a, b = seg
    ab = b - a
    length2 = ab @ ab
    pieces = []
    for p in (gap_p0, gap_p1):
        cross = ab[0] * (p - a)[1] - ab[1] * (p - a)[0]
        if abs(cross) > tol * max(1.0, length2):
            return [seg]
    t0, t1 = sorted(((gap_p0 - a) @ ab / length2, (gap_p1 - a) @ ab / length2))
    if t1 <= 0 or t0 >= 1:
        return [seg]
    if t0 > 0:
        pieces.append(np.array([a, a + t0 * ab]))
    if t1 < 1:
        pieces.append(np.array([a + t1 * ab, b]))
    return pieces


@dataclass
class World:
    rooms: list
    doors: list = field(default_factory=list)
    things: list = field(default_factory=list)

    def room(self, name):
        return next(r for r in self.rooms if r.name == name)

    def room_walls(self, name):
        """Wall segments (M, 2, 2) of one room, with its door gaps cut out."""
        segs = list(segments_of(self.room(name).polygon))
        for door in self.doors:
            if name in door.rooms:
                segs = [piece for s in segs for piece in _cut(s, door.p0, door.p1)]
        return np.array(segs)

    def segments(self):
        """Every reflecting segment and its label: walls, then box objects."""
        segs, labels = [], []
        for room in self.rooms:
            walls = self.room_walls(room.name)
            segs.extend(walls)
            labels.extend(["wall"] * len(walls))
        for thing in self.things:
            if thing.kind == "box":
                edges = segments_of(thing.outline())
                segs.extend(edges)
                labels.extend([thing.name] * len(edges))
        return np.array(segs), np.array(labels)

    def discs(self):
        """Circular objects: centres (K, 2), radii (K,), labels (K,)."""
        ds = [t for t in self.things if t.kind == "disc"]
        if not ds:
            return np.zeros((0, 2)), np.zeros(0), np.array([], dtype=str)
        return (np.array([t.pose[:2] for t in ds]), np.array([t.size[0] for t in ds]),
                np.array([t.name for t in ds]))

    def room_of(self, point):
        """Name of the room containing `point`, or None (e.g. standing in a doorway)."""
        for room in self.rooms:
            if point_in_polygon(point, room.polygon)[0]:
                return room.name
        return None


def two_room_apartment():
    """The course's standard world: a 6x4 m living room and a 4x5 m bedroom joined by a door.

    The rooms are deliberately not squares and not equal, so that each has a discrete (180°)
    symmetry but the apartment as a whole does not.
    """
    living = Room("living", [[0, 0], [6, 0], [6, 4], [0, 4]])
    bedroom = Room("bedroom", [[6, 0], [10, 0], [10, 5], [6, 5]])
    door = Door("door", ("living", "bedroom"), [6, 1.6], [6, 2.5])
    things = [
        Thing("table", "box", [2.2, 2.6, 0.0], (1.2, 0.8)),
        Thing("bottle", "disc", [4.6, 0.9, 0.0], (0.05,)),
        Thing("chair", "disc", [8.4, 3.6, 0.0], (0.25,)),
    ]
    return World([living, bedroom], [door], things)
