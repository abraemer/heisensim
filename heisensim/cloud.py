import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass()
class SimpleBlockade(ABC):
    r_bl: float = 0.1
    max_iter: int = 1000

    def _append_to_pos(self, pos):
        tree = cKDTree(pos)
        i = 0
        while True:
            new_pos = self._new_pos()
            in_range = tree.query_ball_point(new_pos, self.r_bl)
            if not in_range:
                return np.append(pos, [new_pos], axis=0)
            i += 1
            if i > self.max_iter:
                raise RuntimeError("The system didn't reach the required size.")

    @abstractmethod
    def _new_pos(self):
        pass

    def sample_positions(self, n):
        pos = np.array([self._new_pos()])
        for i in range(n - 1):
            pos = self._append_to_pos(pos)
        return pos


@dataclass()
class Box(SimpleBlockade):
    length_x = 1
    length_y = 1
    length_z = 1

    def _new_pos(self):
        return [self.length_x, self.length_y, self.length_z] * np.random.rand(3)


@dataclass()
class Sphere(SimpleBlockade):
    radius: float = 1

    def _new_pos(self):
        while True:
            pos = 1 - 2 * np.random.rand(3)
            if np.linalg.norm(pos) < 1:
                return self.radius * pos
