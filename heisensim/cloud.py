import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass()
class SimpleBlockade(ABC):
    r_bl: float = 0.1
    max_iter: int = 1000

    def _append_to_pos(self, pos):
        return np.append(pos, [self._find_new_point(pos)], axis=0)
    
    def _find_new_point(self, pos):
        tree = cKDTree(pos)
        i = 0
        while True:
            new_pos = self._new_pos()
            in_range = tree.query_ball_point(new_pos, self.r_bl)
            if not in_range:
                return new_pos
            i += 1
            if i > self.max_iter:
                raise RuntimeError(f"The system didn't reach the required size. Obj: {self!r}")

    @abstractmethod
    def _new_pos(self):
        pass

    def sample_positions(self, n):
        pos = np.array([self._new_pos()])
        for _ in range(n - 1):
            pos = self._append_to_pos(pos)
        return pos


@dataclass()
class Box(SimpleBlockade):
    length_x: float = 1
    length_y: float = 1
    length_z: float = 1

    def _new_pos(self):
        return [self.length_x, self.length_y, self.length_z] * np.random.rand(3)

@dataclass
class BoxPBC(Box):
    def _find_new_point(self, pos):
        # generate all copies
        all_pos = pos
        if self.length_x > self.r_bl:
            all_pos = np.vstack([all_pos + [[a*self.length_x,0,0]] for a in (-1,0,1)])
        if self.length_y > self.r_bl:
            all_pos = np.vstack([all_pos + [[0,b*self.length_y,0]] for b in (-1,0,1)])
        if self.length_z > self.r_bl:
            all_pos = np.vstack([all_pos + [[0,0,c*self.length_z]] for c in (-1,0,1)])
        return super()._find_new_point(all_pos)




@dataclass()
class Sphere(SimpleBlockade):
    radius: float = 1

    def _new_pos(self):
        while True:
            pos = 1 - 2 * np.random.rand(3)
            if np.linalg.norm(pos) < 1:
                return self.radius * pos


@dataclass
class RandomChain(SimpleBlockade):
    length: float = 1
    max_iter: int = 1000

    def _new_pos(self):
        return [0, 0, self.length * np.random.rand()]

    def sample_positions(self, n):
        pos = np.array([[0, 0, 0], [0, 0, self.length]])
        for _ in range(n - 2):
            pos = self._append_to_pos(pos)
        return pos
