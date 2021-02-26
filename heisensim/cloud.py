from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

@dataclass()
class SimpleBlockade(ABC):
    r_bl: float = 1.0
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
        "Generates a new point in the geometry. Subclasses need to override this!"

    def sample_positions(self, n):
        "Sample n positions respecting the blockade radius from the cloud's geometry. Throws RuntimeError if not converged."
        pos = np.array([self._new_pos()])
        for _ in range(n - 1):
            pos = self._append_to_pos(pos)
        return pos

    def _metric(self):
        "Return the type of metric used to compute the distance between 2 points."
        return "euclidean"

    def compute_distances(self, pos):
        "Compute the distance matrix for the given particle positions"
        return squareform(pdist(pos, self._metric()))
    
    def interaction_pairwise(self, pos, function):
        "Compute f(a,b) for all particle (assuming symmetry) and return the interation matrix"
        res = np.zeros((pos.shape[0], pos.shape[0]), dtype=np.float64)
        for i in range(pos.shape[0]):
            for j in range(i):
                res[i,j] = res[j,i] = function(*self._pairwise(pos[i], pos[j]))
        return res
    
    def _pairwise(self, a, b):
        "Allow geometries to customize the particle positions when computing pairwise interactions. Used in BoxPBC"
        return a,b


@dataclass()
class Box(SimpleBlockade):
    lengths: np.array = np.ones(3, dtype=np.float64)

    def _new_pos(self):
        return self.lengths * np.random.rand(self.lengths.size)


@dataclass
class BoxPBC(Box):
    def _find_new_point(self, pos):
        # generate all copies
        all_pos = pos
        for dim, l in enumerate(self.lengths):
            if l > 0:
                direction = np.zeros((1, self.lengths.size))
                direction[0,dim] = l
                all_pos = np.vstack((all_pos, all_pos+direction, all_pos-direction))
        return super()._find_new_point(all_pos)
    
    def _metric(self):
        def dist(a, b):
            acc = 0
            for a_coord, b_coord, l in zip(a, b, self.lengths):
                acc += min((a_coord-b_coord)**2 , (a_coord-b_coord-l)**2, (a_coord-b_coord+l)**2)
            return np.sqrt(acc)
        return dist
    
    def _pairwise(self, a, b):
        new_b = np.zeros_like(b)
        for i,(ai,bi,li) in enumerate(zip(a,b,self.lengths)):
            # choose the coord with minimal distance in each dimension
            vals = (ai-bi)**2, (ai-bi-li)**2, (ai-bi+li)**2
            coords = bi, bi+li, bi-li
            new_b[i] = coords[np.argmin(vals)]
        return a, new_b



@dataclass()
class Sphere(SimpleBlockade):
    radius: float = 1
    dim: int = 3

    def _new_pos(self):
        while True:
            pos = 1 - 2 * np.random.rand(self.dim)
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
