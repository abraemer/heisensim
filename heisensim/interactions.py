from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


@dataclass()
class InteractionParams(ABC):
    normalization: Any = None

    def get_interaction(self, geometry, positions, *args, **kwargs):
        int_mat = self._get_interaction(geometry, positions, *args, **kwargs)
        return self.normalize(int_mat, self.normalization)

    @abstractmethod
    def _get_interaction(self, geometry, positions, *args, **kwargs):
        pass

    @staticmethod
    def normalize(int_mat, normalization):
        if normalization is None:
            return int_mat

        mf = int_mat.sum(axis=1)
        if normalization == 'mean':
            int_mat /= np.mean(mf)
        elif normalization == 'median':
            int_mat /= np.median(mf)
        return int_mat


@dataclass()
class PowerLaw(InteractionParams):
    exponent: float = 6
    coupling: float = 1

    def _get_interaction(self, geometry, positions, *args, **kwargs):
        coupling = self.coupling
        exponent = self.exponent

        distance_matrix = geometry.compute_distances(positions)
        np.fill_diagonal(distance_matrix, 1)
        interaction = coupling * distance_matrix**(-exponent)
        np.fill_diagonal(interaction, 0)
        return interaction


class VanDerWaals(PowerLaw):
    def __init__(self, coupling=1, normalization=None):
        self.coupling = coupling
        self.exponent = 6
        self.normalization = normalization


class DipoleCoupling(PowerLaw):
    def __init__(self, coupling, normalization=None):
        self.coupling = coupling
        self.exponent = 3
        self.normalization = normalization

    def dipole_interaction(self, u, v):
        d = u - v
        dist = np.linalg.norm(d)
        return self.coupling * (1 - 3 * (d[2] / dist) ** 2) / dist ** self.exponent

    # noinspection PyTypeChecker
    def _get_interaction(self, geometry, positions, *args, **kwargs):
        return geometry.interaction_pairwise(positions, self.dipole_interaction)
