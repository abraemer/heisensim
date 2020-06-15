from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
from abc import ABC, abstractmethod
from heisensim.spin_half import *


@dataclass()
class InteractionParams(ABC):
    @abstractmethod
    def get_interaction(self, *args):
        pass


@dataclass()
class PowerLaw(InteractionParams):
    exponent: float = 6
    coupling: float = 1

    def get_interaction(self, pos):
        coupling = self.coupling
        exponent = self.exponent

        dist = squareform(pdist(pos))
        np.fill_diagonal(dist, 1)
        interaction = coupling * dist**(-exponent)
        np.fill_diagonal(interaction, 0)
        return interaction


class VanDerWaals(PowerLaw):
    def __init__(self, coupling=1):
        self.coupling = coupling
        self.exponent = 6


class DipoleCoupling(PowerLaw):
    def __init__(self, coupling):
        self.coupling = coupling
        self.exponent = 3


class SpinModel:
    def __init__(self, int_mat, spin_spin_terms=(1, 1, -0.6), symmetric=False):
        self.int_mat = int_mat
        self.symmetric = symmetric
        self.spin_spin_terms = spin_spin_terms

    @property
    def int_mat_mf(self):
        return self.int_mat.sum(axis=1)

    @property
    def J_mean(self):
        return np.mean(self.int_mat_mf)

    @property
    def J_median(self):
        return np.median(self.int_mat_mf)

    @property
    def N(self):
        return self.int_mat.shape[0]

    def hamiltonian(self, hx=0, hy=0, hz=0):
        N = self.N

        sx_list = self.get_op_list(sx)
        sy_list = self.get_op_list(sy)
        sz_list = self.get_op_list(sz)

        alpha, beta, gamma = self.spin_spin_terms

        # Gamma = decay_to_container(self.gamma_d, self.gamma_u, self.N)
        # c_op_list = collapse_operators(self.gamma_ud, N)

        # external field terms
        H_field = self.hamiltonian_field(hx, hy, hz)

        # interaction terms
        H_int = 0
        for i in range(N):
            for j in range(i):
                H_int += self.int_mat[i, j] * (
                        alpha * sx_list[i] * sx_list[j]
                        + beta * sy_list[i] * sy_list[j]
                        + gamma * sz_list[i] * sz_list[j])
        # H_int = self.symmetrize_op(H_int)

        return H_int + H_field

    def hamiltonian_field(self, hx=0, hy=0, hz=0):
        N = self.N
        sx_list = self.get_op_list(sx)
        sy_list = self.get_op_list(sy)
        sz_list = self.get_op_list(sz)

        hx = np.resize(hx, N)
        hy = np.resize(hy, N)
        hz = np.resize(hz, N)

        H = sum(hx[i] * sx_list[i] + hy[i] * sy_list[i] + hz[i] * sz_list[i] for i in range(N))
        return H  # self.symmetrize_op(H)

    def single_spin_op(self, op, n):
        N = self.N
        op = single_spin_op(op, n, N)
        return self.symmetrize_op(op)

    def get_op_list(self, op):
        N = self.N
        return [self.single_spin_op(op, n) for n in range(N)]

    def product_state(self, state=up_x):
        N = self.N
        psi0 = state.unit()
        psi = qt.tensor(N * [psi0])
        return self.symmetrize_state(psi)

    def symmetrize_state(self, state):
        if self.symmetric:
            return symmetrize_state(state)
        return state

    def symmetrize_op(self, op):
        if self.symmetric:
            return symmetrize_op(op)
        return op


class XXZ(SpinModel):
    def __init__(self, pos, int_params=VanDerWaals(), delta=-0.6, normalize=None, symmetric=True):
        self.pos = pos
        self.int_params = int_params
        spin_spin_terms = (1, 1, delta)
        int_mat = self.int_params.get_interaction(self.pos)
        if normalize:
            mf = int_mat.sum(axis=1)
            if normalize is 'mean':
                int_mat /= np.mean(mf)
            elif normalize is 'median':
                int_mat /= np.median(mf)
        super().__init__(int_mat, spin_spin_terms, symmetric)
