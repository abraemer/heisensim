from scipy.optimize import minimize_scalar
import numpy as np


def weights_canonical(beta, ev):
    if beta < 0:
        ev = -ev
        beta = -beta
    # normalize energys - shift such that smallest energy is 0
    # so there are no negative evals that may blow up
    minimum = np.min(ev)
    ev = ev - minimum
    weights = np.exp(-beta * ev)
    return weights / weights.sum()


def energy_diff(beta, ev, E_0):
    weights = weights_canonical(beta, ev)
    return np.abs(weights @ ev - E_0)


def canonical_ensemble(ev, E_0, beta_0=0):
    beta = minimize_scalar(energy_diff, args=(np.array(ev), E_0))
    return weights_canonical(beta.x, ev)


def micro_ensemble(ev, E_0, delta_E=10):
    micro = (ev < E_0 + delta_E) & (ev > E_0 - delta_E)
    if micro.sum() > 0:
        return micro / micro.sum()
    else:
        micro = 0 * ev
        micro[np.argmin(np.array((ev - E_0)**2))] = 1
        return micro


def diagonal_ensemble(eon):
    return eon
