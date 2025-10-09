"""
Wendling Neural Mass Model implementation.

This module implements the Wendling et al. (2002) neural mass model for
simulating local field potentials in epileptic brain tissue.

Reference:
    Wendling, F., Bartolomei, F., Bellanger, J. J., & Chauvel, P. (2002).
    Epileptic fast activity can be explained by a model of impaired GABAergic
    dendritic inhibition. European Journal of Neuroscience, 15(9), 1499-1508.
"""

import numpy as np


def WendlingNMM(y, h, P):
    """
    Wendling neural mass model differential equations.

    This model simulates interactions between three neuronal populations:
    pyramidal cells, excitatory interneurons, and slow/fast inhibitory
    interneurons.

    Parameters
    ----------
    y : array-like
        State vector of length 10 containing membrane potentials and
        their derivatives for different neuronal populations
    h : float
        Time step size (s)
    P : dict
        Dictionary of model parameters including:
        - A, B, G: synaptic gains
        - a, b, g: inverse time constants
        - C1-C7: connectivity constants
        - e0, v0, r: sigmoid parameters
        - meanP, sigmaP: noise parameters

    Returns
    -------
    numpy.ndarray
        Updated state vector after one time step
    """
    noise = np.random.normal(P["meanP"], P["sigmaP"])

    dydx = np.zeros(10)

    dydx[0] = y[5]
    dydx[5] = P["A"] * P["a"] * sigm(y[1]-y[2]-y[3], P) - 2. * P["a"] * y[5] - P["a"] * P["a"] * y[0]
    dydx[1] = y[6]
    dydx[6] = P["A"] * P["a"] * (noise + P["C2"] * sigm(P["C1"]* y[0] + P["SG"], P)) - 2. * P["a"] * y[6] - P["a"]*P["a"] * y[1]
    dydx[2] = y[7]
    dydx[7] = P["B"] * P["b"] * (P["C4"] * sigm(P["C3"] * y[0], P)) - 2. * P["b"] * y[7] - P["b"]*P["b"] * y[2]
    dydx[3] = y[8]
    dydx[8] = P["G"] * P["g"] * (P["C7"] * sigm((P["C5"] * y[0] - P["C6"] * y[4]), P)) - 2. * P["g"] * y[8] - P["g"]*P["g"] * y[3]
    dydx[4] = y[9]
    dydx[9] = P["B"] * P["b"] * (sigm(P["C3"] * y[0], P)) - 2. * P["b"] * y[9] - P["b"]*P["b"] * y[4]

    yout = np.empty((0))

    for i in range(10):
        yout = np.append(yout, y[i] + h * dydx[i])

    return yout


def sigm(v, P):
    """
    Sigmoidal transfer function.

    Converts average membrane potential to average firing rate using a
    sigmoid function.

    Parameters
    ----------
    v : float
        Membrane potential
    P : dict
        Parameter dictionary containing e0, r, and v0

    Returns
    -------
    float
        Firing rate
    """
    return 2. * P["e0"] / (1. + np.exp(P["r"] * (P["v0"] - v)))
