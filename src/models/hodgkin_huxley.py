"""
Hodgkin-Huxley neuron model implementation.

This module implements both normal and pathological (seizure) versions of the
Hodgkin-Huxley model for simulating neuronal electrical activity.

Reference:
    Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of
    membrane current and its application to conduction and excitation in nerve.
    The Journal of Physiology, 117(4), 500-544.
"""

import numpy as np


def HH_ode(t, Y, params):
    """
    Hodgkin-Huxley ODE system for a single neuron in normal conditions.

    Parameters
    ----------
    t : float
        Current time (ms)
    Y : array-like
        State vector [V, m, h, n] where:
        - V: membrane potential (mV)
        - m: sodium activation gating variable
        - h: sodium inactivation gating variable
        - n: potassium activation gating variable
    params : tuple
        Model parameters (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)

    Returns
    -------
    list
        Derivatives [dV/dt, dm/dt, dh/dt, dn/dt]
    """
    # Define the parameters
    V, m, h, n = Y
    C_m, g_Na, g_K, g_L, E_Na, E_K, E_L = params

    # Define time-varying external current
    I_ext = I_injection(t)

    # Gating variable rate functions
    def alpha_m(V):
        return 0.1*(V+40) / (1 - np.exp(-(V+40)/10)) if abs(V+40) > 1e-6 else 1.0

    def beta_m(V):
        return 4.0*np.exp(-(V+65)/18)

    def alpha_h(V):
        return 0.07*np.exp(-(V+65)/20)

    def beta_h(V):
        return 1.0/(1 + np.exp(-(V+35)/10))

    def alpha_n(V):
        return 0.01*(V+55) / (1 - np.exp(-(V+55)/10)) if abs(V+55) > 1e-6 else 0.1

    def beta_n(V):
        return 0.125*np.exp(-(V+65)/80)

    # Ionic currents
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K  = g_K  * n**4 * (V - E_K)
    I_L  = g_L  * (V - E_L)

    # HH ODEs
    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V)*(1 - m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1 - h) - beta_h(V)*h
    dndt = alpha_n(V)*(1 - n) - beta_n(V)*n

    return [dVdt, dmdt, dhdt, dndt]


def I_injection(t):
    """
    External current injection protocol.

    Parameters
    ----------
    t : float
        Current time (ms)

    Returns
    -------
    float
        Injected current (µA/cm²)
    """
    if t < 20:
        return 0.0
    elif t < 100:
        return 7.0
    else:
        return 0.0


def HH_ode_seizure(t, Y, params_normal, params_seizure):
    """
    Hodgkin-Huxley ODE system with seizure-like ion channel changes.

    This version smoothly transitions from normal to seizure state using a
    sigmoid function. Conductances change over time to simulate pathological
    activity.

    Parameters
    ----------
    t : float
        Current time (ms)
    Y : array-like
        State vector [V, m, h, n]
    params_normal : tuple
        Normal state parameters (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_ext)
    params_seizure : tuple
        Seizure state parameters (same structure as params_normal)

    Returns
    -------
    list
        Derivatives [dV/dt, dm/dt, dh/dt, dn/dt]
    """
    V, m, h, n = Y

    # Unpack normal and seizure parameters
    C_m, g_Na_norm, g_K_norm, g_L, E_Na, E_K, E_L, I_ext = params_normal
    _, g_Na_seiz, g_K_seiz, _, _, _, _, _ = params_seizure

    # Compute seizure factor for the transitions around t_onset
    s = seizure_factor(t)

    # Effective conductances
    g_Na_eff = g_Na_norm + s * (g_Na_seiz - g_Na_norm)
    g_K_eff  = g_K_norm  + s * (g_K_seiz  - g_K_norm)

    # Gating variable rate functions
    def alpha_m(V):
        return 0.1*(V+40) / (1 - np.exp(-(V+40)/10)) if abs(V+40) > 1e-6 else 1.0

    def beta_m(V):
        return 4.0*np.exp(-(V+65)/18)

    def alpha_h(V):
        return 0.07*np.exp(-(V+65)/20)

    def beta_h(V):
        return 1.0 / (1 + np.exp(-(V+35)/10))

    def alpha_n(V):
        return 0.01*(V+55) / (1 - np.exp(-(V+55)/10)) if abs(V+55) > 1e-6 else 0.1

    def beta_n(V):
        return 0.125*np.exp(-(V+65)/80)

    # Ionic currents using the effective conductances
    I_Na = g_Na_eff * (m**3) * h * (V - E_Na)
    I_K  = g_K_eff * (n**4) * (V - E_K)
    I_L  = g_L * (V - E_L)

    # ODEs
    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V)*(1 - m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1 - h) - beta_h(V)*h
    dndt = alpha_n(V)*(1 - n) - beta_n(V)*n

    return [dVdt, dmdt, dhdt, dndt]


def seizure_factor(t, t_onset=50, k=0.5):
    """
    Sigmoid function for smooth transition from normal to seizure state.

    Parameters
    ----------
    t : float
        Current time (ms)
    t_onset : float, optional
        Time at which seizure starts (default: 50 ms)
    k : float, optional
        Steepness of transition (default: 0.5)

    Returns
    -------
    float
        Value between 0 (normal) and 1 (seizure)
    """
    return 1.0 / (1.0 + np.exp(-k*(t - t_onset)))
