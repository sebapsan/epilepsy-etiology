"""
Parameter definitions for neural models.

This module contains standard parameter sets for the Hodgkin-Huxley and
Wendling neural mass models.
"""


def get_hh_params():
    """
    Get standard Hodgkin-Huxley model parameters.

    Returns
    -------
    dict
        Dictionary containing:
        - params: tuple of model parameters
        - Y0: initial conditions
        - t_start, t_end: simulation time bounds
    """
    # Constants (as in Hodgkin & Huxley 1952)
    C_m  = 1.0     # µF/cm²
    g_Na = 120.0   # mS/cm²
    g_K  = 36.0    # mS/cm²
    g_L  = 0.3     # mS/cm²
    E_Na = 50.0    # mV
    E_K  = -77.0   # mV
    E_L  = -54.387 # mV

    params = (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)

    # Initial conditions [V, m, h, n]
    Y0 = [-65.0, 0.05, 0.60, 0.32]

    return {
        'params': params,
        'Y0': Y0,
        't_start': 0.0,
        't_end': 120.0
    }


def get_hh_seizure_params():
    """
    Get Hodgkin-Huxley parameters for normal and seizure states.

    Returns
    -------
    dict
        Dictionary containing normal and seizure parameter sets
    """
    params_normal = (1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.387, 10.0)
    params_seizure = (1.0, 180.0, 20.0, 0.3, 50.0, -77.0, -54.387, 10.0)

    Y0 = [-65.0, 0.05, 0.60, 0.32]

    return {
        'params_normal': params_normal,
        'params_seizure': params_seizure,
        'Y0': Y0,
        't_start': 0.0,
        't_end': 120.0
    }


def get_wendling_params(A=4, B=15, G=22):
    """
    Get Wendling neural mass model parameters.

    Parameters
    ----------
    A : float, optional
        Excitatory synaptic gain (default: 4)
    B : float, optional
        Slow inhibitory synaptic gain (default: 15)
    G : float, optional
        Fast inhibitory synaptic gain (default: 22)

    Returns
    -------
    dict
        Complete parameter dictionary for Wendling model
    """
    P = {
        "A": A, "B": B, "G": G,
        "a": 100., "b": 50., "g": 500.,
        "v0": 6., "e0": 2.5, "r": 0.56,
        "C1": 1., "C2": 0.8, "C3": 0.25, "C4": 0.25,
        "C5": 0.3, "C6": 0.1, "C7": 0.8, "C": 135.
    }

    # Scale connectivity constants
    P["C1"] *= P["C"]
    P["C2"] *= P["C"]
    P["C3"] *= P["C"]
    P["C4"] *= P["C"]
    P["C5"] *= P["C"]
    P["C6"] *= P["C"]
    P["C7"] *= P["C"]

    # Noise parameters
    P["SG"] = 1.        # stimulus (synaptic) gain
    P["meanP"] = 90.    # input noise mean
    P["sigmaP"] = 60.   # input noise standard deviation

    return P


def get_seizure_configs():
    """
    Get parameter configurations for different epileptic states.

    Returns a list of 6 configurations representing different types of
    neuronal activity from background to ictal (seizure) states.

    Returns
    -------
    list
        List of dictionaries with A, B, G parameters for each state
    """
    configs = [
        {"A": 3,    "B": 26,   "G": 10},  # Background activity
        {"A": 4,    "B": 26,   "G": 10},  # Sporadic spikes
        {"A": 5,    "B": 22,   "G": 10},  # Rhythmic spikes
        {"A": 4,    "B": 15,   "G": 22},  # Slow quasi-sinusoidal
        {"A": 6.5,  "B": 10,   "G": 22},  # Low voltage rapid activity
        {"A": 7,    "B": 18,   "G": 2}    # Fast activity (ictal)
    ]
    return configs


def get_concatenated_seizure_configs():
    """
    Get smooth transition configurations for realistic seizure simulation.

    These configurations create smooth transitions between different seizure
    states to simulate a complete epileptic event.

    Returns
    -------
    list
        List of dictionaries with A, B, G parameters for smooth transitions
    """
    configs = [
        {"A": 3,    "B": 26, "G": 10},   # Pre-ictal
        {"A": 3.5,  "B": 22, "G": 10},   # Early transition
        {"A": 4.5,  "B": 22, "G": 8},    # Mid transition
        {"A": 5,    "B": 15, "G": 12},   # Late transition
        {"A": 6.5,  "B": 8,  "G": 22},   # Ictal onset
        {"A": 7,    "B": 18, "G": 3}     # Full ictal
    ]
    return configs
