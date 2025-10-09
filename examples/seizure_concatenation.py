"""
Complete seizure event simulation with smooth state transitions.

This script simulates a realistic epileptic seizure by smoothly transitioning
through different neuronal states, creating a continuous 30-second recording.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.wendling import WendlingNMM
from src.utils.parameters import get_wendling_params, get_concatenated_seizure_configs


def get_current_params(t, configs):
    """
    Get current parameters based on time with smooth transitions.

    Transition windows:
    - [4,6]:   config0 -> config1
    - [9,11]:  config1 -> config2
    - [14,16]: config2 -> config3
    - [19,21]: config3 -> config4
    - [24,26]: config4 -> config5

    Parameters
    ----------
    t : float
        Current time (seconds)
    configs : list
        List of configuration dictionaries

    Returns
    -------
    dict
        Current A, B, G parameters
    """
    if t < 4:
        return configs[0]
    elif t < 6:
        f = (t - 4) / 2.0
        return {
            "A": configs[0]["A"] * (1 - f) + configs[1]["A"] * f,
            "B": configs[0]["B"] * (1 - f) + configs[1]["B"] * f,
            "G": configs[0]["G"] * (1 - f) + configs[1]["G"] * f
        }
    elif t < 9:
        return configs[1]
    elif t < 11:
        f = (t - 9) / 2.0
        return {
            "A": configs[1]["A"] * (1 - f) + configs[2]["A"] * f,
            "B": configs[1]["B"] * (1 - f) + configs[2]["B"] * f,
            "G": configs[1]["G"] * (1 - f) + configs[2]["G"] * f
        }
    elif t < 14:
        return configs[2]
    elif t < 16:
        f = (t - 14) / 2.0
        return {
            "A": configs[2]["A"] * (1 - f) + configs[3]["A"] * f,
            "B": configs[2]["B"] * (1 - f) + configs[3]["B"] * f,
            "G": configs[2]["G"] * (1 - f) + configs[3]["G"] * f
        }
    elif t < 19:
        return configs[3]
    elif t < 21:
        f = (t - 19) / 2.0
        return {
            "A": configs[3]["A"] * (1 - f) + configs[4]["A"] * f,
            "B": configs[3]["B"] * (1 - f) + configs[4]["B"] * f,
            "G": configs[3]["G"] * (1 - f) + configs[4]["G"] * f
        }
    elif t < 24:
        return configs[4]
    elif t < 26:
        f = (t - 24) / 2.0
        return {
            "A": configs[4]["A"] * (1 - f) + configs[5]["A"] * f,
            "B": configs[4]["B"] * (1 - f) + configs[5]["B"] * f,
            "G": configs[4]["G"] * (1 - f) + configs[5]["G"] * f
        }
    else:
        return configs[5]


def main():
    """Simulate and plot a complete seizure event."""
    # Simulation parameters
    finalTime = 30       # 30-second simulation
    Fs = 512             # sampling frequency (Hz)
    dt = 1.0 / Fs        # time step
    nb_fonc = 10         # number of ODEs
    nbSamples = int(finalTime * Fs)

    # Get seizure configurations
    configs = get_concatenated_seizure_configs()

    # Initialize model parameters
    P = get_wendling_params()
    P["g"] = 350.  # Adjusted from default

    # Run simulation with time-varying parameters
    simulatedLFP = np.zeros(nbSamples)
    tvec = np.zeros(nbSamples)
    yold = np.zeros(nb_fonc)

    t = 0.0
    for tt in range(nbSamples):
        current_params = get_current_params(t, configs)
        P["A"] = current_params["A"]
        P["B"] = current_params["B"]
        P["G"] = current_params["G"]

        ynew = WendlingNMM(yold, dt, P)
        yold = ynew
        tvec[tt] = t
        t += dt
        simulatedLFP[tt] = ynew[1] - ynew[2] - ynew[3]

    # Normalize the signal using a moving average baseline
    window_size = int(1.0 * Fs)  # 1-second window
    baseline = np.convolve(simulatedLFP, np.ones(window_size)/window_size, mode='same')
    normalizedLFP = simulatedLFP - baseline

    # Plot the full 30-second normalized LFP trace
    plt.figure(figsize=(10, 4))
    plt.plot(tvec, normalizedLFP, 'k-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('')
    plt.title('Simulated Seizure Event (Normalized Baseline)')
    plt.xlim(0, 30)
    plt.ylim(-20, 20)
    plt.gca().set_yticks([])
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'seizure_concatenation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
