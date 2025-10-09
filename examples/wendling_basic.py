"""
Basic Wendling neural mass model simulation.

This script simulates local field potential (LFP) using the Wendling et al.
(2002) model with default parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.wendling import WendlingNMM
from src.utils.parameters import get_wendling_params


def main():
    """Run Wendling simulation and plot LFP."""
    # Simulation parameters
    finalTime = 5        # simulation time (seconds)
    Fs = 512             # sampling frequency (Hz)
    dt = 1.0 / Fs        # time step
    nb_fonc = 10         # number of ODEs
    nbSamples = int(finalTime * Fs)

    # Get model parameters
    P = get_wendling_params(A=4, B=15, G=22)

    # Initialize arrays
    simulatedLFP = np.zeros(nbSamples)
    yold = np.zeros(nb_fonc)
    xstates = np.zeros([nb_fonc, nbSamples])
    tvec = np.zeros(nbSamples)
    t = 0.0

    # Run simulation using Euler integration
    for tt in range(nbSamples):
        ynew = WendlingNMM(yold, dt, P)
        yold = ynew
        tvec[tt] = t
        t += dt
        xstates[:, tt] = ynew
        simulatedLFP[tt] = ynew[1] - ynew[2] - ynew[3]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(tvec, simulatedLFP, 'k-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.title('Simulated Local Field Potential (LFP)')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'wendling_basic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
