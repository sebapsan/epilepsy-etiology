"""
Wendling model simulation showing six different epileptic states.

This script reproduces the different types of neuronal activity patterns
observed in epilepsy, from background activity to full ictal (seizure) state.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.wendling import WendlingNMM
from src.utils.parameters import get_wendling_params, get_seizure_configs


def main():
    """Simulate and plot six different epileptic states."""
    # Simulation parameters
    finalTime = 5
    Fs = 512
    dt = 1.0 / Fs
    nb_fonc = 10
    nbSamples = int(finalTime * Fs)

    # Get configurations for different states
    configs = get_seizure_configs()

    # Create figure with 6 subplots
    fig, axs = plt.subplots(6, 1, figsize=(6, 12), sharex=True)

    state_names = [
        'Background activity',
        'Sporadic spikes',
        'Rhythmic spikes',
        'Slow quasi-sinusoidal',
        'Low voltage rapid',
        'Fast activity (ictal)'
    ]

    # Loop over each configuration and simulate
    for i, config in enumerate(configs):
        # Get parameters for this configuration
        P = get_wendling_params(A=config["A"], B=config["B"], G=config["G"])

        # Initialize simulation arrays
        simulatedLFP = np.zeros(nbSamples)
        yold = np.zeros(nb_fonc)
        tvec = np.zeros(nbSamples)
        t = 0.0

        # Run simulation using Euler integration
        for tt in range(nbSamples):
            ynew = WendlingNMM(yold, dt, P)
            yold = ynew
            tvec[tt] = t
            t += dt
            simulatedLFP[tt] = ynew[1] - ynew[2] - ynew[3]

        # Plot in corresponding subplot
        axs[i].plot(tvec, simulatedLFP, 'k-', linewidth=1)
        axs[i].set_xticks([])
        axs[i].spines['top'].set_visible(True)
        axs[i].spines['right'].set_visible(True)
        axs[i].spines['left'].set_visible(True)
        axs[i].set_title(f'Type {i+1}: {state_names[i]}', loc='left', fontsize=10)

        # Vertical dashed lines every second
        for j in range(1, 5):
            axs[i].axvline(j, linestyle='dashed', color='gray', linewidth=0.8)

    # Time scale bar
    fig.text(0.87, 0.99, "1 sec", fontsize=10, ha='center', color="black")
    fig.add_artist(plt.Line2D([0.78, 0.97], [0.98, 0.98], color='black',
                              linewidth=3, transform=fig.transFigure, clip_on=False))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'wendling_six_states.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
