"""
Hodgkin-Huxley model with seizure-like ion channel changes.

This script simulates pathological neuronal activity by smoothly transitioning
ion channel conductances from normal to seizure states.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os
import warnings

# Suppress overflow warnings from exponentials
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hodgkin_huxley import HH_ode_seizure, seizure_factor
from src.utils.parameters import get_hh_seizure_params


def main():
    """Run the seizure simulation and plot results."""
    # Get parameters
    config = get_hh_seizure_params()
    params_normal = config['params_normal']
    params_seizure = config['params_seizure']
    Y0 = config['Y0']
    t_start = config['t_start']
    t_end = config['t_end']

    # Create time vector
    t_eval = np.linspace(t_start, t_end, 5000)

    # Solve the equations
    solution = solve_ivp(
        HH_ode_seizure,
        [t_start, t_end],
        Y0,
        args=(params_normal, params_seizure),
        t_eval=t_eval,
        method='RK45'
    )

    # Plot membrane potential
    plt.figure(figsize=(10, 5))
    plt.plot(solution.t, solution.y[0], label='Membrane Potential')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Hodgkin-Huxley Model: Seizure-like Ion Channel Changes")
    plt.axvspan(50, 120, color='red', alpha=0.3, label='Seizure Period')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'hh_seizure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    plt.show()

    # Plot conductance changes
    s_vals = seizure_factor(solution.t)
    g_Na_eff = params_normal[1] + s_vals * (params_seizure[1] - params_normal[1])
    g_K_eff = params_normal[2] + s_vals * (params_seizure[2] - params_normal[2])

    plt.figure(figsize=(10, 5))
    plt.plot(solution.t, g_Na_eff, label="Effective g_Na")
    plt.plot(solution.t, g_K_eff, label="Effective g_K")
    plt.xlabel("Time (ms)")
    plt.axvspan(50, 120, color='red', alpha=0.3, label='Seizure Period')
    plt.ylabel("Conductance (mS/cm²)")
    plt.title("Time-dependent Ion Channel Conductances")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'hh_conductances.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
