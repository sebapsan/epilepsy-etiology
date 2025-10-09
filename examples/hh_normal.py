"""
Hodgkin-Huxley model simulation for a single neuron in normal conditions.

This script demonstrates basic neuronal spiking behavior in response to
external current injection.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hodgkin_huxley import HH_ode, I_injection
from src.utils.parameters import get_hh_params


def main():
    """Run the Hodgkin-Huxley simulation and plot results."""
    # Get parameters
    config = get_hh_params()
    params = config['params']
    Y0 = config['Y0']
    t_start = config['t_start']
    t_end = config['t_end']

    # Create time vector
    t_eval = np.linspace(t_start, t_end, 5000)

    # Solve the equations using Runge-Kutta method
    solution = solve_ivp(
        HH_ode,
        [t_start, t_end],
        Y0,
        args=(params,),
        t_eval=t_eval,
        method='RK45'
    )

    # Extract membrane potential
    V_trace = solution.y[0]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(solution.t, V_trace, color='blue')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('Hodgkin-Huxley with Step Current Injection')
    ax1.grid(False)

    # Plot the input current for reference
    I_trace = [I_injection(t) for t in solution.t]
    ax2.plot(solution.t, I_trace, color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('I_ext (µA/cm²)')
    ax2.grid(False)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'hh_normal.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
