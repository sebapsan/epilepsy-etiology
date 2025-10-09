"""
Models module containing Hodgkin-Huxley and Wendling implementations
"""

from .hodgkin_huxley import HH_ode, HH_ode_seizure
from .wendling import WendlingNMM, sigm

__all__ = ['HH_ode', 'HH_ode_seizure', 'WendlingNMM', 'sigm']
