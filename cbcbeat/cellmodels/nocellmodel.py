"""This module contains a dummy cardiac cell model."""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["NoCellModel"]

from cbcbeat.cellmodels.cardiaccellmodel import CardiacCellModel
from collections import OrderedDict

# FIXME: This class represents a design flaw rather than anything
# else. Remove in a clean-up of the solvers.


class NoCellModel(CardiacCellModel):
    """
    Class representing no cell model (only bidomain equations). It
    actually just represents a single completely decoupled ODE.
    """

    def __init__(self, params=None, init_conditions=None):
        CardiacCellModel.__init__(self, params, init_conditions)

    def I(self, v, s, time=None):
        return 0

    def F(self, v, s, time=None):
        return -s

    def num_states(self):
        return 1

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("V", 0.0), ("S", 0.0)])
        return ic

    def __str__(self):
        return "No cardiac cell model"
