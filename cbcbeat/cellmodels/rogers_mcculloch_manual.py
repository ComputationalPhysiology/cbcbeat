"""This module contains a Rogers-McCulloch cardiac cell model which is
a modified version of the FitzHughNagumo model.

This formulation is based on the description on page 2 of "Optimal
control approach ..." by Nagaiah, Kunisch and Plank, 2013, J Math
Biol.

The module was written by hand, in particular it was not
autogenerated.

"""

from __future__ import division

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["RogersMcCulloch"]

from collections import OrderedDict
from cbcbeat.cellmodels.cardiaccellmodel import CardiacCellModel


class RogersMcCulloch(CardiacCellModel):
    """The Rogers-McCulloch model is a modified FitzHughNagumo model. This
    formulation follows the description on page 2 of "Optimal control
    approach ..." by Nagaiah, Kunisch and Plank, 2013, J Math Biol
    with w replaced by s. Note that this model introduces one
    additional parameter compared to the original 1994
    Rogers-McCulloch model.

    This is a model containing two nonlinear, ODEs for the evolution
    of the transmembrane potential v and one additional state variable
    s:

    .. math::

      \frac{dv}{dt} = - I_{ion}(v, s)

      \frac{ds}{dt} = F(v, s)

    where

    .. math::

      I_{ion}(v, s) = g v (1 - v/v_th)(1 - v/v_p) + \eta_1 v s

          F(v, s) = \eta_2  (v/vp - \eta_3 s)

    """

    def __init__(self, params=None, init_conditions=None):
        "Create cardiac cell model, optionally from given parameters."
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict(
            [
                ("g", 1.5),  # S/(cm^2)
                ("v_th", 13.0),  # mV
                ("v_p", 100.0),  # mV
                ("eta_1", 4.4),  # S/(cm^2)
                ("eta_2", 0.012),
                ("eta_3", 1.0),
            ]
        )
        # The original parameters of Rogers and McCulloch, 1994.
        # params = OrderedDict([("a", 0.13),
        #                      ("b", 0.013),
        #                      ("d", 1.),
        #                      ("c_1", 0.26),
        #                      ("c_2", 0.1)])

        return params

    def I(self, v, s, time=None):
        "Return the ionic current."

        # Extract parameters
        g = self._parameters["g"]
        v_p = self._parameters["v_p"]
        v_th = self._parameters["v_th"]
        eta_1 = self._parameters["eta_1"]
        # Define current
        i = g * v * (1 - v / v_th) * (1 - v / v_p) + eta_1 * v * s

        # Original R&McC current
        # a = self._parameters["a"]
        # c_1 = self._parameters["c_1"]
        # c_2 = self._parameters["c_2"]
        # i = - (c_1*v*(v - a)*(1 - v) - c_2*v*s)

        return i

    def F(self, v, s, time=None):
        "Return right-hand side for state variable evolution."

        # Extract parameters
        eta_2 = self._parameters["eta_2"]
        eta_3 = self._parameters["eta_3"]
        v_p = self._parameters["v_p"]
        f = eta_2 * (v / v_p - eta_3 * s)

        # The original from R&McC 1994
        # b = self._parameters["b"]
        # d = self._parameters["d"]
        # f = b*(v - d*s)
        return f

    @staticmethod
    def default_initial_conditions():
        ic = OrderedDict([("V", 0.0), ("S", 0.0)])
        return ic

    def num_states(self):
        "Return number of state variables."
        return 1

    def __str__(self):
        "Return string representation of class."
        return "Rogers-McCulloch 1994 cardiac cell model"
