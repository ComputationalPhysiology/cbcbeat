"""This module contains a Fitzhughnagumo cardiac cell model

The module was autogenerated from a gotran form file
"""
from __future__ import division
from collections import OrderedDict
import ufl

from cbcbeat.dolfinimport import *
from cbcbeat.cellmodels import CardiacCellModel


class Fitzhughnagumo(CardiacCellModel):
    """
    NOT_IMPLEMENTED
    """

    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model

        *Arguments*
         params (dict, :py:class:`dolfin.Mesh`, optional)
           optional model parameters
         init_conditions (dict, :py:class:`dolfin.Mesh`, optional)
           optional initial conditions
        """
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict(
            [
                ("a", 0.13),
                ("b", 0.013),
                ("c_1", 0.26),
                ("c_2", 0.1),
                ("c_3", 1.0),
                ("stim_amplitude", 0),
                ("stim_duration", 1),
                ("stim_period", 1000),
                ("stim_start", 1),
                ("v_peak", 40.0),
                ("v_rest", -85.0),
            ]
        )
        return params

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("v", -85.0), ("s", 0.0)])
        return ic

    def I(self, v, s, time=None):
        """
        Transmembrane current
        """
        # Imports
        # No imports for now

        time = time if time else Constant(0.0)
        # Assign states
        _is_vector = False

        # Assign parameters
        a = self._parameters["a"]
        stim_start = self._parameters["stim_start"]
        stim_amplitude = self._parameters["stim_amplitude"]
        c_1 = self._parameters["c_1"]
        c_2 = self._parameters["c_2"]
        v_rest = self._parameters["v_rest"]
        stim_duration = self._parameters["stim_duration"]
        v_peak = self._parameters["v_peak"]

        current = (
            -(v - v_rest)
            * (v_peak - v)
            * (-(v_peak - v_rest) * a + v - v_rest)
            * c_1
            / ((v_peak - v_rest) * (v_peak - v_rest))
            + (v - v_rest) * c_2 * s / (v_peak - v_rest)
            - (1.0 - 1.0 / (1.0 + ufl.exp(-5.0 * stim_start + 5.0 * time)))
            * stim_amplitude
            / (1.0 + ufl.exp(-5.0 * stim_start + 5.0 * time - 5.0 * stim_duration))
        )

        return current

    def F(self, v, s, time=None):
        """
        Right hand side for ODE system
        """
        # Imports
        # No imports for now

        time = time if time else Constant(0.0)
        # Assign states
        _is_vector = False

        # Assign parameters
        c_3 = self._parameters["c_3"]
        b = self._parameters["b"]
        v_rest = self._parameters["v_rest"]

        F_expressions = [
            # Derivative for state s
            (-c_3 * s + v - v_rest)
            * b,
        ]

        return as_vector(F_expressions) if _is_vector else F_expressions[0]

    def num_states(self):
        return 1

    def __str__(self):
        return "Fitzhughnagumo cardiac cell model"
