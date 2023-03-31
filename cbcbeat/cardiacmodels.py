"""This module contains a container class for cardiac models:
:py:class:`~cbcbeat.cardiacmodels.CardiacModel`.  This class
should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2016-04-21

__all__ = ["CardiacModel"]

import dolfin
from cbcbeat.dolfinimport import backend
from cbcbeat.markerwisefield import Markerwise, handle_markerwise
from cbcbeat.cellmodels import CardiacCellModel
from ufl.log import error

# ------------------------------------------------------------------------------
# Cardiac models
# ------------------------------------------------------------------------------


class CardiacModel(object):
    """
    A container class for cardiac models. Objects of this class
    represent a specific cardiac simulation set-up and should provide

    * A computational domain
    * A cardiac cell model
    * Intra-cellular and extra-cellular conductivities
    * Various forms of stimulus (optional).

    This container class is designed for use with the splitting
    solvers (:py:mod:`cbcbeat.splittingsolver`), see their
    documentation for more information on how the attributes are
    interpreted in that context.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        the computational domain in space
      time (:py:class:`dolfin.Constant` or None )
        A constant holding the current time.
      M_i (:py:class:`ufl.Expr`)
        the intra-cellular conductivity as an ufl Expression
      M_e (:py:class:`ufl.Expr`)
        the extra-cellular conductivity as an ufl Expression
      cell_models (:py:class:`~cbcbeat.cellmodels.cardiaccellmodel.CardiacCellModel`)
        a cell model or a dict with cell models associated with a cell model domain
      stimulus (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.
      applied_current (:py:class:`ufl.Expr`, optional)
        an applied current as an ufl Expression

    """

    def __init__(
        self, domain, time, M_i, M_e, cell_models, stimulus=None, applied_current=None
    ):
        "Create CardiacModel from given input."

        self._handle_input(
            domain, time, M_i, M_e, cell_models, stimulus, applied_current
        )

    def _handle_input(
        self, domain, time, M_i, M_e, cell_models, stimulus=None, applied_current=None
    ):
        # Check input and store attributes
        msg = "Expecting domain to be a Mesh instance, not %r" % domain
        assert isinstance(domain, dolfin.Mesh), msg
        self._domain = domain

        msg = "Expecting time to be a Constant instance, not %r." % time
        assert isinstance(time, backend.Constant) or time is None, msg
        self._time = time

        self._intracellular_conductivity = M_i
        self._extracellular_conductivity = M_e

        # Handle cell_models
        self._cell_models = handle_markerwise(cell_models, CardiacCellModel)
        if isinstance(self._cell_models, Markerwise):
            msg = "Different cell_models are currently not supported."
            error(msg)

        # Handle stimulus
        self._stimulus = handle_markerwise(
            stimulus, dolfin.cpp.function.GenericFunction
        )

        # Handle applied current
        ac = applied_current
        self._applied_current = handle_markerwise(
            ac, dolfin.cpp.function.GenericFunction
        )

    def applied_current(self):
        "An applied current: used as a source in the elliptic bidomain equation"
        return self._applied_current

    def stimulus(self):
        "A stimulus: used as a source in the parabolic bidomain equation"
        return self._stimulus

    def conductivities(self):
        """Return the intracellular and extracellular conductivities
        as a tuple of UFL Expressions.

        *Returns*
        (M_i, M_e) (:py:class:`tuple` of :py:class:`ufl.Expr`)
        """
        return (self.intracellular_conductivity(), self.extracellular_conductivity())

    def intracellular_conductivity(self):
        "The intracellular conductivity (:py:class:`ufl.Expr`)."
        return self._intracellular_conductivity

    def extracellular_conductivity(self):
        "The intracellular conductivity (:py:class:`ufl.Expr`)."
        return self._extracellular_conductivity

    def time(self):
        "The current time (:py:class:`dolfin.Constant` or None)."
        return self._time

    def domain(self):
        "The spatial domain (:py:class:`dolfin.Mesh`)."
        return self._domain

    def cell_models(self):
        "Return the cell models"
        return self._cell_models
