"""This module contains a container class for torso models to be coupled 
with cardiac models::py:class:`~cbcbeat.cardiacmodels.TorsoModel`.  
This class should be instantiated for setting up specific cardiac simulation
scenarios.
"""

# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2016-04-21

__all__ = ["TorsoModel"]

from dolfinimport import Parameters, Mesh, Constant, GenericFunction, log
from markerwisefield import Markerwise, handle_markerwise

# ------------------------------------------------------------------------------
# Torso models
# ------------------------------------------------------------------------------

class TorsoModel(object):
    """
    A container class for torso models. This class is used to define
    the torso model to be coupled with the cardiac model in the 
    cardiac simulation. This class should provide

    * A computational domain (embedding the heart domain)
    * A conductivity

    This container class is designed for use with the splitting
    solvers (:py:mod:`cbcbeat.splittingsolver`), see their
    documentation for more information on how the attributes are
    interpreted in that context.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        the computational domain in space
      M_T (:py:class:`ufl.Expr`)
        the conductivity as an ufl Expression
    """
    def __init__(self, domain, M_T):
        "Create TorsoModel from given input."

        self._handle_input(domain, M_T)

    def _handle_input(self, domain, M_T):

        # Check input and store attributes
        msg = "Expecting domain to be a Mesh instance, not %r" % domain
        assert isinstance(domain, Mesh), msg

        self._domain = domain
        self._conductivity = M_T

    def conductivity(self):
        """Return the conductivity as anUFL Expressions.

        *Returns*
        M_T (:py:class:`ufl.Expr`)
        """
        return self._conductivity

    def domain(self):
        "The spatial domain (:py:class:`dolfin.Mesh`)."
        return self._domain
