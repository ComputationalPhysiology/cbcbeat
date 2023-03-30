"""This module contains a base class for cardiac cell models."""
__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["CardiacCellModel", "MultiCellModel"]

from ufl.log import error
import dolfin
from cbcbeat.dolfinimport import backend
from collections import OrderedDict


class CardiacCellModel:
    """
    Base class for cardiac cell models. Specialized cell models should
    subclass this class.

    Essentially, a cell model represents a system of ordinary
    differential equations. A cell model is here described by two
    (Python) functions, named F and I. The model describes the
    behavior of the transmembrane potential 'v' and a number of state
    variables 's'

    The function F gives the right-hand side for the evolution of the
    state variables:

      d/dt s = F(v, s)

    The function I gives the ionic current. If a single cell is
    considered, I gives the (negative) right-hand side for the
    evolution of the transmembrane potential

    (*)  d/dt v = - I(v, s)

    If used in a bidomain setting, the ionic current I enters into the
    parabolic partial differential equation of the bidomain equations.

    If a stimulus is provided via

      cell = CardiacCellModel()
      cell.stimulus = Expression("I_s(t)", degree=1)

    then I_s is added to the right-hand side of (*), which thus reads

       d/dt v = - I(v, s) + I_s

    Note that the cardiac cell model stimulus is ignored when the cell
    model is used a spatially-varying setting (for instance in the
    bidomain setting). In this case, the user is expected to specify a
    stimulus for the cardiac model instead.
    """

    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model

        *Arguments*
         params (dict or  :py:class:`dolfin.Parameters`, optional)
           optional model parameters
         init_conditions (dict, optional)
           optional initial conditions
        """

        # FIXME: MER: Does this need to be this complicated?
        self._parameters = self.default_parameters()
        self._initial_conditions = self.default_initial_conditions()

        params = params or OrderedDict()
        init_conditions = init_conditions or OrderedDict()

        if params:
            assert isinstance(
                params, dict
            ), "expected a dict or a Parameters, as the params argument"
            if isinstance(params, dolfin.Parameters):
                params = params.to_dict()
            self.set_parameters(**params)

        if init_conditions:
            assert isinstance(
                init_conditions, dict
            ), "expected a dict or a Parameters, as the init_condition argument"
            if isinstance(init_conditions, dolfin.Parameters):
                init_conditions = init_conditions.to_dict()
            self.set_initial_conditions(**init_conditions)

        self.stimulus = None

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        return OrderedDict()

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        return OrderedDict()

    def set_parameters(self, **params):
        "Update parameters in model"
        for param_name, param_value in params.items():
            if param_name not in self._parameters:
                error("'%s' is not a parameter in %s" % (param_name, self))
            if not isinstance(param_value, (float, int)) and not isinstance(
                param_value._cpp_object, dolfin.cpp.function.GenericFunction
            ):
                error(
                    "'%s' is not a scalar or a dolfin.cpp.function.GenericFunction"
                    % param_name
                )
                if (
                    hasattr(param_value, "_cpp_object")
                    and isinstance(
                        param_value._cpp_object, dolfin.cpp.function.GenericFunction
                    )
                    and param_value._cpp_object.value_size() != 1
                ):
                    error("expected the value_size of '%s' to be 1" % param_name)

            self._parameters[param_name] = param_value

    def set_initial_conditions(self, **init):
        "Update initial_conditions in model"
        for init_name, init_value in init.items():
            if init_name not in self._initial_conditions:
                error("'%s' is not a parameter in %s" % (init_name, self))
            if not isinstance(init_value, (float, int)) and not isinstance(
                init_value._cpp_object, dolfin.cpp.function.GenericFunction
            ):
                error(
                    "'%s' is not a scalar or a dolfin.cpp.function.GenericFunction"
                    % init_name
                )
            if (
                hasattr(init_value, "_cpp_object")
                and isinstance(
                    init_value._cpp_object, dolfin.cpp.function.GenericFunction
                )
                and init_value._cpp_object.value_size() != 1
            ):
                error("expected the value_size of '%s' to be 1" % init_name)
            self._initial_conditions[init_name] = init_value

    def initial_conditions(self):
        "Return initial conditions for v and s as an Expression."
        return dolfin.Expression(
            list(self._initial_conditions.keys()), degree=1, **self._initial_conditions
        )

    def parameters(self):
        "Return the current parameters."
        return self._parameters

    def F(self, v, s, time=None):
        "Return right-hand side for state variable evolution."
        error("Must define F = F(v, s)")

    def I(self, v, s, time=None):
        "Return the ionic current."
        error("Must define I = I(v, s)")

    def num_states(self):
        """Return number of state variables (in addition to the
        membrane potential)."""
        error("Must overload num_states")

    def __str__(self):
        "Return string representation of class."
        return "Some cardiac cell model"


class MultiCellModel(CardiacCellModel):
    """
    MultiCellModel
    """

    def __init__(self, models, keys, markers):
        """
        *Arguments*
        models (tuple)
          tuple of existing CardiacCellModels
        keys (tuple)
          integers demarking the domain for each cell model
        markers (:py:class:`dolfin.MeshFunction`)
          MeshFunction defining the partitioning of the mesh (which model where)
        """
        self._cell_models = models
        self._keys = keys
        self._key_to_cell_model = dict(zip(keys, range(len(keys))))
        self._markers = markers

        self._num_states = max(c.num_states() for c in self._cell_models)

    def __getitem__(self, key):
        return self.models()[self._key_to_cell_model[key]]

    def models(self):
        return self._cell_models

    def markers(self):
        return self._markers

    def keys(self):
        return self._keys

    def mesh(self):
        return self._markers.mesh()

    def num_models(self):
        return len(self._cell_models)

    def num_states(self):
        """Return number of state variables (in addition to the
        membrane potential)."""
        return self._num_states

    def F(self, v, s, time=None, index=None):
        if index is None:
            error("(Domain) index must be specified for multi cell models")
        # Extract which cell model index (given by index in incoming tuple)
        k = self._key_to_cell_model[index]
        return self._cell_models[k].F(v, s, time)

    def I(self, v, s, time=None, index=None):
        if index is None:
            error("(Domain) index must be specified for multi cell models")
        # Extract which cell model index (given by index in incoming tuple)
        k = self._key_to_cell_model[index]
        return self._cell_models[k].I(v, s, time)

    def initial_conditions(self):
        "Return initial conditions for v and s as a dolfin.dolfin.cpp.function.GenericFunction."

        n = self.num_states()  # (Maximal) Number of states in MultiCellModel
        VS = dolfin.VectorFunctionSpace(self.mesh(), "DG", 0, n + 1)
        vs = backend.Function(VS)

        markers = self.markers()
        u = dolfin.TrialFunction(VS)
        v = dolfin.TestFunction(VS)

        dy = dolfin.Measure("dx", domain=self.mesh(), subdomain_data=markers)

        # Define projection into multiverse
        a = dolfin.inner(u, v) * dy()

        Ls = list()
        for k, model in enumerate(self.models()):
            ic = model.initial_conditions()  # Extract initial conditions
            n_k = model.num_states()  # Extract number of local states
            i_k = self.keys()[k]  # Extract domain index of cell model k
            L_k = sum(ic[j] * v[j] * dy(i_k) for j in range(n_k))
            Ls.append(L_k)
        L = sum(Ls)
        backend.solve(a == L, vs)
        return vs


# from dolfin import *
# from cbcbeat import *

# mesh = UnitSquareMesh(20, 20)
# markers = CellFunction("uint", mesh, 0)
# markers.array()[::2] = 1
# c0 = Beeler_reuter_1977()
# c1 = FitzHughNagumoManual()
# cell_model = MultiCellModel((c0, c1), (0, 1), markers)
# print cell_model.num_states()
