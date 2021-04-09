"""
Unit tests for various types of solvers for cardiac cell models.
"""
__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestCardiacODESolver", "TestBasicSingleCellSolver"]


import itertools
import pytest
from testutils import slow, assert_almost_equal, parametrize, cell_model

from dolfin import info, UnitIntervalMesh
from ufl.log import info_red, info_green
from cbcbeat import supported_cell_models, \
    CardiacODESolver, BasicSingleCellSolver, \
    Constant, Expression
from cbcbeat.cellmodels import *


supported_schemes = ["ForwardEuler", "BackwardEuler", "CrankNicolson",
                     "RK4", "ESDIRK3", "ESDIRK4", "RL1", "RL2", "GRL1", "GRL2"]
print([Model.__name__ for Model in supported_cell_models])
supported_cell_models_str = [Model.__name__ for Model in supported_cell_models]
#supported_cell_models_str = ["Tentusscher_2004_mcell"]

class TestCardiacODESolver(object):
    """ Tests the cardiac ODE solver on different cell models. """

    # Note that these should be essentially identical to the ones
    # for the BasicSingleCellSolver
    references = {"NoCellModel":
                   {"BackwardEuler": (0, 0.3),
                    "CrankNicolson": (0, 0.2),
                    "ForwardEuler": (0, 0.1),
                    "RL1": (0, 0.1),
                    "RL2": (0, 0.2),
                    "GRL1": (0, 0.1),
                    "GRL2": (0, 0.2),
                    "RK4": (0, 0.2),
                    "ESDIRK3": (0, 0.2),
                    "ESDIRK4": (0, 0.2),
                    },

                   "FitzHughNagumoManual":
                   {"BackwardEuler": (0, -84.70013280019053),
                    "CrankNicolson": (0, -84.80005016079546),
                    "ForwardEuler": (0, -84.9),
                    "RL1": (0, -84.9),
                    "RL2": (0, -84.80003356232497),
                    "GRL1": (0, -84.90001689809608),
                    "GRL2": (0, -84.8000834179),
                    "RK4": (0, -84.80004467770296),
                    "ESDIRK3": (0, -84.80004459269247),
                    "ESDIRK4": (0, -84.80004468281632),
                    },

                   "Fitzhughnagumo":
                   {"BackwardEuler": (0, -84.70013280019053),
                    "CrankNicolson": (0, -84.8000501607955),
                    "ForwardEuler":  (0, -84.9),
                    "RL1": (0, -84.9),
                    "RL2": (0, -84.80003356232497),
                    "GRL1": (0, -84.90001689809608),
                    "GRL2": (0, -84.8000834179),
                    "RK4":  (0, -84.80004467770296),
                    "ESDIRK3":  (0, -84.80004467770296),
                    "ESDIRK4":  (0, -84.80004468281632),
                    },

                   "Tentusscher_2004_mcell":
                   {"BackwardEuler": (0, -85.89745525156506),
                    "CrankNicolson": (0, -85.99685674414921),
                    "ForwardEuler":  (0, -86.09643254164848),
                    "RL1": (0, -86.09648383320031),
                    "RL2": (0, -85.99673587692514),
                    "GRL1": (0, -86.09661529223865),
                    "GRL2": (0, -85.99710694560402),
                    "RK4":  (0, "nan"),
                    "ESDIRK3":  (0, -85.99681862337053),
                    "ESDIRK4":  (0, -85.99681796046603),
                    }
                   }

    def compare_against_reference(self, sol, Model, Scheme):
        ''' Compare the model solution with the reference solution. '''
        try:
            ind, ref_value = self.references[Model][Scheme]
        except KeyError:
            info_red("Missing references for %s, %s: value is %g"
                 % (Model, Scheme, sol[0]))
            return

        info("Value for %s, %s is %g" % (Model, Scheme, sol[ind]))
        if ref_value != "nan":
            assert_almost_equal(float(sol[ind]), float(ref_value), tolerance=1e-6)

    def replace_with_constants(self, params):
        ''' Replace all float values in params by Constants. '''
        for param_name in list(params.keys()):
            value = params[param_name]
            params[param_name] = Constant(value)

    def _setup_solver(self, Model, Scheme, time=0.0, stim=None, params=None):
        ''' Generate a new solver object with the given start time, stimulus and parameters. '''
        # Create model instance
        if isinstance(Model, str):
            Model = eval(Model)
        model = Model(params=params)

        # Initialize time and stimulus (note t=time construction!)
        if stim is None:
            stim = Expression("1000*t", t=time, degree=1)

        # Initialize solver
        mesh = UnitIntervalMesh(5)
        params = CardiacODESolver.default_parameters()
        params["scheme"] = Scheme

        solver = CardiacODESolver(mesh, time, model, I_s=stim, params=params)

        # Create scheme
        info_green("\nTesting %s with %s scheme" % (model, Scheme))

        # Start with native initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())
        vs.assign(vs_)

        return solver

    @slow
    @parametrize(("Model","Scheme"),
        list(itertools.product(supported_cell_models_str,supported_schemes))
        )
    def test_compare_against_reference(self, Model, Scheme):
        ''' Runs the given cell model with the numerical scheme
            and compares the result with the reference value. '''

        solver = self._setup_solver(Model, Scheme, time=Constant(0))
        (vs_, vs) = solver.solution_fields()

        next_dt = 0.01
        solver.step((0.0, next_dt))
        vs_.assign(vs)
        solver.step((next_dt, 2*next_dt))

        self.compare_against_reference(vs.vector(), Model, Scheme)

    @slow
    @parametrize(("Model","Scheme"),
        list(itertools.product(supported_cell_models_str,supported_schemes))
        )
    def test_compare_against_reference_constant(self, Model, Scheme):
        ''' Runs the given cell model with the numerical scheme
            and compares the result with the reference value. '''

        Model = eval(Model)
        params = Model.default_parameters()
        self.replace_with_constants(params)

        solver = self._setup_solver(Model, Scheme, time=Constant(0), params=params)
        (vs_, vs) = solver.solution_fields()

        next_dt = 0.01
        solver.step((0.0, next_dt))
        vs_.assign(vs)
        solver.step((next_dt, 2*next_dt))

        self.compare_against_reference(vs.vector(), Model, Scheme)
