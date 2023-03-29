"""
Unit tests for various types of solvers for cardiac cell models.
"""
__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestCardiacODESolver", "TestBasicSingleCellSolver"]


import pytest
from testutils import medium, assert_almost_equal, parametrize

from ufl.log import info_red, info_green
from cbcbeat import (
    supported_cell_models,
    BasicSingleCellSolver,
    NoCellModel,
    FitzHughNagumoManual,
    Tentusscher_2004_mcell,
    Constant,
    Expression,
)


class TestBasicSingleCellSolver(object):
    "Test functionality for the basic single cell solver."

    references = {
        NoCellModel: {1.0: (0, 0.3), 0.5: (0, 0.2), 0.0: (0, 0.1)},
        FitzHughNagumoManual: {
            1.0: (0, -84.70013280019053),
            0.5: (0, -84.8000503072239979),
            0.0: (0, -84.9),
        },
        Tentusscher_2004_mcell: {
            1.0: (0, -85.89745525156506),
            0.5: (0, -85.99686000794499),
            0.0: (0, -86.09643254164848),
        },
    }

    def _run_solve(self, model, time, theta):
        "Run two time steps for the given model with the given theta solver."
        dt = 0.01
        T = 2 * dt
        interval = (0.0, T)

        # Initialize solver
        params = BasicSingleCellSolver.default_parameters()
        params["theta"] = theta

        params["enable_adjoint"] = False
        solver = BasicSingleCellSolver(model, time, params=params)

        # Assign initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())

        # Solve for a couple of steps
        solutions = solver.solve(interval, dt)
        for ((t0, t1), vs) in solutions:
            pass

        # Check that we are at the end time
        assert_almost_equal(t1, T, 1e-10)
        return vs.vector()

    @medium
    @parametrize(("theta"), [0.0, 0.5, 1.0])
    def test_default_basic_single_cell_solver(self, cell_model, theta):
        "Test basic single cell solver."
        time = Constant(0.0)
        model = cell_model
        Model = cell_model.__class__

        if Model == supported_cell_models[3] and theta > 0:
            pytest.xfail("failing configuration (but should work)")

        model.stimulus = Expression("1000*t", t=time, degree=1)

        info_green("\nTesting %s" % model)
        vec_solve = self._run_solve(model, time, theta)

        if Model == supported_cell_models[3] and theta == 0:
            pytest.xfail("failing configuration (but should work)")

        if Model in self.references and theta in self.references[Model]:
            ind, ref_value = self.references[Model][theta]
            print("vec_solve", vec_solve.get_local())
            print("ind", ind, "ref", ref_value)

            assert_almost_equal(vec_solve[ind], ref_value, 1e-10)
        else:
            info_red("Missing references for %r, %r" % (Model, theta))
