"""
Unit tests for various types of bidomain solver
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestSplittingSolver"]

from testutils import assert_almost_equal, medium, parametrize

import dolfin
from dolfin import UnitCubeMesh, Expression
from cbcbeat import (
    FitzHughNagumoManual,
    CardiacModel,
    backend,
    BasicSplittingSolver,
    SplittingSolver,
    dolfinimport,
)

try:
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
except Exception:
    dolfin.set_log_level(dolfin.WARNING)
    pass


class TestSplittingSolver(object):
    "Test functionality for the splitting solvers."

    def setup(self):
        self.mesh = UnitCubeMesh(5, 5, 5)

        # Create time
        self.time = backend.Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0*t", t=self.time, degree=1)

        # Create ac
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time, degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = CardiacModel(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            self.cell_model,
            self.stimulus,
            self.applied_current,
        )

        dt = 0.1
        self.t0 = 0.0
        self.dt = [(0.0, dt), (dt * 2, dt / 2), (dt * 4, dt)]
        # Test using variable dt interval but using the same dt.

        self.T = self.t0 + 5 * dt
        self.ics = self.cell_model.initial_conditions()

    @medium
    @parametrize(("solver_type"), ["direct", "iterative"])
    def test_basic_and_optimised_splitting_solver_exact(self, solver_type):
        """Test that basic and optimised splitting solvers yield
        very comparative results when configured identically."""

        # Create basic solver
        params = BasicSplittingSolver.default_parameters()
        params["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
        params["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
        solver = BasicSplittingSolver(self.cardiac_model, params=params)

        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for interval, fields in solutions:
            (vs_, vs, vur) = fields
        a = vs.vector().norm("l2")
        c = vur.vector().norm("l2")
        assert_almost_equal(interval[1], self.T, 1e-10)

        if dolfinimport.has_dolfin_adjoint:
            import dolfin_adjoint

            dolfin_adjoint.adj_reset()

        # Create optimised solver with direct solution algorithm
        params = SplittingSolver.default_parameters()
        params["BidomainSolver"]["linear_solver_type"] = solver_type
        params["enable_adjoint"] = False
        if solver_type == "direct":
            params["BidomainSolver"]["use_avg_u_constraint"] = True
        solver = SplittingSolver(self.cardiac_model, params=params)

        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for interval, fields in solutions:
            (vs_, vs, vur) = fields
        assert_almost_equal(interval[1], self.T, 1e-10)
        b = vs.vector().norm("l2")
        d = vur.vector().norm("l2")

        print("a, b = ", a, b)
        print("c, d = ", c, d)
        print("a - b = ", (a - b))
        print("c - d = ", (c - d))

        # Compare results, discrepancy is in difference in ODE
        # solves.
        assert_almost_equal(a, b, tolerance=1.0)
        assert_almost_equal(c, d, tolerance=1.0)
