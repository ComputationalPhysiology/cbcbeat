"""
Unit tests for GOSS splitting solver
"""

__author__ = "Cécile Daversin Catty (cecile@simula.no), 2022"
__all__ = ["TestGOSSplittingSolver"]

from testutils import assert_almost_equal, medium, parametrize

from dolfin import info

from cbcbeat import *
from cbcbeat.gossplittingsolver import GOSSplittingSolver

import goss
from goss.dolfinutils import DOLFINParameterizedODE

try:
    set_log_level(LogLevel.WARNING)
except:
    set_log_level(WARNING)
    pass


class TestSplittingSolver(object):
    "Test functionality for the splitting solvers."

    def setup(self):
        self.mesh = UnitCubeMesh(5, 5, 5)

        # Create time
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0*t", t=self.time, degree=1)

        # Create ac
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time,
                                          degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        # Splitting Solver cell and cardiac models
        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = CardiacModel(self.mesh, self.time,
                                          self.M_i, self.M_e,
                                          self.cell_model,
                                          self.stimulus,
                                          self.applied_current)

        # GOSS Splitting Solver cell and cardiac models (expect different object)
        self.goss_cell_model = goss.dolfinutils.DOLFINParameterizedODE(
            "../../cbcbeat/cellmodels/fitzhughnagumo",
            field_states=["v", "s"])

        self.goss_cardiac_model = CardiacModel(self.mesh, self.time,
                                               self.M_i, self.M_e,
                                               self.goss_cell_model,
                                               self.stimulus,
                                               self.applied_current)

        dt = 0.1
        self.t0 = 0.0
        self.dt = [(0.0, dt), (dt*2, dt/2), (dt*4, dt)]
        # Test using variable dt interval but using the same dt.

        self.T = self.t0 + 5*dt
        self.ics = self.cell_model.initial_conditions()

    @medium
    @parametrize(("solver_type"), ["direct", "iterative"])
    def test_basic_and_optimised_goss_splitting_solver_exact(self, solver_type):
        """Test that basic and optimised splitting solvers yield
        very comparative results when configured identically."""

        # Create GOSS splitting solver
        params = SplittingSolver.default_parameters()
        params["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
        params["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
        solver = SplittingSolver(self.cardiac_model, params=params)
        
        (vs_, vs, vur) = solver.solution_fields()
        vs_.assign(self.ics)

        # Solve
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (vs_, vs, vur) = fields
        a = vs.vector().norm("l2")
        c = vur.vector().norm("l2")
        assert_almost_equal(interval[1], self.T, 1e-10)

        if dolfin_adjoint:
            adj_reset()

        # Create optimised solver with direct solution algorithm
        params = GOSSplittingSolver.default_parameters()
        params["BidomainSolver"]["linear_solver_type"] = solver_type
        params["enable_adjoint"] = False
        if solver_type == "direct":
            params["BidomainSolver"]["use_avg_u_constraint"] = True
        solver = GOSSplittingSolver(self.goss_cardiac_model, params=params)

        (vs_, vs, vur) = solver.solution_fields()

        # Solve again
        solutions = solver.solve((self.t0, self.T), self.dt)
        for (interval, fields) in solutions:
            (vs_, vs, vur) = fields
        assert_almost_equal(interval[1], self.T, 1e-10)
        b = vs.vector().norm("l2")
        d = vur.vector().norm("l2")

        print("a, b = ", a, b)
        print("c, d = ", c, d)
        print("a - b = ", (a - b))
        print("c - d = ", (c - d))

        err_ab = abs((a-b)*100/a)
        err_cd = abs((c-d)*100/c)
        print("Error a - b = ", err_ab, "%")
        print("Error c - d = ", err_cd, "%")

        # Compare results, discrepancy is in difference in ODE
        # solves.
        assert_almost_equal(a, b, tolerance=5.)
        assert_almost_equal(c, d, tolerance=5.)
        # Discrepancy in % (less than 1% error)
        assert_almost_equal(err_ab, 0, tolerance=1.)
        assert_almost_equal(err_cd, 0, tolerance=1.)
