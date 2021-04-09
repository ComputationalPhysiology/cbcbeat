"""
Unit tests for various types of bidomain solver
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

from testutils import assert_almost_equal, assert_equal, fast

from dolfin import *
from cbcbeat import BasicBidomainSolver, BasicMonodomainSolver, \
        MonodomainSolver, BidomainSolver, \
        Constant

class TestBasicBidomainSolver(object):
    "Test functionality for the basic bidomain solver."

    def setUp(self):
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create ac
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time,
                                          degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_basic_solve(self):
        "Test that solver runs."
        self.setUp()

        Solver = BasicBidomainSolver

        # Create solver
        solver = Solver(self.mesh, self.time,
                        self.M_i, self.M_e, I_s=self.stimulus,
                        I_a=self.applied_current)

        # Solve
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

    @fast
    def test_compare_solve_step(self):
        "Test that solve gives same results as single step"
        self.setUp()

        Solver = BasicBidomainSolver
        solver = Solver(self.mesh, self.time,
                        self.M_i, self.M_e, I_s=self.stimulus,
                        I_a=self.applied_current)

        (v_, vs) = solver.solution_fields()

        # Solve
        interval = (self.t0, self.t0 + self.dt)
        solutions = solver.solve(interval, self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
            a = vur.vector().norm("l2")

        # Reset v_
        v_.vector()[:] = 0.0

        # Step
        solver.step(interval)
        b = vs.vector().norm("l2")

        # Check that result from solve and step match.
        assert_equal(a, b)


class TestBasicMonodomainSolver(object):
    "Test functionality for the basic monodomain solver."

    def setUp(self):
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_basic_solve(self):
        "Test that solver runs."
        self.setUp()

        Solver = BasicMonodomainSolver

        # Create solver
        solver = Solver(self.mesh, self.time,
                        self.M_i, I_s=self.stimulus)

        # Solve
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

    @fast
    def test_compare_solve_step(self):
        "Test that solve gives same results as single step"
        self.setUp()

        Solver = BasicMonodomainSolver
        solver = Solver(self.mesh, self.time,
                        self.M_i, I_s=self.stimulus)

        (v_, vs) = solver.solution_fields()

        # Solve
        interval = (self.t0, self.t0 + self.dt)
        solutions = solver.solve(interval, self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
            a = vur.vector().norm("l2")

        # Reset v_
        v_.vector()[:] = 0.0

        # Step
        solver.step(interval)
        b = vs.vector().norm("l2")

        # Check that result from solve and step match.
        assert_equal(a, b)

class TestBidomainSolver(object):
    def setUp(self):
        N = 5
        self.mesh = UnitCubeMesh(N, N, N)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create ac
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time,
                                          degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_solve(self):
        "Test that solver runs."
        self.setUp()

        # Create solver and solve
        solver = BidomainSolver(self.mesh, self.time,
                                self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields

    @fast
    def test_compare_with_basic_solve(self):
        """Test that solver with direct linear algebra gives same
        results as basic bidomain solver."""
        self.setUp()

        # Create solver and solve
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = True
        solver = BidomainSolver(self.mesh, self.time,
                                self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current, params=params)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
        bidomain_result = vur.vector().norm("l2")

        # Create other solver and solve
        solver = BasicBidomainSolver(self.mesh, self.time,
                                     self.M_i, self.M_e,
                                     I_s=self.stimulus,
                                     I_a=self.applied_current)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
        basic_bidomain_result = vur.vector().norm("l2")

        print(bidomain_result)
        print(basic_bidomain_result)
        assert_almost_equal(bidomain_result, basic_bidomain_result,
                               1e-13)

    @fast
    def test_compare_direct_iterative(self):
        "Test that direct and iterative solution give comparable results."
        self.setUp()

        # Create solver and solve
        params = BidomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        params["use_avg_u_constraint"] = True
        solver = BidomainSolver(self.mesh, self.time,
                                self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current,
                                params=params)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
            (v, u, r) = vur.split(deepcopy=True)
            a = v.vector().norm("l2")

        # Create solver and solve using iterative means
        params = BidomainSolver.default_parameters()
        params["petsc_krylov_solver"]["monitor_convergence"] = True
        solver = BidomainSolver(self.mesh, self.time,
                                self.M_i, self.M_e,
                                I_s=self.stimulus,
                                I_a=self.applied_current,
                                params=params)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vu) = fields
            (v, u) = vu.split(deepcopy=True)
            b = v.vector().norm("l2")

        print("lu gives ", a)
        print("krylov gives ", b)
        assert_almost_equal(a, b, 1e-4)

class TestMonodomainSolver(object):
    def setUp(self):
        N = 5
        self.mesh = UnitCubeMesh(N, N, N)
        self.time = Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.1

    @fast
    def test_solve(self):
        "Test that solver runs."
        self.setUp()

        # Create solver and solve
        solver = MonodomainSolver(self.mesh, self.time,
                                  self.M_i, I_s=self.stimulus)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields

    @fast
    def test_compare_with_basic_solve(self):
        """Test that solver with direct linear algebra gives same
        results as basic monodomain solver."""
        self.setUp()

        # Create solver and solve
        params = MonodomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        solver = MonodomainSolver(self.mesh, self.time,
                                  self.M_i, I_s=self.stimulus,
                                  params=params)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
        monodomain_result = vur.vector().norm("l2")

        # Create other solver and solve
        solver = BasicMonodomainSolver(self.mesh, self.time,
                                       self.M_i, I_s=self.stimulus)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
        basic_monodomain_result = vur.vector().norm("l2")

        print("monodomain_result = ", monodomain_result)
        print("basic_monodomain_result = ", basic_monodomain_result)
        assert_almost_equal(monodomain_result, basic_monodomain_result,
                               1e-13)

    @fast
    def test_compare_direct_iterative(self):
        "Test that direct and iterative solution give comparable results."
        self.setUp()

        # Create solver and solve
        params = MonodomainSolver.default_parameters()
        params["linear_solver_type"] = "direct"
        solver = MonodomainSolver(self.mesh, self.time,
                                  self.M_i, I_s=self.stimulus,
                                  params=params)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, v) = fields
            a = v.vector().norm("l2")

        # Create solver and solve using iterative means
        params = MonodomainSolver.default_parameters()
        params["linear_solver_type"] = "iterative"
        params["krylov_solver"]["monitor_convergence"] = True
        solver = MonodomainSolver(self.mesh, self.time,
                                  self.M_i, I_s=self.stimulus,
                                  params=params)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, v) = fields
            b = v.vector().norm("l2")

        print("lu gives ", a)
        print("krylov gives ", b)
        assert_almost_equal(a, b, 1e-4)
