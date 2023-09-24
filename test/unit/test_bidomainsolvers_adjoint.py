"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = (
    "Marie E. Rognes (meg@simula.no), 2013, and Simon W. Funke (simon@simula.no) 2014"
)
__all__ = ["TestBidomainSolversAdjoint"]

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl
from cbcbeat import backend, BasicBidomainSolver, BidomainSolver
from modelparameters.logger import info_green
from dolfin import parameters, UnitCubeMesh, Expression
from testutils import (
    assert_equal,
    fast,
    slow,
    adjoint,
    parametrize,
    assert_greater,
    require_dolfin_adjoint,
)

import sys

try:
    import dolfin_adjoint
except ImportError:
    pass

args = (
    sys.argv[:1]
    + """
                      --petsc.bidomain_ksp_monitor_true_residual
                      --petsc.bidomain_ksp_view
                      --petsc.ksp_view
                      """.split()
)
parameters.parse(args)


@require_dolfin_adjoint
class TestBidomainSolversAdjoint(object):
    """Test adjoint functionality for the bidomain solver."""

    def setup(self):
        self.mesh = UnitCubeMesh(5, 5, 5)
        self.time = backend.Constant(0.0)

        # Create stimulus
        self.stimulus = Expression("2.0", degree=1)

        # Create applied current
        self.applied_current = Expression("sin(2*pi*x[0])*t", t=self.time, degree=3)

        # Create conductivity "tensors"
        self.M_i = 1.0
        self.M_e = 2.0

        self.t0 = 0.0
        self.dt = 0.1
        self.T = 5 * self.dt

    def _setup_solver(self, Solver, solver_type, enable_adjoint=True):
        """Creates the bidomain solver."""

        # Create solver
        params = Solver.default_parameters()

        if Solver == BasicBidomainSolver:
            params.update(
                {
                    "linear_variational_solver": {
                        "linear_solver": "cg" if solver_type == "iterative" else "lu"
                    }
                }
            )
            params.update(
                {
                    "linear_variational_solver": {
                        "krylov_solver": {"relative_tolerance": 1e-12}
                    }
                }
            )
            params.update({"linear_variational_solver": {"preconditioner": "ilu"}})
        else:
            params.update(
                {"linear_solver_type": solver_type, "enable_adjoint": enable_adjoint}
            )
            if solver_type == "iterative":
                params.update({"petsc_krylov_solver": {"relative_tolerance": 1e-12}})
                # params.petsc_krylov_solver.relative_tolerance = 1e-12
            else:
                params.update(
                    {"use_avg_u_constraint": True}
                )  # NOTE: In contrast to iterative
                # solvers, the direct solver does not handle nullspaces consistently,
                # i.e. the solution differes from solve to solve, and hence the Taylor
                # testes would not pass.

        self.solver = Solver(
            self.mesh,
            self.time,
            self.M_i,
            self.M_e,
            I_s=self.stimulus,
            I_a=self.applied_current,
            params=params,
        )

    def _solve(self, ics=None):
        """Runs the forward model with the basic bidomain solver."""
        print("Running forward basic model")

        (vs_, vs) = self.solver.solution_fields()

        solutions = self.solver.solve((self.t0, self.t0 + self.T), self.dt)

        # Set initial conditions
        if ics is not None:
            vs_.interpolate(ics)

        # Solve
        for interval, fields in solutions:
            pass

        return vs

    @adjoint
    @fast
    @parametrize(
        ("Solver", "solver_type", "tol"),
        [
            (BasicBidomainSolver, "direct", 1e-15),
            (BasicBidomainSolver, "iterative", 1e-15),
            (BidomainSolver, "direct", 1e-15),
            (
                BidomainSolver,
                "iterative",
                1e-10,
            ),  # NOTE: The replay is not exact because
            # dolfin-adjoint's overloaded Krylov method is not constent with DOLFIN's
            # (it orthogonalizes the rhs vector as an additional step)
        ],
    )
    def test_replay(self, Solver, solver_type, tol):
        "Test that replay of basic bidomain solver reports success."

        self._setup_solver(Solver, solver_type)
        self._solve()

        # Check replay
        info_green("Running replay basic (%s)" % solver_type)
        success = dolfin_adjoint.replay_dolfin(stop=True, tol=tol)
        assert_equal(success, True)

    def tlm_adj_setup(self, Solver, solver_type):
        """Common code for test_tlm and test_adjoint."""
        self._setup_solver(Solver, solver_type)
        self._solve()
        (vs_, vs) = self.solver.solution_fields()

        # Define functional
        def form(w):
            return ufl.inner(w, w) * ufl.dx

        J = dolfin_adjoint.Functional(
            form(vs) * dolfin_adjoint.dt[dolfin_adjoint.FINISH_TIME]
        )
        m = dolfin_adjoint.Control(vs_)

        # Compute value of functional with current ics
        Jics = backend.assemble(form(vs))

        # Define reduced functional
        def Jhat(ics):
            self._setup_solver(Solver, solver_type, enable_adjoint=False)
            vs = self._solve(ics)
            return backend.assemble(form(vs))

        # Stop annotating
        parameters["adjoint"]["stop_annotating"] = True

        return J, Jhat, m, Jics

    @adjoint
    @slow
    @parametrize(
        ("Solver", "solver_type"),
        [
            (BasicBidomainSolver, "direct"),
            (BasicBidomainSolver, "iterative"),
            (BidomainSolver, "iterative"),
            (BidomainSolver, "direct"),
        ],
    )
    def test_tlm(self, Solver, solver_type):
        """Test that tangent linear model of basic bidomain solver converges at 2nd order."""
        info_green("Running tlm basic (%s)" % solver_type)

        J, Jhat, m, Jics = self.tlm_adj_setup(Solver, solver_type)

        # Check TLM correctness
        dJdics = dolfin_adjoint.compute_gradient_tlm(J, m, forget=False)
        assert dJdics is not None, "Gradient is None (#fail)."
        conv_rate_tlm = dolfin_adjoint.taylor_test(Jhat, m, Jics, dJdics)

        # Check that minimal convergence rate is greater than some given number
        assert_greater(conv_rate_tlm, 1.9)

    @adjoint
    @slow
    @parametrize(
        ("Solver", "solver_type"),
        [
            (BasicBidomainSolver, "direct"),
            (BasicBidomainSolver, "iterative"),
            (BidomainSolver, "iterative"),
            (BidomainSolver, "direct"),
        ],
    )
    def test_adjoint(self, Solver, solver_type):
        """Test that adjoint model of basic bidomain solver converges at 2nd order."""
        info_green("Running adjoint basic (%s)" % solver_type)

        J, Jhat, m, Jics = self.tlm_adj_setup(Solver, solver_type)

        # Check adjoint correctness
        dJdics = dolfin_adjoint.compute_gradient(J, m, forget=False)
        assert dJdics is not None, "Gradient is None (#fail)."
        conv_rate = dolfin_adjoint.taylor_test(Jhat, m, Jics, dJdics, seed=1e-3)

        # Check that minimal convergence rate is greater than some given number
        assert_greater(conv_rate, 1.9)
