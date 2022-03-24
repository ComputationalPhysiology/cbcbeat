"""
Verify the correctness of the test splitting solver with an analytic solution
"""

__author__ = "Simon W. Funke (simon@simula.no) and Jakob Schreiner, 2018"
import pytest
import cbcbeat
from pathlib import Path


from testutils import medium
from cbcbeat.gossplittingsolver import GOSSplittingSolver
from cbcbeat.dolfinimport import (
    Expression,
    Constant,
    UnitSquareMesh,
)
from cbcbeat import errornorm
from cbcbeat.utils import convergence_rate

try:
    import goss

    has_goss = True
except ImportError:
    has_goss = False

require_goss = pytest.mark.skipif(
    not has_goss,
    reason="goss is required to run the test",
)

here = Path(__file__).parent.absolute()


def main(N, dt, T, theta=0.5):
    """Run bidomain MMA."""

    # Exact solutions
    u_exact_str = "-cos(pi*x[0])*cos(pi*x[1])*sin(t)/2.0"
    v_exact_str = "cos(pi*x[0])*cos(pi*x[1])*sin(t)"
    s_exact_str = "-cos(pi*x[0])*cos(pi*x[1])*cos(t)"

    # Source term
    ac_str = (
        "cos(t)*cos(pi*x[0])*cos(pi*x[1]) + pow(pi, 2)*cos(pi*x[0])*cos(pi*x[1])*sin(t)"
    )
    ac_str += " - " + s_exact_str
    # Create data
    mesh = UnitSquareMesh(N, N)
    time = Constant(0.0)
    # V = FunctionSpace(mesh, "CG", 1)

    # We choose the FHN parameters such that s=1 and I_s=v
    model = goss.dolfinutils.DOLFINParameterizedODE(
        here.joinpath("../../cbcbeat/cellmodels/simple_model"), field_states=["v", "s"]
    )

    model.set_initial_conditions(v=0, s=Expression(s_exact_str, degree=5, t=0))

    ps = GOSSplittingSolver.default_parameters()
    ps["pde_solver"] = "bidomain"
    ps["theta"] = theta
    ps["ode_solver"]["solver"] = "RK4"
    ps["enable_adjoint"] = False
    ps["BidomainSolver"]["linear_solver_type"] = "direct"
    ps["BidomainSolver"]["use_avg_u_constraint"] = True
    ps["apply_stimulus_current_to_pde"] = True

    stimulus = Expression(ac_str, t=time, dt=dt, degree=5)
    M_i = 1.0
    M_e = 1.0
    heart = cbcbeat.CardiacModel(mesh, time, M_i, M_e, model, stimulus)
    splittingsolver = GOSSplittingSolver(heart, params=ps)

    # Define exact solution (Note: v is returned at end of time
    # interval(s), u is computed at somewhere in the time interval
    # depending on theta)
    v_exact = Expression(v_exact_str, t=T, degree=1)
    u_exact = Expression(u_exact_str, t=T - (1.0 - theta) * dt, degree=3)

    pde_vs_, pde_vs, vur = splittingsolver.solution_fields()
    pde_vs_.assign(model.initial_conditions())

    solutions = splittingsolver.solve((0, T), dt)
    for (t0, t1), (vs_, vs, vur) in solutions:
        pass
    # Compute errors
    v = vs.split(deepcopy=True)[0]
    u = vur.split(deepcopy=True)[1]
    v_error = errornorm(v_exact, v, "L2", degree_rise=2)
    u_error = errornorm(u_exact, u, "L2", degree_rise=2)

    return v_error, u_error, mesh.hmin(), dt, T


@require_goss
@pytest.mark.goss
@medium
def test_spatial_convergence():
    """Take a very small timestep, reduce mesh size, expect 2nd order convergence."""
    v_errors = []
    u_errors = []
    hs = []
    dt = 0.01
    T = 1.0
    for N in (5, 10, 20, 40):
        v_error, u_error, h, dt_, T = main(N, dt, T)
        v_errors.append(v_error)
        u_errors.append(u_error)
        hs.append(h)

    v_rates = convergence_rate(hs, v_errors)
    u_rates = convergence_rate(hs, u_errors)
    print("dt, T = ", dt, T)
    print("v_errors = ", v_errors)
    print("u_errors = ", u_errors)
    print("v_rates = ", v_rates)
    print("u_rates = ", u_rates)

    assert all(v > 1.9 for v in v_rates), "Failed convergence for v"
    assert all(u > 1.85 for u in u_rates), "Failed convergence for u"


@require_goss
@pytest.mark.goss
@medium
def test_temporal_convergence():
    """Take a small mesh, reduce timestep size, expect 2nd order convergence."""
    v_errors = []
    u_errors = []
    dts = []
    dt = 1.0
    T = 1.0
    N = 150
    for level in (0, 1, 2, 3):
        a = dt / (2**level)
        v_error, u_error, h, a, T = main(N, a, T)
        v_errors.append(v_error)
        u_errors.append(u_error)
        dts.append(a)

    v_rates = convergence_rate(dts, v_errors)
    u_rates = convergence_rate(dts, u_errors)
    print("v_errors = ", v_errors)
    print("u_errors = ", u_errors)
    print("v_rates = ", v_rates)
    print("u_rates = ", u_rates)

    assert v_rates[-1] > 1.95, "Failed convergence for v"
    # Increase spatial resolution to get better temporal convergence for u
    assert u_rates[-1] > 1.78, "Failed convergence for u"
