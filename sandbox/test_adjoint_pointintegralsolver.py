"""
Basic test for replaying and computing the gradient of a model that
uses PointIntegralSolver.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = []

from dolfin import *
from dolfin_adjoint import *
from cbcbeat import *

def main(CellModel, Solver):

    parameters["form_compiler"]["quadrature_degree"] = 2
    parameters["form_compiler"]["cpp_optimize"] = True
    #flags = ["-O3", "-ffast-math", "-march=native"]
    #parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

    # Create cell model
    cell = CellModel()
    num_states = cell.num_states()

    # Create function spaces
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    S = BasicSplittingSolver.state_space(mesh, num_states)
    VS = V*S

    # Create solution function and set its initial value
    vs = Function(VS, name="vs")
    vs.assign(project(cell.initial_conditions(), VS, annotate=False),
              annotate=True)
    (v, s) = split(vs)

    # Define the right-hand-side of the system of ODEs: Dt(u) = rhs(u)
    # Note that sign of the ionic current
    (w, r) = TestFunctions(VS)
    rhs = inner(cell.F(v, s), r) + inner(- cell.I(v, s), w)
    form = rhs*dP

    # In the beginning...
    time = Constant(0.0)

    # Create scheme
    scheme = Solver(form, vs, time)
    scheme.t().assign(float(time))  # FIXME: Should this be scheme or
                                    # solver and why is this needed?
    info(scheme)

    # Create solver
    solver = PointIntegralSolver(scheme)

    # Time step
    k = 0.1
    solver.step(k)

    return vs

if __name__ == "__main__":

    # Run forward
    vs = main(FitzHughNagumoManual, BackwardEuler)

    # Output what has happened
    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    # Replay
    print "Replaying"
    success = replay_dolfin(tol=0.0, stop=True)
    assert (success == True), "Replay failed." # !? How does this actually work

    # Define favorite functional
    print "Computing gradient"
    J = Functional(inner(vs, vs)*dx*dt[FINISH_TIME])
    dJdic = compute_gradient(J, Control(vs))

    # Plot gradient
    plot(dJdic, interactive=True, title="dJdic")


