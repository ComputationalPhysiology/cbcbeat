"""
This demo shows how to:
- Solve a cardiac cell model over a domain (for each vertex in that domain)
- How to set heterogeneous (spatially varying) cell model parameters
- How to set FEniCS parameters for improved computational efficiency
- How to replay the forward solve using via dolfin-adjoint
- How to output the recorded forward solve
"""

__author__ = "Marie E. Rognes (meg@simula.no)"
from ufl.log import info_green
import dolfin
from cbcbeat import backend, Tentusscher_2004_mcell, CardiacODESolver

# For computing faster
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = flags
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4


def forward():
    info_green("Running forward model")

    # Set-up domain in space and time
    N = 10
    mesh = dolfin.UnitSquareMesh(N, N)
    time = backend.Constant(0.0)

    # Choose your favorite cell model
    model = Tentusscher_2004_mcell()

    # You can set spatially varying cell model parameters e.g. as:
    model.set_parameters(K_mNa=dolfin.Expression("40*sin(pi*x[0])", degree=4))

    # Add some stimulus
    stimulus = dolfin.Expression("100*t", t=time, degree=0)

    Solver = CardiacODESolver
    params = Solver.default_parameters()
    solver = Solver(mesh, time, model, I_s=stimulus, params=params)

    # Set-up initial conditions
    (vs_, vs) = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Set-up other solution parameters
    dt = 0.2
    interval = (0.0, 1.0)

    # Generator for solutions
    solutions = solver.solve(interval, dt)

    # Do something with the solutions
    times = []
    for (t0, t1), vs in solutions:
        times.append(t1)
        print(vs.vector().get_local())
    dolfin.plot(vs[0], title="v")
    import matplotlib.pyplot as plt

    plt.savefig("vs.png")


def replay():
    import dolfin_adjoint

    info_green("Replaying forward model")

    # Output some html
    dolfin_adjoint.adj_html("forward.html", "forward")

    # Replay
    dolfin.parameters["adjoint"]["stop_annotating"] = True
    success = dolfin_adjoint.replay_dolfin(tol=0.0, stop=True)
    if success:
        info_green("Replay successful")
    else:
        info_green("Replay failed")


if __name__ == "__main__":
    # Run forward model
    forward()

    import cbcbeat

    if cbcbeat.dolfinimport.has_dolfin_adjoint:
        # Replay
        replay()
