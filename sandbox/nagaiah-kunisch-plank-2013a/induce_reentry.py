from cbcbeat import *

class S1S2Stimulation(Expression):
    "A S1-S2 stimulation protocol."
    def __init__(self, t):
        self.t = t # ms
    def eval(self, value, x):
        # S1 stimulus
        if (float(self.t) >= 0 and float(self.t) <= 5
            and near(x[0], 0.0)):
            value[0] = 100. # mV
        else:
            value[0] = 0.0

def main():

    # Create mesh
    n = 128
    mesh = RectangleMesh(0.0, 0.0, 2.0, 2.0, n, n)

    # Define time and timestep dt
    time = Constant(0.0)
    dt = 0.04 # ms

    # Create conductivity tensors
    M_i = diag(as_vector([2.0e-3, 3.1e-4])) # S/cm
    M_e = diag(as_vector([2.0e-3, 1.3e-3])) # S/cm

    # Create cell model
    cell_model = RogersMcCulloch()

    # Create stimulus
    I_s = S1S2Stimulation(t=time)

    # Create cardiac model from above inputs
    tissue = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus={0: I_s})

    # Create solver and preset timestep for
    params = SplittingSolver.default_parameters()
    params["BidomainSolver"]["linear_solver_type"] = "direct"
    params["BidomainSolver"]["default_timestep"] = dt
    params["theta"] = 1.0
    params["CardiacODESolver"]["scheme"] = "BackwardEuler"
    solver = SplittingSolver(tissue, params=params)

    # Extract initial conditions from the cell model and set the solver
    (vs_, vs, vu) = solver.solution_fields()
    ics = tissue.cell_model().initial_conditions()
    vs_.assign(ics, solver.VS)

    # Set-up solve
    T = 100*dt # ms
    solutions = solver.solve((0, T), dt)
    v = Function(solver.VS.sub(0).collapse())
    for (timestep, fields) in solutions:
        print timestep
        v.assign(vs.split(deepcopy=True)[0], annotate=False)
        plot(v, title="v")


if __name__ == "__main__":
    main()
    interactive()
