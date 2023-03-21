import dolfin
import math
from pathlib import Path
import goss
import cbcbeat
import gotran

from cbcbeat.gossplittingsolver import GOSSplittingSolver

here = Path(__file__).absolute().parent


def extract_subfunction(vs, index=0):
    VS = vs.function_space()
    V = VS.sub(index).collapse()
    fa = dolfin.FunctionAssigner(V, VS.sub(index))
    v = dolfin.Function(V)
    fa.assign(v, vs.sub(index))
    return v


def error(v1, v2, v3=1.0):
    return dolfin.assemble(((v1 - v2) / v3) ** 2 * dolfin.dx)


def test_monodomain_goss_DG():
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 3

    # First we initialize some parameters for the solver

    ps = GOSSplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = 0.5
    ps["MonodomainSolver"]["family"] = "DG"
    ps["theta"] = 0.5
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    ps["ode_solver"]["solver"] = "GRL1"
    ps["ode_solver"]["num_threads"] = 6
    ps["ode_solver"]["space"] = "DG_1"

    # We define the domain, i.e a unit square that we scale by a factor of 10.0
    domain = dolfin.UnitSquareMesh(10, 10)
    domain.coordinates()[:] *= 10

    # Next we define the stimulus domain which will be a circle with
    #

    class StimSubDomain(dolfin.SubDomain):
        def __init__(self, center, radius):
            self.x0, self.y0 = center
            self.radius = radius
            super().__init__()

        def inside(self, x, on_boundary):
            r = math.sqrt((x[0] - self.x0) ** 2 + (x[1] - self.y0) ** 2)
            if r < self.radius:
                return True
            return False

    # Create a mesh function containing markers for the stimulus domain
    stim_domain = dolfin.MeshFunction("size_t", domain, domain.topology().dim(), 0)
    # Set all markers to zero
    stim_domain.set_all(0)
    # We mark the stimulus domain with a different marker
    stim_marker = 1
    domain_size = domain.coordinates().max()
    # Create a domain
    stim_subdomain = StimSubDomain(
        center=(domain_size / 2.0, 0.0),
        radius=domain_size / 5.0,
    )
    # And mark the domain
    stim_subdomain.mark(stim_domain, stim_marker)

    # Make a constant representing time
    time = dolfin.Constant(0.0)
    # Next we create the stimulus protocol
    stimulus = dolfin.Expression("10*t*x[0]", t=time, degree=1)
    # We load the ode representing the ionic model

    gotran_ode = gotran.load_ode(
        here / "../../cbcbeat/cellmodels/tentusscher_panfilov_2006_M_cell.ode"
    )

    # and create a cell model with the membrane potential as a field state
    # This will make the membrane potential being spatial dependent-
    states = gotran_ode.state_symbols
    cellmodel = goss.dolfinutils.DOLFINParameterizedODE(
        gotran_ode,
        field_states=states,
    )

    # Do not apply any stimulus in the cell model since we will do this through the bidomain model
    cellmodel.set_parameter("stim_amplitude", 0)

    # Next we need to define the conductivity tensors. Here we use the same parameters as in {ref}`bidomain`

    chi = 12000.0  # cm^{-1}
    s_il = 300.0 / chi  # mS
    s_it = s_il / 2  # mS
    s_el = 200.0 / chi  # mS
    s_et = s_el / 1.2  # mS

    # and make a new conductivity tensor by taking the harmonic mean of the parameters

    sl = s_il * s_el / (s_il + s_el)
    st = s_it * s_et / (s_it + s_et)
    M_i = dolfin.as_tensor(((dolfin.Constant(sl), 0), (0, dolfin.Constant(st))))

    # Create and the CardiacModel in `cbcbeat`. Since we use a monodomain model we don't need to pass the extracellular conductivity

    heart = cbcbeat.CardiacModel(
        domain=domain,
        time=time,
        M_i=M_i,
        M_e=None,
        cell_models=cellmodel,
        stimulus=stimulus,
    )

    # and initialize the solver
    solver_DG = GOSSplittingSolver(heart, ps, V_index=states.index("V"))

    ps["MonodomainSolver"]["family"] = "CG"
    ps["ode_solver"]["space"] = "CG_1"
    solver_CG = GOSSplittingSolver(heart, ps, V_index=states.index("V"))

    # We extract the membrane potential from the solution fields
    interval = (0, 10.0)
    dt = 0.1

    _, vs0_CG, _ = solver_CG.solution_fields()
    v0_CG = extract_subfunction(vs0_CG, index=states.index("V"))

    for (timestep, fields) in solver_CG.solve(interval, dt):

        _, vs_CG, _ = fields
    v_CG = extract_subfunction(vs_CG, index=states.index("V"))

    _, vs0_DG, _ = solver_DG.solution_fields()
    v0_DG = extract_subfunction(vs0_DG, index=states.index("V"))

    for (timestep, fields) in solver_DG.solve(interval, dt):

        _, vs_DG, _ = fields

    v_DG = extract_subfunction(vs_DG, index=states.index("V"))

    v0_DG_intCG = dolfin.Function(v0_CG.function_space())
    v0_DG_intCG.interpolate(v0_DG)

    # Check initial conditions
    assert error(v0_CG, v0_DG_intCG) < 1e-12

    v_DG_intCG = dolfin.Function(v_CG.function_space())
    v_DG_intCG.interpolate(v_DG)

    # Check final solution
    assert error(v_CG, v_DG_intCG, v_CG) < 0.1  #

    # fig = plt.figure()
    # cbcbeat.plot(v_CG)
    # plt.title("CG")
    # fig.savefig("CG.png")
    # fig = plt.figure()
    # cbcbeat.plot(v_DG)
    # plt.title("DG")
    # fig.savefig("DG.png")
