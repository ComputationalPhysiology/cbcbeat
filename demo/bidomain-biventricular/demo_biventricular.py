#!/usr/bin/env python
#  -*- coding: utf-8 -*-
#
# How to use the cbcbeat module with non-trivial mesh and conductivities
# ======================================================================
#
# This is a fairly realistic demo illustrating how to use cbcbeat for
# a non-trivial bidomain simulation involving real meshes of the left
# and right ventricle, stimulation data based on mesh functions etc.
#
# The run time is fairly long, so for demo purposes, just try the
# following. First, run the script 'preprocess_conductivities.py' to
# generate conductivities from the other data (press q after):
#
# $ python preprocess_conductivities.py
#
# Next, run the demo with a low end time to see if things seem to be
# running.
#
# $ python demo_biventricular.py --T 1.0
#
# To run in parallel using MPI, try for instance
#
# $ mpirun -n 2 python demo_biventricular.py --T 10.0
#

__author__ = "Marie E. Rognes (meg@simula.no) and Johan E. Hake"
try:
    from ufl_legacy.log import info, info_blue, error
except ImportError:
    from ufl.log import info, info_blue, error
import dolfin
from cbcbeat import (
    backend,
    FitzHughNagumoManual,
    Tentusscher_panfilov_2006_M_cell,
    CardiacModel,
    SplittingSolver,
)
import time


def setup_application_parameters():
    # Setup application parameters and parse from command-line
    application_parameters = dolfin.Parameters("Application")
    application_parameters.add("T", 10.0)  # End time  (ms)
    application_parameters.add("timestep", 0.1)  # Time step (ms)
    application_parameters.add(
        "directory", "results_%s" % time.strftime("%Y_%d%b_%Hh_%Mm")
    )
    application_parameters.add("stimulus_amplitude", 30.0)
    application_parameters.add("healthy", True)
    application_parameters.add("cell_model", "FitzHughNagumo")
    application_parameters.parse()
    info(application_parameters, True)
    return application_parameters


def setup_general_parameters():
    # Adjust some general FEniCS related parameters
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2


def setup_conductivities(mesh, application_parameters):
    # Load fibers and sheets
    Vv = dolfin.VectorFunctionSpace(mesh, "DG", 0)
    fiber = backend.Function(Vv)
    dolfin.File("data/fibers.xml.gz") >> fiber
    sheet = backend.Function(Vv)
    dolfin.File("data/sheet.xml.gz") >> sheet
    cross_sheet = backend.Function(Vv)
    dolfin.File("data/cross_sheet.xml.gz") >> cross_sheet

    # Extract stored conductivity data.
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    if application_parameters["healthy"] is True:
        info_blue("Using healthy conductivities")
        g_el_field = backend.Function(V, "data/healthy_g_el_field.xml.gz", name="g_el")
        g_et_field = backend.Function(V, "data/healthy_g_et_field.xml.gz", name="g_et")
        g_en_field = backend.Function(V, "data/healthy_g_en_field.xml.gz", name="g_en")
        g_il_field = backend.Function(V, "data/healthy_g_il_field.xml.gz", name="g_il")
        g_it_field = backend.Function(V, "data/healthy_g_it_field.xml.gz", name="g_it")
        g_in_field = backend.Function(V, "data/healthy_g_in_field.xml.gz", name="g_in")
    else:
        info_blue("Using ischemic conductivities")
        g_el_field = backend.Function(V, "data/g_el_field.xml.gz", name="g_el")
        g_et_field = backend.Function(V, "data/g_et_field.xml.gz", name="g_et")
        g_en_field = backend.Function(V, "data/g_en_field.xml.gz", name="g_en")
        g_il_field = backend.Function(V, "data/g_il_field.xml.gz", name="g_il")
        g_it_field = backend.Function(V, "data/g_it_field.xml.gz", name="g_it")
        g_in_field = backend.Function(V, "data/g_in_field.xml.gz", name="g_in")

    # Construct conductivity tensors from directions and conductivity
    # values relative to that coordinate system
    A = dolfin.as_matrix(
        [
            [fiber[0], sheet[0], cross_sheet[0]],
            [fiber[1], sheet[1], cross_sheet[1]],
            [fiber[2], sheet[2], cross_sheet[2]],
        ]
    )
    from ufl import diag

    M_e_star = diag(dolfin.as_vector([g_el_field, g_et_field, g_en_field]))
    M_i_star = diag(dolfin.as_vector([g_il_field, g_it_field, g_in_field]))
    M_e = A * M_e_star * A.T
    M_i = A * M_i_star * A.T

    gs = (g_il_field, g_it_field, g_in_field, g_el_field, g_et_field, g_en_field)

    return (M_i, M_e, gs)


def setup_cell_model(params):
    option = params["cell_model"]
    if option == "FitzHughNagumo":
        # Setup cell model based on parameters from G. T. Lines, which
        # seems to be a little more excitable than the default
        # FitzHugh-Nagumo parameters from the Sundnes et al book.
        k = 0.00004
        Vrest = -85.0
        Vthreshold = -70.0
        Vpeak = 40.0
        k = 0.00004
        l = 0.63
        b = 0.013
        v_amp = Vpeak - Vrest
        cell_parameters = {
            "c_1": k * v_amp**2,
            "c_2": k * v_amp,
            "c_3": b / l,
            "a": (Vthreshold - Vrest) / v_amp,
            "b": l,
            "v_rest": Vrest,
            "v_peak": Vpeak,
        }
        cell_model = FitzHughNagumoManual(cell_parameters)
    elif option == "tenTusscher":
        cell_model = Tentusscher_panfilov_2006_M_cell()
        # cell_model = Tentusscher_2004_mcell()
    else:
        error("Unrecognized cell model option: %s" % option)

    return cell_model


def setup_cardiac_model(application_parameters):
    # Initialize the computational domain in time and space
    time = backend.Constant(0.0)
    mesh = dolfin.Mesh("data/mesh115_refined.xml.gz")
    mesh.coordinates()[:] /= 1000.0  # Scale mesh from micrometer to millimeter
    mesh.coordinates()[:] /= 10.0  # Scale mesh from millimeter to centimeter
    mesh.coordinates()[:] /= 4.0  # Scale mesh as indicated by Johan/Molly

    # Setup conductivities
    (M_i, M_e, gs) = setup_conductivities(mesh, application_parameters)

    # Setup cell model
    cell_model = setup_cell_model(application_parameters)

    # Define some simulation protocol (use cpp expression for speed)
    stimulation_cells = dolfin.MeshFunction(
        "size_t", mesh, "data/stimulation_cells.xml.gz"
    )

    V = dolfin.FunctionSpace(mesh, "DG", 0)
    from stimulation import cpp_stimulus

    amp = application_parameters["stimulus_amplitude"]
    pulse = dolfin.CompiledExpression(
        dolfin.compile_cpp_code(cpp_stimulus).Stimulus(),
        element=V.ufl_element(),
        t=time._cpp_object,
        amplitude=amp,
        duration=10.0,
        cell_data=stimulation_cells,
    )

    # Initialize cardiac model with the above input
    heart = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus=pulse)
    return (heart, gs)


def main(store_solutions=True):
    dolfin.set_log_level(dolfin.LogLevel.INFO)

    dolfin.begin("Setting up application parameters")
    application_parameters = setup_application_parameters()
    setup_general_parameters()
    dolfin.end()

    dolfin.begin("Setting up cardiac model")
    (heart, gs) = setup_cardiac_model(application_parameters)
    dolfin.end()

    # Extract end time and time-step from application parameters
    T = application_parameters["T"]
    k_n = application_parameters["timestep"]

    # Since we know the time-step we want to use here, set it for the
    # sake of efficiency in the bidomain solver
    dolfin.begin("Setting up splitting solver")
    params = SplittingSolver.default_parameters()
    params["theta"] = 1.0
    params["CardiacODESolver"]["scheme"] = "GRL1"
    # params["BidomainSolver"]["linear_solver_type"] = "direct"
    # params["BidomainSolver"]["default_timestep"] = k_n
    solver = SplittingSolver(heart, params=params)
    dolfin.end()

    # Extract solution fields from solver
    (vs_, vs, vu) = solver.solution_fields()

    # Extract and assign initial condition
    vs_.assign(heart.cell_models().initial_conditions())
    # Store parameters
    directory = application_parameters["directory"]
    application_params_file = dolfin.File("%s/application_parameters.xml" % directory)
    application_params_file << application_parameters
    solver_params_file = dolfin.File("%s/solver_parameters.xml" % directory)
    solver_params_file << solver.parameters
    params_file = dolfin.File("%s/parameters.xml" % directory)
    params_file << dolfin.parameters

    # Set-up solve
    solutions = solver.solve((0, T), k_n)

    # Set up storage
    mpi_comm = heart.domain().mpi_comm()
    vs_file = dolfin.HDF5File(mpi_comm, "%s/vs" % directory, "w")
    u_file = dolfin.HDF5File(mpi_comm, "%s/u" % directory, "w")

    # Store initial solutions:
    if store_solutions:
        vs_file.write(vs_, "/function", 0.0)
        u = vu.split()[1]
        u_file.write(u, "/function", 0.0)

    # (Compute) and store solutions
    timer = dolfin.Timer("Forward solve")
    theta = params["theta"]
    for timestep, fields in solutions:
        # Store hdf5
        print("Solving on ", timestep)
        if store_solutions:
            (t0, t1) = timestep
            vs_file.write(vs, "/function", t1)
            u_file.write(u, "/function", t0 + theta * (t1 - t0))
    dolfin.plot(vs[0], title="v")

    vs_file.close()
    u_file.close()
    timer.stop()

    # List timings
    dolfin.list_timings(
        dolfin.TimingClear.keep,
        [
            dolfin.TimingType.wall,
        ],
    )
    return (gs, solver)


if __name__ == "__main__":
    main()
    import matplotlib.pyplot as plt

    plt.savefig("vs.png")
