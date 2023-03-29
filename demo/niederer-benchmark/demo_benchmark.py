#!/usr/bin/env python
#  -*- coding: utf-8 -*-
#
# How to use the cbcbeat module to solve the Niederer et al 2011 benchmark
# ========================================================================
#
# This demo shows how to
# * Use cbcbeat to solve the 2011 Niederer benchmark [Phil. Trans. R. Soc.]
# * Use petsc4py to customize the PETSc linear solvers in detail
# * Use the FEniCS Parameter system for handling application parameters
# * Run cbcbeat in parallel (just use mpirun -n N python ....)
#
# Run this demo with e.g.
# $ mpirun -n 2 python demo_benchmark.py --T 100 --casedir results-100
#
# Recommend analyzing the outputs in serial

__author__ = "Johan Hake and Simon W. Funke (simon@simula.no)"

# Modified by Marie E. Rognes (meg@simula.no), 2014

try:
    pass
except:
    print("Cannot import petsc4py")

from dolfin import *
from cbcbeat import *

import sys
import numpy

args = (
    [sys.argv[0]]
    + """
                       --petsc.ksp_type cg
                       --petsc.pc_type gamg
                       --petsc.pc_gamg_verbose 10
                       --petsc.pc_gamg_square_graph 0
                       --petsc.pc_gamg_coarse_eq_limit 3000
                       --petsc.mg_coarse_pc_type redundant
                       --petsc.mg_coarse_sub_pc_type lu
                       --petsc.mg_levels_ksp_type richardson
                       --petsc.mg_levels_ksp_max_it 3
                       --petsc.mg_levels_pc_type sor
                       """.split()
)
parameters.parse(argv=args)

parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["representation"] = "uflacs"


def define_conductivity_tensor(chi, C_m):

    # Conductivities as defined by page 4339 of Niederer benchmark
    sigma_il = 0.17  # mS / mm
    sigma_it = 0.019  # mS / mm
    sigma_el = 0.62  # mS / mm
    sigma_et = 0.24  # mS / mm

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a * b / (a + b)

    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l / (C_m * chi)  # mm^2 / ms
    s_t = sigma_t / (C_m * chi)  # mm^2 / ms

    # Define conductivity tensor
    M = as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))

    return M


def setup_model(cellmodel, mesh):
    """Set-up cardiac model based on benchmark parameters."""

    # Define time
    time = Constant(0.0)

    # Surface to volume ratio
    chi = 140.0  # mm^{-1}
    # Membrane capacitance
    C_m = 0.01  # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Mark stimulation region defined as [0, L]^3
    S1_marker = 1
    L = 1.5
    S1_subdomain = CompiledSubDomain(
        "x[0] <= L + DOLFIN_EPS && x[1] <= L + DOLFIN_EPS && x[2] <= L + DOLFIN_EPS",
        L=L,
    )
    S1_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    duration = 2.0  # ms
    A = 50000.0  # mu A/cm^3
    cm2mm = 10.0
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms
    I_s = Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )
    # Store input parameters in cardiac model
    stimulus = Markerwise((I_s,), (1,), S1_markers)
    heart = CardiacModel(mesh, time, M, None, cellmodel, stimulus)

    return heart


def cell_model_initial_conditions():
    """Return initial conditions specified in the Niederer benchmark
    for the ten Tuscher & Panfilov cell model."""
    ic = {
        "V": -85.23,  # mV
        "Xr1": 0.00621,
        "Xr2": 0.4712,
        "Xs": 0.0095,
        "m": 0.00172,
        "h": 0.7444,
        "j": 0.7045,
        "d": 3.373e-05,
        "f": 0.7888,
        "f2": 0.9755,
        "fCass": 0.9953,
        "s": 0.999998,
        "r": 2.42e-08,
        "Ca_i": 0.000126,  # millimolar
        "R_prime": 0.9073,
        "Ca_SR": 3.64,  # millimolar
        "Ca_ss": 0.00036,  # millimolar
        "Na_i": 8.604,  # millimolar
        "K_i": 136.89,  # millimolar
    }
    return ic


def run_splitting_solver(mesh, application_parameters):

    # Extract parameters
    T = application_parameters["T"]
    dt = application_parameters["dt"]
    application_parameters["dx"]
    theta = application_parameters["theta"]
    scheme = application_parameters["scheme"]
    preconditioner = application_parameters["preconditioner"]
    store = application_parameters["store"]
    casedir = application_parameters["casedir"]

    # cell model defined by benchmark specifications
    CellModel = Tentusscher_panfilov_2006_epi_cell

    # Set-up solver
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = theta
    ps["MonodomainSolver"]["preconditioner"] = preconditioner
    ps["MonodomainSolver"]["default_timestep"] = dt
    ps["MonodomainSolver"]["use_custom_preconditioner"] = False
    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    ps["CardiacODESolver"]["scheme"] = scheme

    # Disable adjoint annotating and recording (saves memory)
    import cbcbeat

    if cbcbeat.dolfin_adjoint:
        parameters["adjoint"]["stop_annotating"] = True

    # Customize cell model parameters based on benchmark specifications
    cell_inits = cell_model_initial_conditions()
    cellmodel = CellModel(init_conditions=cell_inits)

    # Set-up cardiac model
    heart = setup_model(cellmodel, mesh)

    # Set-up solver and time it
    timer = Timer("SplittingSolver: setup")
    solver = SplittingSolver(heart, ps)
    timer.stop()

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())

    # Set-up separate potential function for post processing
    VS0 = vs.function_space().sub(0)
    V = VS0.collapse()
    v = Function(V)

    # Set-up object to optimize assignment from a function to subfunction
    assigner = FunctionAssigner(V, VS0)
    assigner.assign(v, vs_.sub(0))

    # Output some degrees of freedom
    total_dofs = vs.function_space().dim()
    pde_dofs = V.dim()
    if MPI.rank(MPI.comm_world) == 0:
        print("Total degrees of freedom: ", total_dofs)
        print("PDE degrees of freedom: ", pde_dofs)

    t0 = 0.0

    # Store initial v
    if store:
        vfile = HDF5File(mesh.mpi_comm(), "%s/v.h5" % casedir, "w")
        vfile.write(v, "/function", t0)
        vfile.write(mesh, "/mesh")

    # Solve
    timer = Timer("SplittingSolver: solve and store")
    solutions = solver.solve((t0, T), dt)

    for (i, ((t0, t1), fields)) in enumerate(solutions):
        if (i % 20 == 0) and MPI.rank(MPI.comm_world) == 0:
            info("Reached t=%g/%g, dt=%g" % (t0, T, dt))
        if store:
            assigner.assign(v, vs.sub(0))
            vfile.write(v, "/function", t1)
            vfile.flush()

    if store:
        vfile.close()

    timer.stop()

    return vs


def create_mesh(dx, refinements=0):
    # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
    # with resolution prescribed by benchmark or more refinements

    # Define geometry parameters
    Lx = 20.0  # mm
    Ly = 7.0  # mm
    Lz = 3.0  # mm

    def N(v):
        return int(numpy.rint(v))
    mesh = BoxMesh(
        MPI.comm_world,
        Point(0.0, 0.0, 0.0),
        Point(Lx, Ly, Lz),
        N(Lx / dx),
        N(Ly / dx),
        N(Lz / dx),
    )

    for i in range(refinements):
        print("Performing refinement", i + 1)
        mesh = refine(mesh, redistribute=False)

    return mesh


def forward(application_parameters):

    # Create mesh
    dx = application_parameters["dx"]
    R = application_parameters["refinements"]
    mesh = create_mesh(dx, R)

    # Run solver
    vs = run_splitting_solver(mesh, application_parameters)
    print("Results stored in %s" % application_parameters["casedir"])

    return vs


def init_application_parameters():
    begin("Setting up application parameters")
    application_parameters = Parameters("Niederer-benchmark")
    application_parameters.add("casedir", "results")
    application_parameters.add("theta", 0.5)
    application_parameters.add("store", True)
    application_parameters.add("dt", 0.05)
    application_parameters.add("dx", 0.5)
    application_parameters.add("T", 100.0)
    application_parameters.add("scheme", "GRL1")
    application_parameters.add("preconditioner", "sor")
    application_parameters.add("refinements", 0)
    application_parameters.parse()
    end()

    return application_parameters


if __name__ == "__main__":

    # Default application parameters and parse from command-line
    application_parameters = init_application_parameters()
    application_parameters.parse()

    # Solve benchmark problem with given specifications
    if True:
        timer = Timer("Total forward time")
        vs = forward(application_parameters)
        timer.stop()

        # List timings
        list_timings(TimingClear.keep, [TimingType.wall])

    # Set this to True to also analyze outputs
    if False:
        from analyze_output import compute_activation_times

        compute_activation_times(application_parameters["casedir"])
