#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# A basic practical example of how to use the cbcbeat module, in
# particular how to solve the monodomain equations coupled to a
# moderately complex cell model using the splitting solver provided by
# cbcbeat.
#
# How to use the cbcbeat module to solve a cardiac EP problem
# ===========================================================
#
# This demo shows how to
# * Use a cardiac cell model from supported cell models
# * Define a cardiac model based on a mesh and other input
# * Use and customize the main solver (SplittingSolver)

# Import the cbcbeat module
import matplotlib.pyplot as plt
import dolfin

# Turn off adjoint functionality
import cbcbeat
from cbcbeat import (
    Tentusscher_panfilov_2006_epi_cell,
    CardiacModel,
    SplittingSolver,
    backend,
)

# Turn on FFC/FEniCS optimizations
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 3


if cbcbeat.dolfinimport.has_dolfin_adjoint:
    dolfin.parameters["adjoint"]["stop_annotating"] = True

# Define the computational domain
mesh = dolfin.UnitSquareMesh(100, 100)
time = backend.Constant(0.0)

# Define the conductivity (tensors)
M_i = 2.0
M_e = 1.0

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = Tentusscher_panfilov_2006_epi_cell()

# Define some external stimulus
stimulus = dolfin.Expression("10*t*x[0]", t=time, degree=1)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, time, M_i, M_e, cell_model, stimulus)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5  # Second order splitting scheme
ps["pde_solver"] = "monodomain"  # Use Monodomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "RL1"  # 1st order Rush-Larsen for the ODEs
ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
ps["MonodomainSolver"]["algorithm"] = "cg"
ps["MonodomainSolver"]["preconditioner"] = "petsc_amg"

solver = SplittingSolver(cardiac_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
dt = 0.1
T = 1.0
interval = (0.0, T)

timer = dolfin.Timer("XXX Forward solve")  # Time the total solve

# Solve!
for timestep, fields in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)

    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

timer.stop()

# Visualize some results
plt.figure()
dolfin.plot(vs[0], title="Transmembrane potential (v) at end time")
plt.savefig("TransmembranePot.png")
plt.figure()
dolfin.plot(vs[-1], title="1st state variable (s_0) at end time")
plt.savefig("s_0(T).png")
# List times spent
dolfin.list_timings(dolfin.TimingClear.keep, [dolfin.TimingType.user])

print("Success!")
