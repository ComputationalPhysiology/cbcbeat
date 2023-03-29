#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# A basic practical example of how to use the cbcbeat module, in
# particular how to solve the bidomain equations coupled to a
# moderately complex cell model using the splitting solver provided by
# cbcbeat and to compute a sensitivity.
#
# How to compute a sensitivity (functional gradient) using cbcbeat
# ================================================================
#
# This demo shows how to
# * Use a cardiac cell model from supported cell models
# * Define a cardiac model based on a mesh and other input
# * Use and customize the main solver (SplittingSolver)
# * Compute the sensitivity (gradient) of an objective functional


# Import the cbcbeat module
from cbcbeat import *
import numpy.random

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Define the computational domain
mesh = UnitSquareMesh(100, 100)
time = Constant(0.0)

# Create synthetic conductivity
Q = FunctionSpace(mesh, "DG", 0)
M_i = Function(Q)
M_i.vector()[:] = 0.1 * (numpy.random.rand(Q.dim()) + 1.0)

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = Tentusscher_panfilov_2006_epi_cell()

# Define some external stimulus
stimulus = Expression("(x[0] > 0.9 && t <= 1.0) ? 30.0 : 0.0", t=time, degree=0)

# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(mesh, time, M_i, None, cell_model, stimulus)

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
k = 0.01
T = 0.1
interval = (0.0, T)

# Solve forward problem
for (timestep, fields) in solver.solve(interval, k):
    print("(t_0, t_1) = (%g, %g)" % timestep)
    (vs_, vs, vur) = fields

# Define functional of interest
j = inner(vs, vs) * dx * dt[FINISH_TIME]
J = Functional(j)

# Indicate the control parameter of interest
m = Control(M_i)

# Compute the gradient, and project it into the right space
dJdm = compute_gradient(J, m, project=True)

# Visualize some results
plot(vs[0], title="Transmembrane potential (v) at end time")
plot(vs[1], title="1st state variable (s_0) at end time")
plot(dJdm, title="Sensitivity with respect to M_i")

interactive()
