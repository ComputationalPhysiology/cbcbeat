#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# .. _first_example
#
# How to use the cbcbeat to solve multiple cardiac ODEs in one mesh
# =================================================================
#
# This demo shows how to
# * Use MultiCellModel to define domains with multiple cell models and
#   solve using the BasicCardiacODESolver
#
# Warning: This functionality is experimental and not yet very practical

__author__ = "Marie E Rognes"

import numpy
import sys

from cbcbeat import *

# Set FFC some parameters
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

parameters["adjoint"]["stop_annotating"] = True

# Define space and time
n = 40
mesh = UnitSquareMesh(n, n)
time = Constant(0.0)

# Surface to volume ratio and membrane capacitance
chi = 140.0     # mm^{-1}
C_m = 0.01      # mu F / mm^2

# Define conductivity tensor
M_i = 1.0
M_e = 1.0

# Define two different cell models on the mesh
c0 = Beeler_reuter_1977()
#c0 = Fenton_karma_1998_BR_altered()
c1 = FitzHughNagumoManual()
markers = CellFunction("uint", mesh, 0)
markers.array()[0:mesh.num_cells()/2] = 2
cell_model = MultiCellModel((c0, c1), (2, 0), markers)
plot(markers, title="Markers")

solver = BasicCardiacODESolver(mesh, time, cell_model,
                               I_s=Expression("100*x[0]*exp(-t)",
                                              t=time, degree=1),
                               params=None)
dt = 0.01
T = 100*dt

# Assign initial conditions
(vs_, vs) = solver.solution_fields()
ic = cell_model.initial_conditions()
vs_.assign(cell_model.initial_conditions())
vs.assign(vs_)

solutions = solver.solve((0.0, T), dt)

V = vs.split()[0].function_space().collapse()
v = Function(V)
for ((t0, t1), y) in solutions:
    v.assign(y.split(deepcopy=True)[0])
    plot(v)

interactive()
