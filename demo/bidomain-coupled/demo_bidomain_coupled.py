#!/usr/bin/env python
#  -*- coding: utf-8 -*-
#
# How to use the cbcbeat module to handle the coupling Heart / Torso
# ==================================================================
#
# This demo shows how to
# * Use a cardiac cell model from supported cell models
# * Define a cardiac model based on a mesh and other input
# * Take into account the coupling with the surrounding torso

# Import the cbcbeat module
from cbcbeat import *

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Turn off adjoint functionality
if cbcbeat.dolfin_adjoint:
    parameters["adjoint"]["stop_annotating"] = True

# Define the shape of the subdomain - Used to mark the cells of the mesh
# This function returns 1 if the point (x,y) is in our subdomain and 0 otherwise
def beutel_heart(x,y):
    a = 0.05
    xshift = x - 0.5
    yshift = y - 0.5
    return (xshift*xshift + yshift*yshift - a)*(xshift*xshift + yshift*yshift - a)*(xshift*xshift + yshift*yshift - a) < xshift*xshift*yshift*yshift*yshift

# Define the computational domain
mesh = UnitSquareMesh(50, 50)
marker = MeshFunction("size_t", mesh, 2, 0)
for c in cells(mesh):
    marker[c] = beutel_heart(c.midpoint().x(), c.midpoint().y())
submesh = MeshView.create(marker, 1)

time = Constant(0.0)

# Conductivities
Cm = 1.0 # Capacitance of the cell membrane [µF.cm-2]
chi = 2000 # area of cell membrane per unit volume [cm-1]
sigmai_t = 1.0
sigmae_t = 1.65
M_i = (1/(Cm*chi))*sigmai_t
M_e = (1/(Cm*chi))*sigmae_t

M_T = 0.25*M_e

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = NoCellModel()

# Define some external stimulus
stimulus = Expression("5*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.55, 2)) / 0.02)", degree=2)

# Collect this information into the CardiacModel and TorsoModel classes
cardiac_model = CardiacModel(submesh, time, M_i, M_e, cell_model, stimulus)
torso_model = TorsoModel(mesh, M_T)

# Customize and create a splitting solve
ps = SplittingSolver.default_parameters()
ps["theta"] = 0.5                        # Second order splitting scheme
ps["pde_solver"] = "bidomain"          # Use Monodomain model for the PDEs
ps["CardiacODESolver"]["scheme"] = "RL1" # 1st order Rush-Larsen for the ODEs

ps["CoupledBidomainSolver"]["linear_solver_type"] = "iterative"
ps["CoupledBidomainSolver"]["algorithm"] = "cg"
ps["CoupledBidomainSolver"]["preconditioner"] = "petsc_amg"

solver = SplittingSolver(cardiac_model, torso_model=torso_model, params=ps)

# Time stepping parameters
dt = 0.25
T = 10
interval = (0.0, T)

timer = Timer("XXX Forward solve") # Time the total solve

# Save solution in xdmf format
vfile = XDMFFile(MPI.comm_world, "./XDMF/v.xdmf")
ufile = XDMFFile(MPI.comm_world, "./XDMF/u.xdmf")

# Solve!
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = ", timestep)

    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

    vur.sub(0).rename("v", "v")
    vfile.write(vur.sub(0),timestep[0]) # transmembrane potential
    vur.sub(1).rename("u", "u")
    ufile.write(vur.sub(1),timestep[0]) # potential in the whole domain

timer.stop()

# List times spent
list_timings(TimingClear.keep, [TimingType.user])

