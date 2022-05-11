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

import matplotlib.pyplot as plt
from cbcbeat import *
import numpy as np
import fiber_utils

# Turn on FFC/FEniCS optimizations
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3

# Turn off adjoint functionality
import cbcbeat
if cbcbeat.dolfin_adjoint:
    parameters["adjoint"]["stop_annotating"] = True

# Define the computational domain
mesh = UnitSquareMesh(20, 20)
time = Constant(0.0)

def circle_heart(x,y):
    r = 0.25
    xshift = x - 0.5
    yshift = y - 0.5
    return xshift*xshift + yshift*yshift < r*r
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

for c in cells(mesh):
    marker[c] = circle_heart(c.midpoint().x(), c.midpoint().y()) ## Beutel heart
heart_mesh = MeshView.create(marker, 1)

def setup_conductivities(mesh, chi, C_m):
    # Load fibers and sheets
    fiber = fiber_utils.generate_fibers(mesh, "fibers.txt")

    # Extract stored conductivity data.
    V = FunctionSpace(mesh, "CG", 1)

    info_blue("Using healthy conductivities")
    g_el_field = Function(V, name="g_el")
    g_et_field = Function(V, name="g_et")
    g_il_field = Function(V, name="g_il")
    g_it_field = Function(V, name="g_it")

    g_el_field.vector()[:] = 2.0/(C_m*chi)
    g_et_field.vector()[:] = 1.65/(C_m*chi)
    g_il_field.vector()[:] = 3.0/(C_m*chi)
    g_it_field.vector()[:] = 1.0/(C_m*chi)

    # Construct conductivity tensors from directions and conductivity
    # values relative to that coordinate system
    A = as_matrix([[fiber[0]], [fiber[1]]])

    from ufl import diag
    M_e_star = diag(as_vector([g_el_field, g_et_field]))
    M_i_star = diag(as_vector([g_il_field, g_it_field]))
    M_e = A*M_e_star*A.T
    M_i = A*M_i_star*A.T

    return M_i, M_e

chi = 90
C_m = 1.0
M_i, M_e = setup_conductivities(heart_mesh, chi, C_m)
M_T = 1.0/(C_m*chi)

# Pick a cell model (see supported_cell_models for tested ones)
cell_model = FitzHughNagumoManual()

# Define stimulus
S1_subdomain = CompiledSubDomain("(pow(x[0] - 0.5, 2) + pow(x[1] - 0.55, 2)) <= pow(0.15, 2)", degree=2)
S1_markers = MeshFunction("size_t", heart_mesh, heart_mesh.topology().dim())
S1_subdomain.mark(S1_markers, 1)

duration = 5.  # ms
amplitude = 10 # mV/ms
I_s = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                  time=time,
                  start=0.0,
                  duration=duration,
                  amplitude=amplitude,
                  degree=0)
# Store input parameters in cardiac model
stimulus = Markerwise((I_s,), (1,), S1_markers)


# Collect this information into the CardiacModel class
cardiac_model = CardiacModel(heart_mesh, time, M_i, M_e, cell_model, stimulus)
torso_model = TorsoModel(mesh, M_T)

# Customize and create a splitting solver
ps = SplittingSolver.default_parameters()
ps['apply_stimulus_current_to_pde'] = True
ps["theta"] = 0.5
ps["pde_solver"] = "bidomain"
ps["CardiacODESolver"]["scheme"] = "RL1"

solver = SplittingSolver(cardiac_model, torso_model=torso_model, params=ps)

# Extract the solution fields and set the initial conditions
(vs_, vs, vur) = solver.solution_fields()
vs_.assign(cell_model.initial_conditions())

# Time stepping parameters
#N = 3200
#T = 400.
N = 100
T = 100
dt = T/N
interval = (0.0, T)

vfile = XDMFFile(MPI.comm_world, "./XDMF/demo_fibers_unitsquare/v.xdmf")
ufile = XDMFFile(MPI.comm_world, "./XDMF/demo_fibers_unitsquare/u.xdmf")

plot_figures = True
plot_frequency = 25 # Plottin every 25 timesteps
for (timestep, fields) in solver.solve(interval, dt):
    print("(t_0, t_1) = (%g, %g)", timestep)
    # Extract the components of the field (vs_ at previous timestep,
    # current vs, current vur)
    (vs_, vs, vur) = fields

    vur.sub(0).rename("v", "v")
    vfile.write(vur.sub(0),timestep[0]) # transmembrane potential
    vur.sub(1).rename("u", "u")
    ufile.write(vur.sub(1),timestep[0]) # potential in the whole domain
    
    if plot_figures == True:
        plot_path = os.getcwd() + "/plots"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        if timestep[0] % plot_frequency == 0:
            plt.figure()
            c = plot(vur.sub(0), title="v at time=%d ms" %(timestep[0]), mode='color')
            c.set_cmap("jet")
            plt.colorbar(c, orientation='vertical')
            plt.savefig(plot_path + "/demo_fibers_unitsquare_v_%d.png" %(timestep[0]))
                
            plt.figure()
            c = plot(vur.sub(1), title="u_e at time=%d ms" %(timestep[0]), mode='color')
            c.set_cmap("jet")
            plt.colorbar(c, orientation='vertical')
            plt.savefig(plot_path + "/demo_fibers_unitsquare_u_e_%d.png" %(timestep[0]))
