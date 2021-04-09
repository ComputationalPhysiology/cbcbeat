__author__ = "Johan Hake and Simon W. Funke (simon@simula.no), 2014"
__all__ = []

# Modified by Marie E. Rognes, 2014

#from dolfin import *
from cbcbeat import *
import numpy
#set_log_level(PROGRESS)

#parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 2

# MER says: should use compiled c++ expression here for vastly
# improved efficiency.
class StimSubDomain(SubDomain):
    "This represents the stimulation domain: [0, L]^3 mm."
    def __init__(self, L):
        self.L = L
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return numpy.all(x <= self.L + DOLFIN_EPS)

def define_conductivity_tensor(chi, C_m):

    # Conductivities as defined by page 4339 of Niederer benchmark
    sigma_il = 0.17  # mS / mm
    sigma_it = 0.019 # mS / mm
    sigma_el = 0.62  # mS / mm
    sigma_et = 0.24  # mS / mm

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a*b/(a + b)
    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l/(C_m*chi) # mm^2 / ms
    s_t = sigma_t/(C_m*chi) # mm^2 / ms

    # Define conductivity tensor
    M = as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))

    return M

def run_monodomain_solver(mesh, dt, T, theta):

    # Define time
    time = Constant(0.0)

    # Surface to volume ratio
    chi = 140.0     # mm^{-1}
    # Membrane capacitance
    C_m = 0.01 # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Define stimulation region defined as [0, L]^3
    stimulus_domain_marker = 1
    L = 1.5
    stimulus_subdomain = StimSubDomain(L)
    markers = CellFunction("size_t", domain, 0)
    markers.set_all(0)
    stimulus_subdomain.mark(markers, stimulus_domain_marker)
    File("output/simulation_region.pvd") << markers

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    stimulation_protocol_duration = dt # ms
    A = 50000. # mu A/cm^3
    cm2mm = 10.
    factor = 1.0/(chi*C_m) # NB: cbcbeat convention
    stimulation_protocol_amplitude = factor*A*(1./cm2mm)**3 # mV/ms
    stim = Expression("time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
                      time=time,
                      start=0.0,
                      duration=stimulation_protocol_duration,
                      amplitude=stimulation_protocol_amplitude)

    # Store input parameters in cardiac model
    I_s = Markerwise((stim,), (stimulus_domain_marker,), markers)

    total = Timer("Total solver time")

    # Set-up solver
    ps = MonodomainSolver.default_parameters()
    ps["default_timestep"] = dt
    ps["linear_solver_type"] = "iterative"
    ps["theta"] = theta
    ps["use_custom_preconditioner"] = True
    ps["algorithm"] = "cg"
    ps["preconditioner"] = "jacobi"

    # Set-up solver
    solver = MonodomainSolver(mesh, time, M, I_s=I_s, params=ps)

    # Extract the solution fields and set the initial conditions
    (v_, v) = solver.solution_fields()
    solutions = solver.solve((0, T), dt)

    V = v.function_space()

    # Solve
    for (timestep, (v_, v)) in solutions:
        print "Solving on %s" % str(timestep)
    total.stop()

    list_timings()

if __name__ == "__main__":

    parameters["adjoint"]["stop_annotating"] = True

    # Define geometry parameters
    Lx = 20. # mm
    Ly = 7.  # mm
    Lz = 3.  # mm

    # Define solver parameters
    theta = 1.0
    dx = 0.1
    dt = 0.05
    T = 10*dt # mS 500.0

    # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
    # with resolution prescribed by benchmark
    N = lambda v: int(numpy.rint(v))
    domain = BoxMesh(0.0, 0.0, 0.0, Lx, Ly, Lz, N(Lx/dx), N(Ly/dx), N(Lz/dx))

    # Run solver
    run_monodomain_solver(domain, dt, T, theta)
