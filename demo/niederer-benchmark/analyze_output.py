from dolfin import *
import numpy
import matplotlib.pyplot as pyplot

def plot_p1p8_line(a, casedir=None):

    Lx = 20. # mm
    Ly = 7.  # mm
    Lz = 3.  # mm

    # Compute n points on the the P1 - P8 line
    n = 101
    x_coords = numpy.linspace(0, Lx, n)
    y_coords = numpy.linspace(0, Ly, n)
    z_coords = numpy.linspace(0, Lz, n)
    points = zip(x_coords, y_coords, z_coords)

    # Evaluate activation time at these points
    times = [a(p) for p in points]

    # Compute distances from origin (P1)
    distances = [numpy.linalg.norm(p) for p in points]

    # Plot activation times versus distance
    pyplot.plot(distances, times)
    pyplot.xlabel('Distance (mm)')
    pyplot.ylabel('Activation time (ms)')
    if casedir:
        pyplot.savefig("%s/activation_times.pdf" % casedir)

    pyplot.show()
    
def compute_activation_times_at_p1p8_line(casedir):

    #evaluation_points = [(0, 0, 0), (0, 7, 0), (20, 0, 0), (20, 7, 0), 
    #                     (0, 0, 3), (0, 7, 3), (20, 0, 3), (20, 7, 3), 
    #                     (10, 3.5, 1.5)]

    # Open mesh

    # Open stored v
    vfile = HDF5File(mpi_comm_world(), "%s/v.h5" % casedir, "r")
    mesh = Mesh()
    vfile.read(mesh, "/mesh", False)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)

    # Set-up data structures for computed activation times
    #times = []
    #values = {}

    threshold = 0.0
    # Field to store the activation times. a(x) = first time when v(x)
    # exceeds given threshold
    a = Function(V)
    a.vector()[:] = -1
    threshold = 0.0
    t0 = 0.0
    
    dofs = range(V.dim())

    for n in range(0, 1000):
        try:
            vector_name = "/function/vector_%d" % n
            vfile.read(v, vector_name)
            t = vfile.attributes(vector_name)["timestamp"]
            print "Computing activation times for t = %g" % t
            
            for i in dofs:
                if (v.vector()[i] >= threshold and a.vector()[i] < t0):
                    a.vector()[i] = t
        except:
            break

    vfile.close()

    # Store output in same directory
    afile = HDF5File(mesh.mpi_comm(), "%s/a.h5" % casedir, "w")
    afile.write(a, "/function", t0)
    afile.close()

    # Plot resulting activation times
    plot(a, title="Activation times")

    return a

def compute_activation_times(casedir):
    a = compute_activation_times_at_p1p8_line(casedir)
    plot_p1p8_line(a, casedir=casedir)

if __name__ == "__main__":

    import sys
    compute_activation_times(sys.argv[1])
