import os.path
import dolfin
import cPickle

class MyTimeSeries:

    def __init__(self, directory, filename, v, modifier):
        # Store input
        # FIXME: Can be simplified with os.path magic
        self._directory = directory
        self._filename = filename
        self._v = v
        self._times = []
        self._time2index = {}

        # Extract MPI communicator from function space/mesh
        mpi_comm = self._v.function_space().mesh().mpi_comm()

        # Initiate HDF5 file
        name = os.path.join(self._directory, filename)
        self._file = dolfin.HDF5File(mpi_comm, name, modifier)

        # Store the function/functionspace
        if modifier == "w":
            self._file.write(v, "function")
        elif modifier == "r":
            self._file.read(v, "function")

    def store(self, v, time):
        # Store the vector only
        self._file.write(v.vector(), "/values_{}".format(len(self._times)))

        # Register this time
        self._times.append(time)
        self._time2index[str(time)] = len(self._times) - 1

    def store_times(self):
        name = os.path.join(self._directory, "times.cpickle")
        cPickle.dump(self._times, open(name, "w"))
        name = os.path.join(self._directory, "time2index.cpickle")
        cPickle.dump(self._time2index, open(name, "w"))

    def retrieve_times(self):
        name = os.path.join(self._directory, "times.cpickle")
        times = cPickle.load(open(name))
        name = os.path.join(self._directory, "time2index.cpickle")
        index = cPickle.load(open(name))
        return (times, index)

    def retrieve(self, v, time):
        (times, index) = self.retrieve_times()
        num = index[str(time)]
        self._file.read(v.vector(), "/values_{}".format(num), True)

if __name__ == "__main__":

    mesh = dolfin.UnitSquareMesh(2,2)
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V)

    # How to store
    series = MyTimeSeries("results", "foo-series.h5", u, "w")
    series.store(u, 0.1)
    u.vector()[:] = 1.0
    series.store(u, 0.2)
    series.store_times()

    # How to retrieve
    series = MyTimeSeries("results", "foo-series.h5", u, "r")
    (times, index) = series.retrieve_times()
    print "Retrieved times and index: ", times, index
    series.retrieve(u, 0.2)
    dolfin.plot(u, title="u", interactive=True)
    print "Success!"
