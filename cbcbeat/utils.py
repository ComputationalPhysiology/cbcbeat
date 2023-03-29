"""This module provides various utilities for internal use."""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = ["state_space", "end_of_time", "convergence_rate", "Projecter"]

import math
import dolfin
from cbcbeat.dolfinimport import backend, has_dolfin_adjoint
from dolfin.cpp.log import log, LogLevel


def annotate_kwargs(ba_parameters):
    if not has_dolfin_adjoint:
        return {}
    if not ba_parameters["enable_adjoint"]:
        return {"annotate": False}
    if dolfin.parameters["adjoint"]["stop_annotating"]:
        return {"annotate": False}

    return {"annotate": True}


def splat(vs, dim):
    if vs.function_space().ufl_element().num_sub_elements() == dim:
        v = vs[0]
        if dim == 2:
            s = vs[1]
        else:
            s = dolfin.as_vector([vs[i] for i in range(1, dim)])
    else:
        v, s = dolfin.split(vs)

    return v, s


def state_space(domain, d, family=None, k=1):
    """Return function space for the state variables.

    *Arguments*
      domain (:py:class:`dolfin.Mesh`)
        The computational domain
      d (int)
        The number of states
      family (string, optional)
        The finite element family, defaults to "CG" if None is given.
      k (int, optional)
        The finite element degree, defaults to 1

    *Returns*
      a function space (:py:class:`dolfin.FunctionSpace`)
    """
    if family is None:
        family = "CG"
    if d > 1:
        S = dolfin.VectorFunctionSpace(domain, family, k, d)
    else:
        S = dolfin.FunctionSpace(domain, family, k)
    return S


def end_of_time(T, t0, t1, dt):
    """
    Return True if the interval (t0, t1) is the last before the end
    time T, otherwise False.
    """
    return (t1 + dt) > (T + dolfin.DOLFIN_EPS)


class TimeStepper:
    """
    A helper object that keep track of simulated time
    """

    def __init__(self, interval, dt, annotate=False):
        """
        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (:py:class:`int or :py:class:`list` of py:class:`tuples` of py:class:`float`)
            The timestep for the solve. A list of tuples of floats can
            also be passed. Each tuple should contain two floats where the
            first includes the start time and the second the dt.
          annotate (:py:class:`bool)
            If enabling dolfin_adjoint timestep annotation
        """

        self.annotate = annotate

        if (
            not isinstance(interval, (tuple, list))
            or len(interval) != 2
            or not all(isinstance(value, (float, int)) for value in interval)
        ):
            raise TypeError(
                "expected tuple or list of size 2 with scalars for "
                "the interval argument"
            )

        if interval[0] >= interval[1]:
            raise ValueError(
                "Start time need to be larger than stop time: "
                "interval[0] < interval[1]"
            )

        # Store time interval
        (self.T0, self.T1) = interval

        if not isinstance(dt, (float, int, list)):
            raise TypeError("expected float or list of tuples for dt argument")

        if isinstance(dt, (float, int)):
            dt = [(self.T0, dt)]

        # Check that all dt are tuples of size 2 with either floats or ints.
        if any(
            (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, (float, int)) for value in item)
            )
            for item in dt
        ):
            raise TypeError(
                "expected list of tuples of size 2 with scalars for " "the dt argument"
            )

        # Check that first time value of dt is the same as the first given in interval
        if dt[0][0] != self.T0:
            raise ValueError(
                "expected first time value of dt to be the same as "
                "the first value of time interval."
            )

        # Check that all time values given in dt are increasing
        if not all(dt[i][0] < dt[i + 1][0] for i in range(len(dt) - 1)):
            raise ValueError("expected all time values in dt to be increasing")

        # Check that all time step values given in dt are positive
        if not all(dt[i][1] > 0 for i in range(len(dt))):
            raise ValueError("expected all time step values in dt to be positive")

        # Store dt
        self._dt = dt

        # Add a dummy dt including stop interval time
        if self._dt[-1][0] < self.T1:
            self._dt.append((self.T1, self._dt[-1][1]))

        # Keep track of dt index
        self._dt_ind = 0
        self.t0 = self.T0
        # self.t1 = self.T0 + self.dt

        # Step through time steps until at end time.
        if self.annotate and has_dolfin_adjoint:
            backend.adj_start_timestep(self.T0)

    def __iter__(self):
        """
        Return an iterator over time intervals
        """
        eps = 1e-10

        while True:
            # Get next t1
            t1 = self.next_t1()

            # Yield time interval
            yield self.t0, t1

            # Break if this is the last step
            if abs(t1 - self.T1) < eps:
                if self.annotate and has_dolfin_adjoint:
                    backend.adj_inc_timestep(time=t1, finished=True)
                break

            # Move to next time
            if self.annotate and has_dolfin_adjoint:
                backend.adj_inc_timestep(time=t1)

            self.t0 = t1

    def next_t1(self):
        """
        Return the time of next end interval
        """
        assert self._dt_ind < len(self._dt) + 1
        dt = self._dt[self._dt_ind][1]
        time_to_switch_dt = self._dt[self._dt_ind + 1][0]
        if time_to_switch_dt - dolfin.DOLFIN_EPS > self.t0 + dt:
            return self.t0 + dt

        # Update dt index
        self._dt_ind += 1
        return time_to_switch_dt


def convergence_rate(hs, errors):
    """
    Compute and return rates of convergence :math:`r_i` such that

    .. math::

      errors = C hs^r

    """
    assert len(hs) == len(errors), "hs and errors must have same length."
    # Compute converence rates
    rates = [
        (math.log(errors[i + 1] / errors[i])) / (math.log(hs[i + 1] / hs[i]))
        for i in range(len(hs) - 1)
    ]

    # Return convergence rates
    return rates


class Projecter(object):
    """Customized class for repeated projection.

    *Arguments*
      V (:py:class:`dolfin.FunctionSpace`)
        The function space to project into
      solver_type (string, optional)
        "iterative" (default) or "direct"

    *Example of usage*::
      my_project = Projecter(V, solver_type="direct")
      u = Function(V)
      f = Function(W)
      my_project(f, u)
    """

    def __init__(self, V, params=None):
        # Set parameters
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up mass matrix for L^2 projection
        self.V = V
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)
        self.m = dolfin.inner(self.u, self.v) * dolfin.dx()
        self.M = backend.assemble(self.m)
        self.b = dolfin.Vector(V.mesh().mpi_comm(), V.dim())

        solver_type = self.parameters["linear_solver_type"]
        assert (
            solver_type == "lu" or solver_type == "cg"
        ), "Expecting 'linear_solver_type' to be 'lu' or 'cg'"
        if solver_type == "lu":
            log(LogLevel.TRACE, "Setting up direct solver for projecter")

            # Customize LU solver (reuse everything)
            solver = backend.LUSolver(self.M)
        else:
            log(LogLevel.TRACE, "Setting up iterative solver for projecter")
            # Customize Krylov solver (reuse everything)
            solver = backend.KrylovSolver("cg", "ilu")
            solver.set_operator(self.M)
            if solver.parameters.has_parameter("preconditioner"):
                solver.parameters["preconditioner"]["structure"] = "same"
            # solver.parameters["nonzero_initial_guess"] = True
        self.solver = solver

    @staticmethod
    def default_parameters():
        parameters = dolfin.Parameters("Projecter")
        parameters.add("linear_solver_type", "cg")
        return parameters

    def __call__(self, f, u):
        """
        Carry out projection of ufl Expression f and store result in
        the function u. The user must make sure that u lives in the
        right space.

        *Arguments*
          f (:py:class:`ufl.Expr`)
            The thing to be projected into this function space
          u (:py:class:`dolfin.Function`)
            The result of the projection
        """
        L = dolfin.inner(f, self.v) * dolfin.dx()
        backend.assemble(L, tensor=self.b)
        self.solver.solve(u.vector(), self.b)
