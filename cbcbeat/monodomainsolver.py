"""
These solvers solve the (pure) monodomain equations on the form: find
the transmembrane potential :math:`v = v(x, t)` such that

.. math::

   v_t - \mathrm{div} ( G_i v) = I_s

where the subscript :math:`t` denotes the time derivative; :math:`G_i`
denotes a weighted gradient: :math:`G_i = M_i \mathrm{grad}(v)` for,
where :math:`M_i` is the intracellular cardiac conductivity tensor;
:math:`I_s` ise prescribed input. In addition, initial conditions are
given for :math:`v`:

.. math::

   v(x, 0) = v_0

Finally, boundary conditions must be prescribed. For now, this solver
assumes pure homogeneous Neumann boundary conditions for :math:`v`.

"""

# Copyright (C) 2013 Johan Hake (hake@simula.no)
# Use and modify at will
# Last changed: 2013-04-18

__all__ = ["BasicMonodomainSolver", "MonodomainSolver"]
import ufl
import dolfin
from cbcbeat.dolfinimport import backend
from cbcbeat.markerwisefield import rhs_with_markerwise_field
from cbcbeat.utils import end_of_time, annotate_kwargs
from ufl.log import debug, info, error


class BasicMonodomainSolver:
    """This solver is based on a theta-scheme discretization in time
    and CG_1 elements in space.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      mesh (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      M_i (:py:class:`ufl.Expr`)
        The intracellular conductivity tensor (as an UFL expression)

      I_s (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.

      v\_ (:py:class:`ufl.Expr`, optional)
        Initial condition for v. A new :py:class:`dolfin.Function`
        will be created if none is given.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(self, mesh, time, M_i, I_s=None, v_=None, params=None):
        # Check some input
        assert isinstance(mesh, dolfin.Mesh), (
            "Expecting mesh to be a Mesh instance, not %r" % mesh
        )
        assert (
            isinstance(time, backend.Constant) or time is None
        ), "Expecting time to be a Constant instance (or None)."
        assert (
            isinstance(params, dolfin.Parameters) or params is None
        ), "Expecting params to be a Parameters instance (or None)"

        # Store input
        self._mesh = mesh
        self._M_i = M_i
        self._I_s = I_s
        self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        k = self.parameters["polynomial_degree"]
        family = self.parameters["family"]

        element = dolfin.FiniteElement(
            family=family,
            cell=self._mesh.ufl_cell(),
            degree=k,
            quad_scheme="default",
        )
        V = dolfin.FunctionSpace(self._mesh, element)

        self.V = V

        # Set-up solution fields:
        if v_ is None:
            self.v_ = backend.Function(V, name="v_")
        else:
            debug("Experimental: v_ shipped from elsewhere.")
            self.v_ = v_

        self.v = backend.Function(self.V, name="v")

        # Figure out whether we should annotate or not
        self._annotate_kwargs = annotate_kwargs(self.parameters)

    @property
    def time(self):
        "The internal time of the solver."
        return self._time

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous v, current v) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.v_, self.v)

    def solve(self, interval, dt=None):
        """
        Solve the discretization on a given time interval (t0, t1)
        with a given timestep dt and return generator for a tuple of
        the interval and the current solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, solution_field) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, v = solution_fields
            # do something with the solutions
        """

        # Initial set-up
        # Solve on entire interval if no interval is given.
        (T0, T) = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while True:
            info("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            if isinstance(self.v_, backend.Function):
                self.v_.assign(self.v)
            else:
                debug("Assuming that v_ is updated elsewhere. Experimental.")

            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        """
        Solve on the given time interval (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.v in correct state at t1.
        """

        # Extract interval and thus time-step
        (t0, t1) = interval
        k_n = backend.Constant(t1 - t0)
        theta = self.parameters["theta"]

        # Extract conductivities
        M_i = self._M_i

        # Set time
        t = t0 + theta * (t1 - t0)
        self.time.assign(t)

        # Define variational formulation
        v = dolfin.TrialFunction(self.V)
        w = dolfin.TestFunction(self.V)
        Dt_v = (v - self.v_) / k_n
        v_mid = theta * v + (1.0 - theta) * self.v_

        (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._mesh, w)
        theta_parabolic = ufl.inner(M_i * ufl.grad(v_mid), ufl.grad(w)) * dz()
        G = Dt_v * w * dz() + theta_parabolic - rhs

        # Define variational problem
        a, L = dolfin.system(G)
        pde = backend.LinearVariationalProblem(a, L, self.v)

        # Set-up solver
        solver = backend.LinearVariationalSolver(pde)
        solver.parameters.update(self.parameters["linear_variational_solver"])
        solver.solve()

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicMonodomainSolver.default_parameters(), True)
        """

        params = dolfin.Parameters("BasicMonodomainSolver")
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("family", "CG")
        params.add("enable_adjoint", True)

        params.add(backend.LinearVariationalSolver.default_parameters())
        return params


class MonodomainSolver(BasicMonodomainSolver):
    __doc__ = BasicMonodomainSolver.__doc__

    def __init__(self, mesh, time, M_i, I_s=None, v_=None, params=None):
        # Call super-class
        BasicMonodomainSolver.__init__(
            self, mesh, time, M_i, I_s=I_s, v_=v_, params=params
        )

        # Create variational forms
        self._timestep = backend.Constant(self.parameters["default_timestep"])
        (self._lhs, self._rhs, self._prec) = self.variational_forms(self._timestep)

        # Preassemble left-hand side (will be updated if time-step
        # changes)
        debug("Preassembling monodomain matrix (and initializing vector)")
        self._lhs_matrix = backend.assemble(self._lhs, **self._annotate_kwargs)
        self._rhs_vector = dolfin.Vector(mesh.mpi_comm(), self._lhs_matrix.size(0))
        self._lhs_matrix.init_vector(self._rhs_vector, 0)

        # Create linear solver (based on parameter choices)
        self._linear_solver, self._update_solver = self._create_linear_solver()

    @property
    def linear_solver(self):
        """The linear solver (:py:class:`dolfin.LUSolver` or
        :py:class:`dolfin.KrylovSolver`)."""
        return self._linear_solver

    def _create_linear_solver(self):
        "Helper function for creating linear solver based on parameters."
        solver_type = self.parameters["linear_solver_type"]

        if solver_type == "direct":
            solver = backend.LUSolver(self._lhs_matrix, self.parameters["lu_type"])
            solver.parameters.update(self.parameters["lu_solver"])
            update_routine = self._update_lu_solver

        elif solver_type == "iterative":
            # Preassemble preconditioner (will be updated if time-step
            # changes)
            debug("Preassembling preconditioner")
            # Initialize KrylovSolver with matrix and preconditioner
            alg = self.parameters["algorithm"]
            prec = self.parameters["preconditioner"]
            if self.parameters["use_custom_preconditioner"]:
                self._prec_matrix = backend.assemble(
                    self._prec, **self._annotate_kwargs
                )
                solver = backend.PETScKrylovSolver(alg, prec)
                solver.parameters.update(self.parameters["krylov_solver"])
                solver.set_operators(self._lhs_matrix, self._prec_matrix)
                solver.ksp().setFromOptions()
            else:
                solver = backend.PETScKrylovSolver(alg, prec)
                solver.parameters.update(self.parameters["krylov_solver"])
                solver.set_operator(self._lhs_matrix)
                solver.ksp().setFromOptions()

            update_routine = self._update_krylov_solver
        else:
            error("Unknown linear_solver_type given: %s" % solver_type)

        return (solver, update_routine)

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(MonodomainSolver.default_parameters(), True)
        """

        params = dolfin.Parameters("MonodomainSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("polynomial_degree", 1)
        params.add("default_timestep", 1.0)
        params.add("family", "CG")

        # Set default solver type to be iterative
        params.add("linear_solver_type", "iterative")
        params.add("lu_type", "default")

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "cg")
        params.add("preconditioner", "petsc_amg")
        params.add("use_custom_preconditioner", True)

        # Add default parameters from both LU and Krylov solvers
        params.add(backend.LUSolver.default_parameters())
        params.add(backend.KrylovSolver.default_parameters())

        # Customize default parameters for KrylovSolver
        # params["krylov_solver"]["preconditioner"]["structure"] = "same"

        return params

    def variational_forms(self, k_n):
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs, prec) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """

        # Extract theta parameter and conductivities
        theta = self.parameters["theta"]
        M_i = self._M_i

        # Define variational formulation
        v = dolfin.TrialFunction(self.V)
        w = dolfin.TestFunction(self.V)

        # Set-up variational problem
        Dt_v_k_n = v - self.v_
        v_mid = theta * v + (1.0 - theta) * self.v_

        (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._mesh, w)
        theta_parabolic = ufl.inner(M_i * ufl.grad(v_mid), ufl.grad(w)) * dz()
        G = Dt_v_k_n * w * dz + k_n * theta_parabolic - k_n * rhs

        # Define preconditioner based on educated(?) guess by Marie
        prec = (v * w + k_n / 2.0 * ufl.inner(M_i * ufl.grad(v), ufl.grad(w))) * dz

        (a, L) = dolfin.system(G)
        return (a, L, prec)

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.v in correct state at t1.
        """

        timer = dolfin.Timer("PDE Step")

        # Extract interval and thus time-step
        (t0, t1) = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta * dt
        self.time.assign(t)

        # Update matrix and linear solvers etc as needed
        timestep_unchanged = abs(dt - float(self._timestep)) < 1.0e-12
        self._update_solver(timestep_unchanged, dt)

        # Assemble right-hand-side
        timer0 = dolfin.Timer("Assemble rhs")
        backend.assemble(self._rhs, tensor=self._rhs_vector, **self._annotate_kwargs)
        del timer0

        # Solve problem
        self.linear_solver.solve(
            self.v.vector(), self._rhs_vector, **self._annotate_kwargs
        )
        timer.stop()

    def _update_lu_solver(self, timestep_unchanged, dt):
        """Helper function for updating an LUSolver depending on
        whether timestep has changed."""

        # Update stored timestep
        # FIXME: dolfin_adjoint still can't annotate constant assignment.
        self._timestep.assign(backend.Constant(dt))  # , annotate=annotate)

        # Reassemble matrix
        backend.assemble(self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs)

    def _update_krylov_solver(self, timestep_unchanged, dt):
        """Helper function for updating a KrylovSolver depending on
        whether timestep has changed."""

        annotate_kwargs(self.parameters)
        # Update reuse of preconditioner parameter in accordance with
        # changes in timestep
        if timestep_unchanged:
            debug("Timestep is unchanged, reusing preconditioner")
            # self.linear_solver.parameters["preconditioner"]["structure"] = "same"
        else:
            debug("Timestep has changed, updating preconditioner")
            # self.linear_solver.parameters["preconditioner"]["structure"] = \
            #                                            "same_nonzero_pattern"

            # Update stored timestep
            self._timestep.assign(backend.Constant(dt))

            # Reassemble matrix
            backend.assemble(
                self._lhs, tensor=self._lhs_matrix, **self._annotate_kwargs
            )

            # Reassemble preconditioner
            if self.parameters["use_custom_preconditioner"]:
                backend.assemble(
                    self._prec, tensor=self._prec_matrix, **self._annotate_kwargs
                )

        # Set nonzero initial guess if it indeed is nonzero
        # if (self.v.vector().norm("l2") > 1.e-12):
        #    debug("Initial guess is non-zero.")
        #    self.linear_solver.parameters["nonzero_initial_guess"] = True
