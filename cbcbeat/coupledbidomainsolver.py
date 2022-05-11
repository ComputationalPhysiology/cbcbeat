"""
These solvers solve the (pure) bidomain equations on the form: find
the transmembrane potential :math:`v = v(x, t)` and the couple
extracellular potential - potential in the surrounding torso
:math:`u = (u_e(x, t), u_T(x,t))` in mV U mT (u_e in mV and u_T in mT)

.. math::

   v_t - \mathrm{div} ( G_i v + G_i u_e) = I_s

   \mathrm{div} (G_i v + (G_i + G_e) u_e) = I_a

   \mathrm{div} (G_T u_T) = 0

where the subscript :math:`t` denotes the time derivative; :math:`G_x`
denotes a weighted gradient: :math:`G_x = M_x \mathrm{grad}(v)` for
:math:`x \in \{i, e, T\}`, where :math:`M_i` and :math:`M_e` are the
intracellular and extracellular cardiac conductivity tensors respectively,
and :math:`M_T` is the conductivity tensor in the torso;
:math:`I_s` and :math:`I_a` are prescribed input. In
addition, initial conditions are given for :math:`v`:

.. math::

   v(x, 0) = v_0

Finally, boundary conditions must be prescribed. For now, this solver
assumes pure homogeneous Neumann boundary conditions for :math:`v` and
:math:`u` and enforces the additional average value zero constraint
for u.

"""

# Copyright (C) 2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-18

__all__ = ["CoupledBasicBidomainSolver", "CoupledBidomainSolver"]

from cbcbeat.dolfinimport import *
from cbcbeat.markerwisefield import *
from cbcbeat.utils import end_of_time, annotate_kwargs
from cbcbeat import debug

class CoupledBasicBidomainSolver(object):
    """This solver is based on a theta-scheme discretization in time
    and CG_1 x CG_1 (x R) elements in space.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      cardiac_mesh (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh) related with the cardiac model

      torso_mesh (:py:class:`dolfin.Mesh`)
        The spatial domain (mesh) for the torso, embedding the cardiac domain

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      M_i (:py:class:`ufl.Expr`)
        The intracellular conductivity tensor (as an UFL expression)

      M_e (:py:class:`ufl.Expr`)
        The extracellular conductivity tensor (as an UFL expression)

      M_T (:py:class:`ufl.Expr`)
        The conductivity tensor in the torso (as an UFL expression)

      I_s (:py:class:`dict`, optional)
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfin.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.

      I_a (:py:class:`dolfin.Expression`, optional)
        A (typically time-dependent) external applied current

      v\_ (:py:class:`ufl.Expr`, optional)
        Initial condition for v. A new :py:class:`dolfin.Function`
        will be created if none is given.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

      """
    def __init__(self, cardiac_mesh, torso_mesh, time, M_i, M_e, M_T,
                 I_s=None, I_a=None, v_=None, params=None):

        # Check some input
        assert isinstance(cardiac_mesh, Mesh), \
            "Expecting cardiac_mesh to be a Mesh instance, not %r" % cardiac_mesh
        assert isinstance(torso_mesh, Mesh), \
            "Expecting torso_mesh to be a Mesh instance, not %r" % torso_mesh
        assert isinstance(time, Constant) or time is None, \
            "Expecting time to be a Constant instance (or None)."
        assert isinstance(params, Parameters) or params is None, \
            "Expecting params to be a Parameters instance (or None)"

        self._nullspace_basis = None

        # Store input
        self._cardiac_mesh = cardiac_mesh
        self._mesh = torso_mesh        
        self._time = time
        self._M_i = M_i
        self._M_e = M_e
        self._M_T = M_T
        self._I_s = I_s
        self._I_a = I_a

        # Check if the cardiac mesh is a submesh of the whole (torso) mesh
        mapping = self._cardiac_mesh.topology().mapping()[self._mesh.id()]
        cell_map = mapping.cell_map()
        assert mapping and mapping.mesh().id() == self._mesh.id(), \
            "The CardiacModel mesh should be built from TorsoModel mesh (MeshView)"

        # Marking heart domain cells in the torso mesh
        self._heart_cells = MeshFunction("size_t", self._mesh, self._mesh.topology().dim(), 0)
        for c in cells(self._cardiac_mesh):
            idx = int(cell_map[c.index()])
            self._heart_cells[idx] = 1;
        
        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Set-up function spaces
        kc = self.parameters["cardiac_polynomial_degree"]
        kt = self.parameters["torso_polynomial_degree"]
        V = FunctionSpace(self._cardiac_mesh, "CG", kc)
        U = FunctionSpace(self._mesh, "CG", kt)

        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
            R = FunctionSpace(self._mesh, "R", 0)
            self.VUR = MixedFunctionSpace(V, U, R)
        else:
            self.VUR = MixedFunctionSpace(V, U)

        self.V = V

        # Set-up solution fields:
        if v_ is None:
            self.merger = FunctionAssigner(V, self.VUR.sub_space(0))
            self.v_ = Function(self.VUR.sub_space(0), name="v_")
        else:
            debug("Experimental: v_ shipped from elsewhere.")
            self.merger = None
            self.v_ = v_

        self.vur = Function(self.VUR)

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
          (previous v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.v_, self.vur)

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
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, vur = solution_fields
            # do something with the solutions
        """    
        timer = Timer("PDE step")

        # Initial set-up
        # Solve on entire interval if no interval is given.
        (T0, T) = interval
        if dt is None:
            dt = (T - T0)
        t0 = T0
        t1 = T0 + dt

       # Step through time steps until at end time
        while (True) :
            info("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if end_of_time(T, t0, t1, dt):
                break

            # If not: update members and move to next time
            # Subfunction assignment would be good here.
            if isinstance(self.v_, Function):
                self.merger.assign(self.v_, self.vur.sub(0))
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
          self.vur in correct state at t1.
        """

        timer = Timer("PDE step")

        # Extract interval and thus time-step
        (t0, t1) = interval
        k_n = Constant(t1 - t0)
        theta = self.parameters["theta"]

        # Extract conductivities
        M_i, M_e = self._M_i, self._M_e
        M_T = self._M_T
        
        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
             (v, u, l) = TrialFunctions(self.VUR)
             (w, q, lamda) = TestFunctions(self.VUR)
        else:
             (v, u) = TrialFunctions(self.VUR)
             (w, q) = TestFunctions(self.VUR)

        # Set time
        t = t0 + theta*(t1 - t0)
        self.time.assign(t)

        # Set-up measure and rhs from stimulus
        (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._cardiac_mesh, w)

        # Define spatial integration domains:
        dV = Measure("dx", domain=self.VUR.sub_space(0).mesh())
        dU = Measure("dx", domain=self.VUR.sub_space(1).mesh(), subdomain_data=self._heart_cells)

        a = v * w * dV \
            + theta * k_n * inner(M_i * grad(v), grad(w)) * dV \
            + (k_n/theta) * inner((M_i + M_e) * grad(u), grad(q)) * dU(1) \
            + (k_n/theta) * inner(M_T * grad(u), grad(q)) * dU(0) \
            + k_n * inner(M_i * grad(u), grad(w)) * dV \
            + k_n * inner(M_i * grad(v), grad(q)) * dV

        L = self.v_ * w * dV \
            - (1. - theta) * k_n * inner(M_i * grad(self.v_), grad(w)) * dV \
            - ((1. - theta)/theta) * k_n * inner(M_i * grad(self.v_), grad(q)) * dU(1)

        # Add applied stimulus as source in parabolic equation if
        # applicable
        L += k_n * rhs

        if use_R:
            a += k_n*(lamda*u + l*q)*dV

        # Add applied current as source in elliptic equation if
        # applicable
        if self._I_a:
            L += k_n*self._I_a*q*dV

        solve(a == L, self.vur)


    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(CoupledBasicBidomainSolver.default_parameters(), True)
        """

        params = Parameters("CoupledBasicBidomainSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("cardiac_polynomial_degree", 1)
        params.add("torso_polynomial_degree", 1)
        params.add("use_avg_u_constraint", True)

        params.add(LinearVariationalSolver.default_parameters())
        return params

class CoupledBidomainSolver(CoupledBasicBidomainSolver):
    __doc__ = CoupledBasicBidomainSolver.__doc__

    def __init__(self, cardiac_mesh, torso_mesh, time, M_i, M_e, M_T,
                 I_s=None, I_a=None, v_=None, params=None):

        # Call super-class
        CoupledBasicBidomainSolver.__init__(self, cardiac_mesh, torso_mesh, time, M_i, M_e, M_T,
                                     I_s=I_s, I_a=I_a, v_=v_, params=params)

        # Check consistency of parameters first
        if self.parameters["enable_adjoint"] and not dolfin_adjoint:
            warning("'enable_adjoint' is set to True, but no "\
                    "dolfin_adjoint installed.")

        # Mark the timestep as unset
        self._timestep = None

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BidomainSolver.default_parameters(), True)
        """

        params = Parameters("CoupledBidomainSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5)
        params.add("cardiac_polynomial_degree", 1)
        params.add("torso_polynomial_degree", 1)

        # Set default solver type to be iterative
        params.add("linear_solver_type", "iterative")
        params.add("use_avg_u_constraint", False)

        # Set default iterative solver choices (used if iterative
        # solver is invoked)
        params.add("algorithm", "cg")
        params.add("preconditioner", "petsc_amg")
        #params.add("preconditioner", "fieldsplit") # This seg faults

        # Add default parameters from both LU and Krylov solvers
        params.add(LUSolver.default_parameters())
        petsc_params = PETScKrylovSolver.default_parameters()
        petsc_params.update({"convergence_norm_type": "preconditioned"})
        params.add(petsc_params)

        # Customize default parameters for LUSolver
        #params["lu_solver"]["same_nonzero_pattern"] = True

        # Customize default parameters for PETScKrylovSolver
        #params["petsc_krylov_solver"]["preconditioner"]["structure"] = "same"
        ## params["petsc_krylov_solver"]["absolute_tolerance"] = 1e-6

        return params

    def variational_forms(self, k_n):
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        *Arguments*
          k_n (:py:class:`ufl.Expr` or float)
            The time step

        *Returns*
          (lhs, rhs) (:py:class:`tuple` of :py:class:`ufl.Form`)

        """

        # Extract theta parameter and conductivities
        theta = self.parameters["theta"]
        M_i = self._M_i
        M_e = self._M_e
        M_T = self._M_T

        # Define variational formulation
        use_R = self.parameters["use_avg_u_constraint"]
        if use_R:
             (v, u, l) = TrialFunctions(self.VUR)
             (w, q, lamda) = TestFunctions(self.VUR)
        else:
             (v, u) = TrialFunctions(self.VUR)
             (w, q) = TestFunctions(self.VUR)

        # Set-up measure and rhs from stimulus
        (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._cardiac_mesh, w)

        # Set up integration domain
        dV = Measure("dx", domain=self.VUR.sub_space(0).mesh())
        dU = Measure("dx", domain=self.VUR.sub_space(1).mesh(), subdomain_data=self._heart_cells)

        # Set-up variational problem
        a = v*w*dV \
            + k_n*theta*inner(M_i*grad(v), grad(w))*dV \
            + (k_n/theta)*inner((M_i + M_e)*grad(u), grad(q))*dU(1) \
            + (k_n/theta)*inner(M_T*grad(u), grad(q))*dU(0) \
            + k_n*inner(M_i*grad(u), grad(w))*dV \
            + k_n*inner(M_i*grad(v), grad(q))*dV
        
        L = self.v_*w*dV \
            - k_n*(1.0 - theta)*inner(M_i*grad(self.v_), grad(w))*dV \
            - k_n*((1.0 - theta)/theta)*inner(M_i*grad(self.v_), grad(q))*dU(1) \
            + k_n*rhs

        if use_R:
            a += k_n*(lamda*u + l*q)*dV

        # Add applied current as source in elliptic equation if
        # applicable
        if self._I_a:
            L += k_n*self._I_a*q*dV

        return (a, L)

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v\_ is in the correct state for t0, gives
          self.vur in correct state at t1.
        """

        timer = Timer("PDE step")

        # Extract interval and thus time-step
        (t0, t1) = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta*dt
        self.time.assign(t)

        # Define variational problem
        if self._timestep is None:
            self._timestep = Constant(dt)
        (self._lhs, self._rhs) = self.variational_forms(self._timestep)

        solve(self._lhs == self._rhs, self.vur)
