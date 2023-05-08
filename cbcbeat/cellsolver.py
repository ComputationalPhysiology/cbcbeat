"This module contains solvers for (subclasses of) CardiacCellModel."

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"

__all__ = [
    "BasicSingleCellSolver",
    "BasicCardiacODESolver",
    "CardiacODESolver",
    "SingleCellSolver",
]

import dolfin
from dolfin import (
    ForwardEuler,  # noqa:F401
    BackwardEuler,  # noqa:F401
    CrankNicolson,  # noqa:F401
    RK4,  # noqa:F401
    ESDIRK3,  # noqa:F401
    ESDIRK4,  # noqa:F401
    RL1,  # noqa:F401
    RL2,  # noqa:F401
    GRL1,  # noqa:F401
    GRL2,  # noqa:F401
)
from cbcbeat.dolfinimport import backend
from cbcbeat import CardiacCellModel, MultiCellModel
from cbcbeat.markerwisefield import (
    handle_markerwise,
    Markerwise,
    rhs_with_markerwise_field,
)
from cbcbeat.utils import state_space, TimeStepper, splat, annotate_kwargs
import ufl
import ufl.classes
import cbcbeat
from ufl.log import info_blue, error


def point_integral_solver_default_parameters():
    try:
        p = backend.PointIntegralSolver.default_parameters()
    except AttributeError:
        p = dolfin.Parameters("point_integral_solver")
        p.add("reset_stage_solutions", True)
        # Set parameters for NewtonSolver
        pn = dolfin.Parameters("newton_solver")
        pn.add("maximum_iterations", 40)
        pn.add("always_recompute_jacobian", False)
        pn.add("recompute_jacobian_each_solve", True)
        pn.add("relaxation_parameter", 1.0, 0.0, 1.0)
        pn.add("relative_tolerance", 1e-10, 1e-20, 2.0)
        pn.add("absolute_tolerance", 1e-15, 1e-20, 2.0)
        pn.add("kappa", 0.1, 0.05, 1.0)
        pn.add("eta_0", 1.0, 1e-15, 1.0)
        pn.add("max_relative_previous_residual", 1e-1, 1e-5, 1.0)
        pn.add("reset_each_step", True)
        pn.add("report", False)
        pn.add("report_vertex", 0, 0, 32767)
        pn.add("verbose_report", False)
        p.add(pn)
    return p


class BasicCardiacODESolver(object):
    """A basic, non-optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(x, t)` and a vector field :math:`s = s(x, t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, and :math:`I_s` is some prescribed stimulus.

    Here, this nonlinear ODE system is solved via a theta-scheme.  By
    default theta=0.5, which corresponds to a Crank-Nicolson
    scheme. This can be changed by modifying the solver parameters.

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

      model (:py:class:`cbcbeat.CardiacCellModel`)
        A representation of the cardiac cell model(s)

      I_s (optional) A typically time-dependent external stimulus
        given as a :py:class:`dolfin.cpp.function.GenericFunction` or a
        Markerwise. NB: it is assumed that the time dependence of I_s
        is encoded via the 'time' Constant.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters
    """

    def __init__(self, mesh, time, model, I_s=None, params=None):
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = model

        # Extract some information from cell model
        self._F = self._model.F
        self._I_ion = self._model.I
        self._num_states = self._model.num_states()

        # Handle stimulus
        self._I_s = handle_markerwise(I_s, dolfin.cpp.function.GenericFunction)

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (mixed) function space for potential + states
        v_family = self.parameters["V_polynomial_family"]
        v_degree = self.parameters["V_polynomial_degree"]
        s_family = self.parameters["S_polynomial_family"]
        s_degree = self.parameters["S_polynomial_degree"]

        if v_family == s_family and s_degree == v_degree:
            self.VS = dolfin.VectorFunctionSpace(
                self._mesh, v_family, v_degree, dim=self._num_states + 1
            )
        else:
            V = dolfin.FunctionSpace(self._mesh, v_family, v_degree)
            S = state_space(self._mesh, self._num_states, s_family, s_degree)
            Mx = dolfin.MixedElement(V.ufl_element(), S.ufl_element())
            self.VS = dolfin.FunctionSpace(self._mesh, Mx)

        # Initialize solution fields
        self.vs_ = backend.Function(self.VS, name="vs_")
        self.vs = backend.Function(self.VS, name="vs")

    @property
    def time(self):
        "The internal time of the solver."
        return self._time

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = dolfin.Parameters("BasicCardiacODESolver")
        params.add("theta", 0.5)
        params.add("V_polynomial_degree", 0)
        params.add("V_polynomial_family", "DG")
        params.add("S_polynomial_degree", 0)
        params.add("S_polynomial_family", "DG")
        params.add("enable_adjoint", True)

        # Use iterative solver as default.
        params.add(backend.NonlinearVariationalSolver.default_parameters())
        params["nonlinear_variational_solver"]["newton_solver"][
            "linear_solver"
        ] = "gmres"

        return params

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs)

    def solve(self, interval, dt=None):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, current vs) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, vs) in solutions:
            # do something with the solutions

        """

        # Initial time set-up
        (T0, T) = interval

        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = T - T0

        # Create timestepper
        time_stepper = TimeStepper(
            interval, dt, annotate=self.parameters["enable_adjoint"]
        )
        for t0, t1 in time_stepper:
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.vs

            self.vs_.assign(self.vs)

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """

        timer = dolfin.Timer("ODE step")

        # Check for cell meshs
        self._mesh.topology().dim()

        # Extract time mesh
        (t0, t1) = interval
        k_n = backend.Constant(t1 - t0)

        # Extract previous solution(s)
        (v_, s_) = splat(self.vs_, self._num_states + 1)

        # Set-up current variables
        self.vs.assign(self.vs_)  # Start with good guess
        (v, s) = splat(self.vs, self._num_states + 1)
        (w, r) = splat(dolfin.TestFunction(self.VS), self._num_states + 1)

        # Define equation based on cell model
        Dt_v = (v - v_) / k_n
        Dt_s = (s - s_) / k_n

        theta = self.parameters["theta"]

        # Set time (propagates to time-dependent variables defined via
        # self.time)
        t = t0 + theta * (t1 - t0)
        self.time.assign(t)

        v_mid = theta * v + (1.0 - theta) * v_
        s_mid = theta * s + (1.0 - theta) * s_

        if isinstance(self._model, MultiCellModel):
            # assert(model.mesh() == self._mesh)

            model = self._model
            mesh = model.mesh()
            dy = dolfin.Measure("dx", domain=mesh, subdomain_data=model.markers())

            # Only allowing trivial forcing functions here
            if isinstance(self._I_s, Markerwise):
                error("Not implemented")
            rhs = self._I_s * w * dy()

            n = model.num_states()  # Extract number of global states

            # Collect contributions to lhs by iterating over the different cell models
            domains = self._model.keys()
            lhs = list()
            for k, model_k in enumerate(model.models()):
                n_k = (
                    model_k.num_states()
                )  # Extract number of local (non-trivial) states

                # Extract right components of coefficients and test functions
                # () is not the same as (1,)
                if n_k == 1:
                    s_mid_k = s_mid[0]
                    r_k = r[0]
                    Dt_s_k = Dt_s[0]
                else:
                    s_mid_k = dolfin.as_vector(tuple(s_mid[j] for j in range(n_k)))
                    r_k = dolfin.as_vector(tuple(r[j] for j in range(n_k)))
                    Dt_s_k = dolfin.as_vector(tuple(Dt_s[j] for j in range(n_k)))

                i_k = domains[k]  # Extract domain index of cell model k

                # Extract right currents and ion channel expressions
                F_theta_k = self._F(v_mid, s_mid_k, time=self.time, index=i_k)
                I_theta_k = -self._I_ion(v_mid, s_mid_k, time=self.time, index=i_k)

                # Variational contribution over the relevant domain
                a_k = (
                    (Dt_v - I_theta_k) * w
                    + dolfin.inner(Dt_s_k, r_k)
                    + dolfin.inner(-F_theta_k, r_k)
                ) * dy(i_k)

                # Add s_trivial = 0 on Omega_{i_k} in variational form:
                a_k += sum(s[j] * r[j] for j in range(n_k, n)) * dy(i_k)
                lhs.append(a_k)
            lhs = sum(lhs)

        else:
            (dz, rhs) = rhs_with_markerwise_field(self._I_s, self._mesh, w)

            # Evaluate currents at averaged v and s. Note sign for I_theta
            F_theta = self._F(v_mid, s_mid, time=self.time)
            I_theta = -self._I_ion(v_mid, s_mid, time=self.time)
            lhs = (Dt_v - I_theta) * w * dz + dolfin.inner(Dt_s - F_theta, r) * dz

        # Set-up system of equations
        G = lhs - rhs

        # Solve system
        pde = backend.NonlinearVariationalProblem(
            G, self.vs, J=dolfin.derivative(G, self.vs)
        )
        solver = backend.NonlinearVariationalSolver(pde)
        solver_params = self.parameters["nonlinear_variational_solver"]
        solver.parameters.update(solver_params)
        solver.solve()
        timer.stop()


class CardiacODESolver(object):
    """An optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(x, t)` and a vector field :math:`s = s(x, t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, and :math:`I_s` is some prescribed stimulus.

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
        The spatial mesh (mesh)

      time (:py:class:`dolfin.Constant` or None)
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      model (:py:class:`cbcbeat.CardiacCellModel`)
        A representation of the cardiac cell model(s)

      I_s (:py:class:`dolfin.Expression`, optional)
        A typically time-dependent external stimulus. NB: it is
        assumed that the time dependence of I_s is encoded via the
        'time' Constant.

      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(self, mesh, time, model, I_s=None, params=None):
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = model

        # Extract some information from cell model
        self._F = self._model.F
        self._I_ion = self._model.I
        self._num_states = self._model.num_states()

        self._I_s = handle_markerwise(I_s, dolfin.cpp.function.GenericFunction)

        # Create time if not given, otherwise use given time
        if time is None:
            self._time = backend.Constant(0.0)
        else:
            self._time = time

        # Initialize and update parameters if given
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Create (vector) function space for potential + states
        self.VS = dolfin.VectorFunctionSpace(
            self._mesh,
            "CG",
            self.parameters["polynomial_degree"],
            dim=self._num_states + 1,
        )

        # Initialize solution field
        self.vs_ = backend.Function(self.VS, name="vs_")
        self.vs = backend.Function(self.VS, name="vs")

        # Initialize scheme
        (v, s) = splat(self.vs, self._num_states + 1)
        (w, q) = splat(dolfin.TestFunction(self.VS), self._num_states + 1)

        # Workaround to get algorithm in RL schemes working as it only
        # works for scalar expressions
        F_exprs = self._F(v, s, self._time)

        # MER: This looks much more complicated than it needs to be!
        # If we have a dolfin.as_vector expression
        F_exprs_q = ufl.zero()
        if isinstance(F_exprs, ufl.classes.ListTensor):
            # for i, expr_i in enumerate(F_exprs.operands()):
            for i, expr_i in enumerate(F_exprs.ufl_operands):
                F_exprs_q += expr_i * q[i]
        else:
            F_exprs_q = F_exprs * q

        rhs = F_exprs_q - self._I_ion(v, s, self._time) * w

        # Handle stimulus: only handle single function case for now
        msg = "Markerwise stimulus not supported by PointIntegralSolver."
        assert not isinstance(self._I_s, Markerwise), msg
        if self._I_s:
            rhs += self._I_s * w

        # FIXME: The application of dP was moved so adding an integral
        # is done just once. Otherwise ufl could not figure out that
        # we had only one integral...
        self._rhs = rhs * dolfin.dP()

        # sys.exit()
        name = self.parameters["scheme"]
        Scheme = self._name_to_scheme(name)
        self._scheme = Scheme(self._rhs, self.vs, self._time)

        # Figure out whether we should annotate or not
        self._annotate_kwargs = annotate_kwargs(self.parameters)

        # Initialize solver and update its parameters
        self._pi_solver = backend.PointIntegralSolver(self._scheme)
        self._pi_solver.parameters.update(self.parameters["point_integral_solver"])

    def _name_to_scheme(self, name):
        """Return scheme class with given name

        *Arguments*
          name (string)

        *Returns*
          the Scheme (:py:class:`dolfin.MultiStageScheme`)

        """
        return eval(name)

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)
        """
        params = dolfin.Parameters("CardiacODESolver")
        params.add("scheme", "BackwardEuler")
        params.add("polynomial_degree", 1)
        params.add(point_integral_solver_default_parameters())
        params.add("enable_adjoint", True)

        return params

    def solution_fields(self):
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs_, current vs) (:py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs)

    def step(self, interval):
        """
        Solve on the given time step (t0, t1).

        End users are recommended to use solve instead.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step
        """
        # NB: The point integral solver operates on vs directly, map
        # initial condition in vs_ to vs:

        timer = dolfin.Timer("ODE step")
        self.vs.assign(self.vs_)

        (t0, t1) = interval
        dt = t1 - t0

        self._annotate_kwargs = annotate_kwargs(self.parameters)
        if cbcbeat.dolfinimport.has_dolfin_adjoint:
            self._pi_solver.step(dt, **self._annotate_kwargs)
        else:
            self._pi_solver.step(dt)
        timer.stop()

    def solve(self, interval, dt=None):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, current vs) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, vs) in solutions:
            # do something with the solutions

        """

        # Initial time set-up
        (T0, T) = interval

        # Solve on entire interval if no interval is given.
        if dt is None:
            dt = T - T0

        # Create timestepper
        time_stepper = TimeStepper(
            interval, dt, annotate=self.parameters["enable_adjoint"]
        )

        for t0, t1 in time_stepper:
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.vs

            # FIXME: This eventually breaks in parallel!?
            self.vs_.assign(self.vs)


class BasicSingleCellSolver(BasicCardiacODESolver):
    """A basic, non-optimised solver for systems of ODEs typically
    encountered in cardiac applications of the form: find a scalar
    field :math:`v = v(t)` and a vector field :math:`s = s(t)`

    .. math::

      v_t = - I_{ion}(v, s) + I_s

      s_t = F(v, s)

    where :math:`I_{ion}` and :math:`F` are given non-linear
    functions, :math:`I_s` is some prescribed stimulus. If :math:`I_s`
    depends on time, it is assumed that :math:`I_s` is a
    :py:class:`dolfin.Expression` with parameter 't'.

    Use this solver if you just want to test the results from a
    cardiac cell model without any spatial mesh dependence.

    Here, this nonlinear ODE system is solved via a theta-scheme.  By
    default theta=0.5, which corresponds to a Crank-Nicolson
    scheme. This can be changed by modifying the solver parameters.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      model (:py:class:`~cbcbeat.cellmodels.cardiaccellmodel.CardiacCellModel`)
        A cardiac cell model
      time (:py:class:`~dolfin.Constant` or None)
        A constant holding the current time.
      params (:py:class:`dolfin.Parameters`, optional)
        Solver parameters

    """

    def __init__(self, model, time, params=None):
        "Create solver from given cell model and optional parameters."

        assert isinstance(model, CardiacCellModel), (
            "Expecting model to be a CardiacCellModel, not %r" % model
        )
        assert isinstance(time, backend.Constant), (
            "Expecting time to be a Constant instance, not %r" % time
        )
        assert isinstance(params, dolfin.Parameters) or params is None, (
            "Expecting params to be a Parameters (or None), not %r" % params
        )

        # Store model
        self._model = model

        # Define carefully chosen dummy mesh
        mesh = dolfin.UnitIntervalMesh(1)

        # Extract information from cardiac cell model and ship off to
        # super-class.
        BasicCardiacODESolver.__init__(
            self, mesh, time, model, I_s=model.stimulus, params=params
        )


class SingleCellSolver(CardiacODESolver):
    def __init__(self, model, time, params=None):
        "Create solver from given cell model and optional parameters."

        assert isinstance(model, CardiacCellModel), (
            "Expecting model to be a CardiacCellModel, not %r" % model
        )
        assert isinstance(time, backend.Constant), (
            "Expecting time to be a Constant instance, not %r" % time
        )
        assert isinstance(params, dolfin.Parameters) or params is None, (
            "Expecting params to be a Parameters (or None), not %r" % params
        )

        # Store model
        self._model = model

        # Define carefully chosen dummy mesh
        mesh = dolfin.UnitIntervalMesh(1)

        # Extract information from cardiac cell model and ship off to
        # super-class.
        CardiacODESolver.__init__(
            self, mesh, time, model, I_s=model.stimulus, params=params
        )
