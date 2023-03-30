"""
This module contains splitting solvers for CardiacModel objects. In
particular, the classes

  * SplittingSolver
  * BasicSplittingSolver

These solvers solve the bidomain (or monodomain) equations on the
form: find the transmembrane potential :math:`v = v(x, t)` in mV, the
extracellular potential :math:`u = u(x, t)` in mV, and any additional
state variables :math:`s = s(x, t)` such that

.. math::

   v_t - \mathrm{div} (M_i \mathrm{grad} v + M_i \mathrm{grad} u) = - I_{ion}(v, s) + I_s

         \mathrm{div} (M_i \mathrm{grad} v + (M_i + M_e) \mathrm{grad} u) = I_a

   s_t = F(v, s)

where

  * the subscript :math:`t` denotes the time derivative,
  * :math:`M_i` and :math:`M_e` are conductivity tensors (in mm^2/ms)
  * :math:`I_s` is prescribed input current (in mV/ms)
  * :math:`I_a` is prescribed input current (in mV/ms)
  * :math:`I_{ion}` and :math:`F` are typically specified by a cell model

Note that M_i and M_e can be viewed as scaled by :math:`\chi*C_m` where
  * :math:`\chi` is the surface-to volume ratio of cells (in 1/mm) ,
  * :math:`C_m` is the specific membrane capacitance (in mu F/(mm^2) ),

In addition, initial conditions are given for :math:`v` and :math:`s`:

.. math::

   v(x, 0) = v_0

   s(x, 0) = s_0

Finally, boundary conditions must be prescribed. These solvers assume
pure Neumann boundary conditions for :math:`v` and :math:`u` and
enforce the additional average value zero constraint for u.

The solvers take as input a
:py:class:`~cbcbeat.cardiacmodels.CardiacModel` providing the
required input specification of the problem. In particular, the
applied current :math:`I_a` is extracted from the
:py:attr:`~cbcbeat.cardiacmodels.CardiacModel.applied_current`
attribute, while the stimulus :math:`I_s` is extracted from the
:py:attr:`~cbcbeat.cardiacmodels.CardiacModel.stimulus` attribute.

It should be possible to use the solvers interchangably. However, note
that the BasicSplittingSolver is not optimised and should be used for
testing or debugging purposes primarily.
"""

# Copyright (C) 2012-2013 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2013-04-15

__all__ = [
    "SplittingSolver",
    "BasicSplittingSolver",
]

import dolfin
from cbcbeat.dolfinimport import backend

from cbcbeat import CardiacModel
from cbcbeat.cellsolver import BasicCardiacODESolver, CardiacODESolver
from cbcbeat.bidomainsolver import BasicBidomainSolver, BidomainSolver
from cbcbeat.monodomainsolver import BasicMonodomainSolver, MonodomainSolver
from cbcbeat.utils import TimeStepper, annotate_kwargs
from ufl.log import info_blue

progress = dolfin.LogLevel.PROGRESS


class Interpolator:
    """A simple helper class to mimic the API of a FunctionAssigner
    assuming that it is possible to interpolate between the two
    functions passed in to the `assign` method
    """

    def assign(self, v_ode, v, **kwargs):
        v_ode.interpolate(v)


def setup_merger(ode_space, vur_space):
    ode_el = ode_space.ufl_element()
    vur_el = vur_space.ufl_element()

    if ode_el.family() == vur_el.family():
        return backend.FunctionAssigner(ode_space, vur_space)

    return Interpolator()


class BasicSplittingSolver:
    """

    A non-optimised solver for the bidomain equations based on the
    operator splitting scheme described in Sundnes et al 2006, p. 78
    ff.

    The solver computes as solutions:

      * "vs" (:py:class:`dolfin.Function`) representing the solution
        for the transmembrane potential and any additional state
        variables, and
      * "vur" (:py:class:`dolfin.Function`) representing the
        transmembrane potential in combination with the extracellular
        potential and an additional Lagrange multiplier.

    The algorithm can be controlled by a number of parameters. In
    particular, the splitting algorithm can be controlled by the
    parameter "theta": "theta" set to 1.0 corresponds to a (1st order)
    Godunov splitting while "theta" set to 0.5 to a (2nd order) Strang
    splitting.

    This solver has not been optimised for computational efficiency
    and should therefore primarily be used for debugging purposes. For
    an equivalent, but more efficient, solver, see
    :py:class:`cbcbeat.splittingsolver.SplittingSolver`.

    *Arguments*
      model (:py:class:`cbcbeat.cardiacmodels.CardiacModel`)
        a CardiacModel object describing the simulation set-up
      params (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters
      V_index (int)
        Index for the membrane potential among the state
        variables, by default 0

    *Assumptions*
      * The cardiac conductivities do not vary in time

    """

    def __init__(self, model, params=None, V_index=0):
        "Create solver from given Cardiac Model and (optional) parameters."

        assert isinstance(
            model, CardiacModel
        ), "Expecting CardiacModel as first argument"

        # Set model and parameters
        self._model = model
        self._V_index = V_index
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Extract solution domain
        self._domain = self._model.domain()
        self._time = self._model.time()

        # Create ODE solver and extract solution fields
        self.ode_solver = self._create_ode_solver()
        (self.vs_, self.vs) = self.ode_solver.solution_fields()
        self.VS = self.vs.function_space()

        # Create PDE solver and extract solution fields
        self.pde_solver = self._create_pde_solver()
        (self.v_, self.vur) = self.pde_solver.solution_fields()

        # Create function assigner for merging v from self.vur into self.vs[0]
        if self.parameters["pde_solver"] == "bidomain":
            V = self.vur.function_space().sub(0)
        else:
            V = self.vur.function_space()

        self.merger = setup_merger(self._ode_membrane_potential_space, V)

        self._annotate_kwargs = annotate_kwargs(self.parameters)

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_model = self._model.cell_models()

        # Extract stimulus from the cardiac model(!)
        if self.parameters["apply_stimulus_current_to_pde"]:
            stimulus = None
        else:
            stimulus = self._model.stimulus()

        # Extract ode solver parameters
        params = self.parameters["BasicCardiacODESolver"]
        # Propagate enable_adjoint to Bidomain solver
        if params.has_parameter("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]

        solver = BasicCardiacODESolver(
            self._domain, self._time, cell_model, I_s=stimulus, params=params
        )
        return solver

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from
        the cardiac model."""

        # Extract applied current from the cardiac model
        applied_current = self._model.applied_current()

        # Extract stimulus from the cardiac model if we should apply
        # it to the PDEs (in the other case, it is handled by the ODE
        # solver)
        if self.parameters["apply_stimulus_current_to_pde"]:
            stimulus = self._model.stimulus()
        else:
            stimulus = None

        # Extract conductivities from the cardiac model
        (M_i, M_e) = self._model.conductivities()

        if self.parameters["pde_solver"] == "bidomain":
            PDESolver = BasicBidomainSolver
            params = self.parameters["BasicBidomainSolver"]
            args = (self._domain, self._time, M_i, M_e)
            kwargs = dict(
                I_s=stimulus,
                I_a=applied_current,
                v_=self._ode_membrane_potential_func,
                params=params,
            )
        else:
            PDESolver = BasicMonodomainSolver
            params = self.parameters["BasicMonodomainSolver"]
            args = (self._domain, self._time, M_i)
            kwargs = dict(
                I_s=stimulus, v_=self._ode_membrane_potential_func, params=params
            )

        # Propagate enable_adjoint to Bidomain solver
        if params.has_parameter("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]

        solver = PDESolver(*args, **kwargs)

        return solver

    @property
    def _ode_membrane_potential_space(self):
        """Get the functio space for the membrane potential
        in the ODE
        """
        if self.VS.num_sub_spaces() == 0:
            return self.VS
        return self.VS.sub(self._V_index)

    def _extract_V(self, solution):
        """Extract the membrane potential from the solution"""
        if self.VS.num_sub_spaces() == 0:
            return solution
        return solution.sub(self._V_index)

    @property
    def _ode_membrane_potential_func(self):
        """Get the membrane potential from the ODE"""
        if self.vs.value_rank() == 0:
            # The solution from the ODE does not come from a
            # VectorFunctionSpace. Assme that this is the membrane
            # potential
            return self.vs
        else:
            # We retrieve the membrane potential from the
            # correct subspace
            return self.vs[self._V_index]

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters for the
        splitting solver

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(BasicSplittingSolver.default_parameters(), True)

        """

        params = dolfin.Parameters("BasicSplittingSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5, 0.0, 1.0)
        params.add("apply_stimulus_current_to_pde", False)
        try:
            params.add("pde_solver", "bidomain", set(["bidomain", "monodomain"]))
        except Exception:
            params.add("pde_solver", "bidomain", ["bidomain", "monodomain"])

        # Add default parameters from ODE solver, but update for V
        # space
        ode_solver_params = BasicCardiacODESolver.default_parameters()
        ode_solver_params["V_polynomial_degree"] = 1
        ode_solver_params["V_polynomial_family"] = "CG"
        params.add(ode_solver_params)

        pde_solver_params = BasicBidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        pde_solver_params = BasicMonodomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        return params

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (previous vs, current vs, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.vs_, self.vs, self.vur)

    def solve(self, interval, dt):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the time step and the solution fields.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, list of tuples of floats)
            The timestep for the solve. A list of tuples of floats can
            also be passed. Each tuple should contain two floats where the
            first includes the start time and the second the dt.

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          dts = [(0., 0.1), (1.0, 0.05), (2.0, 0.1)]
          solutions = solver.solve((0.0, 1.0), dts)

          # Iterate over generator (computes solutions as you go)
          for ((t0, t1), (vs_, vs, vur)) in solutions:
            # do something with the solutions

        """

        # Create timestepper
        time_stepper = TimeStepper(
            interval, dt, annotate=self.parameters["enable_adjoint"]
        )

        for t0, t1 in time_stepper:
            info_blue("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()

            # Update previous solution
            self.vs_.assign(self.vs)

    def step(self, interval):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with timestep given by the interval length.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)

        *Invariants*
          Given self._vs in a correct state at t0, provide v and s (in
          self.vs) and u (in self.vur) in a correct state at t1. (Note
          that self.vur[0] == self.vs[0] only if theta = 1.0.)
        """

        # Extract some parameters for readability
        theta = self.parameters["theta"]

        # Extract time domain
        (t0, t1) = interval
        dt = t1 - t0
        t = t0 + theta * dt

        # Compute tentative membrane potential and state (vs_star)
        dolfin.begin(progress, "Tentative ODE step")
        # Assumes that its vs_ is in the correct state, gives its vs
        # in the current state
        self.ode_solver.step((t0, t))
        dolfin.end()

        # Compute tentative potentials vu = (v, u)
        dolfin.begin(progress, "PDE step")
        # Assumes that its vs_ is in the correct state, gives vur in
        # the current state
        self.pde_solver.step((t0, t1))
        dolfin.end()

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and the s part of its
            # vs are in the correct state, provides input argument(in
            # this case self.vs) in its correct state
            self.merge(self.vs)
            return

        # Otherwise, we do another ode_step:
        dolfin.begin(progress, "Corrective ODE step")

        # Update vs_ based on vs, the s part of vs and vs_ are now in
        # the correct state.
        self.vs_.assign(self.vs)

        # Set the v part of vs_ to be v from vur. Now vs_ is in its
        # correct state.
        self.merge(self.vs_)

        # Assumes that its vs_ is in the correct state, provides vs in
        # the correct state
        self.ode_solver.step((t, t1))

        dolfin.end()

    def merge(self, solution):
        """
        Combine solutions from the PDE solve and the ODE solve to form
        a single mixed function.

        *Arguments*
          solution (:py:class:`dolfin.Function`)
            Function holding the combined result
        """
        timer = dolfin.Timer("Merge step")

        dolfin.begin(progress, "Merging")
        if self.parameters["pde_solver"] == "bidomain":
            v = self.vur.sub(0)
        else:
            v = self.vur

        self.merger.assign(self._extract_V(solution), v, **self._annotate_kwargs)
        dolfin.end()

        timer.stop()


class SplittingSolver(BasicSplittingSolver):
    """

    An optimised solver for the bidomain equations based on the
    operator splitting scheme described in Sundnes et al 2006, p. 78
    ff.

    The solver computes as solutions:

      * "vs" (:py:class:`dolfin.Function`) representing the solution
        for the transmembrane potential and any additional state
        variables, and
      * "vur" (:py:class:`dolfin.Function`) representing the
        transmembrane potential in combination with the extracellular
        potential and an additional Lagrange multiplier.

    The algorithm can be controlled by a number of parameters. In
    particular, the splitting algorithm can be controlled by the
    parameter "theta": "theta" set to 1.0 corresponds to a (1st order)
    Godunov splitting while "theta" set to 0.5 to a (2nd order) Strang
    splitting.

    *Arguments*
      model (:py:class:`cbcbeat.cardiacmodels.CardiacModel`)
        a CardiacModel object describing the simulation set-up
      params (:py:class:`dolfin.Parameters`, optional)
        a Parameters object controlling solver parameters

    *Example of usage*::

      from cbcbeat import *

      # Describe the cardiac model
      mesh = UnitSquareMesh(100, 100)
      time = Constant(0.0)
      cell_model = FitzHughNagumoManual()
      stimulus = Expression("10*t*x[0]", t=time, degree=1)
      cm = CardiacModel(mesh, time, 1.0, 1.0, cell_model, stimulus)

      # Extract default solver parameters
      ps = SplittingSolver.default_parameters()

      # Examine the default parameters
      info(ps, True)

      # Update parameter dictionary
      ps["theta"] = 1.0 # Use first order splitting
      ps["CardiacODESolver"]["scheme"] = "GRL1" # Use Generalized Rush-Larsen scheme

      # Use monodomain equations as the PDE model
      ps["pde_solver"] = "monodomain"
      # Use direct linear solver of the PDEs
      ps["MonodomainSolver"]["linear_solver_type"] = "direct"
      # Use backward Euler for temporal discretization for the PDEs
      ps["MonodomainSolver"]["theta"] = 1.0

      solver = SplittingSolver(cm, params=ps)

      # Extract the solution fields and set the initial conditions
      (vs_, vs, vur) = solver.solution_fields()
      vs_.assign(cell_model.initial_conditions())

      # Solve
      dt = 0.1
      T = 1.0
      interval = (0.0, T)
      for (timestep, fields) in solver.solve(interval, dt):
          (vs_, vs, vur) = fields
          # Do something with the solutions


    *Assumptions*
      * The cardiac conductivities do not vary in time

    """

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters for the
        splitting solver

        *Returns*
          The set of default parameters (:py:class:`dolfin.Parameters`)

        *Example of usage*::

          info(SplittingSolver.default_parameters(), True)
        """

        params = dolfin.Parameters("SplittingSolver")
        params.add("enable_adjoint", True)
        params.add("theta", 0.5, 0, 1)
        params.add("apply_stimulus_current_to_pde", False)
        try:
            params.add("pde_solver", "bidomain", set(["bidomain", "monodomain"]))
            params.add(
                "ode_solver_choice",
                "CardiacODESolver",
                set(["BasicCardiacODESolver", "CardiacODESolver"]),
            )
        except Exception:
            params.add("pde_solver", "bidomain", ["bidomain", "monodomain"])
            params.add(
                "ode_solver_choice",
                "CardiacODESolver",
                ["BasicCardiacODESolver", "CardiacODESolver"],
            )
            pass

        # Add default parameters from ODE solver
        ode_solver_params = CardiacODESolver.default_parameters()
        ode_solver_params["scheme"] = "RL1"
        params.add(ode_solver_params)

        # Add default parameters from ODE solver
        basic_ode_solver_params = BasicCardiacODESolver.default_parameters()
        basic_ode_solver_params["V_polynomial_degree"] = 1
        basic_ode_solver_params["V_polynomial_family"] = "CG"
        params.add(basic_ode_solver_params)

        pde_solver_params = BidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        pde_solver_params = MonodomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        return params

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_model = self._model.cell_models()

        # Extract stimulus from the cardiac model(!)
        if self.parameters["apply_stimulus_current_to_pde"]:
            stimulus = None
        else:
            stimulus = self._model.stimulus()

        Solver = eval(self.parameters["ode_solver_choice"])
        params = self.parameters[Solver.__name__]
        if params.has_parameter("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]
        solver = Solver(
            self._domain, self._time, cell_model, I_s=stimulus, params=params
        )

        return solver

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from
        the cardiac model."""

        # Extract applied current from the cardiac model (stimulus
        # invoked in the ODE step)
        applied_current = self._model.applied_current()

        # Extract stimulus from the cardiac model
        if self.parameters["apply_stimulus_current_to_pde"]:
            stimulus = self._model.stimulus()
        else:
            stimulus = None

        # Extract conductivities from the cardiac model
        (M_i, M_e) = self._model.conductivities()

        if self.parameters["pde_solver"] == "bidomain":
            PDESolver = BidomainSolver
            params = self.parameters["BidomainSolver"]
            args = (self._domain, self._time, M_i, M_e)
            kwargs = dict(
                I_s=stimulus,
                I_a=applied_current,
                v_=self._ode_membrane_potential_func,
                params=params,
            )
        else:
            PDESolver = MonodomainSolver
            params = self.parameters["MonodomainSolver"]
            args = (self._domain, self._time, M_i)
            kwargs = dict(
                I_s=stimulus, v_=self._ode_membrane_potential_func, params=params
            )

        # Propagate enable_adjoint to Bidomain solver
        if params.has_parameter("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]

        solver = PDESolver(*args, **kwargs)

        return solver
