"""This module contains a CellSolver that uses JIT compiled Gotran
models together with GOSS (General ODE System Solver), which can be
interfaced by the GossSplittingSolver"""

__author__ = "Johan Hake (hake.dev@gmail.com), 2013"

__all__ = ["GOSSplittingSolver"]

import numpy as np
import types

from dolfin import *
from dolfin.cpp.log import log, LogLevel

# Goss and Gotran imports
import goss
import gotran

from goss.dolfinutils import DOLFINODESystemSolver

#if "DOLFINODESystemSolver" not in goss.__dict__:
#    raise ImportError("goss could not import DOLFINODESystemSolver")

# Beatadjoint imports
from cbcbeat.bidomainsolver import BidomainSolver
from cbcbeat.monodomainsolver import MonodomainSolver
from cbcbeat.cardiacmodels import CardiacModel
from cbcbeat.utils import TimeStepper, Projecter


class GOSSplittingSolver:

    def __init__(self, model, params=None):

        # Check some input
        assert isinstance(model, CardiacModel), \
            "Expecting the model to be CardiacModel, not %r" % model

        # Set model and parameters
        self._model = model
        self.parameters = self.default_parameters()
        if params is not None:
            self.parameters.update(params)

        # Extract solution domain and time
        self._domain = self._model.domain()
        self._time = self._model.time()

        # Create PDE solver and extract solution fields
        self.pde_solver = self._create_pde_solver()
        (self.v, self.vur) = self.pde_solver.solution_fields()

        # Create ODE solver and extract solution fields
        self.ode_solver = self._create_ode_solver()

        if params and "enable_adjoint" in params and params["enable_adjoint"]:
            raise RuntimeError("GOSS is not compatible with dolfin-adjoint. "\
                               "params['enable_adjoint'] must be false ")

        # Set-up projection solver (for optimised merging) of fields
        self.vs_projecter = Projecter(self.v.function_space(),
                                      params=self.parameters["Projecter"])

    @staticmethod
    def default_parameters():
        """Initialize and return a set of default parameters for the
        splitting solver

        *Returns*
          A set of parameters (:py:class:`dolfin.Parameters`)

        To inspect all the default parameters, do::

          info(SplittingSolver.default_parameters(), True)
        """

        params = Parameters("GOSSplittingSolver")

        # Have to be false as GOSS is not compatible with dolfin-adjoint
        params.add("apply_stimulus_current_to_pde", False)
        params.add("enable_adjoint", False)
        params.add("theta", 0.5, 0, 1)
        try:
            params.add("pde_solver", "bidomain", set(["bidomain", "monodomain"]))
        except:
            params.add("pde_solver", "bidomain", ["bidomain", "monodomain"])
            pass


        # Add default parameters from ODE solver
        ode_solver_params = DOLFINODESystemSolver.default_parameters_dolfin()
        ode_solver_params.rename("ode_solver")
        #ode_solver_params.add("membrane_potential", "V")
        params.add(ode_solver_params)

        pde_solver_params = BidomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        pde_solver_params = MonodomainSolver.default_parameters()
        pde_solver_params["polynomial_degree"] = 1
        params.add(pde_solver_params)

        projecter_params = Projecter.default_parameters()
        params.add(projecter_params)

        return params

    def _create_pde_solver(self):
        """Helper function to initialize a suitable PDE solver from
        the cardiac model."""

        # Extract applied current from the cardiac model (stimulus
        # invoked in the ODE step)
        applied_current = self._model.applied_current

        # Extract stimulus from the cardiac model(!)
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
            kwargs = dict(I_s=stimulus, I_a=applied_current, params=params)
        else:
            PDESolver = MonodomainSolver
            params = self.parameters["MonodomainSolver"]
            args = (self._domain, self._time, M_i,)
            kwargs = dict(I_s=stimulus, params=params)

        # Propagate enable_adjoint to Bidomain solver
        if params.has_parameter("enable_adjoint"):
            params["enable_adjoint"] = self.parameters["enable_adjoint"]

        solver = PDESolver(*args, **kwargs)

        return solver

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_models = self._model.cell_models()

        # Create DOLFINODESystemSolver
        solver = DOLFINODESystemSolver(self._domain, \
                                       dict(zip(cell_models.keys(), cell_models.models())),
                                       domains=cell_models.markers(), \
                                       params=self.parameters["ode_solver"])

        return solver

    def solution_fields(self):
        """
        Return tuple of previous and current solution objects.

        Modifying these will modify the solution objects of the solver
        and thus provides a way for setting initial conditions for
        instance.

        *Returns*
          (current v, current vur) (:py:class:`tuple` of :py:class:`dolfin.Function`)
        """
        return (self.v, self.vur)

    def solve(self, interval, dt):
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the time step and the solution fields.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int)
            The timestep for the solve

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for ((t0, t1), (v, vur)) in solutions:
            # do something with the solutions

        """
        # Create timestepper
        time_stepper = TimeStepper(interval, dt, \
                                   annotate=self.parameters["enable_adjoint"])

        for t0, t1 in time_stepper:

            log(LogLevel.INFO, "Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            yield (t0, t1), self.solution_fields()


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
        dt = (t1 - t0)
        t = t0 + theta*dt

        # Compute tentative membrane potential and state (vs_star)
        begin("Tentative ODE step")

        # Assumes that ode_solver is in the correct state
        self.ode_solver.step((t0, t), self.v)
        end()

        # Compute tentative potentials vu = (v, u)
        begin("PDE step")
        # Assumes that its v is in the correct state, gives vur in
        # the current state
        self.pde_solver.step((t0, t1))
        self.merge(self.v)
        end()

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and v is in same state
            return

        # Otherwise, we do another ode_step:
        begin("Corrective ODE step")

        # Assumes that v is in the correct state, updates v in
        # the correct state
        self.ode_solver.step((t, t1), self.v)
        end()

    def merge(self, solution):
        """
        Extract membrane potential from solutions from the PDE solve and put
        it into membrane potential used for the ODE solve.

        *Arguments*
          solution (:py:class:`dolfin.Function`)
            Function holding the combined result
        """
        # Disabled for now (does not pass replay)

        begin("Merging using custom projecter")
        if self.parameters["pde_solver"] == "bidomain":
            v = split(self.vur)[0]
            solution.vector()[:] = project(v, solution.function_space()).vector()
        else:
            solution.vector()[:] = self.vur.vector()

        # FIXME: We should not need to do a projection. A sub function assign would
        # FIXME: be sufficient.
        # FIXME: Does not work in parallel!!!
        #self.vs_projecter(v, solution)
        end()
