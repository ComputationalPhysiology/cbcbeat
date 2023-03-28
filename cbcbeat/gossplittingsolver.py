"""This module contains a CellSolver that uses JIT compiled Gotran
models together with GOSS (General ODE System Solver), which can be
interfaced by the GossSplittingSolver"""

__author__ = "Johan Hake (hake.dev@gmail.com), 2013"

__all__ = ["GOSSplittingSolver"]

from cbcbeat.cellmodels.cardiaccellmodel import MultiCellModel

from dolfin import Parameters

from goss.dolfinutils import DOLFINODESystemSolver

# cbcbeat imports
from cbcbeat.bidomainsolver import BidomainSolver
from cbcbeat.monodomainsolver import MonodomainSolver
from cbcbeat.splittingsolver import SplittingSolver
from cbcbeat.utils import Projecter


class GOSSplittingSolver(SplittingSolver):
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
        except Exception:
            params.add("pde_solver", "bidomain", ["bidomain", "monodomain"])
            pass

        # Add default parameters from ODE solver
        ode_solver_params = DOLFINODESystemSolver.default_parameters_dolfin()
        ode_solver_params.rename("ode_solver")
        # ode_solver_params.add("membrane_potential", "V")
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

    def _create_ode_solver(self):
        """Helper function to initialize a suitable ODE solver from
        the cardiac model."""

        # Extract cardiac cell model from cardiac model
        cell_models = self._model.cell_models()

        kwargs = {"odes": cell_models}
        if isinstance(cell_models, MultiCellModel):
            kwargs = {
                "odes": dict(zip(cell_models.keys(), cell_models.models())),
                "domains": cell_models.markers(),
            }

        solver = DOLFINODESystemSolver(
            self._domain, params=self.parameters["ode_solver"], **kwargs
        )
        return solver
