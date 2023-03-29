"""
The cbcbeat Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from cbcbeat import *

"""
import warnings as _warnings

_warnings.filterwarnings("ignore", category=DeprecationWarning)


# Import all of dolfin with possibly dolfin-adjoint on top


# Model imports
from cbcbeat.cellmodels import (
    FitzHughNagumoManual,
    NoCellModel,
    RogersMcCulloch,
    Beeler_reuter_1977,
    Tentusscher_2004_mcell,
    Tentusscher_panfilov_2006_epi_cell,
    Fenton_karma_1998_MLR1_altered,
    Fenton_karma_1998_BR_altered,
    CardiacCellModel,
    MultiCellModel,
)
from cbcbeat.markerwisefield import (
    rhs_with_markerwise_field,
    Markerwise,
    handle_markerwise,
)

from cbcbeat.bidomainsolver import BasicBidomainSolver, BidomainSolver
from cbcbeat.cardiacmodels import CardiacModel
from cbcbeat.cellsolver import (
    CardiacODESolver,
    BasicCardiacODESolver,
    SingleCellSolver,
    BasicSingleCellSolver,
)
from cbcbeat.monodomainsolver import BasicMonodomainSolver, MonodomainSolver
from cbcbeat.splittingsolver import SplittingSolver, BasicSplittingSolver

# Solver imports

# Various utility functions, mainly for internal use

# NB: Workaround for FEniCS 1.7.0dev
import ufl as _ufl

_ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
import dolfin as _dolfin

# Set-up some global parameters
beat_parameters = _dolfin.Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", True)


__all__ = [
    "FitzHughNagumoManual",
    "NoCellModel",
    "RogersMcCulloch",
    "Beeler_reuter_1977",
    "Tentusscher_2004_mcell",
    "Tentusscher_panfilov_2006_epi_cell",
    "Fenton_karma_1998_MLR1_altered",
    "Fenton_karma_1998_BR_altered",
    "CardiacCellModel",
    "MultiCellModel",
    "rhs_with_markerwise_field",
    "Markerwise",
    "handle_markerwise",
    "BasicBidomainSolver",
    "BidomainSolver",
    "CardiacModel",
    "CardiacODESolver",
    "BasicCardiacODESolver",
    "SingleCellSolver",
    "BasicSingleCellSolver",
    "BasicMonodomainSolver",
    "MonodomainSolver",
    "SplittingSolver",
    "BasicSplittingSolver",
]
