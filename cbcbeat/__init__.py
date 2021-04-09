"""
The cbcbeat Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from cbcbeat import *

"""
import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)


# Import all of dolfin with possibly dolfin-adjoint on top
from cbcbeat.dolfinimport import *

# Model imports
from cbcbeat.cardiacmodels import CardiacModel
from cbcbeat.cellmodels import *
from cbcbeat.markerwisefield import *

# Solver imports
from cbcbeat.splittingsolver import BasicSplittingSolver
from cbcbeat.splittingsolver import SplittingSolver
from cbcbeat.cellsolver import BasicSingleCellSolver, SingleCellSolver
from cbcbeat.cellsolver import BasicCardiacODESolver, CardiacODESolver
from cbcbeat.bidomainsolver import BasicBidomainSolver
from cbcbeat.bidomainsolver import BidomainSolver
from cbcbeat.monodomainsolver import BasicMonodomainSolver
from cbcbeat.monodomainsolver import MonodomainSolver

# Various utility functions, mainly for internal use
import cbcbeat.utils

# NB: Workaround for FEniCS 1.7.0dev
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

# Set-up some global parameters
beat_parameters = dolfinimport.Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", True)
