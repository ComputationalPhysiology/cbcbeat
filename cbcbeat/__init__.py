"""
The cbcbeat Python module is a problem and solver collection for
cardiac electrophysiology models.

To import the module, type::

  from cbcbeat import *

"""
import warnings as _warnings


# Import all of dolfin with possibly dolfin-adjoint on top


# Model imports
from cbcbeat.cellmodels import (
    FitzHughNagumoManual,
    NoCellModel,
    RogersMcCulloch,
    Beeler_reuter_1977,
    Tentusscher_2004_mcell,
    Tentusscher_panfilov_2006_epi_cell,
    Tentusscher_panfilov_2006_M_cell,
    Fenton_karma_1998_MLR1_altered,
    Fenton_karma_1998_BR_altered,
    CardiacCellModel,
    MultiCellModel,
    supported_cell_models,
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
from cbcbeat.dolfinimport import backend
from cbcbeat import dolfinimport

# Solver imports

# Various utility functions, mainly for internal use

# NB: Workaround for FEniCS 1.7.0dev
import ufl as _ufl

_ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
import dolfin as _dolfin

parameters = _dolfin.parameters
# Set-up some global parameters
beat_parameters = _dolfin.Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", True)

import ufl
import ufl.log
from ufl import *
from ufl.log import *
from dolfin import *


try:
    import dolfin_adjoint
    from dolfin_adjoint import *
except ImportError:
    dolfin_adjoint = None
    dolfin_adjoint_all = []
else:
    dolfin_adjoint_all = dolfin_adjoint.__all__


class StarImportDeprecationWarning(DeprecationWarning):
    def __init__(self, *args: object) -> None:
        msg = """
*** ===================================================== ***
*** CBCBeat: Using * imports will be deprecated in the    ***
*** next version of cbcbeat. Please change all imports    ***
*** such as 'from cbcbeat import *' with 'import cbcbeat' ***
*** or 'from cbcbeat import foo, bar'                     ***
*** ===================================================== ***"""
        super().__init__(msg)


_warnings.simplefilter("always", StarImportDeprecationWarning)


def __getattr__(name):
    if name == "__all__":
        _warnings.warn(StarImportDeprecationWarning(), stacklevel=2)
        return (
            dolfin_adjoint_all
            + _dolfin.__all__
            + ufl.__all__
            + ufl.log.__all__
            + [
                "FitzHughNagumoManual",
                "NoCellModel",
                "RogersMcCulloch",
                "Beeler_reuter_1977",
                "Tentusscher_2004_mcell",
                "Tentusscher_panfilov_2006_epi_cell",
                "Tentusscher_panfilov_2006_M_cell",
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
                "backend",
                "supported_cell_models",
                "dolfinimport",
                "parameters",
                "beat_parameters",
            ]
        )

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
