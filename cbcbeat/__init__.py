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
from cbcbeat.cellmodels import *
from cbcbeat.markerwisefield import *

# Solver imports

# Various utility functions, mainly for internal use

# NB: Workaround for FEniCS 1.7.0dev
import ufl

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

# Set-up some global parameters
beat_parameters = dolfinimport.Parameters("beat-parameters")
beat_parameters.add("enable_adjoint", True)
