"""This module handles all dolfin import in cbcbeat. Here dolfin and
dolfin_adjoint gets imported. If dolfin_adjoint is not present it will not
be imported."""

__author__ = "Johan Hake (hake.dev@gmail.com), 2013"

# FIXME: This is here for readthedocs Mock purposes. Better fix would
# be, duh, better.


try:
    pass
except:
    pass

try:
    pass
except:
    pass

from dolfin import *

try:
    from dolfin_adjoint import *
    import dolfin_adjoint

except:
    # FIXME: Should we raise some sort of warning?
    dolfin_adjoint = None
    pass
