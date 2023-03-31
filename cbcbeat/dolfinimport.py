"""This module handles all dolfin import in cbcbeat. Here dolfin and
dolfin_adjoint gets imported. If dolfin_adjoint is not present it will not
be imported."""

__author__ = "Johan Hake (hake.dev@gmail.com), 2013"


import dolfin as backend

try:
    import dolfin_adjoint as backend  # noqa: F811

    has_dolfin_adjoint = True

except ImportError:
    # FIXME: Should we raise some sort of warning?
    has_dolfin_adjoint = False


__all__ = ["backend"]
