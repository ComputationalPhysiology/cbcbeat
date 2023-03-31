# Copyright (C) 2014 Marie E. Rognes (meg@simula.no)
# Use and modify at will
# Last changed: 2016-10-19

__all__ = ["Markerwise", "handle_markerwise", "rhs_with_markerwise_field"]

from dolfin import dx, Measure
from ufl.log import error


def handle_markerwise(g, classtype):
    # Handle stimulus
    if (
        g is None
        or isinstance(g, classtype)
        or isinstance(g, Markerwise)
        or isinstance(g, object)
    ):
        return g
    else:
        msg = "Expecting stimulus to be a %s or Markerwise, not %r " % (
            str(classtype),
            g,
        )
        error(msg)


def rhs_with_markerwise_field(g, mesh, v):
    if g is None:
        dz = dx
        rhs = 0.0
    elif isinstance(g, Markerwise):
        markers = g.markers()
        dz = Measure("dx", domain=mesh, subdomain_data=markers)
        rhs = sum([g * v * dz(i) for (i, g) in zip(g.keys(), g.values())])
    else:
        dz = dx
        rhs = g * v * dz()
    return (dz, rhs)


class Markerwise(object):
    """A container class representing an object defined by a number of
    objects combined with a mesh function defining mesh domains and a
    map between the these.

    *Arguments*
      objects (tuple)
        the different objects
      keys (tuple of ints)
        a map from the objects to the domains marked in markers
      markers (:py:class:`dolfin.MeshFunction`)
        a mesh function mapping which domains the mesh consist of

    *Example of usage*

    Given (g0, g1), (2, 5) and markers, let

      g = g0 on domains marked by 2 in markers
      g = g1 on domains marked by 5 in markers

    letting::

      g = Markerwise((g0, g1), (2, 5), markers)

    """

    def __init__(self, objects, keys, markers):
        "Create Markerwise from given input."

        # Check input
        assert len(objects) == len(
            keys
        ), "Expecting the number of objects to equal the number of keys"

        # Store attributes:
        self._objects = dict(zip(keys, objects))
        self._markers = markers

    def values(self):
        "The objects"
        return self._objects.values()

    def keys(self):
        "The keys or domain numbers"
        return self._objects.keys()

    def markers(self):
        "The markers"
        return self._markers

    def __getitem__(self, key):
        "The objects"
        return self._objects[key]
