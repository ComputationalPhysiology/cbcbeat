"Various utilies, fixtures and marks intended for test functionality."

__author__ = "Marie E. Rognes (meg@simula.no), 2014"


import numpy.linalg
import pytest

from cbcbeat import (
    dolfinimport,
)


try:
    has_goss = True
except ImportError:
    has_goss = False

require_goss = pytest.mark.skipif(
    not has_goss,
    reason="goss is required to run the test",
)
require_dolfin_adjoint = pytest.mark.skipif(
    not dolfinimport.has_dolfin_adjoint,
    reason="dolfin-adjoint is required to run the test",
)

# Marks
fast = pytest.mark.fast
medium = pytest.mark.medium
slow = pytest.mark.slow
adjoint = pytest.mark.adjoint
parametrize = pytest.mark.parametrize
disabled = pytest.mark.disabled
xfail = pytest.mark.xfail


# Assertions
def assert_almost_equal(a, b, tolerance):
    c = a - b
    try:
        assert abs(float(c)) < tolerance
    except TypeError:
        c_inf = numpy.linalg.norm(c, numpy.inf)
        assert c_inf < tolerance


def assert_equal(a, b):
    assert a == b


def assert_true(a):
    assert a is True


def assert_greater(a, b):
    assert a > b
