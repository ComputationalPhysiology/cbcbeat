import sys
import os
import random
import pytest
import cbcbeat
if cbcbeat.dolfin_adjoint:
    import dolfin_adjoint

default_params = cbcbeat.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    cbcbeat.parameters.update(default_params)

    # Reset adjoint state
    if cbcbeat.dolfin_adjoint:
        cbcbeat.adj_reset()

    # Fix the seed to avoid random test failures due to slight
    # tolerance variations
    random.seed(21)
