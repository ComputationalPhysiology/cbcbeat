import random
import cbcbeat
import pytest
import dolfin

try:
    from ufl_legacy import inner, dP
except ImportError:
    from ufl import inner, dP

from dolfin import UnitSquareMesh, FunctionSpace, MixedElement, split, TestFunctions
from cbcbeat.utils import state_space
from cbcbeat import (
    supported_cell_models,
    backend,
    NoCellModel,  # noqa:F401
    FitzHughNagumoManual,  # noqa: F401
    Tentusscher_2004_mcell,  # noqa: F401
    RogersMcCulloch,  # noqa: F401
    Beeler_reuter_1977,  # noqa: F401
    Tentusscher_panfilov_2006_epi_cell,  # noqa: F401
    Fenton_karma_1998_MLR1_altered,  # noqa: F401
    Fenton_karma_1998_BR_altered,  # noqa: F401
)


default_params = dolfin.parameters.copy()


def pytest_runtest_setup(item):
    """Hook function which is called before every test"""

    # Reset dolfin parameter dictionary
    cbcbeat.parameters.update(default_params)

    # Reset adjoint state
    if cbcbeat.dolfinimport.has_dolfin_adjoint:
        cbcbeat.adj_reset()

    # Fix the seed to avoid random test failures due to slight
    # tolerance variations
    random.seed(21)


# Fixtures
supported_cell_models_str = [Model.__name__ for Model in supported_cell_models]


@pytest.fixture(params=supported_cell_models_str)
def cell_model(request):
    Model = eval(request.param)
    return Model()


@pytest.fixture(params=supported_cell_models_str)
def ode_test_form(request):
    Model = eval(request.param)
    model = Model()
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    S = state_space(mesh, model.num_states())
    Mx = MixedElement((V.ufl_element(), S.ufl_element()))
    VS = FunctionSpace(mesh, Mx)
    vs = backend.Function(VS)
    vs.assign(backend.project(model.initial_conditions(), VS))
    (v, s) = split(vs)
    (w, r) = TestFunctions(VS)
    rhs = inner(model.F(v, s), r) + inner(-model.I(v, s), w)
    form = rhs * dP
    return form
