"""
Unit tests for the merger in splitting solver
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestMerger"]

from testutils import fast, parametrize

import numpy as np
from cbcbeat import CardiacModel, \
        BasicSplittingSolver, SplittingSolver, \
        FitzHughNagumoManual, UnitCubeMesh, Constant

class TestMerger(object):
    "Test functionality for the splitting solvers."

    def setup(self):
        self.mesh = UnitCubeMesh(2, 2, 2)
        self.cell_model = FitzHughNagumoManual()
        self.cardiac_model = CardiacModel(self.mesh, None,
                                          1.0, 2.0,
                                          self.cell_model)

    @fast
    @parametrize("Solver", [SplittingSolver, BasicSplittingSolver])
    def test_basic_and_optimised_splitting_solver_merge(self, Solver):
        """Test that the merger in basic and optimised splitting
        solvers works.
        """

        # Create basic solver
        solver = Solver(self.cardiac_model)

        (vs_, vs, vur) = solver.solution_fields()

        vs.vector()[:] = 2.0
        vur.vector()[:] = 1.0
        solver.merge(vs)

        tol = 1e-13
        assert np.abs(vs.sub(0, deepcopy=1).vector().array()-1.0).max() < tol
        assert np.abs(vs.sub(1, deepcopy=1).vector().array()-2.0).max() < tol
