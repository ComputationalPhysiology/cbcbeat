"""
Regression and correctness test for FitzHughNagumoManual model and pure
BasicSingleCellSolver: compare (in eyenorm) time evolution with results from
Section 2.4.1 p. 36 in Sundnes et al, 2006 (checked 2012-09-19), and
check that maximal v/s values do not regress
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2012--2013"
__all__ = ["TestBasicSingleCellSolver"]

import pytest

from cbcbeat import Expression, Constant, Parameters, dolfin_adjoint
from cbcbeat import FitzHughNagumoManual, CardiacCellModel
from cbcbeat import BasicSingleCellSolver

from testutils import slow

class TestBasicSingleCellSolver:

    @slow
    def test_fitzhugh_nagumo_manual(self):
        """Test that the manually written FitzHugh-Nagumo model gives
        comparable results to a given reference from Sundnes et al,
        2006."""

        class Stimulus(Expression):
            def __init__(self, **kwargs):
                self.t = kwargs["t"]
            def eval(self, value, x):
                if float(self.t) >= 50 and float(self.t) < 60:
                    v_amp = 125
                    value[0] = 0.05*v_amp
                else:
                    value[0] = 0.0

        if dolfin_adjoint:
            from dolfin_adjoint import adj_reset
            adj_reset()

        cell = FitzHughNagumoManual()
        time = Constant(0.0)
        cell.stimulus = Stimulus(t=time, degree=0)
        solver = BasicSingleCellSolver(cell, time)

        # Setup initial condition
        (vs_, vs) = solver.solution_fields()
        ic = cell.initial_conditions()
        vs_.assign(ic)

        # Initial set-up
        interval = (0, 400)
        dt = 1.0

        times = []
        v_values = []
        s_values = []

        # Solve
        solutions = solver.solve(interval, dt=dt)
        for (timestep, vs) in solutions:
            (t0, t1) = timestep
            times += [(t0 + t1)/2]

            v_values += [vs.vector()[0]]
            s_values += [vs.vector()[1]]

        # Regression test
        v_max_reference = 2.6883308148064152e+01
        s_max_reference = 6.8660144687023219e+01
        tolerance = 1.e-14
        print("max(v_values) %.16e" % max(v_values))
        print("max(s_values) %.16e" % max(s_values))
        msg = "Maximal %s value does not match reference: diff is %.16e"

        v_diff = abs(max(v_values) - v_max_reference)
        s_diff = abs(max(s_values) - s_max_reference)
        assert (v_diff < tolerance), msg % ("v", v_diff)
        assert (s_diff < tolerance), msg % ("s", s_diff)

        # Correctness test
        import os
        if int(os.environ.get("DOLFIN_NOPLOT", 0)) != 1:
            import pylab
            pylab.plot(times, v_values, 'b*')
            pylab.plot(times, s_values, 'r-')

    @slow
    def test_fitz_hugh_nagumo_modified(self):

        k = 0.00004
        Vrest = -85.
        Vthreshold = -70.
        Vpeak = 40.
        k = 0.00004
        l = 0.63
        b = 0.013

        class FHN2(CardiacCellModel):
            """ODE model:

            parameters(Vrest,Vthreshold,Vpeak,k,l,b,ist)

            input(u)
            output(g)
            default_states(v=-85, w=0)

            Vrest = -85;
            Vthreshold = -70;
            Vpeak = 40;
            k = 0.00004;
            l = 0.63;
            b = 0.013;
            ist = 0.0

            v = u[0]
            w = u[1]

            g[0] =  -k*(v-Vrest)*(w+(v-Vthreshold)*(v-Vpeak))-ist;
            g[1] = l*(v-Vrest) - b*w;

            [specified by G. T. Lines Sept 22 2012]

            Note the minus sign convention here in the specification of
            I (g[0]) !!
            """

            def __init__(self):
                CardiacCellModel.__init__(self)

            def default_parameters(self):
                parameters = Parameters("FHN2")
                parameters.add("Vrest", Vrest)
                parameters.add("Vthreshold", Vthreshold)
                parameters.add("Vpeak", Vpeak)
                parameters.add("k", k)
                parameters.add("l", l)
                parameters.add("b", b)
                parameters.add("ist", 0.0)
                return parameters

            def I(self, v, w, time=None):
                k = self._parameters["k"]
                Vrest = self._parameters["Vrest"]
                Vthreshold = self._parameters["Vthreshold"]
                Vpeak = self._parameters["Vpeak"]
                ist = self._parameters["ist"]
                i =  -k*(v-Vrest)*(w+(v-Vthreshold)*(v-Vpeak))-ist;
                return -i

            def F(self, v, w, time=None):
                l = self._parameters["l"]
                b = self._parameters["b"]
                Vrest = self._parameters["Vrest"]
                return l*(v-Vrest) - b*w;

            def num_states(self):
                return 1

            def __str__(self):
                return "Modified FitzHugh-Nagumo cardiac cell model"

        def _run(cell):
            if dolfin_adjoint:
                from dolfin_adjoint import adj_reset
                adj_reset()

            solver = BasicSingleCellSolver(cell, Constant(0.0))

            # Setup initial condition
            (vs_, vs) = solver.solution_fields()
            vs_.vector()[0] = 30. # Non-resting state
            vs_.vector()[1] = 0.

            T = 2
            solutions = solver.solve((0, T), 0.25)
            times = []
            v_values = []
            s_values = []
            for ((t0, t1), vs) in solutions:
                times += [0.5*(t0 + t1)]
                v_values.append(vs.vector()[0])
                s_values.append(vs.vector()[1])

            return (v_values, s_values, times)

        # Try the modified one
        cell_mod = FHN2()
        (v_values_mod, s_values_mod, times_mod) = _run(cell_mod)

        # Compare with our standard FitzHugh (reparametrized)
        v_amp = Vpeak - Vrest
        cell_parameters = {"c_1": k*v_amp**2, "c_2": k*v_amp, "c_3": b/l,
                           "a": (Vthreshold - Vrest)/v_amp, "b": l,
                           "v_rest": Vrest, "v_peak": Vpeak}
        cell = FitzHughNagumoManual(cell_parameters)
        (v_values, s_values, times) = _run(cell)

        msg = "Mismatch in %s value comparison, diff = %.16e"
        v_diff = abs(v_values[-1] - v_values_mod[-1])
        s_diff = abs(s_values[-1] - s_values_mod[-1])
        assert (v_diff < 1.e-12), msg % v_diff
        assert (s_diff < 1.e-12), msg % s_diff

        # Look at some plots
        import os
        if int(os.environ.get("DOLFIN_NOPLOT", 0)) != 1:
            import pylab
            pylab.title("Standard FitzHugh-Nagumo")
            pylab.plot(times, v_values, 'b*')
            pylab.plot(times, s_values, 'r-')

            pylab.figure()
            pylab.title("Modified FitzHugh-Nagumo")
            pylab.plot(times_mod, v_values_mod, 'b*')
            pylab.plot(times_mod, s_values_mod, 'r-')
