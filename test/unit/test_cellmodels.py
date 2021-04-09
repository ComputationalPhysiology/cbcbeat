"Basic and more advanced tests for the cell models and their forms."

__author__ = "Marie E. Rognes (meg@simula.no), 2013 -- 2014"

import pytest

from dolfin import *
from cbcbeat.utils import state_space

from testutils import fast, slow, parametrize, assert_almost_equal
from testutils import cell_model, ode_test_form

class TestModelCreation:
    "Test basic features of cell models."
    def test_create_cell_model_has_ics(self, cell_model):
        "Test that cell model has initial conditions."
        model = cell_model
        ics = model.initial_conditions()

class TestFormCompilation:
    "Test form compilation with different optimizations."
    def test_form_compilation(self, ode_test_form):
        "Test that form can be compiled by FFC."
        f = Form(ode_test_form)

    @slow
    def test_optimized_form_compilation(self, ode_test_form):
        "Test that form can be compiled by FFC with optimizations."
        ps = parameters["form_compiler"].copy()
        ps["cpp_optimize"] = True
        f = Form(ode_test_form, form_compiler_parameters=ps)

    @slow
    def test_custom_optimized_compilation(self, ode_test_form):
        "Test that form can be compiled with custom optimizations."
        ps = parameters["form_compiler"].copy()
        ps["cpp_optimize"] = True
        flags = ["-O3", "-ffast-math", "-march=native"]
        ps["cpp_optimize_flags"] = " ".join(flags)
        f = Form(ode_test_form, form_compiler_parameters=ps)

class TestCompilationCorrectness:
    "Test form compilation results with different optimizations."

    def point_integral_step(self, model):
        # Set-up forms
        mesh = UnitSquareMesh(10, 10)
        V = FunctionSpace(mesh, "CG", 1)
        S = state_space(mesh, model.num_states())
        Me = MixedElement((V.ufl_element(), S.ufl_element()))
        VS = FunctionSpace(mesh, Me)
        vs = Function(VS)
        vs.assign(project(model.initial_conditions(), VS))
        (v, s) = split(vs)
        (w, r) = TestFunctions(VS)
        rhs = inner(model.F(v, s), r) + inner(- model.I(v, s), w)
        form = rhs*dP

        # Set-up scheme
        time = Constant(0.0)
        scheme = BackwardEuler(form, vs, time)
        scheme.t().assign(float(time))

        # Create and step solver
        solver = PointIntegralSolver(scheme)
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
        solver.parameters["newton_solver"]["report"] = False
        dt = 0.1
        solver.step(dt)
        return vs

    @slow
    def test_point_integral_solver(self, cell_model):
        "Compare form compilation result with and without optimizations."

        parameters["form_compiler"]["representation"] = "uflacs" #"quadrature"
        parameters["form_compiler"]["quadrature_degree"] = 2
        tolerance = 1.e-12

        # Run with no particular optimizations
        vs = self.point_integral_step(cell_model)
        non_opt_result = vs.vector().get_local()  #vs.vector().array()

        # Compare with results using aggresssive optimizations
        flags = "-O3 -ffast-math -march=native"
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = flags
        vs = self.point_integral_step(cell_model)
        assert_almost_equal(non_opt_result, vs.vector().get_local(), tolerance) # vs.vector().array(), tolerance)

        # Compare with results using standard optimizations
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
        vs = self.point_integral_step(cell_model)
        assert_almost_equal(non_opt_result, vs.vector().get_local(), tolerance) #vs.vector().array(), tolerance)

        # Compare with results using uflacs if installed
        try:
            parameters["form_compiler"]["representation"] = "uflacs"
            vs = self.point_integral_step(cell_model)
            assert_almost_equal(non_opt_result, vs.vector.get_local(), tolerance) #vs.vector().array(), tolerance)
        except:
            pass

        # Reset parameters
        parameters["form_compiler"]["representation"] = "auto"
