"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013 and Simon W. Funke (simon@simula.no), 2014"
__all__ = ["TestCardiacODESolverAdjoint"]

import pytest
from cbcbeat import *
from testutils import assert_true, assert_greater, slow, adjoint, cell_model, parametrize, xfail

supported_schemes = ["CrankNicolson",
                     "RK4",
                     "ESDIRK4"]

fails_with_RK4 = (Tentusscher_2004_mcell,
                  Tentusscher_panfilov_2006_epi_cell,
)

seed_collection_adm = {Tentusscher_2004_mcell: 1e-5,
                       Beeler_reuter_1977: 1e-5,
                       Tentusscher_panfilov_2006_epi_cell: 1e-6,
}

seed_collection_tlm = seed_collection_adm.copy()
seed_collection_tlm[Tentusscher_panfilov_2006_epi_cell] = 4e-5

cellmodel_parameters_seeds = {}
cellmodel_parameters_seeds[FitzHughNagumoManual] = ("a", 1e-5)
cellmodel_parameters_seeds[RogersMcCulloch] = ("g", 1e-5)
cellmodel_parameters_seeds[Tentusscher_2004_mcell] = ("g_CaL", 1e-5)
cellmodel_parameters_seeds[Tentusscher_panfilov_2006_epi_cell] = ("g_to", 1e-6)
cellmodel_parameters_seeds[Beeler_reuter_1977] = ("g_s", 1e-5)

fails_with_forward_euler = ()

class TestCardiacODESolverAdjoint(object):

    def setup_dolfin_parameters(self):
        ''' Set optimisation parameters for these tests '''

        parameters["form_compiler"]["cpp_optimize"] = True
        flags = "-O3 -ffast-math -march=native"
        parameters["form_compiler"]["cpp_optimize_flags"] = flags

    def _setup_solver(self, model, Scheme, mesh):

        # Initialize time and stimulus (note t=time construction!)
        time = Constant(0.0)
        stim = Expression("(time >= stim_start) && (time < stim_start + stim_duration)"
                          " ? stim_amplitude : 0.0 ", time=time, stim_amplitude=52.0,
                          stim_start=0.0, stim_duration=1.0, name="stim", degree=1)

        # Initialize solver
        params = CardiacODESolver.default_parameters()
        params["scheme"] = Scheme
        solver = CardiacODESolver(mesh, time, model, I_s=stim, params=params)

        return solver

    def _run(self, solver, ics):
        # Assign initial conditions

        solver._pi_solver.scheme().t().assign(0)
        (vs_, vs) = solver.solution_fields()
        vs_.assign(ics)

        # Solve for a couple of steps
        dt = 0.01
        T = 4*dt
        dt = [(0.0, dt), (dt*3,dt/2)]
        solver._pi_solver.parameters.update({"reset_stage_solutions" : True})
        solver._pi_solver.parameters.update({"newton_solver":
                                             {"reset_each_step": True}})
        solver._pi_solver.parameters.update({"newton_solver":
                                             {"relative_tolerance": 1.0e-10}})
        solver._pi_solver.parameters.update({"newton_solver":
                                             {"always_recompute_jacobian": True
                                             }})
        solutions = solver.solve((0.0, T), dt)
        for ((t0, t1), vs) in solutions:
            pass

    def tlm_adj_setup_initial_conditions(self, cell_model, Scheme):
        mesh = UnitIntervalMesh(3)
        Model = cell_model.__class__

        # Initiate solver, with model and Scheme
        params = Model.default_parameters()
        model = Model(params=params)

        solver = self._setup_solver(model, Scheme, mesh)
        ics = project(model.initial_conditions(), solver.VS).copy(deepcopy=True, name="ics")

        info_green("Running forward %s with %s (setup)" % (model, Scheme))
        self._run(solver, ics)

        # Define functional
        (vs_, vs) = solver.solution_fields()
        form = lambda w: inner(w, w)*dx
        J = Functional(form(vs)*dt[FINISH_TIME])

        # Compute value of functional with current ics
        Jics = assemble(form(vs))

        # Set-up runner
        def Jhat(ics):
            self._run(solver, ics)
            (vs_, vs) = solver.solution_fields()
            return assemble(form(vs))

        # Stop annotating
        parameters["adjoint"]["stop_annotating"] = True

        m = Control(vs)
        return J, Jhat, m, Jics

    def tlm_adj_setup_cellmodel_parameters(self, cell_model, Scheme):
        mesh = UnitIntervalMesh(3)
        Model = cell_model.__class__

        # Initiate solver, with model and Scheme
        cell_params = Model.default_parameters()
        param_name = cellmodel_parameters_seeds[Model][0]
        cell_params[param_name] = Constant(cell_params[param_name], name=param_name)
        model = Model(params=cell_params)

        solver = self._setup_solver(model, Scheme, mesh)

        info_green("Running forward %s with %s (setup)" % (model, Scheme))
        ics = Function(project(model.initial_conditions(), solver.VS), name="ics")
        self._run(solver, ics)

        # Define functional
        (vs_, vs) = solver.solution_fields()
        form = lambda w: inner(w, w)*dx
        J = Functional(form(vs)*dt[FINISH_TIME])

        # Compute value of functional with current ics
        Jics = assemble(form(vs))

        # Set-up runner
        def Jhat(val):
            cell_params[param_name].assign(val)
            ics = Function(project(model.initial_conditions(), solver.VS), name="ics")
            self._run(solver, ics)
            (vs_, vs) = solver.solution_fields()
            return assemble(form(vs))

        # Stop annotating
        solver.parameters["enable_adjoint"] = False

        m = ConstantControl(cell_params[param_name])
        return J, Jhat, m, Jics

    @adjoint
    @slow
    @parametrize(("Scheme"), supported_schemes)
    def test_replay(self, cell_model, Scheme):
        mesh = UnitIntervalMesh(3)
        Model = cell_model.__class__

        if isinstance(cell_model, fails_with_RK4) and Scheme == "RK4":
            pytest.xfail("RK4 is unstable for some models with this timestep (0.01)")

        # Initiate solver, with model and Scheme
        params = Model.default_parameters()
        model = Model(params=params)

        solver = self._setup_solver(model, Scheme, mesh)
        ics = project(model.initial_conditions(), solver.VS)

        info_green("Running forward %s with %s (replay)" % (model, Scheme))
        self._run(solver, ics)

        print(solver.solution_fields()[0].vector().get_local())

        info_green("Replaying")
        success = replay_dolfin(tol=0, stop=True)
        assert_true(success)

    @adjoint
    @slow
    @parametrize(("Scheme"), supported_schemes)
    def test_tlm_initial(self, cell_model, Scheme):
        "Test that we can compute the gradient for some given functional"

        if Scheme == "ForwardEuler":
            pytest.xfail("Forward Euler is unstable for some models with this timestep (0.01)")

        if isinstance(cell_model, fails_with_RK4) and Scheme == "RK4":
            pytest.xfail("RK4 is unstable for some models with this timestep (0.01)")

        J, Jhat, m, Jics = self.tlm_adj_setup_initial_conditions(cell_model, Scheme)

        # Seed for taylor test
        seed = seed_collection_tlm.get(cell_model.__class__)

        # Check TLM correctness
        info_green("Computing gradient")
        dJdics = compute_gradient_tlm(J, m, forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate_tlm = taylor_test(Jhat, m, Jics, dJdics, seed=seed)

        assert_greater(conv_rate_tlm, 1.8)
        return

    @adjoint
    @slow
    @xfail  #  dolfin-adjoint does not support differentiating with respect to
            # ODE parameters yet
    @parametrize(("Scheme"), supported_schemes)
    def test_tlm_cell_model_parameter(self, cell_model, Scheme):
        if Scheme == "ForwardEuler":
            pytest.xfail("Forward Euler is unstable for some models with this timestep (0.01)")

        if isinstance(cell_model, fails_with_RK4) and Scheme == "RK4":
            pytest.xfail("RK4 is unstable for some models with this timestep (0.01)")

        J, Jhat, m, Jics = self.tlm_adj_setup_cellmodel_parameters(cell_model, Scheme)

        # Seed for taylor test
        seed = seed_collection_tlm.get(cell_model.__class__)

        # Stop annotating
        parameters["adjoint"]["stop_annotating"] = True

        # Check TLM correctness
        info_green("Computing gradient")
        dJdics = compute_gradient_tlm(J, m, forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate_tlm = taylor_test(Jhat, m, Jics, dJdics, seed=seed)

        assert_greater(conv_rate_tlm, 1.8)

    @adjoint
    @slow
    @parametrize(("Scheme"), supported_schemes)
    def test_adjoint_initial(self, cell_model, Scheme):
        """ Test that the gradient computed with the adjoint model is correct. """

        if isinstance(cell_model, fails_with_RK4) and Scheme == "RK4":
            pytest.xfail("RK4 is unstable for some models with this timestep (0.01)")

        if isinstance(cell_model, fails_with_forward_euler) and Scheme == "ForwardEuler":
            pytest.xfail("ForwardEuler is unstable for some models with this timestep (0.01)")

        J, Jhat, m, Jics = self.tlm_adj_setup_initial_conditions(cell_model, Scheme)

        # Seed for taylor test
        seed = seed_collection_adm.get(cell_model.__class__)

        # Compute gradient with respect to vs.
        info_green("Computing gradient")
        dJdics = compute_gradient(J, m, forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate = taylor_test(Jhat, m, Jics, dJdics, seed=seed)

        # Check that minimal rate is greater than some given number
        assert_greater(conv_rate, 1.9)

    @adjoint
    @slow
    @xfail  #  dolfin-adjoint does not support differentiating with respect to
            # ODE parameters yet
    @parametrize(("Scheme"), supported_schemes)
    def test_adjoint_cell_model_parameter(self, cell_model, Scheme):
        """ Test that the gradient computed with the adjoint model is correct. """

        if isinstance(cell_model, fails_with_RK4) and Scheme == "RK4":
            pytest.xfail("RK4 is unstable for some models with this timestep (0.01)")

        if isinstance(cell_model, fails_with_forward_euler) and Scheme == "ForwardEuler":
            pytest.xfail("ForwardEuler is unstable for some models with this timestep (0.01)")

        J, Jhat, m, Jics = self.tlm_adj_setup_cellmodel_parameters(cell_model, Scheme)

        # Seed for taylor test
        seed = seed_collection_adm.get(cell_model.__class__)

        # Compute gradient with respect to vs.
        info_green("Computing gradient")
        dJdics = compute_gradient(J, m, forget=False)
        assert (dJdics is not None), "Gradient is None (#fail)."
        conv_rate = taylor_test(Jhat, m, Jics, dJdics, seed=seed)

        # Check that minimal rate is greater than some given number
        assert_greater(conv_rate, 1.9)
