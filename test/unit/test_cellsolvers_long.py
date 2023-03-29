"""
Unit tests for various types of solvers for cardiac cell models.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = ["TestCardiacODESolver"]

import numpy as np
import os

from dolfin import dof_to_vertex_map
from ufl.log import info_green
from cbcbeat import *
from testutils import assert_almost_equal, adjoint, slow


class TestCardiacODESolver(object):

    # Note that these should be essentially identical to the ones
    # for the BasicSingleCellSolver
    references = {
        NoCellModel: {
            "BackwardEuler": (0, 0.3),
            "CrankNicolson": (0, 0.2),
            "ForwardEuler": (0, 0.1),
            "RK4": (0, 0.2),
            "ESDIRK3": (0, 0.2),
            "ESDIRK4": (0, 0.2),
        },
        FitzHughNagumoManual: {
            "BackwardEuler": (0, -84.70013280019053),
            "CrankNicolson": (0, -84.80005016079546),
            "ForwardEuler": (0, -84.9),
            "RK4": (0, -84.80004467770296),
            "ESDIRK3": (0, -84.80004459269247),
            "ESDIRK4": (0, -84.80004468281632),
        },
        Tentusscher_2004_mcell: {
            "BackwardEuler": (1, -85.89745525156506),
            "CrankNicolson": (1, -85.99685674414921),
            "ForwardEuler": (1, -86.09643254164848),
            "RK4": (1, "nan"),
            "ESDIRK3": (1, -85.99681862337053),
            "ESDIRK4": (1, -85.99681796046603),
        },
    }

    def _setup_solver(self, Model, Scheme, mesh, time, stim=None, params=None):
        # Create model instance
        model = Model(params=params)

        # Initialize time and stimulus (note t=time construction!)
        if stim is None:
            stim = Expression("1000*t", t=time, degree=1)

        # Initialize solver
        params = CardiacODESolver.default_parameters()
        params["scheme"] = Scheme
        solver = CardiacODESolver(mesh, time, model, I_s=stim, params=params)

        # Create scheme
        info_green("\nTesting %s with %s scheme" % (model, Scheme))

        # Start with native initial conditions
        (vs_, vs) = solver.solution_fields()
        vs_.assign(model.initial_conditions())
        vs.assign(vs_)

        return solver


@adjoint
@slow
def closure_long_run(Scheme, dt_org, abs_tol, rel_tol):
    def long_run_compare(self):

        mesh = UnitIntervalMesh(5)

        # FIXME: We need to make this run in paralell.
        if MPI.size(mesh.mpi_comm()) > 1:
            return

        Model = Tentusscher_2004_mcell
        tstop = 10
        ind_V = 0
        dt_ref = 0.1
        time_ref = np.linspace(0, tstop, int(tstop / dt_ref) + 1)
        dir_path = os.path.dirname(__file__)
        Vm_reference = np.fromfile(os.path.join(dir_path, "Vm_reference.npy"))
        params = Model.default_parameters()

        time = Constant(0.0)
        stim = Expression(
            "(time >= stim_start) && (time < stim_start + stim_duration)"
            " ? stim_amplitude : 0.0 ",
            time=time,
            stim_amplitude=52.0,
            stim_start=1.0,
            stim_duration=1.0,
            degree=1,
        )

        # Initiate solver, with model and Scheme
        if dolfin_adjoint:
            adj_reset()

        solver = self._setup_solver(Model, Scheme, mesh, time, stim, params)
        solver._pi_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-8
        solver._pi_solver.parameters["newton_solver"]["maximum_iterations"] = 30
        solver._pi_solver.parameters["newton_solver"]["report"] = False

        scheme = solver._scheme
        (vs_, vs) = solver.solution_fields()

        vs.assign(vs_)

        dof_to_vertex_map_values = dof_to_vertex_map(vs.function_space())
        scheme.t().assign(0.0)

        vs_array = np.zeros(
            mesh.num_vertices() * vs.function_space().dofmap().num_entity_dofs(0)
        )
        vs_array[dof_to_vertex_map_values] = vs.vector().get_local()
        output = [vs_array[ind_V]]
        time_output = [0.0]
        dt = dt_org

        # Time step
        next_dt = max(min(tstop - float(scheme.t()), dt), 0.0)
        t0 = 0.0

        while next_dt > 0.0:

            # Step solver
            solver.step((t0, t0 + next_dt))
            vs_.assign(vs)

            # Collect plt output data
            vs_array[dof_to_vertex_map_values] = vs.vector().get_local()
            output.append(vs_array[ind_V])
            time_output.append(float(scheme.t()))

            # Next time step
            t0 += next_dt
            next_dt = max(min(tstop - float(scheme.t()), dt), 0.0)

        # Compare solution from CellML run using opencell
        assert_almost_equal(output[-1], Vm_reference[-1], abs_tol)

        output = np.array(output)
        time_output = np.array(time_output)

        output = np.interp(time_ref, time_output, output)

        value = np.sqrt(np.sum(((Vm_reference - output) / Vm_reference) ** 2)) / len(
            Vm_reference
        )
        assert_almost_equal(value, 0.0, rel_tol)

    return long_run_compare


for Scheme, dt_org, abs_tol, rel_tol in [
    ("BackwardEuler", 0.1, 1e-1, 1e-1),
    ("CrankNicolson", 0.125, 1e-0, 1e-1),
    ("ESDIRK3", 0.125, 1e-0, 1e-1),
    ("ESDIRK4", 0.125, 1e-0, 1e-1),
]:

    func = closure_long_run(Scheme, dt_org, abs_tol, rel_tol)
    setattr(TestCardiacODESolver, "test_{0}_long_run_tentusscher".format(Scheme), func)
