#!/usr/bin/env python
#  -*- coding: utf-8 -*-
#
# How to use the cbcbeat module to just look at one cardiac cell model
# ====================================================================
#
# This demo shows how to
# * Use the SingleCellSolver
# * Adjust cardiac cell model parameters (here following the Arevalo
#   et al, Nature Communications, 2016 set-up

__author__ = "Marie E. Rognes (meg@simula.no), 2017"

import math
from ufl.log import info_green
import pylab
import cbcbeat
import dolfin
from cbcbeat import backend, Tentusscher_panfilov_2006_epi_cell, SingleCellSolver

# Disable adjointing


if cbcbeat.dolfinimport.has_dolfin_adjoint:
    dolfin.parameters["adjoint"]["stop_annotating"] = True

# For easier visualization of the variables
dolfin.parameters["reorder_dofs_serial"] = False

# For computing faster
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = flags


class Stimulus(dolfin.UserExpression):
    "Some self-defined stimulus."

    def __init__(self, time, **kwargs):
        self.t = time
        super().__init__(**kwargs)

    def eval(self, value, x):
        if float(self.t) >= 2 and float(self.t) <= 11:
            v_amp = 125
            value[0] = 0.05 * v_amp
        else:
            value[0] = 0.0

    def value_shape(self):
        return ()


def plot_results(times, values, show=True):
    "Plot the evolution of each variable versus time."

    variables = list(zip(*values))
    pylab.figure(figsize=(20, 10))

    int(math.ceil(math.sqrt(len(variables))))
    for i, var in enumerate(
        [
            variables[0],
        ]
    ):
        # pylab.subplot(rows, rows, i+1)
        pylab.plot(times, var, "*-")
        pylab.title("Var. %d" % i)
        pylab.xlabel("t")
        pylab.grid(True)

    info_green("Saving plot to 'variables.pdf'")
    pylab.savefig("variables.pdf")
    if show:
        pylab.show()


def main(scenario="default"):
    "Solve a single cell model on some time frame."

    # Initialize model and assign stimulus
    params = Tentusscher_panfilov_2006_epi_cell.default_parameters()
    if scenario != "default":
        new = {
            "g_Na": params["g_Na"] * 0.38,
            "g_CaL": params["g_CaL"] * 0.31,
            "g_Kr": params["g_Kr"] * 0.30,
            "g_Ks": params["g_Ks"] * 0.20,
        }
        model = Tentusscher_panfilov_2006_epi_cell(params=new)
    else:
        model = Tentusscher_panfilov_2006_epi_cell()

    time = backend.Constant(0.0)
    model.stimulus = Stimulus(time=time, degree=0)

    # Initialize solver
    params = SingleCellSolver.default_parameters()
    params["scheme"] = "GRL1"
    solver = SingleCellSolver(model, time, params)

    # Assign initial conditions
    (vs_, vs) = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    dt = 0.05
    interval = (0.0, 600.0)

    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for (t0, t1), vs in solutions:
        print(("Current time: %g" % t1))
        times.append(t1)
        values.append(vs.vector().get_local())

    return times, values


def compare_results(times, many_values, legends=(), show=True):
    "Plot the evolution of each variable versus time."

    pylab.figure(figsize=(20, 10))
    for values in many_values:
        variables = list(zip(*values))
        int(math.ceil(math.sqrt(len(variables))))
        for i, var in enumerate(
            [
                variables[0],
            ]
        ):
            # pylab.subplot(rows, rows, i+1)
            pylab.plot(times, var, "-")
            pylab.title("Var. %d" % i)
            pylab.xlabel("t")
            pylab.grid(True)

    pylab.legend(legends)
    info_green("Saving plot to 'variables.pdf'")
    pylab.savefig("variables.pdf")
    if show:
        pylab.show()


if __name__ == "__main__":
    (times, values1) = main("default")
    (times, values2) = main("gray zone")
    compare_results(
        times, [values1, values2], legends=("default", "gray zone"), show=True
    )
