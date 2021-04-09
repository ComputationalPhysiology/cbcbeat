"""Single cell simulation of the response of the cell model
(reparametrised 1994 Rogers and McCulloch) and parameters used in
Nagaiah et al, J Math Biol, 2013.
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2014"

import math
import pylab
from cbcbeat import *

# For easier visualization of the variables
parameters["reorder_dofs_serial"] = False

# For computing faster
parameters["form_compiler"]["cpp_optimize"] = True
flags = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["cpp_optimize_flags"] = flags

class Stimulus(Expression):
    "Applied stimulus"
    def __init__(self, t):
        self.t = t # ms
    def eval(self, value, x):
        if float(self.t) >= 435 and float(self.t) <= 439:
            value[0] = 20. # mV
        else:
            value[0] = 0.0
def main():
    "Solve a single cell model on some time frame."

    # Initialize model and assign stimulus
    #model = RogersMcCulloch()
    model = FitzHughNagumoManual()
    time = Constant(430.0)
    model.stimulus = {0: Stimulus(time)}

    # Initialize solver
    #params = BasicSingleCellSolver.default_parameters()
    solver = BasicSingleCellSolver(model, time)#, params)

    # Assign initial conditions
    (vs_, vs) = solver.solution_fields()
    vs_.assign(model.initial_conditions())

    # Solve and extract values
    dt = 0.5
    T = 700. + 0.0001 # 639
    interval = (float(time), T)

    solutions = solver.solve(interval, dt)
    times = []
    values = []
    for ((t0, t1), vs) in solutions:
        print "Current time: %g" % t1
        times.append(t1)
        values.append(vs.vector().array())

    return times, values

def plot_results(times, values, show=True):
    "Plot the evolution of each variable versus time."

    variables = zip(*values)
    pylab.figure(figsize=(20, 10))

    rows = int(math.ceil(math.sqrt(len(variables))))
    for (i, var) in enumerate(variables):
        pylab.subplot(rows, rows, i+1)
        pylab.plot(times, var, '*-')
        pylab.title("Var. %d" % i)
        pylab.xlabel("t")
        pylab.grid(True)

    # Store plot
    filename = "single_cell_response.pdf"
    info_green("Saving plot to %s" % filename)
    pylab.savefig(filename)
    if show:
        pylab.show()

if __name__ == "__main__":

    (times, values) = main()
    plot_results(times, values, show=True)
