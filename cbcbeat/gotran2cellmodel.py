# Copyright (C) 2012 Johan Hake (hake.dev@gmail.com)
# Use and modify at will
# Last changed: 2014-09-12

try:
    import gotran
except Exception as e:
    print("Gotran not installed. Not possible to convert gotran model to cellmodel.")
    raise e

# Gotran imports
from gotran.model import load_ode
from gotran.model.ode import ODE
from modelparameters.logger import error as gotran_error
from modelparameters.utils import check_arg

from gotran.common.options import parameters as gotran_parameters
from gotran.codegeneration.algorithmcomponents import componentwise_derivative
from gotran.codegeneration.codecomponent import CodeComponent

from gotran.codegeneration.codegenerators import DOLFINCodeGenerator

try:
    from ufl_legacy.log import error
except ImportError:
    from ufl.log import error

_class_template = """
\"\"\"This module contains a {ModelName} cardiac cell model

The module was autogenerated from a gotran ode file
\"\"\"
from __future__ import division
from collections import OrderedDict
import ufl

from cbcbeat.dolfinimport import *
from cbcbeat.cellmodels import CardiacCellModel

class {ModelName}(CardiacCellModel):
    def __init__(self, params=None, init_conditions=None):
        \"\"\"
        Create cardiac cell model

        *Arguments*
         params (dict, :py:class:`dolfin.Mesh`, optional)
           optional model parameters
         init_conditions (dict, :py:class:`dolfin.Mesh`, optional)
           optional initial conditions
        \"\"\"
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        \"Set-up and return default parameters.\"
{default_parameters}
        return params

    @staticmethod
    def default_initial_conditions():
        \"Set-up and return default initial conditions.\"
{initial_conditions}
        return ic

    def _I(self, v, s, time):
        \"\"\"
        Original gotran transmembrane current dV/dt
        \"\"\"
{I_body}

    def I(self, v, s, time=None):
        \"\"\"
        Transmembrane current

           I = -dV/dt

        \"\"\"
        return -self._I(v, s, time)

    def F(self, v, s, time=None):
        \"\"\"
        Right hand side for ODE system
        \"\"\"
{F_body}

    def num_states(self):
        return {num_states}

    def __str__(self):
        return '{ModelName} cardiac cell model'
"""

_class_form = dict(
    ModelName="NOT_IMPLEMENTED",
    default_parameters="NOT_IMPLEMENTED",
    I_body="NOT_IMPLEMENTED",
    F_body="NOT_IMPLEMENTED",
    num_states="NOT_IMPLEMENTED",
    initial_conditions="NOT_IMPLEMENTED",
)


__all__ = ["CellModelGenerator"]


class CellModelGenerator(DOLFINCodeGenerator):
    """
    Convert a Gotran model to a cbcbeat compatible cell model
    """

    def __init__(self, ode, membrane_potential):
        # Init base class
        super(CellModelGenerator, self).__init__()

        check_arg(ode, ODE, 0)
        check_arg(membrane_potential, str, 1)

        assert not ode.is_dae

        # Capitalize first letter of name
        name = ode.name
        self.name = (
            name
            if name[0].isupper()
            else name[0].upper() + (name[1:] if len(name) > 1 else "")
        )

        # Set cbcbeat compatible gotran code generation parameters
        generation_params = gotran_parameters.generation.copy()
        generation_params.code.default_arguments = "stp"
        generation_params.code.time.name = "time"

        generation_params.code.array.index_format = "[]"
        generation_params.code.array.index_offset = 0

        generation_params.code.parameters.representation = "named"
        generation_params.code.parameters.array_name = "parameters"
        generation_params.code.states.representation = "named"
        generation_params.code.states.array_name = "states"
        generation_params.code.body.representation = "named"
        generation_params.code.body.use_cse = False

        if ode.num_full_states < 2:
            gotran_error("expected the ODE to have at least more than 1 state")

        # Check that ode model has the membrane potential state name
        if membrane_potential not in ode.present_ode_objects:
            gotran_error(
                "Cannot find the membrane potential. ODE does not "
                "contain a state with name '{0}'".format(membrane_potential)
            )

        state = ode.present_ode_objects[membrane_potential]
        if not isinstance(state, gotran.model.State):
            gotran_error(
                "Cannot find the membrane potential. ODE does not "
                "contain a state with name '{0}'".format(membrane_potential)
            )

        # The name of the membrane potential
        self.V_name = state.name

        # Get the I and F expressions
        I_ind = [
            ind
            for ind, expr in enumerate(ode.state_expressions)
            if expr.state.name == self.V_name
        ][0]

        # Create gotran code component for I(s,t) (dV/dt)
        I_comp = componentwise_derivative(
            ode, I_ind, generation_params.code, result_name="current"
        )

        # Create gotran code component for F(s,t) (dS/dt)
        F_inds = list(range(ode.num_full_states))
        F_inds.remove(I_ind)
        F_comp = componentwise_derivative(
            ode, F_inds, generation_params.code, result_name="F_expressions"
        )

        # Create the class form and start fill it
        self._class_form = _class_form.copy()
        self._class_form["I_body"] = self.function_code(
            I_comp, indent=2, include_signature=False
        )
        self._class_form["F_body"] = self.function_code(
            F_comp, indent=2, include_signature=False
        )
        self._class_form["num_states"] = ode.num_full_states - 1
        self._class_form["ModelName"] = self.name
        self._class_form["default_parameters"] = self.default_parameters_body(ode)
        self._class_form["initial_conditions"] = self.initial_conditions_body(ode)

    def _init_arguments(self, comp, default_arguments=None):
        check_arg(comp, CodeComponent)
        params = self.params.code
        default_arguments = default_arguments or params.default_arguments

        # Check if comp defines used_states if not use the root components
        # full_states attribute
        # FIXME: No need for full_states here...
        states = [state.name for state in comp.root.full_states]
        used_parameters = comp.used_parameters

        num_states = comp.root.num_full_states

        # Start building body
        body_lines = []
        body_lines.append("time = time if time else Constant(0.0)")
        body_lines.append("")
        body_lines.append("# Assign states")
        if self.V_name != "v":
            body_lines.append("{0} = v".format(self.V_name))

        state_names = [state for state in states if state != self.V_name]
        if len(states) == 1:
            if states[0] != "s":
                body_lines.append("{0} = s".format(states[0]))
        else:
            body_lines.append("assert(len(s) == {0})".format(num_states - 1))
            body_lines.append(", ".join(state_names) + " = s")
        body_lines.append("")

        body_lines.append("# Assign parameters")
        for param in used_parameters:
            body_lines.append(
                "{0} = self._parameters[" '"{1}"]'.format(param.name, param.name)
            )

        # If initilizing results
        if comp.results:
            body_lines.append("")
            body_lines.append("# Init return args")

        for result_name in comp.results:
            shape = comp.shapes[result_name]
            if len(shape) > 1:
                error("expected only result expression with rank 1")

            body_lines.append("{0} = [ufl.zero()]*{1}".format(result_name, shape[0]))

        return body_lines

    def generate(self):
        """
        Return a beat cell model file as a str
        """
        return _class_template.format(**self._class_form)

    def default_parameters_body(self, ode):
        """
        Generate code for the default parameter bod
        """
        if ode.num_parameters > 0:
            params = ode.parameters[:]

            param = params.pop(0)

            body_lines = [
                'params = OrderedDict([("{}", {}),'.format(param.name, param.init)
            ]
            body_lines.extend(
                '                      ("{}", {}),'.format(
                    param.name,
                    param.init
                    if isinstance(param.init, (float, int))
                    else param.init[0],
                )
                for param in params
            )
            body_lines[-1] = body_lines[-1][0:-1] + "])"
        else:
            body_lines = ["params = OrderedDict()"]

        return "\n".join(self.indent_and_split_lines(body_lines, 2))

    def initial_conditions_body(self, ode):
        """
        Generate code for the ic body
        """

        # First get ic for v
        v_init = ode.present_ode_objects[self.V_name].init
        s_init, s_names = zip(
            *[
                (state.init, state.name)
                for state in ode.full_states
                if state.name != self.V_name
            ]
        )
        body_lines = ['ic = OrderedDict([("{}", {}),'.format(self.V_name, v_init)]
        body_lines.extend(
            '                  ("{}", {}),'.format(name, value)
            for name, value in zip(s_names, s_init)
        )
        body_lines[-1] = body_lines[-1][0:-1] + "])"

        return "\n".join(self.indent_and_split_lines(body_lines, 2))


def gotran2beat(filename, params):
    """
    Create a beat cell model from a gotran model
    """

    # Load Gotran model
    ode = load_ode(filename)

    # Create a Beat Cell model code generator
    cell_gen = CellModelGenerator(ode, params.membrane_potential)

    output = params.output

    if output:
        if not (len(output) > 3 and output[-3:] == ".py"):
            output += ".py"
    else:
        output = filename.replace(".ode", "") + ".py"

    f = open(output, "w")

    f.write(cell_gen.generate())


def main():
    import sys
    import os
    from modelparameters.parameterdict import ParameterDict, Param

    params = ParameterDict(
        output=Param("", description="Specify the basename of the output file"),
        membrane_potential=Param(
            "V", description="The name of the " "membrane potential state."
        ),
    )
    params.parse_args(usage="usage: %prog FILE [options]")  # sys.argv[2:])

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2beat(file_name, params)


if __name__ == "__main__":
    raise SystemExit(main())
