#!/usr/bin/env python

import numpy as np
import os

try:
    import gotran
except Exception as e:
    print("Gotran not installed. Not possible to convert gotran model to cellmodel.")
    raise e

from modelparameters.parameterdict import ParameterDict

from gotran.codegeneration.codegenerators import PythonCodeGenerator

from gotran.model.ode import ODE
from gotran.model.odeobjects import Comment

from gotran.codegeneration.algorithmcomponents import *
from gotran.codegeneration.codecomponent import CodeComponent
from gotran.common import check_arg, check_kwarg, error
from gotran.common.options import parameters

__all__ = ["DOLFINCodeGenerator"]

class DOLFINCodeGenerator(PythonCodeGenerator):
    """
    Class for generating a DOLFIN compatible declarations of an ODE from
    a gotran file 
    """

    @staticmethod
    def default_parameters():
        default_params = parameters.generation.code.copy()
        state_repr = dict.__getitem__(default_params.states, "representation")
        param_repr = dict.__getitem__(default_params.parameters, "representation")
        return ParameterDict(state_repr=state_repr,
                             param_repr=param_repr,
                             )

    def __init__(self, code_params=None):
        """
        Instantiate the class
        
        Arguments:
        ----------
        code_params : dict
            Parameters controling the code generation
        """
        code_params = code_params or {}
        check_kwarg(code_params, "code_params", dict)
        
        params = DOLFINCodeGenerator.default_parameters()
        params.update(code_params)
        
        # Get a whole set of gotran code parameters and update with dolfin
        # specific options
        generation_params = parameters.generation.copy()
        generation_params.code.default_arguments = \
                            "st" if params.param_repr == "numerals" else "stp"
        generation_params.code.time.name = "time"

        generation_params.code.array.index_format = "[]"
        generation_params.code.array.index_offset = 0

        generation_params.code.parameters.representation = params.param_repr

        generation_params.code.states.representation = params.state_repr
        generation_params.code.states.array_name = "states"

        generation_params.code.body.array_name = "body"
        generation_params.code.body.representation = "named"
        
        # Init base class
        super(DOLFINCodeGenerator, self).__init__(ns="ufl")

        # Store attributes (over load default PythonCode parameters)
        self.params = generation_params
        
        
    def _init_arguments(self, comp, default_arguments=None):

        check_arg(comp, CodeComponent)
        params = self.params.code
        default_arguments = default_arguments or params.default_arguments

        # Check if comp defines used_states if not use the root components
        # full_states attribute
        # FIXME: No need for full_states here...
        states = comp.root.full_states
        parameters = comp.root.parameters

        num_states = comp.root.num_full_states
        num_parameters = comp.root.num_parameters

        # Start building body
        body_lines = ["# Imports", "import ufl", "import dolfin"]

        if "s" in default_arguments and states:
            
            states_name = params.states.array_name
            body_lines.append("")
            body_lines.append("# Assign states")
            body_lines.append("assert(isinstance({0}, dolfin.Function))"\
                              .format(states_name))
            body_lines.append("assert(states.function_space().depth() == 1)")
            body_lines.append("assert(states.function_space().num_sub_spaces() "\
                              "== {0})".format(num_states))

            # Generate state assign code
            if params.states.representation == "named":
                
                body_lines.append(", ".join(state.name for state in states) + \
                                  " = dolfin.split({0})".format(states_name))
        
        # Add parameters code if not numerals
        if "p" in default_arguments and params.parameters.representation \
               in ["named", "array"]:

            parameters_name = params.parameters.array_name
            body_lines.append("")
            body_lines.append("# Assign parameters")
            body_lines.append("assert(isinstance({0}, (dolfin.Function, "\
                              "dolfin.Constant)))".format(parameters_name))
            body_lines.append("if isinstance({0}, dolfin.Function):".format(\
                parameters_name))
            if_closure = []
            if_closure.append("assert({0}.function_space().depth() == 1)"\
                              .format(parameters_name))
            if_closure.append("assert({0}.function_space().num_sub_spaces() "\
                              "== {1})".format(parameters_name, num_parameters))
            body_lines.append(if_closure)
            body_lines.append("else:")
            body_lines.append(["assert({0}.value_size() == {1})".format(\
                parameters_name, num_parameters)])

            # Generate parameters assign code
            if params.parameters.representation == "named":
                
                body_lines.append(", ".join(param.name for param in \
                parameters) + " = dolfin.split({0})".format(parameters_name))

        # If initilizing results
        if comp.results:
            body_lines.append("")
            body_lines.append("# Init return args")
            
        for result_name in comp.results:
            shape = comp.shapes[result_name]
            if len(shape) > 1:
                error("expected only result expression with rank 1")

            body_lines.append("{0} = [ufl.zero()]*{1}".format(\
                result_name, shape[0]))
            
        return body_lines

    def function_code(self, comp, indent=0, default_arguments=None, \
                      include_signature=True):

        default_arguments = default_arguments or \
                            self.params.code.default_arguments

        check_arg(comp, CodeComponent)
        check_kwarg(default_arguments, "default_arguments", str)
        check_kwarg(indent, "indent", int)
        
        body_lines = self._init_arguments(comp, default_arguments)
        
        # Iterate over any body needed to define the dy
        for expr in comp.body_expressions:
            if isinstance(expr, Comment):
                body_lines.append("")
                body_lines.append("# " + str(expr))
            else:
                body_lines.append(self.to_code(expr.expr, expr.name))

        if comp.results:
            body_lines.append("")
            body_lines.append("# Return results")
            body_lines.append("return {0}".format(", ".join(\
                ("{0}[0]" if comp.shapes[result_name][0] == 1 \
                 else "dolfin.as_vector({0})").format(result_name) \
                 for result_name in comp.results)))

        if include_signature:
            
            # Add function prototype
            body_lines = self.wrap_body_with_function_prototype(\
                body_lines, comp.function_name, \
                self.args(default_arguments), \
                comp.description, self.decorators())
        
        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def init_states_code(self, ode, indent=0):
        """
        Generate code for setting initial condition
        """

        # Start building body
        states = ode.full_states
        body_lines = ["# Imports", "import dolfin",\
                      "", "# Init values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            state.name, state.init) for state in states)))
        body_lines.append("init_values = [{0}]".format(", ".join(\
            "{0}".format(state.init) for state in states)))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        
        body_lines.append("")
        body_lines.append("inf = float(\"inf\")")
        body_lines.append("")
        body_lines.append("# State indices and limit checker")

        body_lines.append("state_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2}, {3!r})".format(\
                state.param.name, i, range_check.format(\
                    **state.param._range.range_formats), \
                state.param._range._not_in_str)\
                for i, state in enumerate(states))))
        body_lines.append("")

        body_lines.append("for state_name, value in values.items():")
        body_lines.append(\
            ["if state_name not in state_ind:",
             ["raise ValueError(\"{{0}} is not a state.\".format(state_name))"],
             "ind, range_check, not_in_format = state_ind[state_name]",
             "if not range_check(value):",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "state_name, not_in_format % str(value)))"],
             "", "# Assign value",
             "init_values[ind] = value"])
            
        body_lines.append("return dolfin.Constant(tuple(init_values))")
        
        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_state_values", "**values", "Init values")
        
        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_parameters_code(self, ode, indent=0):
        """
        Generate code for setting parameters
        """

        parameters = ode.parameters

        # Start building body
        body_lines = ["# Imports", "import dolfin",\
                      "", "# Param values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            param.name, param.init) for param in parameters)))
        body_lines.append("param_values = [{0}]"\
                          .format(", ".join("{0}".format(param.init) \
                                            for param in parameters)))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# Parameter indices and limit checker")

        body_lines.append("parameter_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2}, {3!r})".format(\
                parameter.param.name, i, range_check.format(\
                    **parameter.param._range.range_formats), \
                parameter.param._range._not_in_str)\
                for i, parameter in enumerate(parameters))))
        body_lines.append("")

        body_lines.append("for param_name, value in values.items():")
        body_lines.append(\
            ["if param_name not in parameter_ind:",
             ["raise ValueError(\"{{0}} is not a parameter.\".format(param_name))"],
             "ind, range_check, not_in_format = parameter_ind[param_name]",
             "if not range_check(value):",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "param_name, not_in_format % str(value)))"],
             "", "# Assign value",
             "init_values[ind] = value"])
            
        body_lines.append("return dolfin.Constant(tuple(param_values))")
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "init_parameter_values", \
            "**values", "Parameter values")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))
           

