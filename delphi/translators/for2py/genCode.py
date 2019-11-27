import ast
import sys
import re
from . import For2PyError


class PrintState:
    def __init__(self, sep=None, add=None):
        self.sep = sep if sep is not None else "\n"
        self.add = add if add is not None else "    "

    def copy(self, sep=None, add=None):
        return PrintState(
            self.sep if sep is None else sep, self.add if add is None else add
        )


class genCode:
    def __init__(self,
                 lambda_string=""
                 ):
        self.lambda_string = lambda_string

        self.process_lambda_node = {
            "ast.FunctionDef": self.process_function_definition,
            "ast.arguments": self.process_arguments,
            "ast.arg": self._process_arg,
            "ast.Load": self._process_load,
            "ast.Store": self._process_store,
            "ast.Index": self.process_index,
            "ast.Num": self._process_num,
            "ast.List": self.process_list_ast,
            "ast.Str": self._process_str,
            "ast.For": self.process_for,
            "ast.If": self.process_if,
            "ast.UnaryOp": self.process_unary_operation,
            "ast.BinOp": self.process_binary_operation,
            "ast.Add": self._process_add,
            "ast.Sub": self._process_subtract,
            "ast.Mult": self._process_multiply,
            "ast.Pow": self._process_power,
            "ast.Div": self._process_divide,
            "ast.USub": self._process_unary_subtract,
            "ast.Eq": self._process_equals_to,
            "ast.NotEq": self._process_not_equal_to,
            "ast.Not": self._process_not,
            "ast.LtE": self._process_less_than_or_equal_to,
            "ast.Lt": self._process_less_than,
            "ast.Gt": self._process_greater_than,
            "ast.GtE": self._process_greater_than_or_equal_to,
            "ast.And": self._process_and,
            "ast.Or": self._process_or,
            "ast.Expr": self.process_expression,
            "ast.Compare": self.process_compare,
            "ast.Subscript": self.process_subscript,
            "ast.Name": self._process_name,
            "ast.AnnAssign": self.process_annotated_assign,
            "ast.Assign": self.process_assign,
            "ast.Call": self.process_call,
            "ast.Import": self.process_import,
            "ast.alias": self._process_alias,
            "ast.Module": self.process_module,
            "ast.BoolOp": self.process_boolean_operation,
            "ast.Attribute": self._process_attribute,
            "ast.AST": self.process_ast,
            "ast.Tuple": self.process_tuple,
            "ast.NameConstant": self.process_name_constant,
        }

    def generate_code(self, node, state):
        """
            This function parses the ast node of the python file and generates
            python code relevant to the information in the ast. This is used as
            the statements inside the lambda functions.
        """
        node_name = node.__repr__().split()[0][2:]

        if isinstance(node, list):
            for cur in node:
                self.lambda_string += f"{self.generate_code(cur,state)}" \
                                      f"{state.sep}"
        elif self.process_lambda_node.get(node_name):
            self.lambda_string = self.process_lambda_node[node_name](node,
                                                                     state)
        else:
            sys.stderr.write(
                "No handler for {0} in genCode, value: {1}\n".format(
                    node.__class__.__name__, str(node)
                )
            )

        return self.lambda_string

    def process_function_definition(self, node, state):
        code_string = "def {0}({1}):{2}{3}".format(
            node.name,
            self.generate_code(node.args, state),
            state.sep + state.add,
            self.generate_code(node.body, state.copy(state.sep + state.add)),
        )
        return code_string

    def process_arguments(self, node, state):
        code_string = ", ".join([self.generate_code(arg, state) for arg in
                                 node.args])
        return code_string

    @staticmethod
    def _process_arg(node, *_):
        code_string = node.arg
        return code_string

    @staticmethod
    def _process_load(*_):
        sys.stderr.write("genCode found ast.Load, is there a bug?\n")

    @staticmethod
    def _process_store(*_):
        sys.stderr.write("genCode found ast.Store, is there a bug?\n")

    def process_index(self, node, state):
        code_string = "[{0}]".format(self.generate_code(node.value, state))
        return code_string

    def process_name_constant(self, node, state):
        code_string = str(node.value)
        return code_string

    @staticmethod
    def _process_num(node, *_):
        code_string = str(node.n)
        return code_string

    def process_list_ast(self, node, state):
        elements = [self.generate_code(elmt, state) for elmt in node.elts]
        code_string = (
            str(elements[0])
            if len(elements) == 1
            else "[{0}]".format(", ".join(elements))
        )
        return code_string

    def process_tuple(self, node, state):
        elements = [self.generate_code(elmt, state) for elmt in node.elts]
        # This tuple handler is a very specific method
        # for handling an array declaration lambda.
        code_string = "[0] * ("
        if len(elements) == 1:
            code_string += str(elements[0])
        else:
            low_bound = None
            # Calculate the size of each dimension
            for elem in elements:
                if low_bound == None:
                    low_bound = elem
                else:
                    code_string += f"{elem} - {low_bound}"
                    low_bound = None
        code_string += ")"
        return code_string


    @staticmethod
    def _process_str(node, *_):
        code_string = '"{0}"'.format(node.s)
        return code_string

    def process_for(self, node, state):
        code_string = "for {0} in {1}:{2}{3}".format(
            self.generate_code(node.target, state),
            self.generate_code(node.iter, state),
            state.sep + state.add,
            self.generate_code(node.body, state.copy(state.sep + state.add)),
        )
        return code_string

    def process_if(self, node, state):
        code_string = "if ({0}):{1}{2}{3}else:{4}{5}".format(
            self.generate_code(node.test, state),
            state.sep + state.add,
            self.generate_code(node.body, state.copy(state.sep + state.add)),
            state.sep,
            state.sep + state.add,
            self.generate_code(node.orelse, state.copy(state.sep + state.add)),
        )
        return code_string

    def process_unary_operation(self, node, state):
        code_string = "{0}({1})".format(
            self.generate_code(node.op, state),
            self.generate_code(node.operand, state)
        )
        return code_string

    def process_binary_operation(self, node, state):
        code_string = "({0}{1}{2})".format(
            self.generate_code(node.left, state),
            self.generate_code(node.op, state),
            self.generate_code(node.right, state),
        )
        return code_string

    @staticmethod
    def _process_add(*_):
        code_string = "+"
        return code_string

    @staticmethod
    def _process_subtract(*_):
        code_string = "-"
        return code_string

    @staticmethod
    def _process_multiply(*_):
        code_string = "*"
        return code_string

    @staticmethod
    def _process_power(*_):
        code_string = "**"
        return code_string

    @staticmethod
    def _process_divide(*_):
        code_string = "/"
        return code_string

    @staticmethod
    def _process_unary_subtract(*_):
        code_string = "-"
        return code_string

    @staticmethod
    def _process_equals_to(*_):
        code_string = "=="
        return code_string

    @staticmethod
    def _process_not_equal_to(*_):
        code_string = "!="
        return code_string

    @staticmethod
    def _process_not(*_):
        code_string = "not"
        return code_string

    @staticmethod
    def _process_less_than_or_equal_to(*_):
        code_string = "<="
        return code_string

    @staticmethod
    def _process_less_than(*_):
        code_string = "<"
        return code_string

    @staticmethod
    def _process_greater_than(*_):
        code_string = ">"
        return code_string

    @staticmethod
    def _process_greater_than_or_equal_to(*_):
        code_string = ">="
        return code_string

    @staticmethod
    def _process_and(*_):
        code_string = "and"
        return code_string

    @staticmethod
    def _process_or(*_):
        code_string = "or"
        return code_string

    def process_expression(self, node, state):
        code_string = self.generate_code(node.value, state)
        return code_string

    def process_compare(self, node, state):
        if len(node.ops) > 1:
            sys.stderr.write(
                "Fix Compare in genCode! Don't have an example of what this "
                "will look like\n"
            )
        else:
            code_string = "({0} {1} {2})".format(
                self.generate_code(node.left, state),
                self.generate_code(node.ops[0], state),
                self.generate_code(node.comparators[0], state),
            )
            return code_string

    def process_subscript(self, node, state):
        # typical:
        # lambda_string = '{0}{1}'.format(genCode(node.value, state), genCode(
        # node.slice,
        # state))
        code_string = self.generate_code(node.value, state)
        return code_string

    @staticmethod
    def _process_name(node, *_):
        code_string = node.id
        return code_string

    def process_annotated_assign(self, node, state):
        code_string = self.generate_code(node.value, state)
        return code_string

    def process_assign(self, node, state):
        code_string = self.generate_code(node.value, state)
        return code_string

    def process_call(self, node, state):
        if isinstance(node.func, ast.Attribute):
            function_node = node.func
            if not isinstance(function_node.value, ast.Attribute):
                module = function_node.value.id
            else:
                module = function_node.value.value.id
            function_name = function_node.attr
            function_name = module + "." + function_name
        else:
            function_name = node.func.id

        if function_name is not "Array":
            # Check for setter and getter functions to differentiate between
            # array and string operations
            if ".set_" in function_name and len(node.args) > 1 and \
                    function_name.split('.')[1] != "set_substr":
                # This is an Array operation
                # Remove the first argument of <.set_>
                # function of array as it is not needed
                del node.args[0]
                code_string = ""
                for arg in node.args:
                    code_string += self.generate_code(arg, state)
            elif ".get_" in function_name and \
                    function_name.split('.')[1] != "get_substr":
                # This is an Array operation
                code_string = function_name.replace(".get_", "[")
                for arg in node.args:
                    code_string += self.generate_code(arg, state)
                code_string += "]"
            else:
                # This is a String operation
                code_string = f"{function_name}("
                if len(node.args) > 0:
                    arg_list = []
                    arg_count = len(node.args)
                    for arg_index in range(arg_count):
                        arg_string = self.generate_code(node.args[arg_index],
                                                        state)
                        if ".f_index" in function_name and \
                                arg_count > 1 and arg_index == arg_count - 1:
                            arg_string = f"[{arg_string}]"
                        arg_list.append(arg_string)
                    code_string += ", ".join(arg_list)
                code_string += ")"

        else:
            code_string = self.generate_code(node.args[1], state)

        return code_string

    def process_import(self, node, state):
        code_string = "import {0}{1}".format(
            ", ".join([self.generate_code(name, state) for name in
                       node.names]), state.sep
        )
        return code_string

    @staticmethod
    def _process_alias(node, *_):
        if node.asname is None:
            code_string = node.name
        else:
            code_string = "{0} as {1}".format(node.name, node.asname)
        return code_string

    def process_module(self, node, state):
        code_string = self.generate_code(node.body, state)
        return code_string

    def process_boolean_operation(self, node, state):
        code_string = "({0} {1} {2})".format(
            self.generate_code(node.values[0], state),
            self.generate_code(node.op, state),
            self.generate_code(node.values[1], state))
        return code_string

    @staticmethod
    def _process_attribute(node, *_):
        # Code below will be kept until all tests pass and removed if they do
        # lambda_string = genCode(node.value, state)

        # This is a fix on `feature_save` branch to bypass the SAVE statement
        # feature where a SAVEd variable is referenced as
        # <function_name>.<variable_name>. So the code below only returns the
        # <variable_name> which is stored under `node.attr`.
        code_string = node.attr
        return code_string

    @staticmethod
    def process_ast(node, *_):
        sys.stderr.write(
            "No handler for AST.{0} in genCode, fields: {1}\n".format(
                node.__class__.__name__, node._fields
            )
        )
