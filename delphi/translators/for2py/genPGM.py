#!/usr/bin/python3.6

import ast
import sys
import tokenize
from datetime import datetime
import re
import argparse
from functools import reduce
import json
from delphi.translators.for2py.genCode import genCode, PrintState
from delphi.translators.for2py.mod_index_generator import get_index
from delphi.translators.for2py import For2PyError
from typing import List, Dict, Iterable, Optional
from itertools import chain, product
import operator
import os
import uuid


BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Eq: operator.eq,
    ast.LtE: operator.le,
}


UNNECESSARY_TYPES = (
    ast.Mult,
    ast.Add,
    ast.Sub,
    ast.Pow,
    ast.Div,
    ast.USub,
    ast.Eq,
    ast.LtE,
)


class GrFNGenerator(object):
    def __init__(self, annassigned_list=[], elif_pgm=[], function_defs=[]):
        self.annassigned_list = annassigned_list
        self.elif_pgm = elif_pgm
        self.function_defs = function_defs
        self.exclude_list = []
        self.mode_mapper = {}
        self.alias_dict = {}

    def genPgm(self, node, state, fnNames, call_source):
        types = (list, ast.Module, ast.FunctionDef)

        if state.fnName is None and not any(
            isinstance(node, t) for t in types
        ):
            if isinstance(node, ast.Call):
                return [{"start": node.func.id}]
            elif isinstance(node, ast.Expr):
                return self.genPgm(node.value, state, fnNames, "start")
            elif isinstance(node, ast.If):
                return self.genPgm(node.body, state, fnNames, "start")
            else:
                return []

        if isinstance(node, list):
            return list(
                chain.from_iterable(
                    [
                        self.genPgm(cur, state, fnNames, call_source)
                        for cur in node
                    ]
                )
            )

        # Function: name, args, body, decorator_list, returns
        elif isinstance(node, ast.FunctionDef):

            # List out all the function definitions in the ast
            self.function_defs.append(node.name)

            localDefs = state.lastDefs.copy()
            localNext = state.nextDefs.copy()
            localTypes = state.varTypes.copy()
            scope_path = (
                state.scope_path.copy()
            )  # Tracks the scope of the identifier
            if len(scope_path) == 0:
                scope_path.append("_TOP")
            scope_path.append(node.name)
            fnState = state.copy(
                lastDefs=localDefs,
                nextDefs=localNext,
                fnName=node.name,
                varTypes=localTypes,
            )

            args = self.genPgm(node.args, fnState, fnNames, "functiondef")
            bodyPgm = self.genPgm(node.body, fnState, fnNames, "functiondef")

            body, fns, iden_spec = get_body_and_functions(bodyPgm)

            variables = list(localDefs.keys())
            variables_tmp = []

            for item in variables:
                match = re.match(
                    r"(format_\d+_obj)|(file_\d+)|(write_list_\d+)|(write_line)",
                    item,
                )
                if not match:
                    variables_tmp.append(item)

            variables = variables_tmp

            fnDef = {
                "name": node.name,
                "type": "container",
                "input": [
                    {"name": arg, "domain": localTypes[arg]} for arg in args
                ],
                "variables": [
                    {"name": var, "domain": localTypes[var]}
                    for var in variables
                ],
                "body": body,
            }

            fns.append(fnDef)

            pgm = {"functions": fns, "identifiers": iden_spec}

            return [pgm]

        # arguments: ('args', 'vararg', 'kwonlyargs', 'kw_defaults', 'kwarg',
        # 'defaults')
        elif isinstance(node, ast.arguments):
            return [
                self.genPgm(arg, state, fnNames, "arguments")
                for arg in node.args
            ]

        # arg: ('arg', 'annotation')
        elif isinstance(node, ast.arg):
            if node.annotation != None:
                state.varTypes[node.arg] = getVarType(node.annotation)
            if state.lastDefs.get(node.arg):
                state.lastDefs[node.arg] += 1
            else:
                state.lastDefs[node.arg] = 0
            return node.arg

        # Load: ()
        elif isinstance(node, ast.Load):
            raise For2PyError("Found ast.Load, which should not happen.")

        # Store: ()
        elif isinstance(node, ast.Store):
            raise For2PyError("Found ast.Store, which should not happen.")

        # Index: ('value',)
        elif isinstance(node, ast.Index):
            self.genPgm(node.value, state, fnNames, "index")

        # Num: ('n',)
        elif isinstance(node, ast.Num):
            return [
                {"type": "literal", "dtype": getDType(node.n), "value": node.n}
            ]

        # List: ('elts', 'ctx')
        elif isinstance(node, ast.List):
            elements = [
                element[0]
                for element in [
                    self.genPgm(elmt, state, fnNames, "List")
                    for elmt in node.elts
                ]
            ]
            return elements if len(elements) == 1 else [{"list": elements}]

        # Str: ('s',)
        elif isinstance(node, ast.Str):
            return [{"type": "literal", "dtype": "string", "value": node.s}]

        # For: ('target', 'iter', 'body', 'orelse')
        elif isinstance(node, ast.For):

            scope_path = state.scope_path.copy()
            if len(scope_path) == 0:
                scope_path.append("_TOP")
            scope_path.append("loop")

            state = state.copy(
                lastDefs=state.lastDefs.copy(),
                nextDefs=state.nextDefs.copy(),
                lastDefDefault=state.lastDefDefault,
                fnName=state.fnName,
                varTypes=state.varTypes.copy(),
                lambdaStrings=state.lambdaStrings,
                start=state.start.copy(),
                scope_path=scope_path,
            )

            if self.genPgm(node.orelse, state, fnNames, "for"):
                raise For2PyError("For/Else in for not supported.")

            indexVar = self.genPgm(node.target, state, fnNames, "for")
            if len(indexVar) != 1 or "var" not in indexVar[0]:
                raise For2PyError("Only one index variable is supported.")
            indexName = indexVar[0]["var"]["variable"]
            loopIter = self.genPgm(node.iter, state, fnNames, "for")
            if (
                len(loopIter) != 1
                or "call" not in loopIter[0]
                or loopIter[0]["call"]["function"] != "range"
            ):
                raise For2PyError("Can only iterate over a range.")

            rangeCall = loopIter[0]["call"]
            if (
                len(rangeCall["inputs"]) != 2
                or len(rangeCall["inputs"][0]) != 1
                or len(rangeCall["inputs"][1]) != 1
                or (
                    "type" in rangeCall["inputs"][0]
                    and rangeCall["inputs"][0]["type"] == "literal"
                )
                or (
                    "type" in rangeCall["inputs"][1]
                    and rangeCall["inputs"][1]["type"] == "literal"
                )
            ):
                raise For2PyError("Can only iterate over a constant range.")

            if len(rangeCall["inputs"]) == 2:
                iterationRange = {
                    "start": rangeCall["inputs"][0][0],
                    "end": rangeCall["inputs"][1][0],
                }
            elif len(rangeCall["inputs"]) == 3:
                iterationRange = {
                    "start": rangeCall["inputs"][0][0],
                    "end": rangeCall["inputs"][1][0],
                    "step": rangeCall["inputs"][2][0],
                }

            loopLastDef = {}
            loopState = state.copy(
                lastDefs=loopLastDef, nextDefs={}, lastDefDefault=-1
            )
            loop = self.genPgm(node.body, loopState, fnNames, "for")
            loopBody, loopFns, iden_spec = get_body_and_functions(loop)

            variables = [x for x in loopLastDef if x != indexName]

            # variables: see what changes?
            loopName = getFnName(
                fnNames, f"{state.fnName}__loop_plate__{indexName}", {}
            )

            loopFn = {
                "name": loopName,
                "type": "loop_plate",
                "input": [
                    {
                      "name": variable,
                      "domain": state.varTypes[variable]
                    }
                    for variable in variables
                ],
                "index_variable": indexName,
                "index_iteration_range": iterationRange,
                "body": loopBody,
            }

            id_specList = self.make_identifier_spec(
                loopName, indexName, {}, state
            )

            loopCall = {
                "name": loopName,
                "input": [
                    {
                      "name": variable,
                      "index": -1,
                    }
                    for variable in variables
                ],
                "output": {}
            }
            pgm = {
                "functions": loopFns + [loopFn],
                "body": [loopCall],
                "identifiers": [],
            }

            for id_spec in id_specList:
                pgm["identifiers"].append(id_spec)

            return [pgm]

        # If: ('test', 'body', 'orelse')
        elif isinstance(node, ast.If):

            scope_path = state.scope_path.copy()
            if len(scope_path) == 0:
                scope_path.append("_TOP")
            scope_path.append("if")

            state = state.copy(
                lastDefs=state.lastDefs.copy(),
                nextDefs=state.nextDefs.copy(),
                lastDefDefault=state.lastDefDefault,
                fnName=state.fnName,
                varTypes=state.varTypes.copy(),
                lambdaStrings=state.lambdaStrings,
                start=state.start.copy(),
                scope_path=scope_path,
            )

            if call_source == "if":
                pgm = {"functions": [], "body": [], "identifiers": []}

                condSrcs = self.genPgm(node.test, state, fnNames, "if")

                startDefs = state.lastDefs.copy()
                ifDefs = startDefs.copy()
                elseDefs = startDefs.copy()
                ifState = state.copy(lastDefs=ifDefs)
                elseState = state.copy(lastDefs=elseDefs)
                ifPgm = self.genPgm(node.body, ifState, fnNames, "if")
                elsePgm = self.genPgm(node.orelse, elseState, fnNames, "if")

                updatedDefs = [
                    var
                    for var in set(startDefs.keys())
                    .union(ifDefs.keys())
                    .union(elseDefs.keys())
                    if var not in startDefs
                    or ifDefs[var] != startDefs[var]
                    or elseDefs[var] != startDefs[var]
                ]

                pgm["functions"] += reduce(
                    (lambda x, y: x + y["functions"]), [[]] + ifPgm
                ) + reduce((lambda x, y: x + y["functions"]), [[]] + elsePgm)

                pgm["body"] += reduce(
                    (lambda x, y: x + y["body"]), [[]] + ifPgm
                ) + reduce((lambda x, y: x + y["body"]), [[]] + elsePgm)

                self.elif_pgm = [
                    pgm,
                    condSrcs,
                    node.test,
                    node.lineno,
                    node,
                    updatedDefs,
                    ifDefs,
                ]

                return []

            pgm = {"functions": [], "body": [], "identifiers": []}

            condSrcs = self.genPgm(node.test, state, fnNames, "if")

            condNum = state.nextDefs.get("#cond", state.lastDefDefault + 1)
            state.nextDefs["#cond"] = condNum + 1

            condName = f"IF_{condNum}"
            state.varTypes[condName] = "boolean"
            state.lastDefs[condName] = 0
            fnName = getFnName(
                fnNames, f"{state.fnName}__condition__{condName}", {}
            )
            condOutput = {"variable": condName, "index": 0}

            fn = {
                "name": fnName,
                "type": "condition",
                "target": condName,
                "reference": node.lineno,
                "sources": [
                    {"name": src["var"]["variable"], "type": "variable"}
                    for src in condSrcs
                    if "var" in src
                ],
            }

            id_specList = self.make_identifier_spec(
                fnName,
                condOutput,
                [src["var"] for src in condSrcs if "var" in src],
                state,
            )

            for id_spec in id_specList:
                pgm["identifiers"].append(id_spec)

            body = {
                "name": fnName,
                "output": condOutput,
                "input": [src["var"] for src in condSrcs if "var" in src],
            }
            pgm["functions"].append(fn)
            pgm["body"].append(body)
            lambda_string = genFn(
                node.test,
                fnName,
                None,
                [src["var"]["variable"] for src in condSrcs if "var" in src],
            )
            state.lambdaStrings.append(lambda_string)

            startDefs = state.lastDefs.copy()
            ifDefs = startDefs.copy()
            elseDefs = startDefs.copy()
            ifState = state.copy(lastDefs=ifDefs)
            elseState = state.copy(lastDefs=elseDefs)
            ifPgm = self.genPgm(node.body, ifState, fnNames, "if")
            elsePgm = self.genPgm(node.orelse, elseState, fnNames, "if")

            pgm["functions"] += reduce(
                (lambda x, y: x + y["functions"]), [[]] + ifPgm
            ) + reduce((lambda x, y: x + y["functions"]), [[]] + elsePgm)

            pgm["body"] += reduce(
                (lambda x, y: x + y["body"]), [[]] + ifPgm
            ) + reduce((lambda x, y: x + y["body"]), [[]] + elsePgm)

            updatedDefs = [
                var
                for var in set(startDefs.keys())
                .union(ifDefs.keys())
                .union(elseDefs.keys())
                if var not in startDefs
                or ifDefs[var] != startDefs[var]
                or elseDefs[var] != startDefs[var]
            ]

            defVersions = {
                key: [
                    version
                    for version in [
                        startDefs.get(key),
                        ifDefs.get(key),
                        elseDefs.get(key),
                    ]
                    if version is not None
                ]
                for key in updatedDefs
            }

            for updatedDef in defVersions:
                versions = defVersions[updatedDef]
                inputs = (
                    [
                        condOutput,
                        {"variable": updatedDef, "index": versions[-1]},
                        {"variable": updatedDef, "index": versions[-2]},
                    ]
                    if len(versions) > 1
                    else [
                        condOutput,
                        {"variable": updatedDef, "index": versions[0]},
                    ]
                )

                output = {
                    "variable": updatedDef,
                    "index": getNextDef(
                        updatedDef,
                        state.lastDefs,
                        state.nextDefs,
                        state.lastDefDefault,
                    ),
                }
                fnName = getFnName(
                    fnNames, f"{state.fnName}__decision__{updatedDef}", output
                )
                fn = {
                    "name": fnName,
                    "type": "decision",
                    "target": updatedDef,
                    "reference": node.lineno,
                    "sources": [
                        {
                            "name": f"{var['variable']}_{var['index']}",
                            "type": "variable",
                        }
                        for var in inputs
                    ],
                }

                # Check for buggy __decision__ tag containing of only IF_ blocks
                # More information required on how __decision__ tags are made
                # This seems to be in development phase and documentation is
                # missing from the GrFN spec as well. Actual removal (or not)
                # of this tag depends on further information about this

                if "IF_" in updatedDef:
                    count = 0
                    for var in inputs:
                        if "IF_" in var["variable"]:
                            count += 1
                    if count == len(inputs):
                        continue

                body = {"name": fnName, "output": output, "input": inputs}

                id_specList = self.make_identifier_spec(
                    fnName, output, inputs, state
                )

                for id_spec in id_specList:
                    pgm["identifiers"].append(id_spec)

                lambda_string = genFn(
                    node,
                    fnName,
                    updatedDef,
                    [f"{src['variable']}_{src['index']}" for src in inputs],
                )
                state.lambdaStrings.append(lambda_string)

                pgm["functions"].append(fn)
                pgm["body"].append(body)

                # Previous ELIF Block is filled??
                if len(self.elif_pgm) > 0:

                    condSrcs = self.elif_pgm[1]

                    for item in self.elif_pgm[0]["functions"]:
                        pgm["functions"].append(item)

                    for item in self.elif_pgm[0]["body"]:
                        pgm["body"].append(item)

                    condNum = state.nextDefs.get(
                        "#cond", state.lastDefDefault + 1
                    )
                    state.nextDefs["#cond"] = condNum + 1

                    condName = f"IF_{condNum}"
                    state.varTypes[condName] = "boolean"
                    state.lastDefs[condName] = 0
                    fnName = getFnName(
                        fnNames, f"{state.fnName}__condition__{condName}", {}
                    )
                    condOutput = {"variable": condName, "index": 0}

                    fn = {
                        "name": fnName,
                        "type": "condition",
                        "target": condName,
                        "reference": self.elif_pgm[3],
                        "sources": [
                            {
                                "name": src["var"]["variable"],
                                "type": "variable",
                            }
                            for src in condSrcs
                            if "var" in src
                        ],
                    }

                    id_specList = self.make_identifier_spec(
                        fnName,
                        condOutput,
                        [src["var"] for src in condSrcs if "var" in src],
                        state,
                    )

                    for id_spec in id_specList:
                        pgm["identifiers"].append(id_spec)

                    body = {
                        "name": fnName,
                        "output": condOutput,
                        "input": [
                            src["var"] for src in condSrcs if "var" in src
                        ],
                    }
                    pgm["functions"].append(fn)
                    pgm["body"].append(body)

                    lambda_string = genFn(
                        self.elif_pgm[2],
                        fnName,
                        None,
                        [
                            src["var"]["variable"]
                            for src in condSrcs
                            if "var" in src
                        ],
                    )
                    state.lambdaStrings.append(lambda_string)

                    startDefs = state.lastDefs.copy()
                    ifDefs = self.elif_pgm[6]
                    elseDefs = startDefs.copy()

                    updatedDefs = self.elif_pgm[5]

                    defVersions = {
                        key: [
                            version
                            for version in [
                                startDefs.get(key),
                                ifDefs.get(key),
                                elseDefs.get(key),
                            ]
                            if version is not None
                        ]
                        for key in updatedDefs
                    }

                    for updatedDef in defVersions:
                        versions = defVersions[updatedDef]
                        inputs = (
                            [
                                condOutput,
                                {
                                    "variable": updatedDef,
                                    "index": versions[-1],
                                },
                                {
                                    "variable": updatedDef,
                                    "index": versions[-2],
                                },
                            ]
                            if len(versions) > 1
                            else [
                                condOutput,
                                {"variable": updatedDef, "index": versions[0]},
                            ]
                        )

                        output = {
                            "variable": updatedDef,
                            "index": getNextDef(
                                updatedDef,
                                state.lastDefs,
                                state.nextDefs,
                                state.lastDefDefault,
                            ),
                        }
                        fnName = getFnName(
                            fnNames,
                            f"{state.fnName}__decision__{updatedDef}",
                            output,
                        )
                        fn = {
                            "name": fnName,
                            "type": "decision",
                            "target": updatedDef,
                            "reference": self.elif_pgm[3],
                            "sources": [
                                {
                                    "name": f"{var['variable']}_{var['index']}",
                                    "type": "variable",
                                }
                                for var in inputs
                            ],
                        }

                        # Check for buggy __decision__ tag containing of only IF_ blocks
                        # More information required on how __decision__ tags are made
                        # This seems to be in development phase and documentation is
                        # missing from the GrFN spec as well. Actual removal (or not)
                        # of this tag depends on further information about this

                        if "IF_" in updatedDef:
                            count = 0
                            for var in inputs:
                                if "IF_" in var["variable"]:
                                    count += 1
                            if count == len(inputs):
                                continue

                        body = {
                            "name": fnName,
                            "output": output,
                            "input": inputs,
                        }

                        id_specList = self.make_identifier_spec(
                            fnName, output, inputs, state
                        )

                        for id_spec in id_specList:
                            pgm["identifiers"].append(id_spec)

                        lambda_string = genFn(
                            self.elif_pgm[4],
                            fnName,
                            updatedDef,
                            [
                                f"{src['variable']}_{src['index']}"
                                for src in inputs
                            ],
                        )
                        state.lambdaStrings.append(lambda_string)

                        pgm["functions"].append(fn)
                        pgm["body"].append(body)

                    self.elif_pgm = []

            return [pgm]

        # UnaryOp: ('op', 'operand')
        elif isinstance(node, ast.UnaryOp):
            return self.genPgm(node.operand, state, fnNames, "unaryop")

        # BinOp: ('left', 'op', 'right')
        elif isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Num) and isinstance(
                node.right, ast.Num
            ):
                for op in BINOPS:
                    if isinstance(node.op, op):
                        val = BINOPS[type(node.op)](node.left.n, node.right.n)
                        return [
                            {
                                "value": val,
                                "dtype": getDType(val),
                                "type": "literal",
                            }
                        ]

            return self.genPgm(
                node.left, state, fnNames, "binop"
            ) + self.genPgm(node.right, state, fnNames, "binop")

        # Mult: ()

        elif any(isinstance(node, nodetype) for nodetype in UNNECESSARY_TYPES):
            t = node.__repr__().split()[0][2:]
            sys.stdout.write(f"Found {t}, which should be unnecessary\n")

        # Expr: ('value',)
        elif isinstance(node, ast.Expr):
            exprs = self.genPgm(node.value, state, fnNames, "expr")
            pgm = {"functions": [], "body": [], "identifiers": []}
            for expr in exprs:
                if "call" in expr:
                    call = expr["call"]
                    body = {
                        "function": call["function"],
                        "output": {},
                        "input": [],
                    }
                    if re.match(r"file_\d+\.write", body["function"]):
                        return []
                    for arg in call["inputs"]:
                        if len(arg) == 1:
                            if "var" in arg[0]:
                                body["input"].append(arg[0]["var"])
                        else:
                            raise For2PyError(
                                "Only 1 input per argument supported right now."
                            )
                    pgm["body"].append(body)
                else:
                    raise For2PyError(f"Unsupported expr: {expr}.")
            return [pgm]

        # Compare: ('left', 'ops', 'comparators')
        elif isinstance(node, ast.Compare):
            return self.genPgm(
                node.left, state, fnNames, "compare"
            ) + self.genPgm(node.comparators, state, fnNames, "compare")

        # Subscript: ('value', 'slice', 'ctx')
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.slice.value, ast.Num):
                raise For2PyError("can't handle arrays right now.")

            val = self.genPgm(node.value, state, fnNames, "subscript")

            if val[0]["var"]["variable"] in self.annassigned_list:
                if isinstance(node.ctx, ast.Store):
                    val[0]["var"]["index"] = getNextDef(
                        val[0]["var"]["variable"],
                        state.lastDefs,
                        state.nextDefs,
                        state.lastDefDefault,
                    )
            else:
                self.annassigned_list.append(val[0]["var"]["variable"])
            return val

        # Name: ('id', 'ctx')
        elif isinstance(node, ast.Name):
            lastDef = getLastDef(node.id, state.lastDefs, state.lastDefDefault)
            if (
                isinstance(node.ctx, ast.Store)
                and state.nextDefs.get(node.id)
                and call_source != "annassign"
            ):
                lastDef = getNextDef(
                    node.id,
                    state.lastDefs,
                    state.nextDefs,
                    state.lastDefDefault,
                )
            return [{"var": {"variable": node.id, "index": lastDef}}]

        # AnnAssign: ('target', 'annotation', 'value', 'simple')
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, ast.List):
                targets = self.genPgm(node.target, state, fnNames, "annassign")
                for target in targets:
                    state.varTypes[target["var"]["variable"]] = getVarType(
                        node.annotation
                    )
                    if target["var"]["variable"] not in self.annassigned_list:
                        self.annassigned_list.append(target["var"]["variable"])
                return []

            sources = self.genPgm(node.value, state, fnNames, "annassign")
            targets = self.genPgm(node.target, state, fnNames, "annassign")

            pgm = {"functions": [], "body": [], "identifiers": []}

            for target in targets:
                state.varTypes[target["var"]["variable"]] = getVarType(
                    node.annotation
                )
                name = getFnName(
                    fnNames,
                    f"{state.fnName}__assign__{target['var']['variable']}",
                    {},
                )
                fn = self.make_fn_dict(name, target, sources, node)
                body = self.make_body_dict(name, target, sources, state)

                if len(sources) > 0:
                    lambda_string = genFn(
                        node,
                        name,
                        target["var"]["variable"],
                        [
                            src["var"]["variable"]
                            for src in sources
                            if "var" in src
                        ],
                    )
                    state.lambdaStrings.append(lambda_string)

                # In the case of assignments of the form:    "ud: List[float]"
                # an assignment function will be created with an empty input list.
                # Also, the function dictionary will be empty. We do not want such
                # assignments in the GrFN so check for an empty <fn> dictionary and
                # return [] if found
                if len(fn) == 0:
                    return []
                if not fn["sources"] and len(sources) == 1:
                    fn["body"] = {
                        "type": "literal",
                        "dtype": sources[0]["dtype"],
                        "value": f"{sources[0]['value']}",
                    }

                for id_spec in body[1]:
                    pgm["identifiers"].append(id_spec)

                pgm["functions"].append(fn)
                pgm["body"].append(body[0])

            return [pgm]

        # Assign: ('targets', 'value')
        elif isinstance(node, ast.Assign):

            scope_path = state.scope_path.copy()
            if len(scope_path) == 0:
                scope_path.append("_TOP")
            state.scope_path = scope_path

            sources = self.genPgm(node.value, state, fnNames, "assign")
            targets = reduce(
                (lambda x, y: x.append(y)),
                [
                    self.genPgm(target, state, fnNames, "assign")
                    for target in node.targets
                ],
            )
            pgm = {"functions": [], "body": [], "identifiers": []}

            for target in targets:
                source_list = []
                if target.get("list"):
                    targets = ",".join(
                        [x["var"]["variable"] for x in target["list"]]
                    )
                    target = {"var": {"variable": targets, "index": 1}}

                # Check whether this is an alias assignment i.e. of the form y=x where y is now the alias of variable x
                self.check_alias(target, sources)

                name = getFnName(
                    fnNames,
                    f"{state.fnName}__assign__{target['var']['variable']}",
                    target,
                )
                # If the index is -1, change it to the index in the fnName. This is a hack right now.
                # if target["var"]["index"] < 0:
                #     target["var"]["index"] = name[-1]

                fn = self.make_fn_dict(name, target, sources, node)
                if len(fn) == 0:
                    return []
                body = self.make_body_dict(name, target, sources, state)
                for src in sources:
                    if "var" in src:
                        source_list.append(src["var"]["variable"])
                    elif "call" in src:
                        for ip in src["call"]["inputs"][0]:
                            if "var" in ip:
                                source_list.append(ip["var"]["variable"])

                lambda_string = genFn(
                    node, name, target["var"]["variable"], source_list
                )
                state.lambdaStrings.append(lambda_string)
                if not fn["sources"] and len(sources) == 1:
                    if sources[0].get("list"):
                        dtypes = set()
                        value = list()
                        for item in sources[0]["list"]:
                            dtypes.add(item["dtype"])
                            value.append(item["value"])
                        dtype = list(dtypes)
                    else:
                        dtype = sources[0]["dtype"]
                        value = f"{sources[0]['value']}"
                    fn["body"] = {
                        "type": "literal",
                        "dtype": dtype,
                        "value": value,
                    }

                for id_spec in body[1]:
                    pgm["identifiers"].append(id_spec)

                pgm["functions"].append(fn)
                pgm["body"].append(body[0])
            return [pgm]

        # Tuple: ('elts', 'ctx')
        elif isinstance(node, ast.Tuple):
            elements = []
            for element in [
                self.genPgm(elmt, state, fnNames, "ctx") for elmt in node.elts
            ]:
                elements.append(element[0])

            return elements if len(elements) == 1 else [{"list": elements}]

        # Call: ('func', 'args', 'keywords')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # Check if there is a <sys> call. Bypass it if exists.
                if isinstance(node.func.value, ast.Attribute):
                    if node.func.value.value.id == "sys":
                        return []
                fnNode = node.func
                module = fnNode.value.id
                fnName = fnNode.attr
                fnName = module + "." + fnName
            else:
                fnName = node.func.id
            inputs = []

            for arg in node.args:
                arg = self.genPgm(arg, state, fnNames, "call")
                inputs.append(arg)

            call = {"call": {"function": fnName, "inputs": inputs}}

            return [call]

        # Module: body
        elif isinstance(node, ast.Module):
            pgms = []
            for cur in node.body:
                pgm = self.genPgm(cur, state, fnNames, "module")
                pgms += pgm
            return [mergeDicts(pgms)]

        # BoolOp: body
        elif isinstance(node, ast.BoolOp):
            pgms = []
            boolOp = {ast.And: "and", ast.Or: "or"}

            for key in boolOp:
                if isinstance(node.op, key):
                    pgms.append([{"boolOp": boolOp[key]}])

            for item in node.values:
                pgms.append(self.genPgm(item, state, fnNames, "boolop"))

            return pgms

        elif isinstance(node, ast.AST):
            sys.stderr.write(
                f"No handler for AST.{node.__class__.__name__} in genPgm, "
                f"fields: {node._fields}\n"
            )

        else:
            sys.stderr.write(
                f"No handler for {node.__class__.__name__} in genPgm, "
                f"value: {str(node)}\n"
            )

        return []

    # This function checks whether an assignment is an alias created. An alias
    # is created when an assignment of the form y=x happens such that y is now
    # an alias of x because it is an exact copy of x. If it is an alias
    # assignment, the dictionary alias_dict will get populated.

    def check_alias(self, target, sources):
        target_index = (
            target["var"]["variable"] + "_" + str(target["var"]["index"])
        )
        if len(sources) == 1 and sources[0].get("var") != None:
            if self.alias_dict.get(target_index):
                self.alias_dict[target_index].append(
                    sources[0]["var"]["variable"]
                    + "_"
                    + str(sources[0]["var"]["index"])
                )
            else:
                self.alias_dict[target_index] = [
                    sources[0]["var"]["variable"]
                    + "_"
                    + str(sources[0]["var"]["index"])
                ]

    def make_iden_dict(self, name, targets, scope_path, holder):
        # Check for aliases
        if isinstance(targets, dict):
            aliases = self.alias_dict.get(
                targets["variable"] + "_" + str(targets["index"]), "None"
            )
        elif isinstance(targets, str):
            aliases = self.alias_dict.get(targets, "None")

        # First, check whether the information is from a variable or a holder(assign, loop, if, etc)
        # Assign the base_name accordingly

        if holder == "body":
            # If we are making the identifier specification of a body holder, the base_name will be the holder
            if isinstance(targets, dict):
                base_name = (
                    name
                    + "$"
                    + targets["variable"]
                    + "_"
                    + str(targets["index"])
                )
            elif isinstance(targets, str):
                base_name = name + "$" + targets
            gensyms_tag = "h"

        elif holder == "variable":
            # The base name will just be the name of the identifier
            base_name = targets
            gensyms_tag = "v"

        # The name space should get the entire directory scope of the fortran file under which it is defined.
        # For PETASCE.for, all modules are defined in the same fortran file so the namespace will be the same
        # for all identifiers

        # TODO handle multiple file namespaces that handle multiple fortran file namespacing

        # TODO Is the namespace path for the python intermediates or the original FORTRAN code? Currently, it captures
        #  the intermediate python file's path
        name_space = self.mode_mapper["FileName"][1].split("/")
        name_space = ".".join(name_space)

        # The scope captures the scope within the file where it exists. The context of modules can be implemented here.
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        elif scope_path[0] == "_TOP" and len(scope_path) > 1:
            scope_path.remove("_TOP")
        scope_path = ".".join(scope_path)

        # TODO Source code reference: This is the line number in the Python (or FORTRAN?) file. According to meeting on
        #  the 21st Feb, 2019, this was the same as namespace. Exactly same though? Need clarity.

        source_reference = name_space

        iden_dict = {
            "base_name": base_name,
            "scope": scope_path,
            "namespace": name_space,
            "aliases": aliases,
            "source_references": source_reference,
            "gensyms": generage_gensysm(gensyms_tag),
        }

        return iden_dict

    # Create the identifier specification for each identifier
    def make_identifier_spec(self, name, targets, sources, state):
        scope_path = state.scope_path
        for_id = 1
        if_id = 1
        identifier_list = []

        for item, scope in enumerate(scope_path):
            if scope == "loop":
                scope_path[item] = scope + "$" + str(for_id)
                for_id += 1
            elif scope == "if":
                scope_path[item] = scope + "$" + str(for_id)
                if_id += 1

        # Identify which kind of identifier it is
        name_regex = r"(?P<scope>\w+)__(?P<type>\w+)__(?P<basename>\w+)"
        match = re.match(name_regex, name)
        if match:
            if match.group("type") == "assign":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "condition":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "loop_plate":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "decision":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)

            return identifier_list

    def make_body_dict(self, name, target, sources, state):
        source_list = []
        for src in sources:
            if "var" in src:
                source_list.append(src["var"])
            if "call" in src:
                for ip in src["call"]["inputs"][0]:
                    if "var" in ip:
                        source_list.append(ip["var"])

        id_spec = self.make_identifier_spec(
            name, target["var"], source_list, state
        )

        body = {"name": name, "output": target["var"], "input": source_list}
        return [body, id_spec]

    def make_fn_dict(self, name, target, sources, node):
        source = []
        fn = {}

        # Regular expression to check for all targets that need to be bypassed. This is related to I/O handling
        bypass_regex = (
            r"^format_\d+$|^format_\d+_obj$|^file_\d+$|^write_list_\d+$|^write_line$|^format_\d+_obj"
            r".*|^Format$|^list_output_formats$|^write_list_steam$"
        )

        # Preprocessing and removing certain Assigns which only pertain to the
        # Python code and do not relate to the FORTRAN code in any way.
        bypass_match_target = re.match(bypass_regex, target["var"]["variable"])

        if bypass_match_target:
            self.exclude_list.append(target["var"]["variable"])
            return fn

        for src in sources:
            if "call" in src:
                # Bypassing identifiers who have I/O constructs on their source fields too.
                # Example: (i[0],) = format_10_obj.read_line(file_10.readline())
                # 'i' is bypassed here
                # TODO this is only for PETASCE02.for. Will need to include 'i' in the long run
                bypass_match_source = re.match(
                    bypass_regex, src["call"]["function"]
                )
                if bypass_match_source:
                    if "var" in src:
                        self.exclude_list.append(src["var"]["variable"])
                    self.exclude_list.append(target["var"]["variable"])
                    return fn
                for source_ins in make_call_body_dict(src):
                    source.append(source_ins)
            if "var" in src:
                variable = src["var"]["variable"]
                source.append({"name": variable, "type": "variable"})

            if re.match(r"\d+", target["var"]["variable"]) and "list" in src:
                # This is a write to a file
                return fn

            fn = {
                "name": name,
                "type": "assign",
                "target": target["var"]["variable"],
                "sources": source,
                "reference": node.lineno,
            }

        return fn


class PGMState:
    def __init__(
        self,
        lambdaStrings: Optional[List[str]],
        lastDefs: Optional[Dict] = {},
        nextDefs: Optional[Dict] = {},
        lastDefDefault=0,
        fnName=None,
        varTypes: Optional[Dict] = {},
        start: Optional[Dict] = {},
        scope_path: Optional[List] = [],
    ):
        self.lastDefs = lastDefs
        self.nextDefs = nextDefs
        self.lastDefDefault = lastDefDefault
        self.fnName = fnName
        self.varTypes = varTypes
        self.start = start
        self.scope_path = scope_path
        self.lambdaStrings = lambdaStrings

    def copy(
        self,
        lastDefs: Optional[Dict] = None,
        nextDefs: Optional[Dict] = None,
        lastDefDefault=None,
        fnName=None,
        varTypes: Optional[Dict] = None,
        start: Optional[Dict] = None,
        scope_path: Optional[List] = None,
        lambdaStrings: Optional[List[str]] = None,
    ):
        return PGMState(
            self.lambdaStrings if lambdaStrings is None else lambdaStrings,
            self.lastDefs if lastDefs == None else lastDefs,
            self.nextDefs if nextDefs == None else nextDefs,
            self.lastDefDefault if lastDefDefault == None else lastDefDefault,
            self.fnName if fnName == None else fnName,
            self.varTypes if varTypes == None else varTypes,
            self.start if start == None else start,
            self.scope_path if scope_path == None else scope_path,
        )


def dump(node, annotate_fields=True, include_attributes=False, indent="  "):
    """
    Return a formatted dump of the tree in *node*.  This is mainly useful for
    debugging purposes.  The returned string will show the names and the values
    for fields.  This makes the code impossible to evaluate, so if evaluation
    is wanted *annotate_fields* must be set to False.  Attributes such as line
    numbers and column offsets are not dumped by default. If this is wanted,
    *include_attributes* can be set to True.
    """

    def _format(node, level=0):
        if isinstance(node, ast.AST):
            fields = [(a, _format(b, level)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend(
                    [
                        (a, _format(getattr(node, a), level))
                        for a in node._attributes
                    ]
                )
            return "".join(
                [
                    node.__class__.__name__,
                    "(",
                    ", ".join(
                        ("%s=%s" % field for field in fields)
                        if annotate_fields
                        else (b for a, b in fields)
                    ),
                    ")",
                ]
            )
        elif isinstance(node, list):
            lines = ["["]
            lines.extend(
                (
                    indent * (level + 2) + _format(x, level + 2) + ","
                    for x in node
                )
            )
            if len(lines) > 1:
                lines.append(indent * (level + 1) + "]")
            else:
                lines[-1] += "]"
            return "\n".join(lines)
        return repr(node)

    if not isinstance(node, ast.AST):
        raise TypeError("expected AST, got %r" % node.__class__.__name__)
    return _format(node)


def printPgm(pgmFile, pgm):
    pgmFile.write(json.dumps(pgm, indent=2))


def genFn(node, fnName: str, returnVal: bool, inputs):
    lambda_strings = []
    lambda_strings.append(
        f"def {fnName}({', '.join(sorted(set(inputs), key=inputs.index))}):\n    "
    )
    # If a `decision` tag comes up, override the call to genCode to manually
    # enter the python script for the lambda file.
    if "__decision__" in fnName:
        code = f"{inputs[2]} if {inputs[0]} else {inputs[1]}"
    else:
        code = genCode(node, PrintState("\n    "))
    if returnVal:
        lambda_strings.append(f"return {code}")
    else:
        lines = code.split("\n")
        indent = re.search("[^ ]", lines[-1]).start()
        lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
        lambda_strings.append("\n".join(lines))
    lambda_strings.append("\n\n")
    return "".join(lambda_strings)


def mergeDicts(dicts: Iterable[Dict]) -> Dict:
    fields = set(chain.from_iterable(d.keys() for d in dicts))

    merged_dict = {field: [] for field in fields}
    for field, d in product(fields, dicts):
        if field in d:
            if isinstance(d[field], list):
                merged_dict[field] += d[field]
            else:
                merged_dict[field].append(d[field])

    return merged_dict


def getFnName(fnNames, basename, target):
    # First, check whether the basename is a 'decision' block. If it is, we need to get it's index from the index of
    # its corresponding identifier's 'assign' block. We do not use the index of the 'decision' block as that will not
    # correspond with that of the 'assign' block.
    # For example: for petpt__decision__albedo, its index will be the index of the latest petpt__assign__albedo + 1
    if "__decision__" in basename:
        part_match = re.match(
            r"(?P<body>\S+)__decision__(?P<identifier>\S+)", basename
        )
        if part_match:
            new_basename = (
                part_match.group("body")
                + "__assign__"
                + part_match.group("identifier")
            )
    else:
        new_basename = basename
    fnId = fnNames.get(new_basename, 0)
    if len(target) > 0:
        if target.get("var"):
            fnId = target["var"]["index"]
        elif target.get("variable"):
            fnId = target["index"]
    if fnId < 0:
        fnId = fnNames.get(new_basename, 0)
    fnName = f"{basename}_{fnId}"
    fnNames[basename] = fnId + 1
    return fnName


def getLastDef(var, lastDefs, lastDefDefault):
    index = lastDefDefault
    if var in lastDefs:
        index = lastDefs[var]
    else:
        lastDefs[var] = index
    return index


def getNextDef(var, lastDefs, nextDefs, lastDefDefault):
    index = nextDefs.get(var, lastDefDefault + 1)
    nextDefs[var] = index + 1
    lastDefs[var] = index
    return index


def getVarType(annNode):
    # wrapped in list
    if isinstance(annNode, ast.Subscript):
        dType = annNode.slice.value.id
    else:
        dType = annNode.id
    try:
        if dType == "float":
            return "real"
        if dType == "int":
            return "integer"
        if dType == "list":
            return "array"
        if dType == "str":
            return "string"
        if dType == "bool":
            return "bool"
        else:
            sys.stderr.write(
                "Unsupported type (only float, int, list, and str "
                "supported as of now).\n"
            )
    except AttributeError:
        raise For2PyError("Unsupported type (annNode is None).")


def getDType(val):
    if isinstance(val, int):
        dtype = "integer"
    elif isinstance(val, float):
        dtype = "real"
    elif isinstance(val, str):
        dtype = "string"
    else:
        raise For2PyError(f"num: {type(val)}.")
    return dtype


def get_body_and_functions(pgm):
    body = list(chain.from_iterable(stmt["body"] for stmt in pgm))
    fns = list(chain.from_iterable(stmt["functions"] for stmt in pgm))
    iden_spec = list(chain.from_iterable(stmt["identifiers"] for stmt in pgm))
    return body, fns, iden_spec


def generage_gensysm(tag):

    # The gensym is used to uniquely identify any identifier in the program. Python's uuid library is used to
    # generate a unique 12 digit HEX string. The uuid4() function of 'uuid' focuses on randomness. Each and every bit
    # of a UUID v4 is generated randomly and with no inherent logic. To every gensym, we add a tag signifying the
    # data type it represents. 'v' is for variables and 'h' is for holders.

    return uuid.uuid4().hex[:12] + "_" + tag


def make_call_body_dict(source):
    source_list = []
    name = source["call"]["function"]
    source_list.append({"name": name, "type": "function"})
    for ip in source["call"]["inputs"]:
        if isinstance(ip, list):
            for item in ip:
                if "var" in item:
                    variable = item["var"]["variable"]
                    source_list.append({"name": variable, "type": "variable"})
                if item.get("dtype") == "string":
                    source_list.append(
                        {"name": item["value"], "type": "variable"}
                    )
    return source_list


def importAst(filename: str):
    return ast.parse(tokenize.open(filename).read())


# Get the absolute path of the python files whose PGMs are being generated.
# TODO: For now the path is started from the directory "for2py" but need further discussion on this


def get_path(fileName: str, instance: str):
    absPath = os.path.abspath(fileName)
    if instance == "namespace":
        if re.match(r".*\/(for2py\/.*).py$", absPath):
            return (
                re.match(r".*\/(for2py\/.*).py$", absPath).group(1).split("/")
            )
        else:
            return fileName
    elif instance == "source":
        if re.match(r".*\/(for2py\/.*$)", absPath):
            return re.match(r".*\/(for2py\/.*$)", absPath).group(1).split("/")
        else:
            return fileName


def create_pgm_dict(
    lambdaFile: str,
    asts: List,
    file_name: str,
    mode_mapper_dict: dict,
    save_file=False,
) -> Dict:
    """ Create a Python dict representing the PGM, with additional metadata for
    JSON output. """

    lambdaStrings = ["import math\n\n"]
    state = PGMState(lambdaStrings)
    generator = GrFNGenerator()
    generator.mode_mapper = mode_mapper_dict
    pgm = generator.genPgm(asts, state, {}, "")[0]
    if pgm.get("start"):
        pgm["start"] = pgm["start"][0]
    else:
        pgm["start"] = generator.function_defs[-1]

    pgm["source"] = [[get_path(file_name, "source")]]

    # dateCreated stores the date and time on which the lambda and PGM file was created.
    # It is stored in YYYMMDD format
    pgm["dateCreated"] = f"{datetime.today().strftime('%Y%m%d')}"

    with open(lambdaFile, "w") as f:
        f.write("".join(lambdaStrings))

    # View the PGM file that will be used to build a scope tree
    if save_file:
        json.dump(pgm, open(file_name[:-3] + ".json", "w"))

    return pgm


def get_asts_from_files(files: List[str], printAst=False):
    asts = []
    for f in files:
        asts.append(importAst(f))
        if printAst:
            print(dump(asts[-1]))

    return asts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="A list of python files to generate a PGM for",
    )
    parser.add_argument(
        "-p",
        "--PGMFile",
        nargs=1,
        required=True,
        help="Filename for the output PGM",
    )
    parser.add_argument(
        "-l",
        "--lambdaFile",
        nargs=1,
        required=True,
        help="Filename for output lambda functions",
    )
    parser.add_argument(
        "-o",
        "--out",
        nargs=1,
        required=True,
        help="Text file containing the list of output python files being generated",
    )
    parser.add_argument(
        "-a",
        "--printAst",
        action="store_true",
        required=False,
        help="Print ASTs",
    )
    args = parser.parse_args(sys.argv[1:])

    with open(args.out[0], "r") as f:
        pythonFiles = f.read()

    pythonFileList = pythonFiles.rstrip().split(" ")

    asts = get_asts_from_files(pythonFileList, args.printAst)
    for index, inAst in enumerate(asts):
        # Read the mode_gen file containing all the identifier mappings
        mode_mapperDict = get_index(pythonFileList[index][:-3] + ".xml")

        lambdaFile = pythonFileList[index][:-3] + "_" + args.lambdaFile[0]
        pgmFile = pythonFileList[index][:-3] + "_" + args.PGMFile[0]
        pgm_dict = create_pgm_dict(
            lambdaFile, [inAst], pythonFileList[index], mode_mapperDict
        )

        with open(pgmFile, "w") as f:
            printPgm(f, pgm_dict)
