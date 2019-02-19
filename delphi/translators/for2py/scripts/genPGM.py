#!/usr/bin/python3.6

import ast
import sys
import tokenize
from datetime import datetime
import re
import argparse
from functools import *
import json
from delphi.translators.for2py.scripts.genCode import *
from typing import List, Dict, Iterable, Optional
from itertools import chain, product
import operator
import os

exclude_list = []

class PGMState:
    def __init__(
        self,
        lambdaFile: Optional[str],
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
        self.lambdaFile = lambdaFile
        self.start = start
        self.scope_path = scope_path

    def copy(
        self,
        lastDefs: Optional[Dict] = None,
        nextDefs: Optional[Dict] = None,
        lastDefDefault=None,
        fnName=None,
        varTypes: Optional[Dict] = None,
        lambdaFile: Optional[str] = None,
        start: Optional[Dict] = None,
        scope_path: Optional[List] = None,
    ):
        return PGMState(
            self.lambdaFile if lambdaFile == None else lambdaFile,
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
    for fields.  This makes the code impossible to evaluate, so if evaluation is
    wanted *annotate_fields* must be set to False.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
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


def genFn(fnFile, node, fnName, returnVal, inputs):
    fnFile.write(f"def {fnName}({', '.join(inputs)}):\n    ")
    code = genCode(node, PrintState("\n    "))
    if returnVal:
        fnFile.write(f"return {code}")
    else:
        lines = code.split("\n")
        indent = re.search("[^ ]", lines[-1]).start()
        lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
        fnFile.write("\n".join(lines))
    fnFile.write("\n\n")


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


def getFnName(fnNames, basename):
    fnId = fnNames.get(basename, 0)
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
    try:
        dType = annNode.slice.value.id
        if dType == "float":
            return "real"
        if dType == "int":
            return "integer"
        if dType == "list":
            return "array"
        if dType == "str":
            return "string"
        else:
            sys.stderr.write(
                "Unsupported type (only float and int supported as of now).\n"
            )
    except AttributeError:
        sys.stderr.write("Unsupported type (annNode is None).\n")
    sys.exit(1)


def getDType(val):
    if isinstance(val, int):
        dtype = "integer"
    elif isinstance(val, float):
        dtype = "real"
    elif isinstance(val, str):
        dtype = "string"
    else:
        sys.stderr.write(f"num: {type(node.n)}\n")
        sys.exit(1)
    return dtype


def get_body_and_functions(pgm):
    body = list(chain.from_iterable(stmt["body"] for stmt in pgm))
    fns = list(chain.from_iterable(stmt["functions"] for stmt in pgm))
    return body, fns


def make_fn_dict(name, target, sources, lambdaName, node):
    source = []
    fn = {}

    # Regular expression to check for all targets that need to be bypassed. This is related to I/O handling
    bypass_regex = r'^format_\d+$|^format_\d+_obj$|^file_\d+$|^write_list_\d+$|^write_line$|^format_\d+_obj.*|^Format$|^list_output_formats$|^write_list_steam$'

    # Preprocessing and removing certain Assigns which only pertain to the Python
    # code and do not relate to the FORTRAN code in any way.
    bypass_match_target = re.match(bypass_regex, target["var"]["variable"])

    if bypass_match_target:
        exclude_list.append(target["var"]["variable"])
        return fn
    for src in sources:
        if "call" in src:
            # Bypassing identifiers who have I/O constructs on their source fields too.
            # Example: (i[0],) = format_10_obj.read_line(file_10.readline())
            # 'i' is bypassed here
            # TODO this is only for PETASCE02.for. Will need to include 'i' in the long run
            bypass_match_source = re.match(bypass_regex, src["call"]["function"])
            if bypass_match_source:
                if "var" in src:
                    exclude_list.append(src["var"]["variable"])
                exclude_list.append(target["var"]["variable"])
                return fn
            for source_ins in make_call_body_dict(src):
                source.append(source_ins)
        if "var" in src:
            variable = src["var"]["variable"]
            source.append({"name": variable, "type": "variable"})

        if re.match(r"\d+", target["var"]["variable"]) and "list" in src:
            # This is a write to a file
            # Can delete from here
            # source.append({"name": "write", "type": "function"})
            # for item in src["list"]:
            #     variable = item["var"]["variable"]
            #     source.append({"name": variable, "type": "variable"})
            # To here
            return fn
        fn = {
            "name": name,
            "type": "assign",
            "target": target["var"]["variable"],
            "sources": source,
            "body": [
                {"type": "lambda", "name": lambdaName, "reference": node.lineno}
            ],
        }

    # # File Open Check
    # fn.update(
    #     {
    #         "name": name,
    #         "type": "assign",
    #         "sources": source,
    #         "body": [
    #             {
    #                 "type": "lambda",
    #                 "name": lambdaName,
    #                 "reference": node.lineno,
    #             }
    #         ],
    #     }
    # )
    # if len(source) > 0:
    #     if (
    #         source[0].get("name") == "open"
    #         and source[0].get("type") == "function"
    #     ):
    #         (file_id, source) = handle_file_open(
    #             target["var"]["variable"], source
    #         )
    #         fn["target"] = file_id
    #     else:
    #         fn["target"] = target["var"]["variable"]
    # else:
    #     fn["target"] = target["var"]["variable"]
    return fn


# def handle_file_open(target, source):
#     # This block maps the 'r' and 'w' modes in python file handling to read and write
#     # commands in the source field.
#     #
#     # Currently, the 'read' and 'write' actions are not included in source field but
#     # this function can handle it if necessary.
#     mode_mapping = {"r": "read", "w": "write"}
#     file_id = re.findall(r".*_(\d+)$", target)[0]
#     source[-1]["name"] = mode_mapping[source[-1]["name"]]
#
#     # Return with 'read'/'write' action. Disabled for now
#     # return (file_id, source)
#
#     # Return without the 'read'/'write' action.
#     return (file_id, source[:-1])


def make_call_body_dict(source):
    source_list = []
    # if re.match(r"format_\d+_obj\.read_line", source["call"]["function"]):
    #     source_list.append({"name": "read", "type": "function"})
    #     file_id_reg = r"file_(\d+)\.readline"
    #     id_string = source["call"]["inputs"][0][0]["call"]["function"]
    #     if re.match(file_id_reg, id_string):
    #         match = re.findall(file_id_reg, id_string)
    #         source_list.append({"name": match[0], "type": "variable"})
    #     return source_list

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


def make_body_dict(name, target, sources):
    source_list = []
    # file_read_index = 2

    # file_id_match = re.match(r"file_(\d+)$", target["var"]["variable"])
    for src in sources:
        if "var" in src:
            source_list.append(src["var"])
        if "call" in src:
            for ip in src["call"]["inputs"][0]:
                # if "call" in ip:
                #     read_match = re.match(
                #         r"file_(\d+)\.readline", ip["call"]["function"]
                #     )
                #     if read_match:
                #         source_list.append(
                #             {
                #                 "variable": read_match.group(1),
                #                 "index": file_read_index,
                #             }
                #         )
                #         file_read_index += 1
                if "var" in ip:
                    source_list.append(ip["var"])
        #     if file_id_match:
        #         target["var"]["variable"] = file_id_match.group(1)
        #         source_list = source_list[:-1]
        # if "list" in src and re.match(r"^(\d+)$", target["var"]["variable"]):
        #     for item in src["list"]:
        #         source_list.append(item)

    body = {"name": name, "output": target["var"], "input": source_list}
    return body


def genPgm(node, state, fnNames):
    types = (list, ast.Module, ast.FunctionDef)
    unnecessary_types = (
        ast.Mult,
        ast.Add,
        ast.Sub,
        ast.Pow,
        ast.Div,
        ast.USub,
        ast.Eq,
        ast.LtE,
    )
    # print (node)
    if state.fnName is None and not any(isinstance(node, t) for t in types):
        if isinstance(node, ast.Call):
            if state.start.get("start"):
                state.start["start"].append(node.func.id)
            else:
                state.start["start"] = [node.func.id]
            return [state.start]
        elif isinstance(node, ast.Expr):
            return genPgm(node.value, state, fnNames)
        elif isinstance(node, ast.If):
            return genPgm(node.body, state, fnNames)
        else:
            return []

    if isinstance(node, list):
        return list(
            chain.from_iterable([genPgm(cur, state, fnNames) for cur in node])
        )

    # Function: name, args, body, decorator_list, returns
    elif isinstance(node, ast.FunctionDef):
        localDefs = state.lastDefs.copy()
        localNext = state.nextDefs.copy()
        localTypes = state.varTypes.copy()
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append('_TOP')
        scope_path.append(node.name)
        fnState = state.copy(
            lastDefs=localDefs,
            nextDefs=localNext,
            fnName=node.name,
            varTypes=localTypes,
            scope_path=scope_path,
        )
        base_name = node.name

        args = genPgm(node.args, fnState, fnNames)
        bodyPgm = genPgm(node.body, fnState, fnNames)

        body, fns = get_body_and_functions(bodyPgm)

        variables = list(localDefs.keys())

        variables_tmp = []

        # Remove all the variables which are in the exclude list
        # Converting to set and back to list to remove any duplicate elements in exclude_list
        [variables.remove(item) for item in list(set(exclude_list))]

        # TODO this code section might be redundant. Need to verify
        for item in variables:
            match = re.match(
                r"(format_\d+_obj)|(file_\d+)|(write_list_\d+)|(write_line)|(format_\d+)|^(write_list_stream)$",
                item,
            )
            if not match:
                variables_tmp.append(item)

        variables = variables_tmp

        fnDef = {
            "name": node.name,
            "type": "container",
            "input": [
                {"name": arg, "domain": localTypes[arg[0]]} for arg in args
            ],
            "variables": [
                {"name": var, "domain": localTypes[var]} for var in variables
            ],
            "body": body,
        }

        fns.append(fnDef)
        pgm = {"functions": fns}

        return [pgm]

    # arguments: ('args', 'vararg', 'kwonlyargs', 'kw_defaults', 'kwarg', 'defaults')
    elif isinstance(node, ast.arguments):
        return [genPgm(arg, state, fnNames) for arg in node.args]

    # arg: ('arg', 'annotation')
    elif isinstance(node, ast.arg):
        state.varTypes[node.arg] = getVarType(node.annotation)
        base_name = node.arg
        id_spec = {
            "base_name": base_name,
            "scope": state.scope_path,
        }
        return (node.arg, id_spec)

    # Load: ()
    elif isinstance(node, ast.Load):
        sys.stderr.write("Found ast.Load, which should not happen\n")
        sys.exit(1)

    # Store: ()
    elif isinstance(node, ast.Store):
        sys.stderr.write("Found ast.Store, which should not happen\n")
        sys.exit(1)

    # Index: ('value',)
    elif isinstance(node, ast.Index):
        genPgm(node.value, state, fnNames)

    # Num: ('n',)
    elif isinstance(node, ast.Num):
        return [
            {"type": "literal", "dtype": getDType(node.n), "value": node.n}
        ]

    # List: ('elts', 'ctx')
    elif isinstance(node, ast.List):
        elements = [
            element[0]
            for element in [genPgm(elmt, state, fnNames) for elmt in node.elts]
        ]
        return elements if len(elements) == 1 else [{"list": elements}]

    # Str: ('s',)
    elif isinstance(node, ast.Str):
        return [{"type": "literal", "dtype": "string", "value": node.s}]

    # For: ('target', 'iter', 'body', 'orelse')
    elif isinstance(node, ast.For):
        if genPgm(node.orelse, state, fnNames):
            sys.stderr.write("For/Else in for not supported\n")
            sys.exit(1)

        indexVar = genPgm(node.target, state, fnNames)
        if len(indexVar) != 1 or "var" not in indexVar[0]:
            sys.stderr.write("Only one index variable is supported\n")
            sys.exit(1)
        indexName = indexVar[0]["var"]["variable"]

        loopIter = genPgm(node.iter, state, fnNames)
        if (
            len(loopIter) != 1
            or "call" not in loopIter[0]
            or loopIter[0]["call"]["function"] != "range"
        ):
            sys.stderr.write("Can only iterate over a range\n")
            sys.exit(1)

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
            sys.stderr.write("Can only iterate over a constant range\n")
            sys.exit(1)

        iterationRange = {
            "start": rangeCall["inputs"][0][0],
            "end": rangeCall["inputs"][1][0],
        }

        loopLastDef = {}
        loopState = state.copy(
            lastDefs=loopLastDef, nextDefs={}, lastDefDefault=-1
        )
        loop = genPgm(node.body, loopState, fnNames)
        loopBody, loopFns = get_body_and_functions(loop)

        variables = [x for x in loopLastDef if x != indexName]

        # variables: see what changes?
        loopName = getFnName(
            fnNames, f"{state.fnName}__loop_plate__{indexName}"
        )
        loopFn = {
            "name": loopName,
            "type": "loop_plate",
            "input": variables,
            "index_variable": indexName,
            "index_iteration_range": iterationRange,
            "body": loopBody,
        }

        loopCall = {"name": loopName, "inputs": variables, "output": {}}
        pgm = {"functions": loopFns + [loopFn], "body": [loopCall]}

        return [pgm]

    # If: ('test', 'body', 'orelse')
    elif isinstance(node, ast.If):
        pgm = {"functions": [], "body": []}

        condSrcs = genPgm(node.test, state, fnNames)

        condNum = state.nextDefs.get("#cond", state.lastDefDefault + 1)
        state.nextDefs["#cond"] = condNum + 1

        condName = f"IF_{condNum}"
        state.varTypes[condName] = "boolean"
        state.lastDefs[condName] = 0
        fnName = getFnName(fnNames, f"{state.fnName}__condition__{condName}")
        condOutput = {"variable": condName, "index": 0}
   
        lambdaName = getFnName(fnNames, f"{state.fnName}__lambda__{condName}")
        fn = {
            "name": fnName,
            "type": "condition",
            "target": condName,
            "sources": [
                {"name": src["var"]["variable"], "type": "variable"}
                for src in condSrcs
                if "var" in src
            ],
            "body": [
                {
                    "type": "lambda",
                    "name": lambdaName,
                    "reference": node.lineno,
                }
            ],
        }
        body = {
            "name": fnName,
            "output": condOutput,
            "input": [src["var"] for src in condSrcs if "var" in src],
        }
        pgm["functions"].append(fn)
        pgm["body"].append(body)
        genFn(
            state.lambdaFile,
            node.test,
            lambdaName,
            None,
            [src["var"]["variable"] for src in condSrcs if "var" in src],
        )

        startDefs = state.lastDefs.copy()
        ifDefs = startDefs.copy()
        elseDefs = startDefs.copy()
        ifState = state.copy(lastDefs=ifDefs)
        elseState = state.copy(lastDefs=elseDefs)
        ifPgm = genPgm(node.body, ifState, fnNames)
        elsePgm = genPgm(node.orelse, elseState, fnNames)


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
            name = "test1"
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
                fnNames, f"{state.fnName}__decision__{updatedDef}"
            )
            fn = {
                "name": fnName,
                "type": "decision",
                "target": updatedDef,
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

            if 'IF_' in updatedDef:
                count = 0
                for var in inputs:
                    if 'IF_' in var['variable']:
                        count += 1
                if count == len(inputs):
                    continue

            body = {"name": fnName, "output": output, "input": inputs}

            pgm["functions"].append(fn)
            pgm["body"].append(body)

        return [pgm]

    # UnaryOp: ('op', 'operand')
    elif isinstance(node, ast.UnaryOp):
        return genPgm(node.operand, state, fnNames)

    # BinOp: ('left', 'op', 'right')
    elif isinstance(node, ast.BinOp):
        binops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Eq: operator.eq,
            ast.LtE: operator.le,
        }
        if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
            for op in binops:
                if isinstance(node.op, op):
                    val = binops[type(node.op)](node.left.n, node.right.n)
                    return [
                        {
                            "value": val,
                            "dtype": getDType(val),
                            "type": "literal",
                        }
                    ]

        return genPgm(node.left, state, fnNames) + genPgm(
            node.right, state, fnNames
        )

    # Mult: ()

    elif any(isinstance(node, nodetype) for nodetype in unnecessary_types):
        t = node.__repr__().split()[0][2:]
        sys.stdout.write(f"Found {t}, which should be unnecessary\n")

    # Expr: ('value',)
    elif isinstance(node, ast.Expr):
        exprs = genPgm(node.value, state, fnNames)
        pgm = {"functions": [], "body": []}
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
                        sys.stderr.write(
                            "Only 1 input per argument supported right now\n"
                        )
                        sys.exit(1)
                pgm["body"].append(body)
            else:
                sys.stderr.write(f"Unsupported expr: {expr}\n")
                sys.exit(1)
        return [pgm]

    # Compare: ('left', 'ops', 'comparators')
    elif isinstance(node, ast.Compare):
        return genPgm(node.left, state, fnNames) + genPgm(
            node.comparators, state, fnNames
        )

    # Subscript: ('value', 'slice', 'ctx')
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.slice.value, ast.Num):
            sys.stderr.write("can't handle arrays right now\n")
            sys.exit(1)

        val = genPgm(node.value, state, fnNames)

        if isinstance(node.ctx, ast.Store):
            val[0]["var"]["index"] = getNextDef(
                val[0]["var"]["variable"],
                state.lastDefs,
                state.nextDefs,
                state.lastDefDefault,
            )

        return val

    # Name: ('id', 'ctx')
    elif isinstance(node, ast.Name):
        lastDef = getLastDef(node.id, state.lastDefs, state.lastDefDefault)
        if isinstance(node.ctx, ast.Store):
            lastDef = getNextDef(
                node.id, state.lastDefs, state.nextDefs, state.lastDefDefault
            )

        return [{"var": {"variable": node.id, "index": lastDef}}]

    # AnnAssign: ('target', 'annotation', 'value', 'simple')
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.value, ast.List):
            targets = genPgm(node.target, state, fnNames)
            for target in targets:
                state.varTypes[target["var"]["variable"]] = getVarType(
                    node.annotation
                )
            return []

        sources = genPgm(node.value, state, fnNames)
        targets = genPgm(node.target, state, fnNames)

        pgm = {"functions": [], "body": []}

        for target in targets:
            state.varTypes[target["var"]["variable"]] = getVarType(
                node.annotation
            )
            name = getFnName(
                fnNames, f"{state.fnName}__assign__{target['var']['variable']}"
            )
            lambdaName = getFnName(
                fnNames, f"{state.fnName}__lambda__{target['var']['variable']}"
            )
            fn = make_fn_dict(name, target, sources, lambdaName, node)
            body = make_body_dict(name, target, sources)

            genFn(
                state.lambdaFile,
                node,
                lambdaName,
                target["var"]["variable"],
                [src["var"]["variable"] for src in sources if "var" in src],
            )

            if not fn["sources"] and len(sources) == 1:
                fn["body"] = {
                    "type": "literal",
                    "dtype": sources[0]["dtype"],
                    "value": f"{sources[0]['value']}",
                }

            pgm["functions"].append(fn)
            pgm["body"].append(body)

        return [pgm]

    # Assign: ('targets', 'value')
    elif isinstance(node, ast.Assign):
        sources = genPgm(node.value, state, fnNames)
        targets = reduce(
            (lambda x, y: x.append(y)),
            [genPgm(target, state, fnNames) for target in node.targets],
        )
        pgm = {"functions": [], "body": []}

        for target in targets:
            source_list = []
            if target.get("list"):
                for var in target["list"]:
                    exclude_list.append(var["var"]["variable"])
                continue

            # # Extracting only file_id from the write list variable
            # match = re.match(r"write_list_(\d+)", target["var"]["variable"])
            # if match:
            #     target["var"]["variable"] = match.group(1)
            name = getFnName(
                fnNames, f"{state.fnName}__assign__{target['var']['variable']}"
            )
            lambdaName = getFnName(
                fnNames, f"{state.fnName}__lambda__{target['var']['variable']}"
            )
            fn = make_fn_dict(name, target, sources, lambdaName, node)

            if len(fn) == 0:
                return []
            body = make_body_dict(name, target, sources)
            for src in sources:
                if "var" in src:
                    source_list.append(src["var"]["variable"])
                elif "call" in src:
                    for ip in src["call"]["inputs"][0]:
                        if "var" in ip:
                            source_list.append(ip["var"]["variable"])

            genFn(
                state.lambdaFile,
                node,
                lambdaName,
                target["var"]["variable"],
                source_list,
            )

            # TODO This section might need to be modified depending on working of code
            # if need to be removed, remove inner if-else block and the dtype and value
            # will be the ones in the else block. Check history

            if not fn["sources"] and len(sources) == 1:
                if sources[0].get("list"):
                    return [pgm]
                else:
                    fn["body"] = {
                        "type": "literal",
                        "dtype": sources[0]["dtype"],
                        "value": f"{sources[0]['value']}",
                    }
            pgm["functions"].append(fn)
            pgm["body"].append(body)
        return [pgm]

    # Tuple: ('elts', 'ctx')
    elif isinstance(node, ast.Tuple):
        elements = []
        for element in [genPgm(elmt, state, fnNames) for elmt in node.elts]:
            elements.append(element[0])

        return elements if len(elements) == 1 else [{"list": elements}]

    # Call: ('func', 'args', 'keywords')
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute):
                fnNode = node.func.value
                if fnNode.value.id == "sys":
                    return []
            elif node.func.value.id == "write_stream_obj":
                return []
            fnNode = node.func
            module = fnNode.value.id
            fnName = fnNode.attr
            fnName = module + "." + fnName
        else:
            fnName = node.func.id
        inputs = []

        for arg in node.args:
            arg = genPgm(arg, state, fnNames)
            inputs.append(arg)

        call = {"call": {"function": fnName, "inputs": inputs}}

        return [call]

    # Module: body
    elif isinstance(node, ast.Module):
        pgms = []
        for cur in node.body:
            pgm = genPgm(cur, state, fnNames)
            pgms += pgm
        return [mergeDicts(pgms)]

    # BoolOp: body
    elif isinstance(node, ast.BoolOp):
        pgms = []
        boolOp = {
            ast.And: "and",
            ast.Or: "or"
            }

        for key in boolOp:
            if isinstance(node.op, key):
                pgms.append([{"boolOp": boolOp[key]}])

        for item in node.values:
            pgms.append(genPgm(item, state, fnNames))

        return pgms


    elif isinstance(node, ast.AST):
        sys.stderr.write(
            f"No handler for AST.{node.__class__.__name__} in genPgm, fields: {node._fields}\n"
        )

    else:
        sys.stderr.write(
            f"No handler for {node.__class__.__name__} in genPgm, value: {str(node)}\n"
        )

    return []


def importAst(filename: str):
    return ast.parse(tokenize.open(filename).read())

# Get the absolute path of the python files whose PGMs are being generated.
# TODO: For now the path is started from the directory "for2py" but need further discussion on this
def get_path(fileName: str, instance: str):
    absPath = os.path.abspath(fileName)
    if instance == "namespace":
        return re.match(r'.*\/(for2py\/.*).py$', absPath).group(1).split('/')
    elif instance == "source":
        return re.match(r'.*\/(for2py\/.*$)', absPath).group(1).split('/')

def create_pgm_dict(lambdaFile: str, asts: List, pgm_file, file_name: str) -> Dict:
    """ Create a Python dict representing the PGM, with additional metadata for
    JSON output. """
    with open(lambdaFile, "w") as f:
        f.write("import math\n\n")
        state = PGMState(f)
        pgm = genPgm(asts, state, {})[0]
        if pgm.get("start"):
            pgm["start"] = pgm["start"][0]
        else:
            pgm["start"] = ""
        # pgm["name"] = pgm_file

        pgm["source"] = [get_path(file_name, "source")]

        # dateCreated stores the date and time on which the lambda and PGM file was created.
        # It is stored in YYYMMDD format
        pgm["dateCreated"] = f"{datetime.today().strftime('%Y%m%d')}"

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
        "-a",
        "--printAst",
        action="store_true",
        required=False,
        help="Print ASTs",
    )
    args = parser.parse_args(sys.argv[1:])
    asts = get_asts_from_files(args.files, args.printAst)
    for index, inAst in enumerate(asts):
        lambdaFile = args.files[index][:-3] + '_' + args.lambdaFile[0]
        pgmFile = args.files[index][:-3] + '_' + args.PGMFile[0]
        pgm_dict = create_pgm_dict(lambdaFile, [inAst], pgmFile, args.files[index])

        with open(pgmFile, "w") as f:
            printPgm(f, pgm_dict)
