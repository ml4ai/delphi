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
from typing import List, Dict, Iterable, Optional
from itertools import chain, product
import operator

BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Eq: operator.eq,
    ast.LtE: operator.le,
}
ANNASSIGNED_LIST = []
ELIF_PGM = []

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


class PGMState:
    def __init__(
        self,
        lambdaStrings: Optional[List[str]],
        lastDefs: Optional[Dict] = {},
        nextDefs: Optional[Dict] = {},
        lastDefDefault=0,
        fnName=None,
        varTypes: Optional[Dict] = {},
    ):
        self.lastDefs = lastDefs
        self.nextDefs = nextDefs
        self.lastDefDefault = lastDefDefault
        self.fnName = fnName
        self.varTypes = varTypes
        self.lambdaStrings = lambdaStrings

    def copy(
        self,
        lastDefs: Optional[Dict] = None,
        nextDefs: Optional[Dict] = None,
        lastDefDefault=None,
        fnName=None,
        varTypes: Optional[Dict] = None,
        lambdaStrings: Optional[List[str]] = None,
    ):
        return PGMState(
            self.lambdaStrings if lambdaStrings is None else lambdaStrings,
            self.lastDefs if lastDefs is None else lastDefs,
            self.nextDefs if nextDefs is None else nextDefs,
            self.lastDefDefault if lastDefDefault is None else lastDefDefault,
            self.fnName if fnName is None else fnName,
            self.varTypes if varTypes is None else varTypes,
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


def genFn(lambdaStrings, node, fnName, returnVal, inputs):
    lambdaStrings.append(f"def {fnName}({', '.join(sorted(set(inputs), key=inputs.index))}):\n    ")
    # If a `decision` tag comes up, override the call to genCode to manually
    # enter the python script for the lambda file.
    if "__decision__" in fnName:
        code = f"{inputs[2]} if {inputs[0]} else {inputs[1]}"
    else:
        code = genCode(node, PrintState("\n    "))
    if returnVal:
        lambdaStrings.append(f"return {code}")
    else:
        lines = code.split("\n")
        indent = re.search("[^ ]", lines[-1]).start()
        lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
        lambdaStrings.append("\n".join(lines))
    lambdaStrings.append("\n\n")


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
    if '__decision__' in basename:
        part_match = re.match(r'(?P<body>\S+)__decision__(?P<identifier>\S+)', basename)
        if part_match:
            new_basename = part_match.group('body') + '__assign__' + part_match.group('identifier')
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
    index = nextDefs.get(var, lastDefDefault+1)
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
        else:
            sys.stderr.write(
                "Unsupported type (only float, int, list, and str"
                "supported as of now).\n"
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
        sys.stderr.write(f"num: {type(val)}\n")
        sys.exit(1)
    return dtype


def get_body_and_functions(pgm):
    body = list(chain.from_iterable(stmt["body"] for stmt in pgm))
    fns = list(chain.from_iterable(stmt["functions"] for stmt in pgm))
    return body, fns


def make_fn_dict(name, target, sources, node):
    source = []
    fn = {}

    # Preprocessing and removing certain Assigns which only pertain to the
    # Python code and do not relate to the FORTRAN code in any way.
    if target["var"]["variable"] == "write_line":
        return fn
    for src in sources:
        if "call" in src:
            if src["call"]["function"] == "Format":
                return fn
            for source_ins in make_call_body_dict(src):
                if source_ins["type"] != "function":
                    source.append(source_ins)
        if "var" in src:
            variable = src["var"]["variable"]
            source.append({"name": variable, "type": "variable"})
        if re.match(r"\d+", target["var"]["variable"]) and "list" in src:
            # This is a write to a file
            source.append({"name": "write", "type": "function"})
            for item in src["list"]:
                variable = item["var"]["variable"]
                source.append({"name": variable, "type": "variable"})

    # File Open Check
    fn.update(
        {
            "name": name,
            "type": "assign",
            "reference": node.lineno,
            "sources": source,
        }
    )
    if len(source) > 0:
        if (
            source[0].get("name") == "open"
            and source[0].get("type") == "function"
        ):
            (file_id, source) = handle_file_open(
                target["var"]["variable"], source
            )
            fn["target"] = file_id
        else:
            fn["target"] = target["var"]["variable"]
    else:
        fn["target"] = target["var"]["variable"]
    return fn


def handle_file_open(target, source):
    # This block maps the 'r' and 'w' modes in python file handling to read and
    # write commands in the source field.
    #
    # Currently, the 'read' and 'write' actions are not included in source
    # field but this function can handle it if necessary.
    mode_mapping = {"r": "read", "w": "write"}
    file_id = re.findall(r".*_(\d+)$", target)[0]
    source[-1]["name"] = mode_mapping[source[-1]["name"]]

    # Return with 'read'/'write' action. Disabled for now
    # return (file_id, source)

    # Return without the 'read'/'write' action.
    return (file_id, source[:-1])


def make_call_body_dict(source):
    source_list = []
    if re.match(r"format_\d+_obj\.read_line", source["call"]["function"]):
        source_list.append({"name": "read", "type": "function"})
        file_id_reg = r"file_(\d+)\.readline"
        id_string = source["call"]["inputs"][0][0]["call"]["function"]
        if re.match(file_id_reg, id_string):
            match = re.findall(file_id_reg, id_string)
            source_list.append({"name": match[0], "type": "variable"})
        return source_list

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
    file_read_index = 2

    file_id_match = re.match(r"file_(\d+)$", target["var"]["variable"])
    for src in sources:
        if "var" in src:
            source_list.append(src["var"])
        if "call" in src:
            for ip in src["call"]["inputs"][0]:
                if "call" in ip:
                    read_match = re.match(
                        r"file_(\d+)\.readline", ip["call"]["function"]
                    )
                    if read_match:
                        source_list.append(
                            {
                                "variable": read_match.group(1),
                                "index": file_read_index,
                            }
                        )
                        file_read_index += 1
                if "var" in ip:
                    source_list.append(ip["var"])
            if file_id_match:
                target["var"]["variable"] = file_id_match.group(1)
                source_list = source_list[:-1]
        if "list" in src and re.match(r"^(\d+)$", target["var"]["variable"]):
            for item in src["list"]:
                source_list.append(item)

    body = {"name": name, "output": target["var"], "input": source_list}
    return body


def genPgm(node, state, fnNames, call_source):
    types = (list, ast.Module, ast.FunctionDef)

    if state.fnName is None and not any(isinstance(node, t) for t in types):
        if isinstance(node, ast.Call):
            return [{"start": node.func.id}]
        elif isinstance(node, ast.Expr):
            return genPgm(node.value, state, fnNames, "start")
        elif isinstance(node, ast.If):
            return genPgm(node.body, state, fnNames, "start")
        else:
            return []

    if isinstance(node, list):
        return list(
            chain.from_iterable([genPgm(cur, state, fnNames, call_source) for cur in node])
        )

    # Function: name, args, body, decorator_list, returns
    elif isinstance(node, ast.FunctionDef):
        localDefs = state.lastDefs.copy()
        localNext = state.nextDefs.copy()
        localTypes = state.varTypes.copy()
        fnState = state.copy(
            lastDefs=localDefs,
            nextDefs=localNext,
            fnName=node.name,
            varTypes=localTypes,
        )

        args = genPgm(node.args, fnState, fnNames, "functiondef")
        bodyPgm = genPgm(node.body, fnState, fnNames, "functiondef")

        body, fns = get_body_and_functions(bodyPgm)

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
                {"name": var, "domain": localTypes[var]} for var in variables
            ],
            "body": body,
        }

        fns.append(fnDef)

        pgm = {"functions": fns}

        return [pgm]

    # arguments: ('args', 'vararg', 'kwonlyargs', 'kw_defaults', 'kwarg',
    # 'defaults')
    elif isinstance(node, ast.arguments):
        return [genPgm(arg, state, fnNames, "arguments") for arg in node.args]

    # arg: ('arg', 'annotation')
    elif isinstance(node, ast.arg):
        state.varTypes[node.arg] = getVarType(node.annotation)
        if state.lastDefs.get(node.arg):
            state.lastDefs[node.arg] += 1
        else:
            state.lastDefs[node.arg] = 0
        return node.arg

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
        genPgm(node.value, state, fnNames, "index")

    # Num: ('n',)
    elif isinstance(node, ast.Num):
        return [
            {"type": "literal", "dtype": getDType(node.n), "value": node.n}
        ]

    # List: ('elts', 'ctx')
    elif isinstance(node, ast.List):
        elements = [
            element[0]
            for element in [genPgm(elmt, state, fnNames, "List") for elmt in node.elts]
        ]
        return elements if len(elements) == 1 else [{"list": elements}]

    # Str: ('s',)
    elif isinstance(node, ast.Str):
        return [{"type": "literal", "dtype": "string", "value": node.s}]

    # For: ('target', 'iter', 'body', 'orelse')
    elif isinstance(node, ast.For):
        if genPgm(node.orelse, state, fnNames, "for"):
            sys.stderr.write("For/Else in for not supported\n")
            sys.exit(1)

        indexVar = genPgm(node.target, state, fnNames, "for")
        if len(indexVar) != 1 or "var" not in indexVar[0]:
            sys.stderr.write("Only one index variable is supported\n")
            sys.exit(1)
        indexName = indexVar[0]["var"]["variable"]

        loopIter = genPgm(node.iter, state, fnNames, "for")
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
        loop = genPgm(node.body, loopState, fnNames, "for")
        loopBody, loopFns = get_body_and_functions(loop)

        variables = [x for x in loopLastDef if x != indexName]

        # variables: see what changes?
        loopName = getFnName(
            fnNames, f"{state.fnName}__loop_plate__{indexName}",{}
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
        global ELIF_PGM

        if call_source == "if":
            pgm = {"functions": [], "body": []}

            condSrcs = genPgm(node.test, state, fnNames, "if")

            startDefs = state.lastDefs.copy()
            ifDefs = startDefs.copy()
            elseDefs = startDefs.copy()
            ifState = state.copy(lastDefs=ifDefs)
            elseState = state.copy(lastDefs=elseDefs)
            ifPgm = genPgm(node.body, ifState, fnNames, "if")
            elsePgm = genPgm(node.orelse, elseState, fnNames, "if")

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

            ELIF_PGM = [pgm, condSrcs, node.test, node.lineno, node, updatedDefs, ifDefs]

            return []

        pgm = {"functions": [], "body": []}

        condSrcs = genPgm(node.test, state, fnNames, "if")

        condNum = state.nextDefs.get("#cond", state.lastDefDefault + 1)
        state.nextDefs["#cond"] = condNum + 1

        condName = f"IF_{condNum}"
        state.varTypes[condName] = "boolean"
        state.lastDefs[condName] = 0
        fnName = getFnName(fnNames, f"{state.fnName}__condition__{condName}", {})
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
        body = {
            "name": fnName,
            "output": condOutput,
            "input": [src["var"] for src in condSrcs if "var" in src],
        }
        pgm["functions"].append(fn)
        pgm["body"].append(body)
        genFn(
            state.lambdaStrings,
            node.test,
            fnName,
            None,
            [src["var"]["variable"] for src in condSrcs if "var" in src],
        )

        startDefs = state.lastDefs.copy()
        ifDefs = startDefs.copy()
        elseDefs = startDefs.copy()
        ifState = state.copy(lastDefs=ifDefs)
        elseState = state.copy(lastDefs=elseDefs)
        ifPgm = genPgm(node.body, ifState, fnNames, "if")
        elsePgm = genPgm(node.orelse, elseState, fnNames, "if")

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

            genFn(
                state.lambdaStrings,
                node,
                fnName,
                updatedDef,
                [f"{src['variable']}_{src['index']}" for src in inputs],
            )

            pgm["functions"].append(fn)
            pgm["body"].append(body)

            # Previous ELIF Block is filled??
            if len(ELIF_PGM) > 0:

                condSrcs = ELIF_PGM[1]

                pgm["functions"].append(ELIF_PGM[0]["functions"])
                pgm["body"].append(ELIF_PGM[0]["body"])

                condNum = state.nextDefs.get("#cond", state.lastDefDefault + 1)
                state.nextDefs["#cond"] = condNum + 1

                condName = f"IF_{condNum}"
                state.varTypes[condName] = "boolean"
                state.lastDefs[condName] = 0
                fnName = getFnName(fnNames, f"{state.fnName}__condition__{condName}", {})
                condOutput = {"variable": condName, "index": 0}

                fn = {
                    "name": fnName,
                    "type": "condition",
                    "target": condName,
                    "reference": ELIF_PGM[3],
                    "sources": [
                        {"name": src["var"]["variable"], "type": "variable"}
                        for src in condSrcs
                        if "var" in src
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
                    state.lambdaStrings,
                    ELIF_PGM[2],
                    fnName,
                    None,
                    [src["var"]["variable"] for src in condSrcs if "var" in src],
                )

                startDefs = state.lastDefs.copy()
                ifDefs = ELIF_PGM[6]
                elseDefs = startDefs.copy()

                updatedDefs = ELIF_PGM[5]

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
                        "reference": ELIF_PGM[3],
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

                    genFn(
                        state.lambdaStrings,
                        ELIF_PGM[4],
                        fnName,
                        updatedDef,
                        [f"{src['variable']}_{src['index']}" for src in inputs],
                    )

                    pgm["functions"].append(fn)
                    pgm["body"].append(body)

                    ELIF_PGM = []

        return [pgm]

    # UnaryOp: ('op', 'operand')
    elif isinstance(node, ast.UnaryOp):
        return genPgm(node.operand, state, fnNames, "unaryop")

    # BinOp: ('left', 'op', 'right')
    elif isinstance(node, ast.BinOp):
        if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
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

        return genPgm(node.left, state, fnNames, "binop") + genPgm(
            node.right, state, fnNames, "binop"
        )

    # Mult: ()

    elif any(isinstance(node, nodetype) for nodetype in UNNECESSARY_TYPES):
        t = node.__repr__().split()[0][2:]
        sys.stdout.write(f"Found {t}, which should be unnecessary\n")

    # Expr: ('value',)
    elif isinstance(node, ast.Expr):
        exprs = genPgm(node.value, state, fnNames, "expr")
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
        return genPgm(node.left, state, fnNames, "compare") + genPgm(
            node.comparators, state, fnNames, "compare"
        )

    # Subscript: ('value', 'slice', 'ctx')
    elif isinstance(node, ast.Subscript):
        global ANNASSIGNED_LIST
        if not isinstance(node.slice.value, ast.Num):
            sys.stderr.write("can't handle arrays right now\n")
            sys.exit(1)

        val = genPgm(node.value, state, fnNames, "subscript")

        if val[0]["var"]["variable"] in ANNASSIGNED_LIST:
            if isinstance(node.ctx, ast.Store):
                val[0]["var"]["index"] = getNextDef(
                    val[0]["var"]["variable"],
                    state.lastDefs,
                    state.nextDefs,
                    state.lastDefDefault,
                )
        else:
           ANNASSIGNED_LIST.append(val[0]["var"]["variable"])
        return val

    # Name: ('id', 'ctx')
    elif isinstance(node, ast.Name):
        lastDef = getLastDef(node.id, state.lastDefs, state.lastDefDefault)
        if isinstance(node.ctx, ast.Store) and state.nextDefs.get(node.id) and source != "annassign":
            lastDef = getNextDef(
                node.id, state.lastDefs, state.nextDefs, state.lastDefDefault
            )
        return [{"var": {"variable": node.id, "index": lastDef}}]

    # AnnAssign: ('target', 'annotation', 'value', 'simple')
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.value, ast.List):
            targets = genPgm(node.target, state, fnNames, "annassign")
            for target in targets:
                state.varTypes[target["var"]["variable"]] = getVarType(
                    node.annotation
                )
                if target['var']['variable'] not in ANNASSIGNED_LIST:
                    ANNASSIGNED_LIST.append(target['var']['variable'])
            return []

        sources = genPgm(node.value, state, fnNames, "annassign")
        targets = genPgm(node.target, state, fnNames, "annassign")
        pgm = {"functions": [], "body": []}

        for target in targets:
            state.varTypes[target["var"]["variable"]] = getVarType(
                node.annotation
            )
            name = getFnName(
                fnNames, f"{state.fnName}__assign__{target['var']['variable']}", {}
            )
            fn = make_fn_dict(name, target, sources, node)
            body = make_body_dict(name, target, sources)

            genFn(
                state.lambdaStrings,
                node,
                name,
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
        sources = genPgm(node.value, state, fnNames, "assign")
        targets = reduce(
            (lambda x, y: x.append(y)),
            [genPgm(target, state, fnNames, "assign") for target in node.targets],
        )
        pgm = {"functions": [], "body": []}
        for target in targets:
            source_list = []
            if target.get("list"):
                targets = ",".join(
                    [x["var"]["variable"] for x in target["list"]]
                )
                target = {"var": {"variable": targets, "index": 1}}

            # Extracting only file_id from the write list variable
            match = re.match(r"write_list_(\d+)", target["var"]["variable"])
            if match:
                target["var"]["variable"] = match.group(1)

            name = getFnName(
                fnNames, f"{state.fnName}__assign__{target['var']['variable']}", target
            )
            # If the index is -1, change it to the index in the fnName. This is a hack right now.
            # if target["var"]["index"] < 0:
            #     target["var"]["index"] = name[-1]

            fn = make_fn_dict(name, target, sources, node)
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
                state.lambdaStrings,
                node,
                name,
                target["var"]["variable"],
                source_list,
            )
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
            pgm["functions"].append(fn)
            pgm["body"].append(body)
        return [pgm]

    # Tuple: ('elts', 'ctx')
    elif isinstance(node, ast.Tuple):
        elements = []
        for element in [genPgm(elmt, state, fnNames, "ctx") for elmt in node.elts]:
            elements.append(element[0])

        return elements if len(elements) == 1 else [{"list": elements}]

    # Call: ('func', 'args', 'keywords')
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            fnNode = node.func
            module = fnNode.value.id
            fnName = fnNode.attr
            fnName = module + "." + fnName
        else:
            fnName = node.func.id
        inputs = []

        for arg in node.args:
            arg = genPgm(arg, state, fnNames, "call")
            inputs.append(arg)

        call = {"call": {"function": fnName, "inputs": inputs}}

        return [call]

    # Module: body
    elif isinstance(node, ast.Module):
        pgms = []
        for cur in node.body:
            pgm = genPgm(cur, state, fnNames, "module")
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
            pgms.append(genPgm(item, state, fnNames, "boolop"))

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


def importAst(filename: str):
    return ast.parse(tokenize.open(filename).read())


def create_pgm_dict(
    lambdaFile: str, asts: List, pgm_file="pgm.json", save_file=False
) -> Dict:
    """ Create a Python dict representing the PGM, with additional metadata for
    JSON output. """
    lambdaStrings = ["import math\n\n"]
    state = PGMState(lambdaStrings)
    pgm = genPgm(asts, state, {}, "")[0]
    if pgm.get("start"):
        pgm["start"] = pgm["start"][0]
    else:
        pgm["start"] = ""
    pgm["name"] = pgm_file
    pgm["dateCreated"] = f"{datetime.today().strftime('%Y-%m-%d')}"

    with open(lambdaFile, "w") as f:
        f.write("".join(lambdaStrings))

    # View the PGM file that will be used to build a scope tree
    if save_file:
        json.dump(pgm, open(pgm_file, "w"))
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
    pgm_dict = create_pgm_dict(args.lambdaFile[0], asts, args.PGMFile[0])

    with open(args.PGMFile[0], "w") as f:
        printPgm(f, pgm_dict)
