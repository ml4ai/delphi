#!/usr/bin/python
import xml.etree.ElementTree as ET
import sys
import argparse
from pyTranslate import *

libRtns = ["read", "open", "close", "format", "print", "write"]
libFns = ["MOD", "EXP", "INDEX", "MIN", "MAX", "cexp", "cmplx", "ATAN"]
inputFns = ["read"]
outputFns = ["write"]
summaries = {}
asts = {}

cleanup = True


class ParseState:
    def __init__(self, subroutine=None):
        self.subroutine = subroutine if subroutine != None else {}
        self.args = (
            [arg["name"] for arg in self.subroutine["args"]]
            if "args" in self.subroutine
            else []
        )

    def copy(self, subroutine=None):
        return ParseState(self.subroutine if subroutine == None else subroutine)


def parseTree(root, state):
    if root.tag == "subroutine" or root.tag == "program":
        subroutine = {"tag": root.tag, "name": root.attrib["name"]}
        summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                subroutine["args"] = parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = parseTree(node, subState)
        asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    elif root.tag == "call":
        call = {"tag": "call"}
        for node in root:
            if node.tag == "name":
                call["name"] = node.attrib["id"]
                call["args"] = []
                for arg in node:
                    call["args"] += parseTree(arg, state)
        return [call]

    elif root.tag == "argument":
        return [{"tag": "arg", "name": root.attrib["name"]}]

    elif root.tag == "declaration":
        decVars = []
        decType = {}
        for node in root:
            if node.tag == "type":
                decType = {"type": node.attrib["name"]}
            elif node.tag == "variables":
                decVars = parseTree(node, state)
        prog = []
        for var in decVars:
            prog.append(decType.copy())
            prog[-1].update(var)
            if var["name"] in state.args:
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]
        return prog

    elif root.tag == "variable":
        try:
            return [{"tag": "variable", "name": root.attrib["name"]}]
        except:
            return []

    elif root.tag == "loop" and root.attrib["type"] == "do":
        do = {"tag": "do"}
        for node in root:
            if node.tag == "header":
                do["header"] = parseTree(node, state)
            elif node.tag == "body":
                do["body"] = parseTree(node, state)
        return [do]

    elif root.tag == "index-variable":
        ind = {"tag": "index", "name": root.attrib["name"]}
        for bounds in root:
            if bounds.tag == "lower-bound":
                ind["low"] = parseTree(bounds, state)
            elif bounds.tag == "upper-bound":
                ind["high"] = parseTree(bounds, state)
        return [ind]

    elif root.tag == "if":
        ifs = []
        curIf = None
        for node in root:
            if node.tag == "header" and "type" not in node.attrib:
                curIf = {"tag": "if"}
                curIf["header"] = parseTree(node, state)
                ifs.append(curIf)
            elif node.tag == "header" and node.attrib["type"] == "else-if":
                newIf = {"tag": "if"}
                curIf["else"] = [newIf]
                curIf = newIf
                curIf["header"] = parseTree(node, state)
                # ifs.append(ifStmt)
            elif node.tag == "body" and (
                "type" not in node.attrib or node.attrib["type"] != "else"
            ):
                curIf["body"] = parseTree(node, state)
            elif node.tag == "body" and node.attrib["type"] == "else":
                curIf["else"] = parseTree(node, state)
        return ifs

    elif root.tag == "operation":
        op = {"tag": "op"}
        for node in root:
            if node.tag == "operand":
                if "left" in op:
                    op["right"] = parseTree(node, state)
                else:
                    op["left"] = parseTree(node, state)
            elif node.tag == "operator":
                if "operator" in op:
                    newOp = {
                        "tag": "op",
                        "operator": node.attrib["operator"],
                        "left": [op],
                    }
                    op = newOp
                else:
                    op["operator"] = node.attrib["operator"]
        return [op]

    elif root.tag == "literal":
        for info in root:
            if info.tag == "pause-stmt":
                return [{"tag": "pause", "msg": root.attrib["value"]}]
        return [
            {
                "tag": "literal",
                "type": root.attrib["type"],
                "value": root.attrib["value"],
            }
        ]

    elif root.tag == "stop":
        return [{"tag": "stop"}]

    elif root.tag == "name":
        if root.attrib["id"] in libFns:
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += parseTree(node, state)
            return [fn]
        else:
            ref = {"tag": "ref", "name": root.attrib["id"]}
            subscripts = []
            for node in root:
                subscripts += parseTree(node, state)
            if subscripts:
                ref["subscripts"] = subscripts
            return [ref]

    elif root.tag == "assignment":
        assign = {"tag": "assignment"}
        for node in root:
            if node.tag == "target":
                assign["target"] = parseTree(node, state)
            elif node.tag == "value":
                assign["value"] = parseTree(node, state)
        return [assign]

    elif root.tag == "exit":
        return [{"tag": "exit"}]

    elif root.tag == "return":
        ret = {"tag": "return"}
        return [ret]

    elif root.tag in libRtns:
        fn = {"tag": "call", "name": root.tag, "args": []}
        for node in root:
            fn["args"] += parseTree(node, state)
        return [fn]

    else:
        prog = []
        for node in root:
            prog += parseTree(node, state)
        return prog


def printAstTree(astFile, tree, blockVal):
    parentVal = blockVal
    for node in tree:
        if parentVal != blockVal:
            astFile.write(
                "\tB" + str(parentVal) + " -> B" + str(blockVal) + "\n"
            )
            parentVal = blockVal
        block = "\tB" + str(blockVal) + ' [label="'
        blockVal += 1
        for key in node:
            if not isinstance(node[key], list):
                block += "'" + str(key) + "=" + str(node[key]) + "' "
        block += '"]\n'
        astFile.write(block)
        for key in node:
            if isinstance(node[key], list) and bool(node[key]):
                astFile.write(
                    "\tB"
                    + str(parentVal)
                    + " -> B"
                    + str(blockVal)
                    + ' [label="'
                    + str(key)
                    + '"]\n'
                )
                blockVal = printAstTree(astFile, node[key], blockVal)

    return blockVal


def printPython1(
    pyFile, root, sep, add, printFirst, definedVars, globalVars, indexRef
):

    for node in root:
        if printFirst:
            pyFile.write(sep)
        else:
            printFirst = True

        if node["tag"] == "variable":
            if (
                node["name"] not in definedVars
                and node["name"] not in globalVars
            ):
                definedVars += [node["name"]]
                # dataType = node['type']
                initVal = 0
                if node["type"] == "DOUBLE":
                    # dataType = 'DOUBLE PRECISION'
                    initVal = 0.0
                pyFile.write("{0} = [{1}]".format(node["name"], initVal))
            else:
                printFirst = False
        elif node["tag"] == "subroutine" or node["tag"] == "program":
            pyFile.write("def {0}(".format(node["name"]))
            args = []
            printPython(
                pyFile, node["args"], ", ", "", False, args, globalVars, False
            )
            pyFile.write("):")
            printPython(
                pyFile,
                node["body"],
                sep + add,
                add,
                True,
                args,
                globalVars,
                True,
            )
        elif node["tag"] == "call":
            pyFile.write("{0}(".format(node["name"]))
            printPython(
                pyFile, node["args"], ", ", "", False, [], globalVars, False
            )
            pyFile.write(")")
        elif node["tag"] == "arg":
            pyFile.write("{0}".format(node["name"]))
            definedVars += [node["name"]]
        # elif node['tag'] == 'variable':
        #    pass #skip
        elif node["tag"] == "do":
            pyFile.write("for ")
            printPython(
                pyFile,
                node["header"],
                "",
                "",
                True,
                definedVars,
                globalVars,
                True,
            )
            pyFile.write(":")
            printPython(
                pyFile,
                node["body"],
                sep + add,
                add,
                True,
                definedVars,
                globalVars,
                True,
            )
        elif node["tag"] == "index":
            # pyFile.write("{0} in range({1}, {2}+1)".format(node['name'], node['low'], node['high']))
            pyFile.write("{0}[0] in range(".format(node["name"]))
            printPython(
                pyFile, node["low"], "", "", True, definedVars, globalVars, True
            )
            pyFile.write(", ")
            printPython(
                pyFile,
                node["high"],
                "",
                "",
                True,
                definedVars,
                globalVars,
                True,
            )
            pyFile.write("+1)")
        elif node["tag"] == "if":
            pyFile.write("if ")
            printPython(
                pyFile,
                node["header"],
                "",
                "",
                True,
                definedVars,
                globalVars,
                True,
            )
            pyFile.write(":")
            printPython(
                pyFile,
                node["body"],
                sep + add,
                add,
                True,
                definedVars,
                globalVars,
                True,
            )
            if "else" in node:
                pyFile.write(sep)
                pyFile.write("else:")
                printPython(
                    pyFile,
                    node["else"],
                    sep + add,
                    add,
                    True,
                    definedVars,
                    globalVars,
                    True,
                )
        elif node["tag"] == "op":
            if indexRef == False:
                pyFile.write("[")
            if "right" in node:
                pyFile.write("(")
                printPython(
                    pyFile,
                    node["left"],
                    "",
                    "",
                    True,
                    definedVars,
                    globalVars,
                    True,
                )
                if node["operator"].lower() == ".ne.":
                    pyFile.write(" != ")
                elif node["operator"].lower() == ".gt.":
                    pyFile.write(" > ")
                elif node["operator"].lower() == ".eq.":
                    pyFile.write(" == ")
                elif node["operator"].lower() == ".lt.":
                    pyFile.write(" < ")
                elif node["operator"].lower() == ".le.":
                    pyFile.write(" <= ")
                else:
                    pyFile.write(" {0} ".format(node["operator"]))
                printPython(
                    pyFile,
                    node["right"],
                    "",
                    "",
                    True,
                    definedVars,
                    globalVars,
                    True,
                )
                pyFile.write(")")
            else:
                pyFile.write("{0}".format(node["operator"]))
                pyFile.write("(")
                printPython(
                    pyFile,
                    node["left"],
                    "",
                    "",
                    True,
                    definedVars,
                    globalVars,
                    True,
                )
                pyFile.write(")")
            if indexRef == False:
                pyFile.write("]")
        elif node["tag"] == "literal":
            pyFile.write("{0}".format(node["value"]))
        elif node["tag"] == "ref":
            if indexRef:
                pyFile.write("{0}[0]".format(node["name"]))
            else:
                pyFile.write("{0}".format(node["name"]))
        elif node["tag"] == "assignment":
            printPython(
                pyFile,
                node["target"],
                "",
                "",
                False,
                definedVars,
                globalVars,
                True,
            )
            pyFile.write(" = ")
            printPython(
                pyFile,
                node["value"],
                "",
                "",
                False,
                definedVars,
                globalVars,
                True,
            )
        elif node["tag"] == "return":
            pyFile.write("return")
        elif node["tag"] == "exit":
            pyFile.write("sys.exit(0)")
        else:
            print("unknown tag: {0}".format(node["tag"]))
            sys.exit(1)


def analyze(files, gen):
    global cleanup
    outputFiles = []
    ast = []
    for f in files:
        tree = ET.parse(f)
        ast += parseTree(tree.getroot(), ParseState())

    # printPython(sys.stdout, ast, "\n", "  ", True, [], [], True)
    pyFile = open(gen, "w")
    printPython(pyFile, ast)
    pyFile.close()

    # astFile = open("ast.dot", "w")
    # printAst(astFile, ast)
    # astFile.close()

    # outputFiles.append("ast.dot")
    # print "Building Function Summaries"

    # computeSummaries()

    # dgFile = open("summaries.dot", "w")
    # printSummaries(dgFile)
    # dgFile.close()

    # outputFiles.append("summaries.dot")

    # if not bool(gen):
    #    gen = []
    #    for ast in asts:
    #        if asts[ast][0]['tag'] == 'program':
    #            gen.append(ast)

    # for name in gen:
    #    print "Processing Clean " + name
    #    progDeps = evalSummary(summaries[name], {}, {}, {}, set())
    #    dgFilename = name + "_clean.dot"
    #    dgFile = open(dgFilename, "w")
    #    printDg(dgFile, progDeps)
    #    dgFile.close()
    #    outputFiles.append(dgFilename)

    # cleanup = False

    # for name in summaries:
    #    summaries[name] = None

    # computeSummaries()

    # for name in gen:
    #    print "Processing Dirty " + name
    #    progDeps = evalSummary(summaries[name], {}, {}, {}, set())
    #    dgFilename = name + "_dirty.dot"
    #    dgFile = open(dgFilename, "w")
    #    printDg(dgFile, progDeps)
    #    dgFile.close()
    #    outputFiles.append(dgFilename)

    # return outputFiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gen",
        nargs="*",
        help="Routines for which dependency graphs should be generated",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="A list of AST files in XML format to analyze",
    )
    args = parser.parse_args(sys.argv[1:])
    analyze(args.files, args.gen[0])


if __name__ == "__main__":
    main()
