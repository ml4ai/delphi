import sys
import pickle
import argparse 

GETFRAME_EXPR = "sys._getframe({}).f_code.co_name"

PRINTFN = {}

class PrintState:
    def __init__(
        self,
        sep=None,
        add=None,
        printFirst=True,
        definedVars=None,
        globalVars=None,
        indexRef=True,
        varTypes=None,
    ):
        self.sep = sep if sep != None else "\n"
        self.add = add if add != None else "    "
        self.printFirst = printFirst
        self.definedVars = definedVars if definedVars != None else []
        self.globalVars = globalVars if globalVars != None else []
        self.indexRef = indexRef
        self.varTypes = varTypes if varTypes != None else {}

    def copy(
        self,
        sep=None,
        add=None,
        printFirst=None,
        definedVars=None,
        globalVars=None,
        indexRef=None,
        varTypes=None,
    ):
        return PrintState(
            self.sep if sep == None else sep,
            self.add if add == None else add,
            self.printFirst if printFirst == None else printFirst,
            self.definedVars if definedVars == None else definedVars,
            self.globalVars if globalVars == None else globalVars,
            self.indexRef if indexRef == None else indexRef,
            self.varTypes if varTypes == None else varTypes,
        )


def printSubroutine(pyFile, node, printState):
    pyFile.write(f"\ndef {node['name']}(")
    args = []
    printAst(
        pyFile,
        node["args"],
        printState.copy(
            sep=", ",
            add="",
            printFirst=False,
            definedVars=args,
            indexRef=False,
        ),
    )
    pyFile.write("):")
    printAst(
        pyFile,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add,
            printFirst=True,
            definedVars=args,
            indexRef=True,
        ),
    )


def printFunction(pyFile, node, printState):
    pyFile.write(f"\ndef {node['name']}(")
    args = []
    printAst(
        pyFile,
        node["args"],
        printState.copy(
            sep=", ",
            add="",
            printFirst=False,
            definedVars=args,
            indexRef=False,
        ),
    )
    pyFile.write("):")
    printAst(
        pyFile,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add,
            printFirst=True,
            definedVars=args,
            indexRef=True,
        ),
    )


def printProgram(pyFile, node, printState):
    printSubroutine(pyFile, node, printState)
    pyFile.write(f"\n\n{node['name']}(){printState.sep}")


def printCall(pyFile, node, printState):
    if not printState.indexRef:
        pyFile.write("[")

    pyFile.write(f"{node['name']}(")
    printAst(
        pyFile,
        node["args"],
        printState.copy(
            sep=", ", add="", printFirst=False, definedVars=[], indexRef=False
        ),
    )
    pyFile.write(")")

    if not printState.indexRef:
        pyFile.write("]")


def printArg(pyFile, node, printState):
    if node["type"] == "INTEGER":
        varType = "int"
    elif node["type"] == "DOUBLE":
        varType = "float"
    else:
        print(f"unrecognized type {node['type']}")
        sys.exit(1)
    pyFile.write(f"{node['name']}: List[{varType}]")
    printState.definedVars += [node["name"]]


def printVariable(pyFile, node, printState):
    if (
        node["name"] not in printState.definedVars
        and node["name"] not in printState.globalVars
    ):
        printState.definedVars += [node["name"]]
        if node["type"] == "INTEGER":
            initVal = 0
            varType = "int"
        elif node["type"] == "DOUBLE":
            initVal = 0.0
            varType = "float"
        else:
            print(f"unrecognized type {node['type']}")
            sys.exit(1)
        pyFile.write(f"{node['name']}: List[{varType}] = [{initVal}]")
    else:
        printState.printFirst = False


def printDo(pyFile, node, printState):
    pyFile.write("for ")
    printAst(
        pyFile,
        node["header"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyFile.write(":")
    printAst(
        pyFile,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add, printFirst=True, indexRef=True
        ),
    )


def printIndex(pyFile, node, printState):
    # pyFile.write("{0} in range({1}, {2}+1)".format(node['name'], node['low'], node['high'])) Don't use this
    # pyFile.write(f"{node['name']}[0] in range(") Use this instead
    pyFile.write("{0}[0] in range(".format(node["name"]))
    printAst(
        pyFile,
        node["low"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyFile.write(", ")
    printAst(
        pyFile,
        node["high"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyFile.write("+1)")


def printIf(pyFile, node, printState):
    pyFile.write("if ")
    printAst(
        pyFile,
        node["header"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyFile.write(":")
    printAst(
        pyFile,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add, printFirst=True, indexRef=True
        ),
    )
    if "else" in node:
        pyFile.write(printState.sep)
        pyFile.write("else:")
        printAst(
            pyFile,
            node["else"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                indexRef=True,
            ),
        )


def printOp(pyFile, node, printState):
    if not printState.indexRef:
        pyFile.write("[")
    if "right" in node:
        pyFile.write("(")
        printAst(
            pyFile,
            node["left"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )

        operator_mapping = {
            ".ne.": " != ",
            ".gt.": " > ",
            ".eq.": " == ",
            ".lt.": " < ",
            ".le.": " <= ",
        }
        pyFile.write(
            operator_mapping.get(
                node["operator"].lower(), f" {node['operator']} "
            )
        )
        printAst(
            pyFile,
            node["right"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        pyFile.write(")")
    else:
        pyFile.write(f"{node['operator']}")
        pyFile.write("(")
        printAst(
            pyFile,
            node["left"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        pyFile.write(")")
    if not printState.indexRef:
        pyFile.write("]")


def printLiteral(pyFile, node, printState):
    pyFile.write(f"{node['value']}")


def printRef(pyFile, node, printState):
    if printState.indexRef:
        pyFile.write(f"{node['name']}[0]")
    else:
        pyFile.write(f"{node['name']}")


def printAssignment(pyFile, node, printState):
    printAst(
        pyFile,
        node["target"],
        printState.copy(sep="", add="", printFirst=False, indexRef=True),
    )
    pyFile.write(" = ")
    printAst(
        pyFile,
        node["value"],
        printState.copy(sep="", add="", printFirst=False, indexRef=True),
    )

def printFuncReturn(pyFile, node, printState):
    if printState.indexRef:
        if node.get("name"):
            pyFile.write(f"return {node['name']}[0]")
        else:
            pyFile.write(f"return {node['value']}")
    else:
        if node.get("name"):
            pyFile.write(f"return {node['name']}")
        else:
            if node.get("value"):
                pyFile.write(f"return {node['value']}")
            else:
                pyFile.write(f"return None")


def printExit(pyFile, node, printState):
    pyFile.write("return")


def printReturn(pyFile, node, printState):
    pyFile.write("sys.exit(0)")


def setupPrintFns():
    PRINTFN.update(
        {
            "subroutine": printSubroutine,
            "program": printProgram,
            "call": printCall,
            "arg": printArg,
            "variable": printVariable,
            "do": printDo,
            "index": printIndex,
            "if": printIf,
            "op": printOp,
            "literal": printLiteral,
            "ref": printRef,
            "assignment": printAssignment,
            "exit": printExit,
            "return": printReturn,
            "function": printFunction,
            "ret": printFuncReturn,
          #  "read": printFileRead,
          #  "open": printFileOpen,
          #  "close": printFileClose,
        }
    )

def printAst(pyFile, root, printState):
    for node in root:
        if printState.printFirst:
            pyFile.write(printState.sep)
        else:
            printState.printFirst = True
        if node.get("tag"):
            PRINTFN[node["tag"]](pyFile, node, printState)


def printPython(gen, outputFile):
    pyFile = open(gen, "w")
    pickleFile = open(outputFile[0], "rb")
    outputRoot = pickle.load(pickleFile)
    root = outputRoot["ast"]
    
    pyFile.write("from typing import List")
    setupPrintFns()
    printAst(pyFile, root, PrintState())
    pickleFile.close()
    pyFile.close()

if __name__ == "__main__":
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
        help="Pickled version of the asts together with non-source code information",
    )
    args = parser.parse_args(sys.argv[1:])
    printPython(args.gen[0], args.files)
