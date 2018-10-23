import sys


getframe_expr = 'sys._getframe({}).f_code.co_name'

printFn = {}

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
    pyFile.write("\ndef {0}(".format(node["name"]))
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
    pyFile.write("\ndef {0}(".format(node["name"]))
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
    pyFile.write("\n\n{0}(){1}".format(node["name"], printState.sep))


def printCall(pyFile, node, printState):
    if printState.indexRef == False:
        pyFile.write("[")

    pyFile.write("{0}(".format(node["name"]))
    printAst(
        pyFile,
        node["args"],
        printState.copy(
            sep=", ", add="", printFirst=False, definedVars=[], indexRef=False
        ),
    )
    pyFile.write(")")

    if printState.indexRef == False:
        pyFile.write("]")


def printArg(pyFile, node, printState):
   # print (node)
    #print (eval(getframe_expr.format(2)))
    #print (eval(getframe_expr.format(3)))  
    if node["type"] == "INTEGER":
        varType = "int"
    elif node["type"] == "DOUBLE":
        varType = "float"
    else:
        print("unrecognized type {0}".format(node["type"]))
        sys.exit(1)
    pyFile.write("{0}: List[{1}]".format(node["name"], varType))
    printState.definedVars += [node["name"]]


def printVariable(pyFile, node, printState):
    if (
        node["name"] not in printState.definedVars
        and node["name"] not in printState.globalVars
    ):
        #print (node)
        #print (printState)
        printState.definedVars += [node["name"]]
        if node["type"] == "INTEGER":
            initVal = 0
            varType = "int"
        elif node["type"] == "DOUBLE":
            initVal = 0.0
            varType = "float"
        else:
            print("unrecognized type {0}".format(node["type"]))
            sys.exit(1)
        pyFile.write(
            "{0}: List[{1}] = [{2}]".format(node["name"], varType, initVal)
        )
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
    # pyFile.write("{0} in range({1}, {2}+1)".format(node['name'], node['low'], node['high']))
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
    if printState.indexRef == False:
        pyFile.write("[")
    if "right" in node:
        pyFile.write("(")
        printAst(
            pyFile,
            node["left"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
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
        printAst(
            pyFile,
            node["right"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        pyFile.write(")")
    else:
        pyFile.write("{0}".format(node["operator"]))
        pyFile.write("(")
        printAst(
            pyFile,
            node["left"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        pyFile.write(")")
    if printState.indexRef == False:
        pyFile.write("]")


def printLiteral(pyFile, node, printState):
    pyFile.write("{0}".format(node["value"]))


def printRef(pyFile, node, printState):
    if printState.indexRef:
        pyFile.write("{0}[0]".format(node["name"]))
    else:
        pyFile.write("{0}".format(node["name"]))


def printAssignment(pyFile, node, printState):
   # if node["target"][0]["name"] in functionLst:
   #     printAst(
   #        pyFile,
   #        node["value"],
   #        printState.copy(sep="", add="", printFirst=False, indexRef=True),
   #     )
    #else:
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
        pyFile.write("return {0}[0]".format(node["name"]))
    else:
        pyFile.write("return {0}".format(node["name"]))


def printExit(pyFile, node, printState):
    pyFile.write("return")


def printReturn(pyFile, node, printState):
    pyFile.write("sys.exit(0)")


def setupPrintFns():
    printFn["subroutine"] = printSubroutine
    printFn["program"] = printProgram
    printFn["call"] = printCall
    printFn["arg"] = printArg
    printFn["variable"] = printVariable
    printFn["do"] = printDo
    printFn["index"] = printIndex
    printFn["if"] = printIf
    printFn["op"] = printOp
    printFn["literal"] = printLiteral
    printFn["ref"] = printRef
    printFn["assignment"] = printAssignment
    printFn["exit"] = printExit
    printFn["return"] = printReturn
    printFn["function"] = printFunction
    printFn["ret"] = printFuncReturn  
    printFn["read"] = printFileRead
    printFn["open"] = printFileOpen
    printFn["close"] = printFileClose

def printAst(pyFile, root, printState):
    for node in root:
        if printState.printFirst:
            pyFile.write(printState.sep)
        else:
            printState.printFirst = True

        printFn[node["tag"]](pyFile, node, printState)


def printPython(pyFile, root):
    pyFile.write("from typing import List")
    setupPrintFns()
    printAst(pyFile, root, PrintState())
