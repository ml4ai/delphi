import sys
import ast
import math
import inspect
import networkx as nx


from delphi.GrFN.networks import GroundedFunctionNetwork
from constraints import Interval
# import PETPT_lambdas as lambdas


def main():
    data_dir = "../../tests/data/program_analysis"
    sys.path.insert(0, data_dir)
    petpt = GroundedFunctionNetwork.from_fortran_file(f"{data_dir}/PETPT.for")
    domain_constraints = {
        Interval(-math.inf, math.inf, False, False)
        for input_name in petpt.inputs
    }
    start_variable = petpt.outputs[0]
    print(domain_constraints)
    traverse_GrFN(start_variable, petpt, domain_constraints)


def get_node_computation_str(func_node: str, grfn: nx.DiGraph):
    lambda_func = grfn.nodes[func_node]["lambda_fn"]
    lambda_str = inspect.getsource(lambda_func)
    clean_lambda = lambda_str.split("\n")[1].strip().replace("return ", "")
    return clean_lambda


def get_comp_ast(comp: str):
    return ast.parse(comp).body[0]


def traverse_GrFN(curr_var, grfn, domains):
    preds = list(grfn.predecessors(curr_var))
    if len(preds) == 0:
        return
    func_node = preds[0]
    print(func_node)
    lambda_str = get_node_computation_str(func_node, grfn)
    func_ast = get_comp_ast(lambda_str)
    traverse_lambda_ast(func_ast)
    print("###################################")
    for var_node in grfn.predecessors(func_node):
        traverse_GrFN(var_node, grfn, domains)


def traverse_lambda_ast(tree):
    # print(vars(tree), type(tree))
    if isinstance(tree, ast.Expr):
        traverse_lambda_ast(tree.value)
    elif isinstance(tree, ast.Attribute):
        print("Attr:", vars(tree))
    elif isinstance(tree, ast.Call):
        print("CALL:", tree.func, tree.args)
        traverse_lambda_ast(tree.func)
        for arg in tree.args:
            traverse_lambda_ast(arg)
    elif isinstance(tree, ast.BinOp):
        print("BINOP:", tree.left, tree.op, tree.right)
        traverse_lambda_ast(tree.left)
        traverse_lambda_ast(tree.right)
    elif isinstance(tree, ast.Compare):
        print("BOOLOP:", tree.left)
        traverse_lambda_ast(tree.left)
        # for op in tree.ops:
        #     traverse_lambda_ast(op)
        for comparator in tree.comparators:
            traverse_lambda_ast(comparator)
    elif isinstance(tree, ast.IfExp):
        print("COND (c,t,f):", tree.test, tree.body, tree.orelse)
        traverse_lambda_ast(tree.test)
        traverse_lambda_ast(tree.body)
        traverse_lambda_ast(tree.orelse)
    elif isinstance(tree, ast.Name):
        print("NAME:", tree.id)
    elif isinstance(tree, ast.Num):
        print("NUMBER:", tree.n)
    else:
        print(vars(tree))
        raise ValueError(f"Unknown Tree type: {type(tree)}")


if __name__ == '__main__':
    main()
