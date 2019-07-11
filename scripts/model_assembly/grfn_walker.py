import sys
import ast
import inspect
import networkx as nx

from delphi.GrFN.networks import GroundedFunctionNetwork
# import PETPT_lambdas as lambdas


def main():
    data_dir = "tests/data/GrFN/"
    sys.path.insert(0, "tests/data/program_analysis")
    petpt = GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/PETPT.for")
    # range_cons = get_range_constraints(petpt)
    domain_cons = get_domain_constraints(petpt)


def get_range_constraints(grfn):
    return NotImplemented


def get_domain_constraints(grfn):
    domains = {}
    start_node = grfn.output_node
    parent_func_node = grfn.predecessors(start_node)[0]
    lambda_func = grfn.nodes[parent_func_node]["lambda_fn"]
    lambda_str = inspect.getsource(lambda_func)
    func_str = lambda_str.split("\n")[1].strip().replace("return ", "")
    func_ast = ast.parse(func_str)
    for node in ast.walk(func_ast):
        print(node._fields)
    return NotImplemented


if __name__ == '__main__':
    main()
