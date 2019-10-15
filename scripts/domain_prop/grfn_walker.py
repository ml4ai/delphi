import os
import sys
import ast
import uuid
import inspect
import pickle
import networkx as nx


from delphi.GrFN.networks import GroundedFunctionNetwork


def main():
    data_dir = "../../tests/data/program_analysis"
    sys.path.insert(0, data_dir)
    fortran_file = sys.argv[1]
    fortran_name = os.path.splitext(fortran_file)[0]
    petpt = GroundedFunctionNetwork.from_fortran_file(f"{data_dir}/{fortran_file}")
    start_variable = petpt.outputs[0]
    lambda_trees = traverse_GrFN(start_variable, petpt)
    pickle.dump(lambda_trees, open(f"{fortran_name}--lambda_trees.pkl", "wb"))


def get_node_computation_str(func_node: str, grfn: nx.DiGraph):
    lambda_func = grfn.nodes[func_node]["lambda_fn"]
    lambda_str = inspect.getsource(lambda_func)
    clean_lambda = lambda_str.split("\n")[1].strip().replace("return ", "")
    return clean_lambda


def get_comp_ast(comp: str):
    return ast.parse(comp).body[0]


def get_bin_op(op_ast):
    if isinstance(op_ast, ast.Add):
        return "+"
    elif isinstance(op_ast, ast.Sub) or isinstance(op_ast, ast.USub):
        return "-"
    elif isinstance(op_ast, ast.Mult):
        return "*"
    elif isinstance(op_ast, ast.Div):
        return "/"
    elif isinstance(op_ast, ast.Pow):
        return "^"
    else:
        raise ValueError(f"Operator not found: {op_ast}")


def get_bool_op(op_ast):
    if isinstance(op_ast, ast.Eq):
        return "=="
    elif isinstance(op_ast, ast.NotEq):
        return "!="
    elif isinstance(op_ast, ast.And):
        return "&&"
    elif isinstance(op_ast, ast.Or):
        return "||"
    elif isinstance(op_ast, ast.LtE):
        return "<="
    elif isinstance(op_ast, ast.GtE):
        return ">="
    elif isinstance(op_ast, ast.Lt):
        return "<"
    elif isinstance(op_ast, ast.Gt):
        return ">"
    else:
        raise ValueError(f"Operator not found: {op_ast}")


def get_call_name(func_ast):
    if isinstance(func_ast, ast.Name):
        return func_ast.id
    elif isinstance(func_ast, ast.Attribute):
        return f"{func_ast.value.id}.{func_ast.attr}"
    else:
        raise ValueError(f"Unknown call type: {func_ast}")


def to_agraph(lambda_tree):
    A = nx.nx_agraph.to_agraph(lambda_tree)
    A.graph_attr.update(
        {"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "TB"}
    )
    A.node_attr.update({"fontname": "Menlo"})
    return A


def traverse_GrFN(curr_var, grfn):
    cur_trees = list()
    preds = list(grfn.predecessors(curr_var))
    if len(preds) == 0:
        return []
    func_node = preds[0]
    func_node_name = func_node.split("::")[-1]
    lambda_graph = nx.DiGraph()
    lambda_str = get_node_computation_str(func_node, grfn)
    func_ast = get_comp_ast(lambda_str)
    traverse_lambda_ast(func_ast, lambda_graph)
    cur_trees.append((func_node_name, lambda_graph))

    for var_node in grfn.predecessors(func_node):
        new_trees = traverse_GrFN(var_node, grfn)
        cur_trees.extend(new_trees)
    return cur_trees


def traverse_lambda_ast(tree, graph, parent="", tabs=0):
    child = uuid.uuid4()
    if isinstance(tree, ast.Expr):
        graph.add_node(child, label="ROOT")
        traverse_lambda_ast(tree.value, graph, parent=child)
    elif isinstance(tree, ast.Attribute):
        print(vars(tree.value))
        print("Attr:", vars(tree))
    elif isinstance(tree, ast.Call):
        graph.add_node(child, label=f"CALL\n({get_call_name(tree.func)})")
        graph.add_edge(parent, child)
        for arg in tree.args:
            traverse_lambda_ast(arg, graph, parent=child, tabs=tabs+1)
    elif isinstance(tree, ast.BinOp):
        graph.add_node(child, label=f"BINOP\n({get_bin_op(tree.op)})")
        graph.add_edge(parent, child)
        traverse_lambda_ast(tree.left, graph, parent=child, tabs=tabs+1)
        traverse_lambda_ast(tree.right, graph, parent=child, tabs=tabs+1)
    elif isinstance(tree, ast.UnaryOp):
        graph.add_node(child, label=f"UNARYOP\n({get_bin_op(tree.op)})")
        graph.add_edge(parent, child)
        traverse_lambda_ast(tree.operand, graph, parent=child, tabs=tabs+1)
    elif isinstance(tree, ast.Compare):
        operators = ", ".join([get_bool_op(o) for o in tree.ops])
        graph.add_node(child, label=f"BOOLOP\n({operators})")
        graph.add_edge(parent, child)
        traverse_lambda_ast(tree.left, graph, parent=child, tabs=tabs+1)
        for comparator in tree.comparators:
            traverse_lambda_ast(comparator, graph, parent=child, tabs=tabs+1)
    elif isinstance(tree, ast.IfExp):
        graph.add_node(child, label=f"COND\n(c,t,f)")
        graph.add_edge(parent, child)
        traverse_lambda_ast(tree.test, graph, parent=child, tabs=tabs+1)
        traverse_lambda_ast(tree.body, graph, parent=child, tabs=tabs+1)
        traverse_lambda_ast(tree.orelse, graph, parent=child, tabs=tabs+1)
    elif isinstance(tree, ast.Name):
        graph.add_node(child, label=f"ID\n({tree.id})")
        graph.add_edge(parent, child)
    elif isinstance(tree, ast.Num):
        graph.add_node(child, label=f"NUM\n({tree.n})")
        graph.add_edge(parent, child)
    elif isinstance(tree, ast.Str):
        graph.add_node(child, label=f"STR\n({tree.s})")
        graph.add_edge(parent, child)
    else:
        print(vars(tree))
        raise ValueError(f"Unknown Tree type: {type(tree)}")


if __name__ == '__main__':
    main()
