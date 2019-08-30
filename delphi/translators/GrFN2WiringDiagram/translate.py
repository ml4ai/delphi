import sys
import inspect
import importlib

import itertools

from delphi.GrFN.networks import GroundedFunctionNetwork


def main():
    data_dir = "scripts/SIR_Demo/"
    sys.path.insert(0, data_dir)
    model_file = "SIR-simple"
    json_file = f"{model_file}_GrFN.json"
    lambdas = importlib.__import__(f"{model_file}_lambdas")

    grfn = GroundedFunctionNetwork.from_json_and_lambdas(json_file, lambdas)
    agraph = grfn.to_agraph()
    agraph.draw('SIR-simple.pdf', prog='dot')
    to_wiring_diagram(grfn, lambdas, "SIR-simple")


def sanitize_name(name):
    return name.replace("-->", "__").replace("::", "__").replace("-1", "neg1").replace("-", "_").replace("@", "").replace("$", "_").replace(".", "_")


def define_variables(vars):
    names = [sanitize_name(n) for n in vars]
    symbols = [":" + n for n in names]
    names_str = ", ".join(names)
    symbols_str = ", ".join(symbols)
    return f"{names_str} = Ob(FreeSymmetricMonoidalCategory, {symbols_str})"


def translate_GrFN(out_node, G, homs):
    out_name = sanitize_name(out_node)
    predecessors = list(G.predecessors(out_node))
    if len(predecessors) == 0:
        return []

    func_node = predecessors[0]
    inputs = list(G.predecessors(func_node))

    if len(inputs) == 0:
        return []

    statements, compositions = list(), list()
    for var_node in inputs:
        var_name = sanitize_name(var_node)
        compositions.append(f"compose({var_name}, {out_name})")
        # compositions.append(f"{out_name} ∘ {var_name}")
        statements.extend(translate_GrFN(var_node, G, homs))
    statements.extend(compositions)

    if G.nodes[func_node]["visited"]:
        return statements

    func_name = G.nodes[func_node]["lambda_fn"].__name__
    input_str = f"otimes({', '.join([sanitize_name(i) for i in inputs])})"
    statements.append(f"{out_name} = Hom({func_name}, {input_str}, {out_name})")
    homs.append(out_name)
    G.nodes[func_node]["visited"] = True
    return statements


def py2jl(py_code):
    return py_code.replace("def", "function").replace(":", "") + "end"


def bfs_translate_GrFN(nodes, G, lambdas, outfile, stmts, homs):
    print(nodes)
    for node in nodes:
        if G.nodes[node]["type"] == "function" and not G.nodes[node]["visited"]:
            func_name = G.nodes[node]["lambda_fn"].__name__
            py_func = inspect.getsource(getattr(lambdas, func_name))
            julia_func = py_func.replace("def", "function").replace(":", "")
            outfile.write(julia_func)
            outfile.write("end\n\n")
            inputs = list(G.predecessors(node))
            output = list(G.successors(node))[0]
            out_name = sanitize_name(output)
            input_str = f"{' ⊗ '.join([sanitize_name(i) for i in inputs])}"
            stmts.append(f"{out_name} = Hom({func_name}, {input_str}, {out_name})")
            homs.append(out_name)
            G.nodes[node]["visited"] = True

    new_nodes = itertools.chain.from_iterable([G.successors(n) for n in nodes])
    unique_new_nodes = list(set(new_nodes))
    if len(unique_new_nodes) > 0:
        bfs_translate_GrFN(unique_new_nodes, G, lambdas, outfile, stmts, homs)


def translate_lambdas(G, lambdas, filename):
    with open(filename, "w") as outfile:
        for name in G.call_graph.nodes():
            func_name = name.split("::")[-1]
            py_func = inspect.getsource(getattr(lambdas, func_name))
            julia_func = py_func.replace("def", "function").replace(":", "")
            outfile.write(julia_func)
            outfile.write("end\n\n")


def to_wiring_diagram(G, lambdas, filename):
    stmts, funcs = list(), list()
    var_defs = list()
    all_vars = list()
    for i, func_set in enumerate(G.function_sets):
        variable_nodes = list()
        for j, name in enumerate(func_set):
            inputs = list(G.predecessors(name))

            # Add variables for MonoidalCategory definition
            for var_node in inputs:
                if not (var_node in variable_nodes or var_node in all_vars):
                    variable_nodes.append(var_node)

            # Translate the lambda function code
            func_name = G.nodes[name]["lambda_fn"].__name__
            funcs.append(py2jl(inspect.getsource(getattr(lambdas, func_name))))

            input_str = f"{' ⊗ '.join([sanitize_name(i) for i in inputs])}"
            out_name = sanitize_name(list(G.successors(name))[0])

            stmts.append(f"WD_{out_name} = Hom({func_name}, {input_str}, {out_name})")
        var_def = define_variables(variable_nodes)
        all_vars.extend(variable_nodes)
        var_defs.append(var_def)

    with open(f"{filename}__functions.jl", "w") as func_file:
        func_file.write("\n\n".join(funcs))

    with open(f"{filename}__wiring.jl", "w") as wiring_file:
        wiring_file.write("\n\n".join([
            "using Catlab",
            "using Catlab.WiringDiagrams",
            "using Catlab.Doctrines",
            "import Catlab.Doctrines: ⊗, id",
            "import Base: ∘",
            f'include("{filename}__functions.jl")',
            "⊗(a::WiringDiagram, b::WiringDiagram) = otimes(a, b)",
            "∘(a::WiringDiagram, b::WiringDiagram) = compose(b, a)",
            "⊚(a,b) = b ∘ a"
        ]))
        wiring_file.write("\n\n\n")
        wiring_file.write("\n\n".join(var_defs))

    # funcs_file.write("\n\n".join(funcs))
    # wiring_file.write("\n\n".join(stmts))
    # diagram_name = filename.replace("-", "_")
    # wiring_file.write(f"\n\n{diagram_name} = WiringDiagram({', '.join(homs)})")
    # funcs_file.close()
    # wiring_file.close()


if __name__ == '__main__':
    main()
