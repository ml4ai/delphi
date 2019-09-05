import sys
import inspect
import importlib

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
    (D, I, S, F) = to_wiring_diagram(grfn, lambdas)
    write_files(D, I, S, F, model_file)


def to_wiring_diagram(G, lambdas):
    layer_defs = get_layer_defs(G)

    # Create variable MonoidalCategory definitions per layer
    variable_defs = [
        define_variables(layer_defs[i]["codomain"])
        for i in sorted(layer_defs.keys())
    ]

    stmts, funcs = list(), list()
    ids = list()
    for i, func_set in reversed(list(enumerate(G.function_sets))):
        layer_domain, layer_codomain = list(), list()
        wd_list, new_stmts = list(), list()
        for j, name in enumerate(func_set):
            inputs = G.nodes[name]["func_inputs"]
            input_names = [sanitize_name(i) for i in inputs]
            layer_domain.extend(input_names)

            # Translate the lambda function code
            func_name = G.nodes[name]["lambda_fn"].__name__
            funcs.append(py2jl(inspect.getsource(getattr(lambdas, func_name))))

            input_str = f"{' ⊗ '.join(input_names)}"
            out_name = sanitize_name(list(G.successors(name))[0])
            layer_codomain.append(out_name)

            new_stmts.append(f"WD_{out_name} = WiringDiagram(Hom({func_name}, {input_str}, {out_name}))")
            wd_list.append(f"WD_{out_name}")

        for port in ids:
            wd_list.append(f"id_{port}")

        if i != 0:
            out_line = f"OUT_{i+1} = OUT_{i} ⊚ IN_{i} ⊚ ({' ⊗ '.join(wd_list)})\n\n"
        else:
            out_line = f"OUT_{i+1} = IN_{i} ⊚ ({' ⊗ '.join(wd_list)})\n\n"

        stmts.append(out_line)
        stmts.extend(new_stmts)

        cur_dom = layer_defs[i-1]["codomain"]
        cur_codom = layer_defs[i]["domain"]
        dom_add = list(set(cur_codom) - set(cur_dom))
        ids.extend(dom_add)
        cur_dom.extend(dom_add)
        layer_defs[i-1]["domain"].extend(dom_add)
        dom = " ⊗ ".join(cur_dom)
        codom = " ⊗ ".join(cur_codom)
        stmts.append(f"IN_{i} = WiringDiagram(Hom(:L{i}_REWIRE, {dom}, {codom}))")
    stmts.reverse()
    ids = [f"id_{i} = id(Ports([{i}]))" for i in ids]
    return variable_defs, ids, stmts, funcs


def write_files(defs, ids, stmts, funcs, filename):
    with open(f"{filename}__functions.jl", "w") as func_file:
        func_file.write("\n\n".join(funcs))

    with open(f"{filename}__wiring.jl", "w") as wiring_file:
        wiring_file.write("\n".join([
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
        wiring_file.write("\n".join(defs))
        wiring_file.write("\n\n\n")
        wiring_file.write("\n".join(ids))
        wiring_file.write("\n\n\n")
        wiring_file.write("\n".join(stmts))


def get_layer_defs(G):
    defs = {-1: {"domain": [], "codomain": [sanitize_name(n) for n in G.inputs]}}
    for i, func_set in enumerate(G.function_sets):
        layer_domain, layer_codomain = list(), list()
        for j, name in enumerate(func_set):
            inputs = G.nodes[name]["func_inputs"]
            input_names = [sanitize_name(i) for i in inputs]
            layer_domain.extend(input_names)

            out_name = sanitize_name(list(G.successors(name))[0])
            layer_codomain.append(out_name)
        defs[i] = {"domain": layer_domain, "codomain": layer_codomain}
    return defs


def sanitize_name(name):
    return name.replace("-->", "__").replace("::", "__") \
               .replace("-1", "neg1").replace("-", "_").replace("@", "") \
               .replace("$", "_").replace(".", "_")


def define_variables(vars):
    names = [sanitize_name(n) for n in vars]
    symbols = [":" + n for n in names]
    names_str = ", ".join(names)
    symbols_str = ", ".join(symbols)
    return f"{names_str} = Ob(FreeSymmetricMonoidalCategory, {symbols_str})"


def py2jl(py_code):
    return py_code.replace("def", "function").replace(": Real", "").replace(":", "") + "end"


if __name__ == '__main__':
    main()
