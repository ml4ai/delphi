import sys

from delphi.GrFN.networks import GroundedFunctionNetwork


def main():
    data_dir = "scripts/SIR_Demo/"
    sys.path.insert(0, data_dir)

    grfn = GroundedFunctionNetwork.from_fortran_file(f"{data_dir}SIR-simple.f")
    to_wiring_diagram(grfn, "SIR-simple.jl")
    agraph = grfn.to_agraph()
    agraph.draw('SIR-gillespie.pdf', prog='dot')
    CAG = grfn.to_CAG_agraph()
    CAG.draw('SIR-gillespie-CAG.pdf', prog='dot')


def translate_GrFN(out_node, G, homs):
    func_node = list(G.predecessors(out_node))[0]
    inputs = list(G.predecessors(func_node))
    if len(inputs) == 0:
        return []

    input_str = f"otimes({', '.join(inputs)})"
    statements = list()
    compositions = list()
    for var_node in inputs:
        compositions.append(f"compose({var_node}, {out_node})")
        statements.extend(translate_GrFN(var_node, G))
    statements.extend()(compositions)
    statements.append(f"{out_node} = Hom({func_node}, {input_str}, {out_node})")
    homs.append(out_node)
    return statements


def to_wiring_diagram(G, filename):
    header = [
        "using Catlab",
        "using Catlab.WiringDiagrams",
        "using Catlab.Doctrines",
        "import Catlab.Doctrines.⊗",
        "import Base: ∘",
        "⊗(a::WiringDiagram, b::WiringDiagram) = otimes(a, b)",
        "∘(a::WiringDiagram, b::WiringDiagram) = compose(b, a)",
        "⊚(a,b) = b ∘ a"
    ]
    cur_node = G.output_node
    homs = []
    statements = translate_GrFN(cur_node, G, homs)
    statements.append(f"WiringDiagram({', '.join(homs)})")

    with open(filename, "w+") as outfile:
        outfile.write("\n".join(header))
        outfile.write("\n".join(statements))


if __name__ == '__main__':
    main()
