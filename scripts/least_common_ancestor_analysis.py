import inspect
import Levenshtein

# from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
import networkx as nx
import numpy as np

from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN


def main():
    PNO_GrFN = GrFN.from_fortran_file(
        f"../tests/data/program_analysis/PETPNO.for"
    )
    PEN_GrFN = GrFN.from_fortran_file(
        f"../tests/data/program_analysis/PETPEN.for"
    )

    # Use basenames for variable comparison because the two GrFNs will have those in common
    PNO_nodes = [
        d["basename"]
        for n, d in PNO_GrFN.nodes(data=True)
        if d["type"] == "variable"
    ]
    PEN_nodes = [
        d["basename"]
        for n, d in PEN_GrFN.nodes(data=True)
        if d["type"] == "variable"
    ]

    shared_nodes = list(set(PNO_nodes).intersection(set(PEN_nodes)))
    # Make a map so we can access the original variable names from the basenames
    PNO_input_map = {get_basename(node): node for node in PNO_GrFN.inputs}
    PEN_input_map = {get_basename(node): node for node in PEN_GrFN.inputs}

    PNO_inputs = list(PNO_input_map.keys())
    PEN_inputs = list(PEN_input_map.keys())

    # Reverse the graph so that LCA analysis will work
    mock_PNO_GrFN = nx.DiGraph()
    mock_PNO_GrFN.add_edges_from([(dst, src) for src, dst in PNO_GrFN.edges])

    mock_PEN_GrFN = nx.DiGraph()
    mock_PEN_GrFN.add_edges_from([(dst, src) for src, dst in PEN_GrFN.edges])

    # Find both sets of shared inputs
    shared_input_nodes = list(set(PNO_inputs).intersection(set(shared_nodes)))

    for i, v1 in enumerate(shared_input_nodes):
        for v2 in shared_input_nodes[i + 1 :]:
            (L1, L2) = pairwise_LCAs(
                mock_PNO_GrFN,
                mock_PEN_GrFN,
                PNO_input_map,
                PEN_input_map,
                v1,
                v2,
            )
            if L1 is None and L2 is None:
                print(f"SHARED: {v1}, {v2}\t\tFAILED\n\n")
                continue
            ((L1, L2), LD) = lambda_levenshtein_dist(
                PNO_GrFN, PEN_GrFN, L1, L2
            )
            print(f"SHARED: {v1}, {v2}\tLev Dist: {LD}")
            print(f"LAMBDAS:\n\t{v1}: {L1}\n\t{v2}: {L2}\n\n")


def get_basename(node_name):
    (_, _, _, _, basename, _) = node_name.split("::")
    return basename


def stringified_lambda(source_ref):
    """Use inspect.getsourcelines() to grab the stringified lambda functions"""
    (code, _) = inspect.getsourcelines(source_ref)
    lines = ("".join(code)).split("\n")
    important_lines = ";".join(lines[1:])
    return important_lines


def pairwise_LCAs(G1, G2, imap1, imap2, v1, v2):
    ivar11, ivar12 = imap1[v1], imap1[v2]
    ivar21, ivar22 = imap2[v1], imap2[v2]
    # Get the actual function code for this node via the attribute "lambda_fn"

    LCA_G1 = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
        G1, ivar11, ivar12
    )
    LCA_G2 = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
        G2, ivar21, ivar22
    )

    return LCA_G1, LCA_G2


def lambda_levenshtein_dist(G1, G2, LCA1, LCA2):
    G1_lambda = G1.nodes[LCA1]["lambda_fn"]
    G2_lambda = G2.nodes[LCA2]["lambda_fn"]

    lam_str1 = stringified_lambda(G1_lambda)
    lam_str2 = stringified_lambda(G2_lambda)
    lev_dist = Levenshtein.distance(lam_str1, lam_str2)
    return (lam_str1, lam_str2), lev_dist


# Recover the differences between the strings with dynamic programming
def editdistDP(fn1, fn2):
    M, N = (
        len(fn1),
        len(fn2),
    )

    DP = np.zeros((M + 1, N + 1), dtype=int)

    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0:
                DP[i][j] = j
            elif j == 0:
                DP[i][j] = i
            elif fn1[i - 1] == fn2[j - 1]:
                DP[i][j] = DP[i - 1][j - 1]
            else:
                DP[i][j] = 1 + min(
                    DP[i][j - 1], DP[i - 1][j], DP[i - 1][j - 1]
                )

    return DP[M][N]


if __name__ == "__main__":
    main()
