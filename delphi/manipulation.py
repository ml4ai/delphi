from .AnalysisGraph import AnalysisGraph

# ==========================================================================
# Manipulation
# ==========================================================================


def merge_nodes(
    G: AnalysisGraph, n1: str, n2: str, same_polarity: bool = True
) -> AnalysisGraph:
    """ Merge node n1 into node n2, with the option to specify relative
    polarity.. """

    for p in G.predecessors(n1):
        for st in G[p][n1]["InfluenceStatements"]:
            if not same_polarity:
                st.obj_delta["polarity"] = -st.obj_delta["polarity"]
            st.obj.db_refs["UN"][0] = (
                "/".join(st.obj.db_refs["UN"][0][0].split("/")[:-1] + [n2]),
                st.obj.db_refs["UN"][0][1],
            )

        if not G.has_edge(p, n2):
            G.add_edge(p, n2)
            G[p][n2]["InfluenceStatements"] = G[p][n1]["InfluenceStatements"]

        else:
            G[p][n2]["InfluenceStatements"] += G[p][n1]["InfluenceStatements"]

    for s in G.successors(n1):
        for st in G.edges[n1, s]["InfluenceStatements"]:
            if not same_polarity:
                st.subj_delta["polarity"] = -st.subj_delta["polarity"]
            st.subj.db_refs["UN"][0] = (
                "/".join(st.subj.db_refs["UN"][0][0].split("/")[:-1] + [n2]),
                st.subj.db_refs["UN"][0][1],
            )

        if not G.has_edge(n2, s):
            G.add_edge(n2, s)
            G[n2][s]["InfluenceStatements"] = G[n1][s]["InfluenceStatements"]
        else:
            G[n2][s]["InfluenceStatements"] += G[n1][s]["InfluenceStatements"]

    G.remove_node(n1)
    return G
