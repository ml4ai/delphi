import networkx as nx


def make_link_tables(GrFN):
    # Add links to the link-graph G from the grounding data. Add all variables
    # found to a list of vars. (Use the vars list to walk the graph in a DFS
    # style to recover all rows for the table)
    vars = list()
    G = nx.Graph()
    for link_dict in GrFN["grounding"]:
        id1 = get_id(link_dict["element_1"])
        id2 = get_id(link_dict["element_2"])
        if id1[0] == "<VAR>":
            vars.append(id1)
        if id2[0] == "<VAR>":
            vars.append(id2)
        G.add_edge("\n".join(id1), "\n".join(id2), label=round(link_dict["score"], 3))

    table_data = dict()
    for var in vars:
        var_name = "\n".join(var)
        var_table = list()
        for n1 in G.neighbors(var_name):
            if n1.startswith("<CMS>"):
                for n2 in G.neighbors(n1):
                    if n2.startswith("<TXT>"):
                        for n3 in G.neighbors(n2):
                            if n3.startswith("<EQN>"):
                                var_table.append({
                                    "link_score": min([
                                        G[var_name][n1]["label"],
                                        G[n1][n2]["label"],
                                        G[n2][n3]["label"]
                                    ]),
                                    "comm": n1.split("\n")[1],
                                    "vc_score": G[var_name][n1]["label"],
                                    "txt": n2.split("\n")[1],
                                    "ct_score": G[n1][n2]["label"],
                                    "eqn": n3.split("\n")[1],
                                    "te_score": G[n2][n3]["label"]
                                })
        var_table.sort(
            key=lambda r: (r["link_score"],
                           r["vc_score"],
                           r["ct_score"],
                           r["te_score"]),
            reverse=True
        )
        table_data[var] = var_table
    return table_data


def get_argument_lists(GrFN):
    # Make a dict of the argument lists for each container in the GrFN indexed
    # by the container basename
    return {
        c["name"].split("::")[-1]: c["arguments"] for c in GrFN["containers"]
    }


def get_call_conventions(GrFN):
    # Make a dict of all of the calls in every container in the GrFN. Index them
    # by basename of the callee. Include the caller basename and input list in
    # the value field
    return {
        stmt["function"]["name"].split("::")[-1]: {
            "caller": container["name"].split("::")[-1],
            "inputs": stmt["input"]
        }
        for container in GrFN["containers"]
        for stmt in container["body"]
        if stmt["function"]["type"] == "function_name"
    }


def output_link_graph(G, filename="linking-graph.pdf"):
    A = nx.nx_agraph.to_AGraph(G)
    A.draw(filename, prog="circo")


def print_table_data(table_data):
    for (_, scope, name, idx), table in table_data.items():
        print("::".join([scope, name, idx]))
        print("L-SCORE\tComment\tV-C\tText-span\tC-T\tEquation\tT-E")
        for row in table:
            print(f"{row['link_score']}\t{row['comm']}\t{row['vc_score']}\t{row['txt']}\t{row['ct_score']}\t{row['eqn']}\t{row['te_score']}")
        print("\n\n")


def merge_similar_vars(vars):
    unique_vars = dict()
    for (_, scope, name, idx) in vars:
        if (scope, name) in unique_vars:
            unique_vars[(scope, name)].append(int(idx))
        else:
            unique_vars[(scope, name)] = [int(idx)]
    return unique_vars


def format_long_text(text):
    new_text = list()
    while len(text) > 8:
        new_text.extend(text[:4])
        new_text.append("\n")
        text = text[4:]
    new_text.extend(text)
    return new_text


def get_id(el_data):
    el_type = el_data["type"]
    if el_type == "identifier":
        var_name = el_data["content"].split("::")[-3:]
        return tuple(["<VAR>"] + var_name)
    elif el_type == "comment_span":
        tokens = el_data["content"].split()
        name = tokens[0]
        desc = " ".join(format_long_text(tokens[1:]))
        return ("<CMS>", f"{name}: {desc}")
        return ("<CMS>", name, desc)
    elif el_type == "text_span":
        desc = " ".join(format_long_text(el_data["content"].split()))
        desc = el_data["content"]
        return ("<TXT>", desc)
    elif el_type == "equation_span":
        desc = " ".join(format_long_text(el_data["content"].split()))
        desc = el_data["content"]
        return ("<EQN>", desc)
    else:
        raise ValueError(f"Unrecognized link type: {el_type}")
