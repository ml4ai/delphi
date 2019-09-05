import json
from pygraphviz import AGraph


def main():
    grfn = json.load(open("SIR-Gillespie-SD_GrFN_with_groundings_eq.json", "r"))
    variables = [v["name"] for v in grfn["variables"]]
    potential_links = grfn["grounding"]
    potential_links.sort(key=lambda l: l["score"], reverse=True)

    G = AGraph()
    for link_dict in potential_links:
        id1 = get_id(link_dict["element_1"])
        id2 = get_id(link_dict["element_2"])
        G.add_edge(id1, id2, label=round(link_dict["score"], 3))
    G.draw("linking-graph.pdf", prog="circo")


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
        var_name = el_data["content"].split("::")[-2]
        return f"<VAR>\n{var_name}"
    elif el_type == "comment_span":
        tokens = el_data["content"].split()
        name = tokens[0]
        desc = " ".join(format_long_text(tokens[1:]))
        return f"<CMS>\n{name}\n{desc}"
    elif el_type == "text_span":
        desc = " ".join(format_long_text(el_data["content"].split()))
        return f"<TXT>\n{desc}"
    elif el_type == "equation_span":
        desc = " ".join(format_long_text(el_data["content"].split()))
        return f"<EQN>\n{desc}"
    else:
        raise ValueError(f"Unrecognized link type: {el_type}")


if __name__ == '__main__':
    main()
