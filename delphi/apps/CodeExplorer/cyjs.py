import os
import sys
import json
import inspect
from sympy import latex, sympify
from pygments.lexers import PythonLexer, JsonLexer
from pygments import highlight
from pygments.formatters import HtmlFormatter

PYTHON_LEXER = PythonLexer()
PYTHON_FORMATTER = HtmlFormatter()
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
GRFN_WITH_ALIGNMENTS = os.path.join(
    THIS_FOLDER, "petasce_with_alignments.json"
)
sys.path.insert(0, "/tmp/automates")


def process_tr_dicts():
    grfn_path = os.path.join(THIS_FOLDER, "grfn_with_alignments.json")
    petasce_path = os.path.join(THIS_FOLDER, "petasce_with_alignments_2.json")

    tr_dict_1 = json.load(open(grfn_path, "r", encoding="utf-8"))
    tr_dict_2 = json.load(open(petasce_path, "r", encoding="utf-8"))

    tr_dict = {
        "variables": tr_dict_1["variables"][0] + tr_dict_2["variables"][0],
        "alignments": tr_dict_1["alignments"][0] + tr_dict_2["alignments"][0],
    }

    tr_dict_processed = {}
    variables = {v.pop("name"): v for v in tr_dict["variables"]}
    alignments = tr_dict["alignments"]
    src_comment_alignments = {
        alignment["src"]: alignment["dst"]
        for alignment in alignments
        if "_COMMENT" in alignment["dst"] and alignment["score"] == 1
    }
    comment_text_alignments = {
        alignment["src"]: [
            a["dst"] for a in alignments if a["src"] == alignment["src"]
        ][0]
        for alignment in alignments
    }
    src_text_alignments = {
        src: {
            "from_comments": variables[comment],
            "from_text": variables[comment_text_alignments[comment]],
        }
        for src, comment in src_comment_alignments.items()
    }
    return (
        src_comment_alignments,
        comment_text_alignments,
        src_text_alignments,
    )


src_comments_alignments, comment_text_alignments, src_text_alignments = (
    process_tr_dicts()
)


def get_tooltip(n):
    if n[1]["type"] == "variable":
        metadata = src_text_alignments.get(n[1]["basename"])
        if metadata is not None:
            comment_provenance = metadata["from_comments"]
            text_provenance = metadata["from_text"]
            tooltip = """
            <strong>Metadata extracted using NLP</strong>
            <nav>
                <div class="nav nav-tabs" id="nav-tab-{n[0]}" role="tablist">
                    <a class="nav-item nav-link active" id="nav-comments-tab-{n[0]}"
                        data-toggle="tab" href="#nav-comments-{n[0]}" role="tab"
                        aria-controls="nav-comments-{n[0]}" aria-selected="true">
                        Code comments
                    </a>
                    <a class="nav-item nav-link" id="nav-text-tab-{n[0]}"
                        data-toggle="tab" href="#nav-text-{n[0]}" role="tab"
                        aria-controls="nav-text-{n[0]}" aria-selected="false">
                        Scientific texts
                    </a>
                </div>
            </nav>
            <div class="tab-content" id="nav-tabContent" style="padding-top:1rem; padding-bottom: 0.5rem;">
                <div class="tab-pane fade show active" id="nav-comments-{n[0]}"
                    role="tabpanel" aria-labelledby="nav-comments-tab-{n[0]}">
                    <table style="width:100%">
                        <tr><td><strong>Text</strong>:</td> <td> {from_comments[description][0][text]} </td></tr>
                        <tr><td><strong>Source</strong>:</td> <td> {from_comments[description][0][source]} </td></tr>
                        <tr><td><strong>Sentence ID</strong>:</td> <td> {from_comments[description][0][sentIdx]} </td></tr>
                    </table>
                </div>
                <div class="tab-pane fade" id="nav-text-{n[0]}" role="tabpanel"
                    aria-labelledby="nav-text-tab-{n[0]}">
                    <table style="width:100%">
                        <tr><td><strong>Text</strong>:</td> <td> {from_text[description][0][text]} </td></tr>
                        <tr><td><strong>Source</strong>:</td> <td> {from_text[description][0][source]} </td></tr>
                        <tr><td><strong>Sentence ID</strong>:</td> <td> {from_text[description][0][sentIdx]} </td></tr>
                    </table>
                </div>
            </div>
            """.format(
                n=n,
                metadata=metadata,
                from_comments=comment_provenance,
                from_text=text_provenance,
            )
        else:
            tooltip = None

    else:
        try:
            func_name = n[1]["lambda_fn"].__name__
            out_name = func_name.split("__")[-2].replace("_", "\_")
            src = inspect.getsource(n[1]["lambda_fn"])
            src_lines = src.split("\n")
            computation = src_lines[1].replace("return", "").strip().replace("math.", "")

            ltx = (out_name + " = " +
                   latex(sympify(computation), mul_symbol="dot",).replace("_", "\_")
                   )

            tooltip = """
            <nav>
                <div class="nav nav-tabs" id="nav-tab-{n}" role="tablist">
                    <a class="nav-item nav-link active" id="nav-eq-tab-{n}"
                        data-toggle="tab" href="#nav-eq-{n}" role="tab"
                        aria-controls="nav-eq-{n}" aria-selected="true">
                        Equation
                    </a>
                    <a class="nav-item nav-link" id="nav-code-tab-{n}"
                        data-toggle="tab" href="#nav-code-{n}" role="tab"
                        aria-controls="nav-code-{n}" aria-selected="false">
                        Lambda Function
                    </a>
                </div>
            </nav>
            <div class="tab-content" id="nav-tabContent" style="padding-top:1rem; padding-bottom: 0.5rem;">
                <div class="tab-pane fade show active" id="nav-eq-{n}"
                    role="tabpanel" aria-labelledby="nav-eq-tab-{n}">
                    \({ltx}\)
                </div>
                <div class="tab-pane fade" id="nav-code-{n}" role="tabpanel"
                    aria-labelledby="nav-code-tab-{n}">
                    {src}
                </div>
            </div>
            """.format(
                ltx=ltx, src=highlight(src, PYTHON_LEXER, PYTHON_FORMATTER), n=n
            )
        except Exception as e:
            print(e)
            return ""
    return tooltip


def to_cyjs_grfn(G):
    elements = {
        "nodes": [
            {
                "data": {
                    "id": n[0],
                    "label": n[1]["label"],
                    "parent": n[1]["parent"],
                    "shape": "ellipse"
                    if n[1].get("type") == "variable"
                    else "rectangle",
                    "color": "maroon"
                    if n[1].get("type") == "variable"
                    else "black",
                    "textValign": "center",
                    "tooltip": get_tooltip(n),
                    "width": 10 if n[1].get("type") == "variable" else 7,
                    "height": 10 if n[1].get("type") == "variable" else 7,
                    "padding": n[1]["padding"],
                }
            }
            for n in G.nodes(data=True)
        ]
        + [
            {
                "data": {
                    "id": n[0],
                    "label": n[0],
                    "shape": "roundrectangle",
                    "color": n[1]["color"],
                    "textValign": "top",
                    "tooltip": n[0],
                    "width": "label",
                    "height": "label",
                    "padding": 10,
                    "parent": (
                        list(G.scope_tree.predecessors(n[0]))[0]
                        if len(list(G.scope_tree.predecessors(n[0]))) != 0
                        else n[0]
                    ),
                }
            }
            for n in G.scope_tree.nodes(data=True)
        ],
        "edges": [
            {
                "data": {
                    "id": f"{edge[0]}_{edge[1]}",
                    "source": edge[0],
                    "target": edge[1],
                }
            }
            for edge in G.edges()
        ],
    }
    json_str = json.dumps(elements, indent=2)
    return json_str


def to_cyjs_cag(G):
    elements = {
        "nodes": [
            {
                "data": {
                    "id": n[0],
                    "label": n[0],
                    "parent": "parent",
                    "shape": "ellipse",
                    "color": "maroon",
                    "textValign": "center",
                    "tooltip": get_tooltip(n),
                    "width": "label",
                    "height": "label",
                    "padding": 15,
                }
            }
            for n in G.nodes(data=True)
        ],
        "edges": [
            {
                "data": {
                    "id": f"{edge[0]}_{edge[1]}",
                    "source": edge[0],
                    "target": edge[1],
                }
            }
            for edge in G.edges()
        ],
    }
    json_str = json.dumps(elements, indent=2)
    return json_str


def to_cyjs_fib(G):
    elements = {
        "nodes": [
            {
                "data": {
                    "id": n[0],
                    "label": n[1]["label"],
                    "parent": n[1]["parent"],
                    "shape": "ellipse"
                    if n[1].get("type") == "variable"
                    else "rectangle",
                    "color": n[1].get("color", "black"),
                    "textValign": "center",
                    "tooltip": get_tooltip(n),
                    "width": 10 if n[1].get("type") == "variable" else 7,
                    "height": 10 if n[1].get("type") == "variable" else 7,
                    "padding": n[1]["padding"],
                }
            }
            for n in G.nodes(data=True)
        ],
        "edges": [
            {
                "data": {
                    "id": f"{edge[0]}_{edge[1]}",
                    "source": edge[0],
                    "target": edge[1],
                }
            }
            for edge in G.edges()
        ],
    }
    json_str = json.dumps(elements, indent=2)
    return json_str
