import os
import sys
import ast
import json
from uuid import uuid4
import subprocess as sp
import importlib
from pprint import pprint
from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
    For2PyError,
)
from delphi.utils.fp import flatten
from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.utils import NodeType, get_node_type
import delphi.paths
import xml.etree.ElementTree as ET
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    flash,
)
from flask_wtf import FlaskForm
from flask_codemirror.fields import CodeMirrorField
from wtforms.fields import SubmitField
from flask_codemirror import CodeMirror
import inspect
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from sympy import sympify, latex, symbols


class MyForm(FlaskForm):
    source_code = CodeMirrorField(
        language="fortran",
        config={"lineNumbers": "true", "viewportMargin": 800},
    )
    submit = SubmitField("Submit", render_kw={"class": "btn btn-primary"})


LEXER = PythonLexer()
FORMATTER = HtmlFormatter()


SECRET_KEY = "secret!"
# mandatory
CODEMIRROR_LANGUAGES = ["fortran"]
# optional
CODEMIRROR_ADDONS = (("display", "placeholder"),)

app = Flask(__name__)
app.config.from_object(__name__)
codemirror = CodeMirror(app)


def get_cluster_nodes(A):
    cluster_nodes = []
    for subgraph in A.subgraphs():
        cluster_nodes.append(
            {
                "data": {
                    "id": subgraph.name,
                    "label": subgraph.name.replace("cluster_", ""),
                    "shape": "roundrectangle",
                    "parent": A.name,
                    "color": subgraph.graph_attr["border_color"],
                    "textValign": "top",
                    "tooltip": None,
                }
            }
        )
        cluster_nodes.append(get_cluster_nodes(subgraph))

    return cluster_nodes


def get_tooltip(n, lambdas):
    if n.attr["node_type"] == "ActionNode":
        x = getattr(lambdas, n.attr["lambda_fn"], None)
        if x is None:
            return "None"
        else:
            src = inspect.getsource(x)
            src_lines = src.split("\n")
            symbs = src_lines[0].split("(")[1].split(")")[0].split(", ")
            ltx = (
                src_lines[0].split("__")[2].split("(")[0].replace("_", "\_")
                + " = "
                + latex(
                    sympify(src_lines[1][10:].replace("math.exp", "e^"))
                ).replace("_", "\_")
            )
            return """
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
                ltx=ltx, src=highlight(src, LEXER, FORMATTER), n=n
            )
    else:
        return json.dumps({"index": n.attr["index"]}, indent=2)


def get_cyjs_elementsJSON_from_ScopeTree(A, lambdas) -> str:
    elements = {
        "nodes": [
            {
                "data": {
                    "id": n,
                    "label": n.attr["label"],
                    "parent": n.attr["parent"],
                    "shape": n.attr["shape"],
                    "color": n.attr["color"],
                    "textValign": "center",
                    "tooltip": get_tooltip(n, lambdas)
                    # "tooltip": highlight(get_tooltip(n, lambdas), LEXER, FORMATTER),
                }
            }
            for n in A.nodes()
        ]
        + flatten(get_cluster_nodes(A)),
        "edges": [
            {
                "data": {
                    "id": f"{edge[0]}_{edge[1]}",
                    "source": edge[0],
                    "target": edge[1],
                }
            }
            for edge in A.edges()
        ],
    }
    json_str = json.dumps(elements, indent=2)
    return json_str


@app.route("/")
def index():
    form = MyForm()
    if form.validate_on_submit():
        text = form.source_code.data
    return render_template("index.html", form=form, code="")


@app.errorhandler(For2PyError)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    flash(response.json["message"])
    return render_template("index.html", code=app.code)


def to_cyjs(G):
    elements = {
        "nodes": [
            {
                "data": {
                    "id": n,
                    "label": n,
                    "parent": "parent",
                    "shape": n.attr["shape"],
                    "color": n.attr["color"],
                    "textValign": "center",
                    "tooltip": "tooltips",
                }
            }
            for n in G.nodes()
        ],
        # + flatten(get_cluster_nodes(A)),
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


@app.route("/processCode", methods=["POST"])
def processCode():
    form = MyForm()
    code = form.source_code.data
    app.code = code
    if code == "":
        return render_template("index.html", form=form)
    lines = [
        line.replace("\r", "") + "\n"
        for line in [line for line in code.split("\n")]
        if line != ""
    ]
    filename = f"input_code_{str(uuid4())}"
    input_code_tmpfile = f"/tmp/automates/{filename}.f"

    with open(input_code_tmpfile, "w") as f:
        f.write(preprocessor.process(lines))

    xml_string = sp.run(
        [
            "java",
            "fortran.ofp.FrontEnd",
            "--class",
            "fortran.ofp.XMLPrinter",
            "--verbosity",
            "0",
            input_code_tmpfile,
        ],
        stdout=sp.PIPE,
    ).stdout

    trees = [ET.fromstring(xml_string)]
    comments = get_comments.get_comments(input_code_tmpfile)
    xml_to_json_translator = translate.XMLToJSONTranslator()
    outputDict = xml_to_json_translator.analyze(trees, comments)
    pySrc = pyTranslate.create_python_source_list(outputDict)[0][0]

    lambdas = f"{filename}_lambdas"
    lambdas_path = f"/tmp/automates/{lambdas}.py"
    sys.path.insert(0, "/tmp/automates")
    G = GroundedFunctionNetwork.from_python_src(
        pySrc, lambdas_path, f"{filename}.json", filename
    )

    scopeTree_elementsJSON = to_cyjs(G.to_agraph())
    program_analysis_graph_elementsJSON = to_cyjs(G.to_CAG_agraph())
    os.remove(input_code_tmpfile)
    os.remove(f"/tmp/automates/{lambdas}.py")

    return render_template(
        "index.html",
        form=form,
        code=app.code,
        python_code=highlight(pySrc, LEXER, FORMATTER),
        scopeTree_elementsJSON=scopeTree_elementsJSON,
        program_analysis_graph_elementsJSON=program_analysis_graph_elementsJSON,
    )


@app.route("/modelAnalysis")
def modelAnalysis():
    import delphi.analysis.comparison.utils as utils
    from delphi.analysis.comparison.ForwardInfluenceBlanket import (
        ForwardInfluenceBlanket,
    )

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    asce = utils.nx_graph_from_dotfile(
        os.path.join(THIS_FOLDER, "static/graphviz_dot_files/asce-graph.dot")
    )
    pt = utils.nx_graph_from_dotfile(
        os.path.join(
            THIS_FOLDER, "static/graphviz_dot_files/priestley-taylor-graph.dot"
        )
    )
    shared_nodes = utils.get_shared_nodes(asce, pt)

    cmb_asce = ForwardInfluenceBlanket(asce, shared_nodes).cyjs_elementsJSON()
    cmb_pt = ForwardInfluenceBlanket(pt, shared_nodes).cyjs_elementsJSON()

    return render_template(
        "modelAnalysis.html",
        model1_elementsJSON=cmb_asce,
        model2_elementsJSON=cmb_pt,
    )


if __name__ == "__main__":
    app.run()
