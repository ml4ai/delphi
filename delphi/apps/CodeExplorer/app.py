import os
import sys
import json
from uuid import uuid4
import subprocess as sp
from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
    For2PyError,
)
from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.analysis import S2_surface, get_max_s2_sensitivity
from delphi.GrFN.sensitivity import sobol_analysis
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
from pygments.lexers import PythonLexer, JsonLexer
from pygments.formatters import HtmlFormatter

import plotly.graph_objs as go
import plotly

import pandas as pd

from sympy import latex, sympify

os.makedirs("/tmp/automates/", exist_ok = True)
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
GRFN_WITH_ALIGNMENTS = os.path.join(THIS_FOLDER, "grfn_with_alignments.json")
TMPDIR = "/tmp/automates"
sys.path.insert(0, TMPDIR)

BOUNDS = {
    "petpt::msalb_-1": [0, 1],
    "petpt::srad_-1": [1, 20],
    "petpt::tmax_-1": [-30, 60],
    "petpt::tmin_-1": [-30, 60],
    "petpt::xhlai_-1": [0, 20],
    "petasce::doy_-1": [1, 365],
    "petasce::meevp_-1": [0, 1],
    "petasce::msalb_-1": [0, 1],
    "petasce::srad_-1": [1, 30],
    "petasce::tmax_-1": [-30, 60],
    "petasce::tmin_-1": [-30, 60],
    "petasce::xhlai_-1": [0, 20],
    "petasce::tdew_-1": [-30, 60],
    "petasce::windht_-1": [
        0.1,
        10,
    ],  # HACK: has a hole in 0 < x < 1 for petasce__assign__wind2m_1
    "petasce::windrun_-1": [0, 900],
    "petasce::xlat_-1": [3, 12],  # HACK: south sudan lats
    "petasce::xelev_-1": [0, 6000],
    "petasce::canht_-1": [0.001, 3],
}

PRESETS = {
    "petpt::msalb_-1": 0.5,
    "petpt::srad_-1": 10,
    "petpt::tmax_-1": 20,
    "petpt::tmin_-1": 10,
    "petpt::xhlai_-1": 10,
}

with open(GRFN_WITH_ALIGNMENTS, "r") as f:
    tr_dict = json.load(f)
    tr_dict_processed = {}
    variables = {v.pop("name"): v for v in tr_dict["variables"][0]}
    alignments = tr_dict["alignments"][0]
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


class MyForm(FlaskForm):
    source_code = CodeMirrorField(
        language="fortran",
        config={"lineNumbers": "true", "viewportMargin": 800},
    )
    submit = SubmitField("Submit", render_kw={"class": "btn btn-primary"})


PYTHON_LEXER = PythonLexer()
PYTHON_FORMATTER = HtmlFormatter()

JSON_LEXER = JsonLexer()
JSON_FORMATTER = HtmlFormatter()


SECRET_KEY = "secret!"
# mandatory
CODEMIRROR_LANGUAGES = ["fortran"]
# optional
CODEMIRROR_ADDONS = (("display", "placeholder"),)

app = Flask(__name__)
app.config.from_object(__name__)
codemirror = CodeMirror(app)


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
        src = inspect.getsource(n[1]["lambda_fn"])
        src_lines = src.split("\n")
        ltx = (
            src_lines[0].split("__")[2].split("(")[0].replace("_", "\_")
            + " = "
            + latex(
                sympify(src_lines[1][10:].replace("math.", "")),
                mul_symbol="dot",
            ).replace("_", "\_")
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
    return tooltip


@app.route("/")
def index():
    form = MyForm()
    if form.validate_on_submit():
        form.source_code.data
    return render_template("index.html", form=form, code="")


@app.errorhandler(For2PyError)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    flash(response.json["message"])
    return render_template("index.html", code=app.code)


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
    outputDict = translate.XMLToJSONTranslator().analyze(trees, comments)
    pySrc = pyTranslate.create_python_source_list(outputDict)[0][0]

    lambdas = f"{filename}_lambdas"
    lambdas_path = f"/tmp/automates/{lambdas}.py"
    G = GroundedFunctionNetwork.from_python_src(
        pySrc, lambdas_path, f"{filename}.json", filename, save_file=False
    )

    try:
        args = G.inputs
        Si = sobol_analysis(
            G,
            10,
            {
                "num_vars": len(args),
                "names": args,
                "bounds": [BOUNDS[arg] for arg in args],
            },
        )
        S2 = Si["S2"]
        (s2_max, v1, v2) = get_max_s2_sensitivity(S2)

        x_var = args[v1]
        y_var = args[v2]
        search_space = [(x_var, BOUNDS[x_var]), (y_var, BOUNDS[y_var])]
        preset_vals = {
            arg: PRESETS[arg]
            for i, arg in enumerate(args)
            if i != v1 and i != v2
        }

        (X, Y, Z) = S2_surface(G, (8, 6), search_space, preset_vals)
        z_data = pd.DataFrame(Z, index=X, columns=Y)
        data = [go.Surface(z=z_data.as_matrix(), colorscale="Viridis")]
        layout = dict(
            title="S2 sensitivity surface",
            scene=dict(
                xaxis=dict(title=x_var.split("::")[1]),
                yaxis=dict(title=y_var.split("::")[1]),
                zaxis=dict(title=G.output_node.split("::")[1]),
            ),
            autosize=True,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    except KeyError:
        flash(
            "Bounds information was not found for some variables,"
            "so the S2 sensitivity surface cannot be produced for this "
            "code example."
        )
        graphJSON = "{}"
        layout = "{}"

    scopeTree_elementsJSON = to_cyjs_grfn(G)
    CAG = G.to_CAG()
    program_analysis_graph_elementsJSON = to_cyjs_cag(CAG)

    os.remove(input_code_tmpfile)
    os.remove(f"/tmp/automates/{lambdas}.py")

    return render_template(
        "index.html",
        form=form,
        code=app.code,
        python_code=highlight(pySrc, PYTHON_LEXER, PYTHON_FORMATTER),
        grfn_json=highlight(
            json.dumps(outputDict, indent=2), JSON_LEXER, JSON_FORMATTER
        ),
        scopeTree_elementsJSON=scopeTree_elementsJSON,
        graphJSON=graphJSON,
        layout=json.dumps(layout),
        program_analysis_graph_elementsJSON=program_analysis_graph_elementsJSON,
    )


@app.route("/modelAnalysis")
def modelAnalysis():
    PETPT_GrFN = GroundedFunctionNetwork.from_fortran_file(
        THIS_FOLDER + "/static/example_programs/petPT.f", tmpdir=TMPDIR
    )
    PETASCE_GrFN = GroundedFunctionNetwork.from_fortran_file(
        THIS_FOLDER + "/static/example_programs/petASCE.f", tmpdir=TMPDIR
    )
    FIB = PETPT_GrFN.to_FIB(PETASCE_GrFN)
    return render_template(
        "modelAnalysis.html",
        petpt_elementsJSON=to_cyjs_cag(PETPT_GrFN.to_CAG()),
        petasce_elementsJSON=to_cyjs_cag(PETASCE_GrFN.to_CAG()),
        fib_elementsJSON=to_cyjs_fib(FIB),
    )


if __name__ == "__main__":
    app.run()
