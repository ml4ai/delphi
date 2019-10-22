import os
import sys
import json
from uuid import uuid4
from datetime import datetime
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
from pygments import highlight

from delphi.apps.CodeExplorer.surface_plots import (
    get_grfn_surface_plot,
    get_fib_surface_plot,
)
from delphi.apps.CodeExplorer.cyjs import (
    to_cyjs_grfn,
    to_cyjs_cag,
    to_cyjs_fib,
    PYTHON_LEXER,
    THIS_FOLDER,
    PYTHON_FORMATTER,
)


os.makedirs("/tmp/automates/", exist_ok=True)
os.makedirs("/tmp/automates/input_code/", exist_ok=True)
TMPDIR = "/tmp/automates"
sys.path.insert(0, TMPDIR)


class MyForm(FlaskForm):
    source_code = CodeMirrorField(
        language="fortran",
        config={"lineNumbers": "true", "viewportMargin": 800},
    )
    submit = SubmitField("Submit", render_kw={"class": "btn btn-primary"})


SECRET_KEY = "secret!"
# mandatory
CODEMIRROR_LANGUAGES = ["fortran"]
# optional
CODEMIRROR_ADDONS = (("display", "placeholder"),)

app = Flask(__name__)
app.config.from_object(__name__)
codemirror = CodeMirror(app)


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

    # dir_name = str(uuid4())
    # os.mkdir(f"/tmp/automates/input_code/{dir_name}")
    # input_code_tmpfile = f"/tmp/automates/input_code/{dir_name}/{orig_file}.f"
    filename = f"input_code_{str(uuid4()).replace('-', '_')}"
    input_code_tmpfile = f"/tmp/automates/{filename}.f"
    with open(input_code_tmpfile, "w") as f:
        f.write(preprocessor.process(lines))

    lambdas = f"{filename}_lambdas"
    lambdas_path = f"/tmp/automates/{lambdas}.py"
    G = GroundedFunctionNetwork.from_fortran_file(input_code_tmpfile,
                                                  tmpdir="/tmp/automates/")

    graphJSON, layout = get_grfn_surface_plot(G)

    scopeTree_elementsJSON = to_cyjs_grfn(G)
    CAG = G.to_CAG()
    program_analysis_graph_elementsJSON = to_cyjs_cag(CAG)

    os.remove(input_code_tmpfile)
    os.remove(f"/tmp/automates/{lambdas}.py")

    return render_template(
        "index.html",
        form=form,
        code=app.code,
        scopeTree_elementsJSON=scopeTree_elementsJSON,
        graphJSON=graphJSON,
        layout=layout,
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

    PETPT_FIB = PETPT_GrFN.to_FIB(PETASCE_GrFN)
    PETASCE_FIB = PETASCE_GrFN.to_FIB(PETPT_GrFN)

    asce_inputs = {
        "petasce::msalb_-1": 0.5,
        "petasce::srad_-1": 15,
        "petasce::tmax_-1": 10,
        "petasce::tmin_-1": -10,
        "petasce::xhlai_-1": 10,
    }


    asce_covers = {
        "petasce::canht_-1": 2,
        "petasce::meevp_-1": "A",
        "petasce::cht_0": 0.001,
        "petasce::cn_4": 1600.0,
        "petasce::cd_4": 0.38,
        "petasce::rso_0": 0.062320,
        "petasce::ea_0": 7007.82,
        "petasce::wind2m_0": 3.5,
        "petasce::psycon_0": 0.0665,
        "petasce::wnd_0": 3.5,
    }
    # graphJSON, layout = get_fib_surface_plot(PETASCE_FIB, asce_covers, 10)


    return render_template(
        "modelAnalysis.html",
        petpt_elementsJSON=to_cyjs_cag(PETPT_GrFN.to_CAG()),
        petasce_elementsJSON=to_cyjs_cag(PETASCE_GrFN.to_CAG()),
        fib_elementsJSON=to_cyjs_fib(PETASCE_FIB.to_CAG()),
        # layout=layout,
        # graphJSON=graphJSON,
    )


def main():
    app.run(host='0.0.0.0', port=80, debug=True)


if __name__ == "__main__":
    main()
