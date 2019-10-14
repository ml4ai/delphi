import os
import sys
import json
from uuid import uuid4
from datetime import datetime
import subprocess as sp
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request, redirect
from flask import url_for, jsonify, flash
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

from delphi.translators.for2py import preprocessor, translate, get_comments
from delphi.translators.for2py import pyTranslate, genPGM, For2PyError
from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.linking import make_link_tables


os.makedirs("/tmp/automates/", exist_ok=True)
os.makedirs("/tmp/automates/input_code/", exist_ok=True)
TMPDIR = "/tmp/automates"
sys.path.insert(0, TMPDIR)
SECRET_KEY = "secret!"
# mandatory
CODEMIRROR_LANGUAGES = ["fortran"]
# optional
CODEMIRROR_ADDONS = (("display", "placeholder"),)

app = Flask(__name__)
app.config.from_object(__name__)
codemirror = CodeMirror(app)

SOURCE_FILES = "/Users/phein/Google Drive/ASKE-AutoMATES/Data/source_model_files"


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/get_saved_materials', methods=["GET"])
def get_saved_materials():
    code_files = os.listdir(os.path.join(SOURCE_FILES, "code"))
    docs_files = [f for f in os.listdir(os.path.join(SOURCE_FILES, "docs"))
                  if f.endswith(".pdf")]
    model_files = [f for f in os.listdir(os.path.join(SOURCE_FILES, "models"))
                   if f.endswith(".json")]
    return jsonify({
        "code": code_files,
        "docs": docs_files,
        "models": model_files
    })


@app.route("/get_GrFN", methods=["POST"])
def get_GrFN():
    model_name = request.form["model_json"]
    return NotImplemented
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


@app.route("/get_link_table", methods=["POST"])
def get_link_table():
    model_name = request.form["model_json"]
    grfn = json.load(open(f"{SOURCE_FILES}/models/{model_name}"))
    return jsonify({str(k): v for k, v in make_link_tables(grfn).items()})


@app.route('/upload_doc', methods=["POST"])
def upload_doc():
    return NotImplemented


def main():
    app.run(host='0.0.0.0', port=80, debug=True)


if __name__ == "__main__":
    main()
