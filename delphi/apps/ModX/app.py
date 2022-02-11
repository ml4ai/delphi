import os
import sys
import json
from uuid import uuid4
from datetime import datetime
from pathlib import Path
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

SOURCE_FILES = os.environ["MODEL_FILES"]
sys.path.insert(0, os.path.join(SOURCE_FILES, "code/"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/get_saved_materials', methods=["GET"])
def get_saved_materials():
    code_files = [f for f in os.listdir(os.path.join(SOURCE_FILES, "code"))
                  if f.endswith(".f")]
    docs_files = [f for f in os.listdir(os.path.join(SOURCE_FILES, "docs"))
                  if f.endswith(".pdf")]
    model_files = [f for f in os.listdir(os.path.join(SOURCE_FILES, "models"))
                   if f.endswith(".json")]
    return jsonify({
        "code": code_files,
        "docs": docs_files,
        "models": model_files
    })


@app.route("/process_text_and_code", methods=["POST"])
def process_text_and_code():
    fortran_file = request.form["source_code"]
    basename = fortran_file[:-2]
    pdf_file = request.form["document"]
    conf_file = get_conf_file(pdf_file)

    fortran_path = os.path.join(SOURCE_FILES, "code", fortran_file)
    norm_json_path = os.path.join(SOURCE_FILES, "code", f"{basename}.json")
    if os.path.isfile(norm_json_path):
        os.remove(norm_json_path)

    GroundedFunctionNetwork.from_fortran_file(fortran_path)
    cur_dir = os.getcwd()
    os.chdir(os.path.join(os.environ["AUTOMATES_LOC"], "text_reading/"))
    sp.run([
        "sbt",
        "-Dconfig.file=" + os.path.join(SOURCE_FILES, "configs", conf_file),
        'runMain org.clulab.aske.automates.apps.ExtractAndAlign'
    ])
    os.chdir(cur_dir)
    tr_json_path = os.path.join(SOURCE_FILES, "models", f"{basename}_with_groundings.json")
    norm_json_path = os.path.join(SOURCE_FILES, "models", f"{basename}.json")
    if os.path.isfile(norm_json_path):
        os.remove(norm_json_path)
    if os.path.isfile(tr_json_path):
        os.rename(tr_json_path, norm_json_path)
    grfn = json.load(open(norm_json_path, "r"))
    return jsonify({
        "link_data": {str(k): v for k, v in make_link_tables(grfn).items()},
        "models": [f for f in os.listdir(os.path.join(SOURCE_FILES, "models"))
                   if f.endswith(".json")]
    })


@app.route("/model_comparison")
def model_comparison():
    PETPT_GrFN = GroundedFunctionNetwork.from_fortran_file(
        "static/source_model_files/code/petpt.f", tmpdir=TMPDIR
    )
    PETASCE_GrFN = GroundedFunctionNetwork.from_fortran_file(
        "static/source_model_files/code/petasce.f", tmpdir=TMPDIR
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
        "model_comparison.html",
        petpt_elementsJSON=to_cyjs_cag(PETPT_GrFN.to_CAG()),
        petasce_elementsJSON=to_cyjs_cag(PETASCE_GrFN.to_CAG()),
        fib_elementsJSON=to_cyjs_fib(PETASCE_FIB.to_CAG()),
        # layout=layout,
        # graphJSON=graphJSON,
    )


@app.route("/get_link_table", methods=["POST"])
def get_link_table():
    model_name = request.form["model_json"]
    grfn = json.load(open(os.path.join(SOURCE_FILES, "models", model_name)))
    return jsonify({str(k): v for k, v in make_link_tables(grfn).items()})


@app.route('/upload_doc', methods=["POST"])
def upload_doc():
    return NotImplemented


def get_conf_file(doc_file):
    if "ASCE" in doc_file:
        return os.path.join(SOURCE_FILES, "configs", "petasce.conf")
    elif doc_file.startswith("ideal_sir"):
        return os.path.join(SOURCE_FILES, "configs", "SIR-simple.conf")
    elif doc_file.startswith("sir_gillespie"):
        return os.path.join(SOURCE_FILES, "configs", "gillespie.conf")
    else:
        raise RuntimeError(f"Config file not specified for: {doc_file}")


def main():
    app.run(host='0.0.0.0', port=80, debug=True)


if __name__ == "__main__":
    main()
