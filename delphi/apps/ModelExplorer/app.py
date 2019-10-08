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
    return render_template("index.html")


@app.route('/get_saved_materials', methods=["GET"])
def get_saved_materials():
    save_path = os.path.join(os.getcwd(), "source_model_files")
    code_files = os.listdir(os.path.join(save_path, "code"))
    docs_files = os.listdir(os.path.join(save_path, "docs"))
    return jsonify({"code": code_files, "docs": docs_files})


@app.route('/upload_doc', methods=["POST"])
def upload_doc():
    save_path = os.path.join(os.getcwd(), "source_model_files")
    code_files = os.listdir(os.path.join(save_path, "code"))
    docs_files = os.listdir(os.path.join(save_path, "docs"))
    return jsonify({"code": code_files, "docs": docs_files})


def main():
    app.run(host='0.0.0.0', port=80, debug=True)


if __name__ == "__main__":
    main()
