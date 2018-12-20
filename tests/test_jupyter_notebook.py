import subprocess as sp

def test_delphi_demo_notebook():
    sp.check_call([
        "jupyter",
        "nbconvert",
        "--execute",
        "notebooks/Delphi-Demo-Notebook.ipynb",
        ])
