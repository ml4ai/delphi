import pytest
import subprocess as sp

@pytest.mark.skip
def test_delphi_demo_notebook():
    sp.check_call([
        "jupyter",
        "nbconvert",
        "--execute",
        "--ExecutePreprocessor.timeout=3600",
        "notebooks/Delphi-Demo-Notebook.ipynb",
        ])
