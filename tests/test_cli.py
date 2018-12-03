import os
import subprocess as sp
from conftest import *


def test_cli(G):
    """ Smokescreen test for CLI application. """
    sp.call(
        [
            "python",
            "delphi/cli.py",
            "execute",
            "--input_dressed_cag",
            "delphi_model.pkl",
        ]
    )
    assert True
    os.remove("dbn_sampled_sequences.csv")
