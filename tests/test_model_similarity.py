import pytest

from pathlib import Path
import delphi.analysis.comparison.utils as utils
from delphi.analysis.comparison.CausalMarkovBlanket import CausalMarkovBlanket


def test_pt_asce_comparison():
    pa_graph_example_dir = Path("tests")/"data"/"pa_graph_examples"
    asce = utils.nx_graph_from_dotfile(str(pa_graph_example_dir/"asce-graph.dot"))
    pt = utils.nx_graph_from_dotfile(str(pa_graph_example_dir/"priestley-taylor-graph.dot"))
    shared_nodes = utils.get_shared_nodes(asce, pt)

    cmb_asce = CausalMarkovBlanket(asce, shared_nodes)
    cmb_pt = CausalMarkovBlanket(pt, shared_nodes)

    expected_cover_set = ['u_2', 'C_d', 'C_n', 'e_a', 'gamma', 'G', 'R_so', 'K_e']
    assert set(expected_cover_set) == set(cmb_asce.cover_nodes)
    assert set() == set(cmb_pt.cover_nodes)
