import pytest

from pathlib import Path
import delphi.analysis.comparison.utils as utils
from delphi.analysis.comparison.ForwardInfluenceBlanket import ForwardInfluenceBlanket


def test_pt_asce_comparison():
    pa_graph_example_dir = Path("tests")/"data"/"program_analysis"/"pa_graph_examples"
    asce = utils.nx_graph_from_dotfile(str(pa_graph_example_dir/"asce-graph.dot"))
    pt = utils.nx_graph_from_dotfile(str(pa_graph_example_dir/"priestley-taylor-graph.dot"))
    shared_nodes = utils.get_shared_nodes(asce, pt)

    cmb_asce = ForwardInfluenceBlanket(asce, shared_nodes)
    cmb_pt = ForwardInfluenceBlanket(pt, shared_nodes)

    utils.draw_graph(cmb_asce, "tests/asce-fib.pdf")
    utils.draw_graph(cmb_pt, "tests/pt-fib.pdf")

    expected_cover_set = ["R_so", "K_cb_min", "S_Kc", "K_cb_max", "h", "RH_min",
                          "MEEVP", "u_2", "K_c_min", "f_w", "C_d", "G", "gamma",
                          "e_a", "C_n", "K_r"]

    assert set(expected_cover_set) == set(cmb_asce.cover_nodes)
    assert set() == set(cmb_pt.cover_nodes)


test_pt_asce_comparison()
