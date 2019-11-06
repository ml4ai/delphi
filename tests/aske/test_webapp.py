import csv
from test_GrFN import petpt_grfn


def test_PETPT_GrFN_wiring(petpt_grfn):
    with open("tests/data/GrFN/petpt_grfn_edges.txt", newline="") as csvfile:
        reader = csv.reader(csvfile)
        edges = {tuple(r) for r in reader}
    assert edges == set(petpt_grfn.edges())


def test_PETPT_CAG_wiring(petpt_grfn):
    with open("tests/data/GrFN/petpt_cag_edges.txt", newline="") as csvfile:
        reader = csv.reader(csvfile)
        edges = {tuple(r) for r in reader}
    assert edges == set(petpt_grfn.to_CAG().edges())
