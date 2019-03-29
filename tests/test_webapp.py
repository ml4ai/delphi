import csv
from test_GrFN import PETPT_GrFN

def test_PETPT_GrFN_wiring():
    with open("tests/data/GrFN/petpt_grfn_edges.txt", newline="") as csvfile:
        reader = csv.reader(csvfile)
        edges = {tuple(r) for r in reader}
    assert edges == set(PETPT_GrFN.edges())

def test_PETPT_CAG_wiring():
    with open("tests/data/GrFN/petpt_cag_edges.txt", newline="") as csvfile:
        reader = csv.reader(csvfile)
        edges = {tuple(r) for r in reader}
    assert edges == set(PETPT_GrFN.to_CAG().edges())
