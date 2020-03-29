from delphi.GrFN.interpreter import ImperativeInterpreter


def test_mini_pet():
    ITP = ImperativeInterpreter.from_src_dir("tests/data/model_analysis/")
    assert hasattr(ITP, "containers")
    assert hasattr(ITP, "variables")
    assert hasattr(ITP, "types")
    assert hasattr(ITP, "documentation")
    ITP.gather_container_stats()
    ITP.label_container_code_types()
    grfns = ITP.build_GrFNs()
    # TODO Adarsh: fill this list out
    expected_grfns = sorted(["PETPT", "PETASCE", "PSE", "FLOOD_EVAP"])
    assert sorted(grfns) == expected_grfns
