from delphi.GrFN.interpreter import ImperativeInterpreter


def test_mini_pet():
    ITP = ImperativeInterpreter.from_src_dir("tests/data/model_analysis/")
    assert hasattr(ITP, "containers")
    assert hasattr(ITP, "variables")
    assert hasattr(ITP, "types")
    assert hasattr(ITP, "documentation")
