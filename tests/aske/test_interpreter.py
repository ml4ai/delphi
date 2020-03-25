from delphi.GrFN.interpreter import ImperativeInterpreter


def test_mini_pet():
    ITP = ImperativeInterpreter.from_src_dir("tests/data/model_analysis/")
    print(type(ITP))
    print(ITP.documentation.keys())


test_mini_pet()
