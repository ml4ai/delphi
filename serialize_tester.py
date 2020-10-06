import delphi.evaluation as EN
from delphi.cpp.DelphiPython import AnalysisGraph

G = AnalysisGraph.from_causemos_json_file("tests/data/delphi/causemos_create-model.json")
G.print_training_range()
model_json = G.serialize_to_json_string(verbose=False)
#print('\n\nReturned json\n\n', model_json)
