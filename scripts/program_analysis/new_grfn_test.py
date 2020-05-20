import os

import numpy as np
import networkx as nx

from delphi.GrFN.interpreter import ImperativeInterpreter
from delphi.GrFN.networks import GroundedFactorNetwork
from delphi.GrFN.structures import GenericIdentifier


def interpreter_test(filepath, con_name, outfile):
    ITP = ImperativeInterpreter.from_src_file(filepath)
    con_id = GenericIdentifier.from_str(con_name)

    G = GroundedFactorNetwork.from_AIR(
        con_id, ITP.containers, ITP.variables, ITP.types,
    )

    A = G.to_AGraph()
    A.draw(outfile, prog="dot")


interpreter_test(
    "../../tests/data/program_analysis/PETASCE_simple.for",
    "@container::PETASCE_simple::@global::petasce",
    "PETASCE--GrFN.pdf",
)
# interpreter_test(
#     "../../tests/data/program_analysis/PETPNO.for",
#     "@container::PETPNO::@global::petpno",
#     "PETPNO--GrFN.pdf",
# )
# interpreter_test(
#     "../../tests/data/program_analysis/SIR-Gillespie-SD.f",
#     "@container::SIR-Gillespie-SD::@global::main",
#     "Gillespie-SD--GrFN.pdf",
# )
# interpreter_test(
#     "../../tests/data/model_analysis/CHIME-SIR.for",
#     "@container::CHIME-SIR::@global::main",
#     "CHIME-SIR--GrFN.pdf",
# )
# interpreter_test(
#     "../../tests/data/program_analysis/SIR-simple.f",
#     "@container::SIR-simple::@global::sir",
#     "SIR-simple--GrFN.pdf",
# )


# CAG = PNO_GrFN.CAG_to_AGraph()
# CAG.draw("PETPT--CAG.pdf", prog="dot")
# assert isinstance(PNO_GrFN, GroundedFactorNetwork)
# assert len(PNO_GrFN.inputs) == 5
# assert len(PNO_GrFN.outputs) == 1
#
# outputs = PNO_GrFN(
#     {
#         name: np.array([1.0], dtype=np.float32)
#         for name in PNO_GrFN.input_name_map.keys()
#     }
# )
# res = outputs[0]
# assert res[0] == np.float32(0.02998372)
# os.remove("PETPT--GrFN.pdf")
# os.remove("PETPT--CAG.pdf")
