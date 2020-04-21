from enum import Enum, auto, unique

import networkx as nx


@unique
class CodeType(Enum):
    ACCESSOR = auto()
    CALCULATION = auto()
    CONVERSION = auto()
    FILEIO = auto()
    HELPER = auto()
    LOGGING = auto()
    MODEL = auto()
    PIPELINE = auto()
    UNKNOWN = auto()


def build_code_type_decision_tree():
    G = nx.DiGraph()
    G.add_node(
        "C0",
        type="condition",
        func=lambda d: d["num_switches"] >= 1,
        shape="rectangle",
        label="num_switches >= 1",
    )
    G.add_node(
        "C1",
        type="condition",
        func=lambda d: d["max_call_depth"] <= 2,
        shape="rectangle",
        label="max_call_depth <= 2",
    )
    G.add_node(
        "C2",
        type="condition",
        func=lambda d: d["num_assgs"] >= 1,
        shape="rectangle",
        label="num_assgs >= 1",
    )
    G.add_node(
        "C3",
        type="condition",
        func=lambda d: d["num_math_assgs"] >= 1,
        shape="rectangle",
        label="num_math_assgs >= 1",
    )
    G.add_node(
        "C4",
        type="condition",
        func=lambda d: d["num_data_changes"] >= 1,
        shape="rectangle",
        label="num_data_changes >= 1",
    )
    G.add_node(
        "C5",
        type="condition",
        func=lambda d: d["num_var_access"] >= 1,
        shape="rectangle",
        label="num_var_access >= 1",
    )
    G.add_node(
        "C6",
        type="condition",
        func=lambda d: d["num_math_assgs"] >= 5,
        shape="rectangle",
        label="num_math_assgs >= 5",
    )

    G.add_node("Accessor", type=CodeType.ACCESSOR, color="blue")
    G.add_node("Calculation", type=CodeType.CALCULATION, color="blue")
    G.add_node("Conversion", type=CodeType.CONVERSION, color="blue")
    G.add_node("File I/O", type=CodeType.FILEIO, color="blue")
    # G.add_node("Helper", type=CodeType.HELPER, color='blue')
    # G.add_node("Logging", type=CodeType.LOGGING, color='blue')
    G.add_node("Model", type=CodeType.MODEL, color="blue")
    G.add_node("Pipeline", type=CodeType.PIPELINE, color="blue")
    G.add_node("Unknown", type=CodeType.UNKNOWN, color="blue")

    G.add_edge("C0", "Pipeline", type=True, color="darkgreen")
    G.add_edge("C0", "C1", type=False, color="red")

    G.add_edge("C1", "C2", type=True, color="darkgreen")
    G.add_edge("C1", "Pipeline", type=False, color="red")

    G.add_edge("C2", "File I/O", type=False, color="red")
    G.add_edge("C2", "C3", type=True, color="darkgreen")

    G.add_edge("C3", "C4", type=True, color="darkgreen")
    G.add_edge("C3", "C5", type=False, color="red")

    G.add_edge("C4", "C6", type=True, color="darkgreen")
    G.add_edge("C4", "Conversion", type=False, color="red")

    G.add_edge("C5", "Accessor", type=True, color="darkgreen")
    G.add_edge("C5", "Unknown", type=False, color="red")

    G.add_edge("C6", "Model", type=True, color="darkgreen")
    G.add_edge("C6", "Calculation", type=False, color="red")

    return G
