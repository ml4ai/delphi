import os
import ast
import pickle
import pytest
from datetime import date
from indra.statements import Concept, Influence, Evidence
from delphi.AnalysisGraph import AnalysisGraph
from delphi.program_analysis.ProgramAnalysisGraph import ProgramAnalysisGraph
from delphi.program_analysis.autoTranslate.scripts import (
    f2py_pp,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
)
from delphi.utils.indra import *
from delphi.utils.shell import cd
import xml.etree.ElementTree as ET
import subprocess as sp

conflict = Concept(
    "conflict",
    db_refs={
        "TEXT": "conflict",
        "UN": [("UN/events/human/conflict", 0.8), ("UN/events/crisis", 0.4)],
    },
)

food_security = Concept(
    "food_security",
    db_refs={
        "TEXT": "food security",
        "UN": [("UN/entities/food/food_security", 0.8)],
    },
)

precipitation = Concept("precipitation")


flooding = Concept("flooding")
s1 = Influence(
    conflict,
    food_security,
    {"adjectives": ["large"], "polarity": 1},
    {"adjectives": ["small"], "polarity": -1},
    evidence=Evidence(
        annotations={"subj_adjectives": ["large"], "obj_adjectives": ["small"]}
    ),
)

default_annotations = {"subj_adjectives": [], "obj_adjectives": []}

s2 = Influence(
    precipitation,
    food_security,
    evidence=Evidence(annotations=default_annotations),
)
s3 = Influence(
    precipitation, flooding, evidence=Evidence(annotations=default_annotations)
)

STS = [s1, s2, s3]


@pytest.fixture(scope="session")
def G():
    G = AnalysisGraph.from_statements(get_valid_statements_for_modeling(STS))
    G.assemble_transition_model_from_gradable_adjectives()
    G.map_concepts_to_indicators()
    G.parameterize(date(2014, 12, 1))
    G.to_pickle()
    G.create_bmi_config_file()
    yield G


os.environ["CLASSPATH"] = (
    os.getcwd() + "/delphi/program_analysis/autoTranslate/bin/*"
)


@pytest.fixture(scope="session")
def PAG():
    original_fortran_file = "tests/data/crop_yield.f"
    preprocessed_fortran_file = "crop_yield_preprocessed.f"
    f2py_pp.process(original_fortran_file, preprocessed_fortran_file)
    xml_string = sp.run(
        [
            "java",
            "fortran.ofp.FrontEnd",
            "--class",
            "fortran.ofp.XMLPrinter",
            "--verbosity",
            "0",
            "crop_yield_preprocessed.f",
        ],
        stdout=sp.PIPE,
    ).stdout

    trees = [ET.fromstring(xml_string)]
    comments = get_comments.get_comments(preprocessed_fortran_file)
    outputDict = translate.analyze(trees, comments)
    pySrc = pyTranslate.create_python_string(outputDict)
    asts = [ast.parse(pySrc)]
    pgm_dict = genPGM.create_pgm_dict(
        "crop_yield_lambdas.py", asts, "crop_yield.json"
    )
    yield pgm_dict
    for filename in (
        preprocessed_fortran_file,
        "crop_yield_lambdas.py",
        "crop_yield.json",
    ):
        os.remove(filename)
