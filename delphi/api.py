from indra.statements import Influence
from typing import *
from .types import CausalAnalysisGraph
from .assembly import (filter_statements, contains_relevant_concept,
                       make_cag_skeleton, add_conditional_probabilities,
                       set_indicators)
from .paths import adjectiveData
from .viz import to_agraph
from .export import to_json

def assemble(sts: List[Influence], adj_data: str = adjectiveData,
        relevant_concepts: Optional[List[str]] = None) -> CausalAnalysisGraph:
    """ Construct a Delphi model from INDRA statements """

    filtered_statements = filter_statements(sts)

    if relevant_concepts is not None:
        processed_relevant_concepts = [c.replace(' ', '_') for c in relevant_concepts]
        print(len(filtered_statements))
        filtered_statements = [s for s in filtered_statements
                if contains_relevant_concept(s, processed_relevant_concepts)]
        print(len(filtered_statements))


    cag_skeleton = make_cag_skeleton(filtered_statements)
    cag_with_pdfs = add_conditional_probabilities(cag_skeleton, adjectiveData)
    return cag_with_pdfs

def parameterize(cag: CausalAnalysisGraph) -> CausalAnalysisGraph:
    return set_indicators(cag)

def execute():
    pass

def load():
    pass

def export(cag: CausalAnalysisGraph, format='pkl', pkl_filename = 'delphi_model.pkl'):
    if format == 'pkl':
        with open(pkl_filename, 'wb') as f:
            pickle.dump(cag, f)
    elif format == 'cra':
        to_json(cag)


def visualize(cag: CausalAnalysisGraph, format = 'agraph'):
    if format == 'agraph':
        return to_agraph(cag)
