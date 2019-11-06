""" Helper functions for working with INDRA statements. """

import json
from typing import Dict, List, Set, Tuple
from indra.statements.concept import Concept
from indra.statements.statements import Influence, Event
from indra.statements.evidence import Evidence
from delphi.utils.fp import flatMap
from functools import singledispatch


def get_concepts(sts: List[Influence]) -> Set[str]:
    """ Get a set of all unique concepts in the list of INDRA statements. """
    return set(flatMap(nameTuple, sts))


def get_valid_statements_for_modeling(sts: List[Influence]) -> List[Influence]:
    """ Select INDRA statements that can be used to construct a Delphi model
    from a given list of statements. """

    return [
        s
        for s in sts
        if is_grounded_statement(s)
        and (s.subj.delta.polarity is not None)
        and (s.obj.delta.polarity is not None)
    ]


def influence_stmt_from_dict(d: Dict) -> Influence:
    st = Influence(
        Event(Concept(d["subj"]["name"], db_refs=d["subj"]["db_refs"])),
        Event(Concept(d["obj"]["name"], db_refs=d["obj"]["db_refs"])),
        d.get("delta"),
        d.get("delta"),
        [
            Evidence(
                e["source_api"], text=e["text"], annotations=e["annotations"]
            )
            for e in d["evidence"]
        ],
    )
    st.belief = d["belief"]
    return st


def get_statements_from_json_list(_list: List[Dict]) -> List[Influence]:
    return [
        influence_stmt_from_dict(elem)
        for elem in _list
        if elem["type"] == "Influence"
        and elem["subj"]["name"] is not None
        and elem["obj"]["name"] is not None
    ]


def get_statements_from_json_file(json_file: str) -> List[Influence]:
    with open(json_file, "r") as f:
       _list = json.load(f)
    from indra.statements.io import stmts_from_json
    return stmts_from_json([stmt for stmt in _list if stmt["type"]=="Influence"])


@singledispatch
def is_grounded():
    pass


@is_grounded.register(Concept)
def _(c: Concept) -> bool:
    """ Check if a concept is grounded """
    return (
        "UN" in c.db_refs
        and c.db_refs["UN"][0][0].split("/")[1] != "properties"
    )


@is_grounded.register(Influence)
def _(s: Influence) -> bool:
    """ Check if an Influence statement is grounded """
    return is_grounded(s.subj) and is_grounded(s.obj)


def is_grounded_concept(c: Concept) -> bool:
    """ Check if a concept is grounded """
    return (
        "UN" in c.db_refs
        and c.db_refs["UN"][0][0].split("/")[1] != "properties"
    )


def is_grounded_statement(s: Influence) -> bool:
    """ Check if an Influence statement is grounded """
    return is_grounded_concept(s.subj.concept) and is_grounded_concept(s.obj.concept)


@singledispatch
def is_well_grounded():
    pass


@is_well_grounded.register(Concept)
def _(c: Concept, cutoff: float = 0.7) -> bool:
    """Check if a concept has a high grounding score. """

    return is_grounded(c) and (top_grounding_score(c) >= cutoff)


@is_well_grounded.register(Influence)
def _(s: Influence, cutoff: float = 0.7) -> bool:
    """ Returns true if both subj and obj are grounded to the UN ontology. """

    return all(map(lambda c: is_well_grounded(c, cutoff), s.agent_list()))


def is_well_grounded_concept(c: Concept, cutoff: float = 0.7) -> bool:
    """Check if a concept has a high grounding score. """

    return is_grounded(c) and (top_grounding_score(c) >= cutoff)


def is_well_grounded_statement(s: Influence, cutoff: float = 0.7) -> bool:
    """ Returns true if both subj and obj are grounded to the UN ontology. """

    return all(
        map(lambda c: is_well_grounded_concept(c, cutoff), s.agent_list())
    )


def is_grounded_to_name(c: Concept, name: str, cutoff=0.7) -> bool:
    """ Check if a concept is grounded to a given name. """
    return (top_grounding(c) == name) if is_well_grounded(c, cutoff) else False


def contains_concept(s: Influence, concept_name: str, cutoff=0.7) -> bool:
    return any(
        map(
            lambda c: is_grounded_to_name(c, concept_name, cutoff),
            s.agent_list(),
        )
    )


def contains_relevant_concept(
    s: Influence, relevant_concepts: List[str], cutoff=0.7
) -> bool:
    """ Returns true if a given Influence statement has a relevant concept, and
    false otherwise. """

    return any(
        map(lambda c: contains_concept(s, c, cutoff=cutoff), relevant_concepts)
    )


def top_grounding(c: Concept) -> str:
    """ Return the top-scoring grounding from the UN ontology. """
    return c.db_refs["UN"][0][0] if "UN" in c.db_refs else c.name


def top_grounding_score(c: Concept) -> float:
    return c.db_refs["UN"][0][1]


def nameTuple(s: Influence) -> Tuple[str, str]:
    """ Returns a 2-tuple consisting of the top groundings of the subj and obj
    of an Influence statement. """
    return top_grounding(s.subj.concept), top_grounding(s.obj.concept)
