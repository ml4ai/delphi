""" Helper functions for working with INDRA statements. """

import json
from typing import Dict, List
from indra.statements import Influence, Concept, Evidence

def influence_stmt_from_dict(d: Dict) -> Influence:
    return Influence(
        Concept(d["subj"]["name"], db_refs = d["subj"]["db_refs"]),
        Concept(d["obj"]["name"], db_refs = d["obj"]["db_refs"]),
        d.get("subj_delta"),
        d.get("obj_delta"),
        [
            Evidence(
                e["source_api"], text=e["text"], annotations=e["annotations"]
            )
            for e in d["evidence"]
        ],
    )


def get_statements_from_json(json_serialized_list: str) -> List[Influence]:
    return [
        influence_stmt_from_dict(d)
        for d in json.loads(json_serialized_list)
        if d["type"] == "Influence"
    ]
