""" Helper functions for working with INDRA statements. """

import json
from typing import Dict
from indra.statements import Influence, Concept, Evidence

def influence_stmt_from_dict(d: Dict) -> Influence:
    return Influence(
        Concept(d["subj"]["name"]),
        Concept(d["obj"]["name"]),
        d.get("subj_delta"),
        d.get("obj_delta"),
        [
            Evidence(
                e["source_api"], text=e["text"], annotations=e["annotations"]
            )
            for e in d["evidence"]
        ],
    )


def get_statements_from_json(json_file: str) -> List[Influence]:
    with open(json_file, "r") as f:
        return [
            influence_stmt_from_dict(d)
            for d in json.load(f)
            if d["type"] == "Influence"
        ]
