""" This script produces a JSON file with a dictionary where the keys are UN
concepts that were grounded to in a recent corpus of Eidos results, and the
values are lists of (unique) strings that were grounded to those concepts.

The input files are JSON files containing JSON-serialized INDRA statements.

This script was prepared for ISI. """

import json
import pickle
from glob import glob
from tqdm import tqdm
from indra.statements import stmts_from_json, Statement

files = glob("../data/UN_stmt_jsons/*")

d = {}
for file in tqdm(files):
    with open(file, "r") as f:
        sts = stmts_from_json(json.load(f)["statements"])
        for s in sts:
            for x in (s.subj, s.obj):
                db_refs = x.db_refs
                text = db_refs["TEXT"]
                if db_refs.get("UN") is not None:
                    top_un_grounding = db_refs["UN"][0][0].split("/")[-1]
                    if top_un_grounding not in d:
                        d[top_un_grounding] = [text]
                    else:
                        d[top_un_grounding].append(text)

for k, v in d.items():
    d[k] = list(set(v))

with open("groundings.json", "w") as f:
    f.write(json.dumps(d, indent=2))
