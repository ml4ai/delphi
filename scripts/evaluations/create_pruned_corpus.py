""" Prune the preassembled corpus json file """

import sys
import json
from tqdm import tqdm

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        sts = json.load(f)
    filtered_sts = []
    for s in tqdm(sts):
        if s["type"] == "Influence":
            for c in (s["subj"], s["obj"]):
                for k in list(c["concept"]["db_refs"].keys()):
                    if k!="WM":
                        del c["concept"]["db_refs"][k]

                if c["concept"]["db_refs"].get("WM") is not None:
                    c["concept"]["db_refs"]["WM"] = c["concept"]["db_refs"]["WM"][0:1]
            filtered_sts.append(s)

    with open(sys.argv[2], "w") as f:
        f.write(json.dumps(filtered_sts, indent=2))
