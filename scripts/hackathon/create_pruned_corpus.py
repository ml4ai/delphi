""" Prune the preassembled corpus json file """

import sys
import json

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        sts = json.load(f)
    filtered_sts = []
    for s in sts:
        if s["type"] == "Influence":
            for c in (s["subj"], s["obj"]):
                for key in [k for k in c["db_refs"] if k != "UN"]:
                    del c["db_refs"][key]

            filtered_sts.append(s)
    with open(sys.argv[2], "w") as f:
        f.write(json.dumps(filtered_sts, indent=2))
