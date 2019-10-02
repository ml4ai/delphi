""" Prune the preassembled corpus json file """

import json
import sys

from tqdm import tqdm


def isInfluenceStatement(s):
    return s["type"] == "Influence"


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        sts = json.load(f)
    filtered_sts = []
    hasWMKey = lambda x: x.get("WM") is not None
    hasNonZeroWMGroundingList = lambda x: len(x["WM"]) != 0

    for s in filter(
        lambda s: isInfluenceStatement(s)
        and all(
            map(
                lambda x: hasWMKey(x) and hasNonZeroWMGroundingList(x),
                map(lambda x: s[x]["concept"]["db_refs"], ("subj", "obj")),
            )
        ),
        sts,
    ):

        for c in (s["subj"], s["obj"]):
            for k in list(c["concept"]["db_refs"].keys()):
                if k != "WM":
                    del c["concept"]["db_refs"][k]
                c["concept"]["db_refs"]["WM"] = c["concept"]["db_refs"]["WM"][
                    0:1
                ]
        filtered_sts.append(s)

    with open(sys.argv[2], "w") as f:
        f.write(json.dumps(filtered_sts, indent=2))
