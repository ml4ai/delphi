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
    isNotGroundedToCausalFactor = lambda x: x["WM"][0][0] not in (
        "wm/concept/causal_factor",
        "wm/concept/causal_factor/condition",
        "wm/concept/causal_factor/condition/trend",
        "wm/concept/causal_factor/access",
        "wm/concept/causal_factor/intervention",
        "wm/concept/causal_factor/movement/movement",
        "wm/concept/causal_factor/social_and_political",
        "wm/concept/entity/artifact",
        "wm/concept/entity/geo-location",
        "wm/concept/entity/government_entity",
        "wm/concept/entity/organization",
        "wm/concept/entity/person_and_group/community",
        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market",
        "wm/concept/indicator_and_reported_property/weather"
    )

    for s in filter(
        lambda s: isInfluenceStatement(s)
        and all(
            map(
                lambda x: hasWMKey(x)
                and hasNonZeroWMGroundingList(x)
                and isNotGroundedToCausalFactor(x),
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
                c["concept"]["db_refs"]["WM"][0][0] = (
                    c["concept"]["db_refs"]["WM"][0][0]
                    .replace(" ", "_")
                    .replace(
                        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/price/food_price",
                        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/price_or_cost/food_price",
                    )
                    .replace(
                        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/price/oil_price",
                        "wm/concept/causal_factor/economic_and_commerce/economic_activity/market/price_or_cost/oil_price",
                    )
                    .replace(
                        "wm/concept/causal_factor/intervention/provision_of_goods_and_services/provide_stationary",
                        "wm/concept/causal_factor/intervention/provision_of_goods_and_services/provide_stationery",
                    )
                    .replace(
                        "wm/concept/causal_factor/intervention/provision_of_goods_and_services/provide_moving_of_houseHolds",
                        "wm/concept/causal_factor/intervention/provision_of_goods_and_services/provide_moving_of_households",
                    )
                    .replace(
                        "wm/concept/causal_factor/social_and_political/crime",
                        "wm/concept/causal_factor/social_and_political/crime/crime",
                    )
                    .replace(
                        "wm/concept/causal_factor/social_and_political/education",
                        "wm/concept/causal_factor/social_and_political/education/education",
                    )
                )
        filtered_sts.append(s)

    with open(sys.argv[2], "w") as f:
        f.write(json.dumps(filtered_sts, indent=2))
