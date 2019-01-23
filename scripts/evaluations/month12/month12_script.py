import os
import json
import pickle
from datetime import datetime
from typing import List
from tqdm import tqdm
from delphi.AnalysisGraph import AnalysisGraph
from delphi.export import to_agraph
from delphi.paths import data_dir
from delphi.utils.indra import get_statements_from_json_file


def create_pruned_corpus_json():
    """ Prune the preassembled corpus json file """
    with open(data_dir / "wm_12_month_4_reader_20190118.json", "r") as f:
        sts = json.load(f)
    filtered_sts = []
    for s in sts:
        if s["type"] == "Influence":
            for c in (s['subj'], s['obj']):
                for key in [k for k in c['db_refs'] if k != "UN"]:
                    del c['db_refs'][key]

            filtered_sts.append(s)
    with open("build/pruned_corpus.json", "w") as f:
        f.write(json.dumps(filtered_sts, indent=2))

def get_all_sts():
    """ Get all preassembled statements, prune away unneeded information, pickle
    them. """
    all_sts = get_statements_from_json_file("build/pruned_corpus.json")
    with open("build/all_sts.pkl", "wb") as f:
        pickle.dump(all_sts, f)


def filter_and_process_statements(
    sts,
    grounding_score_cutoff: float = 0.8,
    belief_score_cutoff: float = 0.85,
    concepts_of_interest: List[str] = [],
):
    """ Filter preassembled statements according to certain rules. """
    filtered_sts = []
    counters = {}

    def update_counter(counter_name):
        if counter_name in counters:
            counters[counter_name] += 1
        else:
            counters[counter_name] = 1

    for s in tqdm(sts):

        update_counter("Original number of statements")

        # Apply belief score threshold cutoff

        if not s.belief > belief_score_cutoff:
            continue
        update_counter(f"Statements with belief score > {belief_score_cutoff}")

        # Select statements with UN groundings

        if s.subj.db_refs.get("UN") is None or s.obj.db_refs.get("UN") is None:
            continue
        update_counter("Statements with UN groundings")

        # Apply grounding score cutoffs

        if not all(
            x[1] > grounding_score_cutoff
            for x in (y.db_refs["UN"][0] for y in (s.subj, s.obj))
        ):
            continue
        update_counter(
            f"Statements with subj and obj grounding scores > {grounding_score_cutoff}"
        )

        # Assign default polarities

        if s.subj_delta["polarity"] is None:
            s.subj_delta["polarity"] = 1
        if s.obj_delta["polarity"] is None:
            s.obj_delta["polarity"] = 1

        filtered_sts.append(s)

    for k, v in counters.items():
        print(f"{k}: {v}")

    return filtered_sts


def create_reference_CAG():
    with open("build/all_sts.pkl", "rb") as f:
        all_sts = pickle.load(f)
    filtered_sts = filter_and_process_statements(all_sts, 0.9)
    G = AnalysisGraph.from_statements(filtered_sts)
    G.merge_nodes(
        "UN/events/natural/weather/precipitation",
        "UN/events/weather/precipitation",
    )
    for n in G.nodes():
        G.delete_edge(n, "UN/events/weather/precipitation")

    with open("build/CAG.pkl", "wb") as f:
        pickle.dump(G, f)


def create_precipitation_centered_CAG(filename="CAG.pdf"):
    """ Get a CAG that examines the downstream effects of changes in precipitation. """

    with open("build/CAG.pkl", "rb") as f:
        G = pickle.load(f)
    G = G.get_subgraph_for_concept(
        "UN/events/weather/precipitation", depth=2, flow="outgoing"
    )
    G.prune(cutoff=0)
    A = to_agraph(G)
    A.graph_attr["rankdir"] = "TB"
    A.draw(filename, prog="dot")
    with open("build/precipitation_centered_CAG.pkl", "wb") as f:
        pickle.dump(G, f)

def create_CAG_with_indicators(filename="CAG_with_indicators.pdf"):
    """ Create a CAG with mapped indicators """
    with open("build/precipitation_centered_CAG.pkl", "rb") as f:
        G = pickle.load(f)
    G.map_concepts_to_indicators()
    A = to_agraph(G, indicators=True)
    A.draw(filename, prog="dot")
    with open("build/CAG_with_indicators.pkl", "wb") as f:
        pickle.dump(G, f)

def create_quantified_CAG():
    with open("build/CAG_with_indicators.pkl", "rb") as f:
        G = pickle.load(f)

    G.res = 500
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    with open("build/quantified_CAG.pkl", "wb") as f:
        pickle.dump(G, f)



def create_parameterized_CAG(filename = "CAG_with_indicators_and_values.pdf"):
    """ Create a CAG with mapped and parameterized indicators """
    with open("build/CAG_with_indicators.pkl", "rb") as f:
        G = pickle.load(f)
    G.parameterize(datetime(2017,4,1))
    A = to_agraph(G, indicators=True, indicator_values=True)
    A.draw(filename, prog="dot")

if __name__ == "__main__":
    os.makedirs("build", exist_ok=True)
    # create_pruned_corpus_json()
    # get_all_sts()
    # create_reference_CAG()
    create_precipitation_centered_CAG()
    # create_quantified_CAG()
    # create_CAG_with_indicators()
    # create_parameterized_CAG()
