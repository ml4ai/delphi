import sys
from tqdm import tqdm
from typing import List
import pickle
from delphi.AnalysisGraph import AnalysisGraph

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


        # Apply belief score threshold cutoff

        if not s.belief > belief_score_cutoff:
            continue
        update_counter(f"Statements with belief score > {belief_score_cutoff}")

        # Assign default polarities

        if s.subj_delta["polarity"] is None:
            s.subj_delta["polarity"] = 1
        if s.obj_delta["polarity"] is None:
            s.obj_delta["polarity"] = 1

        filtered_sts.append(s)

    for k, v in counters.items():
        print(f"{k}: {v}")

    return filtered_sts

def create_reference_CAG(inputPickleFile, outputPickleFile):
    with open(inputPickleFile, "rb") as f:
        all_sts = pickle.load(f)
    #Second and Third Argument control grounding score and belief score cutoff,
    #respectively.
    filtered_sts = filter_and_process_statements(all_sts,0.75,.85)
    G = AnalysisGraph.from_statements(filtered_sts)
    G.merge_nodes(
        "UN/events/natural/weather/precipitation",
        "UN/events/weather/precipitation",
    )

    with open(outputPickleFile, "wb") as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    create_reference_CAG(sys.argv[1], sys.argv[2])
