from tqdm import tqdm
from delphi.paths import data_dir
from delphi.utils.indra import (
    get_statements_from_json_file,
    get_valid_statements_for_modeling,
    get_concepts,
)
import networkx as nx
from itertools import permutations
from delphi.utils.fp import pairwise
from typing import List


def show_concepts_of_interest(sts):
    for concept in get_concepts(sts):
        for concept_of_interest in (
            "crop",
            "precipitation",
            "food",
            "commerce",
            "health",
            "wealth",
            "conflict",
        ):
            if concept_of_interest in concept:
                print(concept)



