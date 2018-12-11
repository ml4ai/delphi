import json
import pickle
from datetime import datetime
from functools import partial
from itertools import permutations, cycle, chain
from typing import Dict, List, Optional, Union, Callable, Tuple, List
from uuid import uuid4
import networkx as nx
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from indra.statements import Influence, Concept, Evidence
from .random_variables import LatentVar, Indicator
from .export import export_edge, _get_units, _get_dtype, _process_datetime
from .paths import south_sudan_data, adjectiveData
from .utils.fp import flatMap, ltake, lmap
from .assembly import (
    constructConditionalPDF,
    get_respdevs,
    make_edges,
    construct_concept_to_indicator_mapping,
    get_indicators,
    get_indicator_value,
    get_data,
)


class AnalysisGraph(nx.DiGraph):
    """ The primary data structure for Delphi """

    def __init__(self, *args, **kwargs):
        """ Default constructor, accepts a list of edge tuples. """
        super().__init__(*args, **kwargs)
        self.id = str(uuid4())
        self.t: float = 0.0
        self.Δt: float = 1.0
        self.time_unit: str = "Placeholder time unit"
        self.dateCreated = datetime.now()
        self.name: str = "Linear Dynamical System with Stochastic Transition Model"
        self.res: int = 100

    # ==========================================================================
    # Constructors
    # ==========================================================================

    @classmethod
    def from_statements_file(cls, file: str):
        """ Construct an AnalysisGraph object from a pickle file containing a
        list of INDRA statements. """

        with open(file, "rb") as f:
            sts = pickle.load(f)

        return cls.from_statements(sts)

    def assign_uuids_to_nodes_and_edges(self):
        """ Assign uuids to nodes and edges. """
        for node in self.nodes(data=True):
            node[1]["id"] = str(uuid4())

        for edge in self.edges(data=True):
            edge[2]["id"] = str(uuid4())

    @classmethod
    def from_statements(cls, sts: List[Influence]):
        """ Construct an AnalysisGraph object from a list of INDRA statements. """
        from .utils.indra import (
            get_valid_statements_for_modeling,
            get_concepts,
        )

        sts = get_valid_statements_for_modeling(sts)
        node_permutations = permutations(get_concepts(sts), 2)
        edges = make_edges(sts, node_permutations)
        self = cls(edges)
        self.assign_uuids_to_nodes_and_edges()
        return self

    @classmethod
    def from_pickle(cls, file: str):
        """ Load an AnalysisGraph object from a pickle file. """
        with open(file, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, cls):
            raise TypeError(
                f"The pickled object in {file} is not an instance of AnalysisGraph"
            )
        else:
            return G

    @classmethod
    def from_json_serialized_statements_list(cls, json_serialized_list):
        from delphi.utils.indra import get_statements_from_json

        return cls.from_statements(
            get_statements_from_json(json_serialized_list)
        )

    @classmethod
    def from_json_serialized_statements_file(cls, file):
        with open(file, "r") as f:
            return cls.from_json_serialized_statements_list(f.read())

    @classmethod
    def from_uncharted_json_file(cls, file):
        with open(file, "r") as f:
            _dict = json.load(f)
        return cls.from_uncharted_json_serialized_dict(_dict)

    @classmethod
    def from_uncharted_json_serialized_dict(
        cls, _dict, minimum_evidence_pieces_required: int = 1
    ):
        sts = _dict["statements"]
        G = nx.DiGraph()
        for s in sts:
            if len(s["evidence"]) >= minimum_evidence_pieces_required:
                subj, obj = s["subj"], s["obj"]
                subj_name, obj_name = [
                    s[x]["db_refs"]["concept"].split("/")[-1]
                    for x in ["subj", "obj"]
                ]
                G.add_edge(subj_name, obj_name)
                subj_delta = s["subj_delta"]
                obj_delta = s["obj_delta"]
                for delta in (subj_delta, obj_delta):
                    # TODO : Ensure that all the statements provided by
                    # Uncharted have unambiguous polarities.
                    if delta["polarity"] is None:
                        delta["polarity"] = 1
                influence_stmt = Influence(
                    Concept(subj_name, db_refs=subj["db_refs"]),
                    Concept(obj_name, db_refs=obj["db_refs"]),
                    subj_delta=s["subj_delta"],
                    obj_delta=s["obj_delta"],
                    evidence=[
                        Evidence(
                            source_api=ev["source_api"],
                            annotations=ev["annotations"],
                            text=ev["text"],
                            epistemics=ev.get("epistemics"),
                        )
                        for ev in s["evidence"]
                    ],
                )
                influence_sts = G.edges[subj_name, obj_name].get(
                    "InfluenceStatements", []
                )
                influence_sts.append(influence_stmt)
                G.edges[subj_name, obj_name][
                    "InfluenceStatements"
                ] = influence_sts

        for concept, indicator in _dict[
            "concept_to_indicator_mapping"
        ].items():
            concept_name = concept.split("/")[-1]
            if concept_name != "Unknown":
                if indicator != "???":
                    indicator_source, *indicator_name = indicator.split("/")
                    if concept_name in G:
                        if G.nodes[concept_name].get("indicators") is None:
                            G.nodes[concept_name]["indicators"] = {}
                        G.nodes[concept_name]["indicators"][
                            indicator_name[-1]
                        ] = Indicator(indicator_name[-1], indicator_source)

        self = cls(G)
        self.assign_uuids_to_nodes_and_edges()
        return self

    def get_latent_state_components(self):
        return flatMap(lambda a: (a, f"∂({a})/∂t"), self.nodes())

    def assemble_transition_model_from_gradable_adjectives(
        self, adjective_data: str = None, res: int = 100
    ):
        """ Add probability distribution functions constructed from gradable
        adjective data to the edges of the analysis graph data structure.

        Args:
            adjective_data
            res
        """

        from scipy.stats import gaussian_kde

        self.res = res
        if adjective_data is None:
            adjective_data = adjectiveData

        gb = pd.read_csv(adjective_data, delim_whitespace=True).groupby(
            "adjective"
        )

        rs = gaussian_kde(
            flatMap(
                lambda g: gaussian_kde(get_respdevs(g[1]))
                .resample(res)[0]
                .tolist(),
                gb,
            )
        ).resample(res)[0]

        for e in self.edges(data=True):
            e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)
            e[2]["betas"] = np.tan(
                e[2]["ConditionalProbability"].resample(self.res)[0]
            )

    def map_concepts_to_indicators(
        self, n: int = 1, mapping_file: Optional[str] = None
    ):
        """ Add indicators to the analysis graph.

        Args:
            n
            mapping_file
        """
        from .utils.web import get_data_from_url

        if mapping_file is None:
            url = "http://vision.cs.arizona.edu/adarsh/export/demos/data/concept_to_indicator_mapping.txt"
            mapping_file = get_data_from_url(url)

        mapping = construct_concept_to_indicator_mapping(n, mapping_file)

        for n in self.nodes(data=True):
            n[1]["indicators"] = get_indicators(
                n[0].lower().replace(" ", "_"), mapping
            )

    def default_update_function(self, n: Tuple[str, dict]) -> List[float]:
        rv = n[1]["rv"]
        return [
            rv.dataset[i]
            + (
                rv.partial_t
                + sum(
                    self[p][n[0]]["betas"][i] * self.nodes[p]["rv"].partial_t
                    for p in self.pred[n[0]]
                )
            )
            * self.Δt
            for i in range(self.res)
        ]

    def emission_function(self, s_i, mu_ij, sigma_ij):
        return np.random.normal(s_i * mu_ij, sigma_ij)

    def construct_default_initial_state(self) -> pd.Series:
        comps = self.get_latent_state_components()
        return pd.Series(ltake(len(comps), cycle([1.0, 0.0])), comps)

    # ==========================================================================
    # Basic Modeling Interface (BMI)
    # ==========================================================================

    def initialize(self, config_file: str):
        """ Initialize the executable AnalysisGraph with a config file.

        Args:
            config_file

        Returns:
            AnalysisGraph
        """
        self.s0 = pd.read_csv(
            config_file, index_col=0, header=None, error_bad_lines=False
        )[1]
        for n in self.nodes(data=True):
            n[1]["rv"] = LatentVar(n[0])
            n[1]["update_function"] = self.default_update_function
            node = n[1]["rv"]
            node.dataset = [self.s0[n[0]] for _ in range(self.res)]
            node.partial_t = self.s0[f"∂({n[0]})/∂t"]
            if n[1].get("indicators") is not None:
                for ind in n[1]["indicators"].values():
                    ind.dataset = np.ones(self.res) * ind.mean

    def update(self):
        """ Advance the model by one time step. """

        next_state = {}

        for n in self.nodes(data=True):
            next_state[n[0]] = n[1]["update_function"](n)

        for n in self.nodes(data=True):
            n[1]["rv"].dataset = next_state[n[0]]
            indicators = n[1].get("indicators")
            if (indicators is not None) and (indicators != {}):
                for indicator_name, indicator in n[1]["indicators"].items():
                    if indicator.mean is not None:
                        indicator.dataset = [
                            self.emission_function(
                                x, indicator.mean, indicator.stdev
                            )
                            for x in n[1]["rv"].dataset
                        ]

        self.t += self.Δt

    def update_until(self, t_final: float):
        """ Updates the model to a particular time t_final """
        while self.t < t_final:
            update(self)

    def finalize(self):
        raise NotImplementedError(
            "This BMI function has not been implemented yet."
        )

    # Model information

    def get_component_name(self) -> str:
        """ Return the name of the model. """
        return self.name

    def get_input_var_names(self) -> List[str]:
        """ Returns the input variable names """
        return get_latent_state_components(self)

    def get_output_var_names(self) -> List[str]:
        """ Returns the output variable names. """
        return get_latent_state_components(self)

    def get_time_step(self) -> float:
        """ Returns the time step size """
        return self.Δt

    def get_time_units(self) -> str:
        """ Returns the time unit. """
        return self.time_unit

    def get_current_time(self) -> float:
        """ Returns the current time in the execution of the model. """
        return self.t

    def export_node(self, n) -> Dict[str, Union[str, List[str]]]:
        """ Return dict suitable for exporting to JSON.

        Args:
            n: A dict representing the data in a networkx AnalysisGraph node.

        Returns:
            The node dict with additional fields for name, units, dtype, and
            arguments.

        """
        node_dict = {
            "name": n[0],
            "units": _get_units(n[0]),
            "dtype": _get_dtype(n[0]),
            "arguments": list(self.predecessors(n[0])),
        }

        if not n[1].get("indicators") is None:
            for indicator in n[1]["indicators"].values():
                if "dataset" in indicator.__dict__:
                    del indicator.__dict__["dataset"]

            node_dict["indicators"] = [
                _process_datetime(indicator.__dict__)
                for indicator in n[1]["indicators"].values()
            ]
        else:
            node_dict["indicators"] = None

        return node_dict

    def to_dict(self) -> Dict:
        """ Export the CAG to a dict. """
        return {
            "name": self.name,
            "dateCreated": str(self.dateCreated),
            "variables": lmap(
                lambda n: self.export_node(n), self.nodes(data=True)
            ),
            "timeStep": str(self.Δt),
            "edge_data": lmap(export_edge, self.edges(data=True)),
        }

    def to_pickle(self, filename: str = "delphi_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def create_bmi_config_file(self, bmi_config_file: str = "bmi_config.txt"):
        s0 = self.construct_default_initial_state()
        s0.to_csv(bmi_config_file, index_label="variable")

    def parameterize(self, time: datetime, data=south_sudan_data):
        """ Parameterize the analysis graph.

        Args:
            time
            data
        """

        if not isinstance(data, pd.DataFrame):
            data = get_data(data)

        nodes_with_indicators = [
            n for n in self.nodes(data=True) if n[1]["indicators"] is not None
        ]

        for n in nodes_with_indicators:
            for indicator_name, indicator in n[1]["indicators"].items():
                indicator.mean, indicator.unit = get_indicator_value(
                    indicator, time, data
                )
                indicator.time = time
                if not indicator.mean is None:
                    indicator.stdev = 0.1 * abs(indicator.mean)

            n[1]["indicators"] = {
                k: v
                for k, v in n[1]["indicators"].items()
                if v.mean is not None
            }

    def merge_nodes(self, n1: str, n2: str, same_polarity: bool = True):
        """ Merge node n1 into node n2, with the option to specify relative
        polarity.

        Args:
            n1
            n2
            same_polarity
        """

        for p in self.predecessors(n1):
            for st in self[p][n1]["InfluenceStatements"]:
                if not same_polarity:
                    st.obj_delta["polarity"] = -st.obj_delta["polarity"]
                st.obj.db_refs["UN"][0] = (
                    "/".join(
                        st.obj.db_refs["UN"][0][0].split("/")[:-1] + [n2]
                    ),
                    st.obj.db_refs["UN"][0][1],
                )

            if not self.has_edge(p, n2):
                self.add_edge(p, n2)
                self[p][n2]["InfluenceStatements"] = self[p][n1][
                    "InfluenceStatements"
                ]

            else:
                self[p][n2]["InfluenceStatements"] += self[p][n1][
                    "InfluenceStatements"
                ]

        for s in self.successors(n1):
            for st in self.edges[n1, s]["InfluenceStatements"]:
                if not same_polarity:
                    st.subj_delta["polarity"] = -st.subj_delta["polarity"]
                st.subj.db_refs["UN"][0] = (
                    "/".join(
                        st.subj.db_refs["UN"][0][0].split("/")[:-1] + [n2]
                    ),
                    st.subj.db_refs["UN"][0][1],
                )

            if not self.has_edge(n2, s):
                self.add_edge(n2, s)
                self[n2][s]["InfluenceStatements"] = self[n1][s][
                    "InfluenceStatements"
                ]
            else:
                self[n2][s]["InfluenceStatements"] += self[n1][s][
                    "InfluenceStatements"
                ]

        self.remove_node(n1)

    # ==========================================================================
    # Subgraphs
    # ==========================================================================

    def get_subgraph_for_concept(
        self, concept: str, depth_limit: Optional[int] = None
    ):
        """ Returns a subgraph of the analysis graph for a single concept.

        Args:
            concept
            depth_limit
        """
        rev = self.reverse()
        dfs_edges = nx.dfs_edges(rev, concept, depth_limit)
        return AnalysisGraph(
            self.subgraph(chain.from_iterable(dfs_edges)).copy()
        )

    def get_subgraph_for_concept_pair(
        self, source: str, target: str, cutoff: Optional[int] = None
    ):
        """ Get subgraph comprised of simple paths between the source and the
        target.

        Args:
            source
            target
            cutoff
        """
        paths = nx.all_simple_paths(self, source, target, cutoff=cutoff)
        return AnalysisGraph(self.subgraph(set(chain.from_iterable(paths))))

    def get_subgraph_for_concept_pairs(
        self, concepts: List[str], cutoff: Optional[int] = None
    ):
        """ Get subgraph comprised of simple paths between the source and the
        target.

        Args:
            concepts
            cutoff
        """
        path_generator = (
            nx.all_simple_paths(self, source, target, cutoff=cutoff)
            for source, target in permutations(concepts, 2)
        )
        paths = chain.from_iterable(path_generator)
        return AnalysisGraph(self.subgraph(set(chain.from_iterable(paths))))
