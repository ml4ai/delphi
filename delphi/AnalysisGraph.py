import os
import json
import pickle
import random
from math import exp, log
from datetime import date
from functools import partial
from itertools import permutations, cycle, chain
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Tuple,
    List,
    Iterable,
    Set,
)
from uuid import uuid4
import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde, norm
import pandas as pd
from indra.statements.statements import Influence
from indra.statements.concept import Concept
from indra.statements.evidence import Evidence as INDRAEvidence
from indra.sources.eidos import process_text
from .random_variables import LatentVar, Indicator
from .export import export_edge, _get_units, _get_dtype, _process_datetime
from .utils.fp import flatMap, take, ltake, lmap, pairwise, iterate
from .utils.indra import (
    get_valid_statements_for_modeling,
    get_concepts,
    get_statements_from_json_list,
    get_statements_from_json_file,
)
from .db import engine
from .assembly import (
    constructConditionalPDF,
    get_respdevs,
    make_edges,
    construct_concept_to_indicator_mapping,
    get_indicator_value,
)
from future.utils import lzip
from tqdm import tqdm
from .icm_api.models import (
    Evidence,
    ICMMetadata,
    CausalVariable,
    CausalRelationship,
    DelphiModel,
    ForwardProjection,
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
        self.dateCreated = date.today().isoformat()
        self.name: str = "Linear Dynamical System with Stochastic Transition Model"
        self.res: int = 100
        self.transition_matrix_collection: List[pd.DataFrame] = []
        self.latent_state_sequences = None

    def assign_uuids_to_nodes_and_edges(self):
        """ Assign uuids to nodes and edges. """
        for node in self.nodes(data=True):
            node[1]["id"] = str(uuid4())

        for edge in self.edges(data=True):
            edge[2]["id"] = str(uuid4())

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

    @classmethod
    def from_statements(cls, sts: List[Influence]):
        """ Construct an AnalysisGraph object from a list of INDRA statements.
        Unknown polarities are set to positive by default."""

        for s in sts:
            for delta in (s.subj_delta, s.obj_delta):
                if delta["polarity"] is None:
                    delta["polarity"] = 1

        node_permutations = permutations(get_concepts(sts), 2)
        edges = make_edges(sts, node_permutations)
        G = cls(edges)
        return G

    @classmethod
    def from_text(cls, text: str):
        """ Construct an AnalysisGraph object from text, using Eidos to perform
        machine reading. """
        eidosProcessor = process_text(text)
        return cls.from_statements(eidosProcessor.statements)

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
        return cls.from_statements(
            get_statements_from_json_list(json_serialized_list)
        )

    @classmethod
    def from_json_serialized_statements_file(cls, file):
        return cls.from_statements(get_statements_from_json_file(file))

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
                if (
                    subj["db_refs"]["concept"] is not None
                    and obj["db_refs"]["concept"] is not None
                ):
                    subj_name, obj_name = [
                        "/".join(s[x]["db_refs"]["concept"].split("/")[:])
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
                            INDRAEvidence(
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
            if indicator is not None:
                indicator_source, indicator_name = (
                    indicator["name"].split("/")[0],
                    "/".join(indicator["name"].split('/')[1:])
                    )
                if concept in G:
                    if G.nodes[concept].get("indicators") is None:
                        G.nodes[concept]["indicators"] = {}
                    G.nodes[concept]["indicators"][indicator_name] = Indicator(
                        indicator_name, indicator_source
                    )

        self = cls(G)
        self.assign_uuids_to_nodes_and_edges()
        return self

    # ==========================================================================
    # Utilities
    # ==========================================================================

    def get_latent_state_components(self):
        return flatMap(lambda a: (a, f"∂({a})/∂t"), self.nodes())

    def construct_default_initial_state(self) -> pd.Series:
        components = self.get_latent_state_components()
        return pd.Series(ltake(len(components), cycle([1.0, 0.0])), components)

    # ==========================================================================
    # Sampling and inference
    # ==========================================================================

    def update_log_prior(self, A: pd.DataFrame) -> float:
        _list = [
            edge[2]["ConditionalProbability"].evaluate(
                A[f"∂({edge[0]})/∂t"][edge[1]] / self.Δt
            )
            for edge in self.edges(data=True)
        ]

        self.log_prior = sum(map(log, _list))

    def update_log_likelihood(self):
        _list = []
        for latent_state, observed_state in zip(
            self.latent_state_sequence, self.observed_state_sequence
        ):
            for n in self.nodes(data=True):
                for indicator, value in observed_state[n[0]].items():
                    ind = n[1]["indicators"][indicator]
                    if ind.timeseries is not None:
                        log_likelihood = np.log(
                            norm.pdf(
                                value, latent_state[n[0]] * ind.mean, ind.stdev
                            )
                        )
                        _list.append(log_likelihood)

        self.log_likelihood = sum(_list)

    def update_log_joint_probability(self):
        self.log_joint_probability = self.log_prior + self.log_likelihood

    def assemble_transition_model_from_gradable_adjectives(self):
        """ Add probability distribution functions constructed from gradable
        adjective data to the edges of the analysis graph data structure.

        Args:
            adjective_data
            res
        """

        df = pd.read_sql_table("gradableAdjectiveData", con=engine)
        gb = df.groupby("adjective")

        rs = gaussian_kde(
            flatMap(
                lambda g: gaussian_kde(get_respdevs(g[1]))
                .resample(self.res)[0]
                .tolist(),
                gb,
            )
        ).resample(self.res)[0]

        for edge in self.edges(data=True):
            edge[2]["ConditionalProbability"] = constructConditionalPDF(
                gb, rs, edge
            )
            edge[2]["βs"] = np.tan(
                edge[2]["ConditionalProbability"].resample(self.res)[0]
            )

    def set_latent_state_sequence(self, A, n_timesteps=10):
        self.latent_state_sequence = ltake(
            n_timesteps,
            iterate(
                lambda s: pd.Series(A.values @ s.values, index=s.index),
                self.s0,
            ),
        )

    def sample_from_prior(self):
        """ Sample elements of the stochastic transition matrix from the prior
        distribution, based on gradable adjectives. """

        # simple_path_dict caches the results of the graph traversal that finds
        # simple paths between pairs of nodes, so that it doesn't have to be
        # executed for every sampled transition matrix.

        node_pairs = list(permutations(self.nodes(), 2))
        simple_path_dict = {
            node_pair: [
                list(pairwise(path))
                for path in nx.all_simple_paths(self, *node_pair)
            ]
            for node_pair in node_pairs
        }

        self.transition_matrix_collection = []

        elements = self.get_latent_state_components()

        for i in range(self.res):
            A = pd.DataFrame(
                np.identity(2 * len(self)), index=elements, columns=elements
            )

            for node in self.nodes:
                A[f"∂({node})/∂t"][node] = self.Δt

            for node_pair in node_pairs:
                A[f"∂({node_pair[0]})/∂t"][node_pair[1]] = sum(
                    np.prod(
                        [
                            self.edges[edge[0], edge[1]]["βs"][i]
                            for edge in simple_path_edge_list
                        ]
                    )
                    * self.Δt
                    for simple_path_edge_list in simple_path_dict[node_pair]
                )
            self.transition_matrix_collection.append(A)

    def sample_observed_state(self, s: pd.Series) -> Dict:
        """ Sample observed state vector. This is the implementation of the
        emission function.

        Args:
            s: Latent state vector.

        Returns:
            Observed state vector.
        """

        return {
            n[0]: {
                i.name: np.random.normal(s[n[0]] * i.mean, i.stdev)
                for i in n[1]["indicators"].values()
            }
            for n in self.nodes(data=True)
        }

    def sample_from_likelihood(self, n_timesteps=10):
        """ Sample a collection of observed state sequences from the likelihood
        model given a collection of transition matrices.

        Args:
            n_timesteps: The number of timesteps for the sequences.
        """

        self.latent_state_sequences = lmap(
            lambda A: ltake(
                n_timesteps,
                iterate(
                    lambda s: pd.Series(A @ s.values, index=s.index), self.s0
                ),
            ),
            self.transition_matrix_collection,
        )

        self.observed_state_sequences = [
            [self.sample_observed_state(s) for s in latent_state_sequence]
            for latent_state_sequence in self.latent_state_sequences
        ]

    def sample_from_proposal(self, A: pd.DataFrame) -> None:
        """ Sample a new transition matrix from the proposal distribution,
        given a current candidate transition matrix. In practice, this amounts
        to the in-place perturbation of an element of the transition matrix
        currently being used by the sampler.

        Args
        """

        # Choose the element of A to perturb
        self.source, self.target, self.edge_dict = random.choice(
            list(self.edges(data=True))
        )
        self.original_value = A[f"∂({self.source})/∂t"][self.target]
        A[f"∂({self.source})/∂t"][self.target] += np.random.normal(scale=0.001)

    def get_timeseries_values_for_indicators(
        self,
        resolution: str = "month",
        time_points: List[int] = range(6, 9),
        n_timesteps=3,
        country: Optional[str] = "South Sudan",
        state: Optional[str] = None,
        unit: Optional[str] = None,
        fallback_aggaxes: List[str] = ["year"],
        aggfunc: Callable = np.mean,
    ):
        """ Attach timeseries to indicators, for performing Bayesian inference. """
        if resolution == "month":
            funcs = [
                partial(get_indicator_value, month=month)
                for month in time_points
            ]
        else:
            raise NotImplementedError(
                "Currently, only the 'month' resolution is supported."
            )

        for n in self.nodes(data=True):
            for indicator in n[1]["indicators"].values():
                indicator.timeseries = [
                    func(indicator, year="2017")[0] for func in funcs
                ]
                if len(set(indicator.timeseries)) == 1:
                    indicator.timeseries = None

    def sample_from_posterior(self, A: pd.DataFrame) -> None:
        """ Run Bayesian inference - sample from the posterior distribution. """
        self.sample_from_proposal(A)
        self.set_latent_state_sequence(A)
        self.update_log_prior(A)
        self.update_log_likelihood()

        candidate_log_joint_probability = self.log_prior + self.log_likelihood

        delta_log_joint_probability = (
            candidate_log_joint_probability - self.log_joint_probability
        )

        acceptance_probability = min(1, np.exp(delta_log_joint_probability))
        if acceptance_probability > np.random.rand():
            self.update_log_joint_probability()
        else:
            A[f"∂({self.source})/∂t"][self.target] = self.original_value
            self.set_latent_state_sequence(A)
            self.update_log_likelihood()
            self.update_log_prior(A)
            self.update_log_joint_probability()

    def infer_transition_matrix_coefficient_from_data(
        self,
        source: str,
        target: str,
        state: Optional[str] = None,
        crop: Optional[str] = None,
    ):
        """ Infer the distribution of a particular transition matrix
        coefficient from data.

        Args:
            source: The source of the edge corresponding to the matrix element
                to infer.
            target: The target of the edge corresponding to the matrix element
                to infer.
            state:
                The state in South Sudan for which the transition matrix
                coefficient should be calculated.
            crop:
                The crop for which the transition matrix coefficient should be
                calculated.
        """
        rows = engine.execute(
            f"select * from dssat where `Crop` like '{crop}'"
            f" and `State` like '{state}'"
        )
        xs, ys = lzip(*[(r["Rainfall"], r["Production"]) for r in rows])
        xs_scaled, ys_scaled = xs / np.mean(xs), ys / np.mean(ys)
        p, V = np.polyfit(xs_scaled, ys_scaled, 1, cov=True)
        self.edges[source, target]["βs"] = np.random.normal(
            p[0], np.sqrt(V[0][0]), self.res
        )
        self.sample_from_prior()

    # ==========================================================================
    # Basic Modeling Interface (BMI)
    # ==========================================================================

    def create_bmi_config_file(self, filename: str = "bmi_config.txt") -> None:
        """ Create a BMI config file to initialize the model.

        Args:
            filename: The filename with which the config file should be saved.
        """
        s0 = self.construct_default_initial_state()
        s0.to_csv(filename, index_label="variable")

    def default_update_function(self, n: Tuple[str, dict]) -> List[float]:
        """ The default update function for a CAG node.
            n: A 2-tuple containing the node name and node data.

        Returns:
            A list of values corresponding to the distribution of the value of
            the real-valued variable representing the node.
        """
        return [
            self.transition_matrix_collection[i].loc[n[0]].values
            @ self.s0[i].values
            for i in range(self.res)
        ]

    def initialize(self, config_file: str = "bmi_config.txt", initialize_indicators = True):
        """ Initialize the executable AnalysisGraph with a config file.

        Args:
            config_file

        Returns:
            AnalysisGraph
        """
        self.t = 0.0
        if not os.path.isfile(config_file):
            self.create_bmi_config_file(config_file)

        self.s0 = [
            pd.read_csv(
                config_file, index_col=0, header=None, error_bad_lines=False
            )[1]
            for _ in range(self.res)
        ]
        self.s0_original = self.s0[0].copy(deep=True)

        self.latent_state_vector = self.construct_default_initial_state()

        for n in self.nodes(data=True):
            rv = LatentVar(n[0])
            n[1]["rv"] = rv
            n[1]["update_function"] = self.default_update_function
            rv.dataset = [1.0 for _ in range(self.res)]
            rv.partial_t = self.s0[0][f"∂({n[0]})/∂t"]
            if initialize_indicators:
                for indicator in n[1]["indicators"].values():
                    indicator.samples = np.random.normal(
                        indicator.mean * np.array(n[1]["rv"].dataset), scale=0.01
                    )

    def update(self, τ: float = 1.0, update_indicators = True, dampen=False):
        """ Advance the model by one time step. """

        for n in self.nodes(data=True):
            n[1]["next_state"] = n[1]["update_function"](n)

        for n in self.nodes(data=True):
            n[1]["rv"].dataset = n[1]["next_state"]

        for n in self.nodes(data=True):
            for i in range(self.res):
                self.s0[i][n[0]] = n[1]["rv"].dataset[i]
                if dampen:
                    self.s0[i][f"∂({n[0]})/∂t"] = self.s0_original[
                        f"∂({n[0]})/∂t"
                    ] * exp(-τ * self.t)
            if update_indicators:
                for indicator in n[1]["indicators"].values():
                    indicator.samples = np.random.normal(
                        indicator.mean * np.array(n[1]["rv"].dataset), scale=0.01
                    )

        self.t += self.Δt

    def update_until(self, t_final: float):
        """ Updates the model to a particular time t_final """
        while self.t < t_final:
            self.update()

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
        return self.get_latent_state_components()

    def get_output_var_names(self) -> List[str]:
        """ Returns the output variable names. """
        return self.get_latent_state_components()

    def get_time_step(self) -> float:
        """ Returns the time step size """
        return self.Δt

    def get_time_units(self) -> str:
        """ Returns the time unit. """
        return self.time_unit

    def get_current_time(self) -> float:
        """ Returns the current time in the execution of the model. """
        return self.t

    # ==========================================================================
    # Export
    # ==========================================================================

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

    # ==========================================================================
    # Model parameterization
    # ==========================================================================

    def map_concepts_to_indicators(
        self, n: int = 1, min_temporal_res: Optional[str] = None
    ):
        """ Add indicators to the analysis graph.

        Args:
            n: Number of matches to keep
            min_temp_res: Minimum temporal resolution.
        """

        mapping = construct_concept_to_indicator_mapping(n)

        for node in self.nodes(data=True):

            # TODO Coordinate with Uncharted (Pascale) and CLULab (Becky) to
            # make sure that the intervention nodes are represented consistently
            # in the mapping (i.e. with spaces vs. with underscores.

            if node[0].split("/")[1] == "interventions":
                node_name = node[0].replace("_", " ")
            else:
                node_name = node[0]

            query_parts = [
                "select Indicator from concept_to_indicator_mapping",
                f"where `Concept` like '{node_name}' and `Source` is 'mitre12'",
            ]

            # TODO Implement temporal resolution constraints. Need to delve
            # into SQL/database stuff a bit more deeply for this. Foreign keys?

            query = "  ".join(query_parts)
            results = engine.execute(query)

            node[1]["indicators"] = {
                "/".join(x.split("/")[1:]): Indicator(
                    "/".join(x.split("/")[1:]), "MITRE12"
                )
                for x in [r[0] for r in take(n, results)]
            }

    def parameterize(
        self,
        country: Optional[str] = "South Sudan",
        state: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        unit: Optional[str] = None,
        fallback_aggaxes: List[str] = ["year", "month"],
        aggfunc: Callable = np.mean,
    ):
        """ Parameterize the analysis graph.

        Args:
            country
            year
            month
            fallback_aggaxes:
                An iterable of strings denoting the axes upon which to perform
                fallback aggregation if the desired constraints cannot be met.
            aggfunc: The function that will be called to perform the
                aggregation if there are multiple matches.
        """

        valid_axes = ("country", "state", "year", "month")

        if any(map(lambda axis: axis not in valid_axes, fallback_aggaxes)):
            raise ValueError(
                "All elements of the fallback_aggaxes set must be one of the "
                f"following: {valid_axes}"
            )

        for n in self.nodes(data=True):
            for indicator in n[1]["indicators"].values():
                indicator.mean, indicator.unit = get_indicator_value(
                    indicator,
                    country,
                    state,
                    year,
                    month,
                    unit,
                    fallback_aggaxes,
                    aggfunc,
                )
                indicator.stdev = 0.1 * abs(indicator.mean)

    # ==========================================================================
    # Manipulation
    # ==========================================================================

    def set_indicator(self, concept: str, indicator: str, source: str):
        self.nodes[concept]["indicators"] = {
            indicator: Indicator(indicator, source)
        }

    def delete_nodes(self, nodes: Iterable[str]):
        """ Iterate over a set of nodes and remove the ones that are present in
        the graph. """
        for n in nodes:
            if self.has_node(n):
                self.remove_node(n)

    def delete_node(self, node: str):
        """ Removes a node if it is in the graph. """
        if self.has_node(node):
            self.remove_node(node)

    def delete_edge(self, source: str, target: str):
        """ Removes an edge if it is in the graph. """
        if self.has_edge(source, target):
            self.remove_edge(source, target)

    def delete_edges(self, edges: Iterable[Tuple[str, str]]):
        """ Iterate over a set of edges and remove the ones that are present in
        the graph. """
        for edge in edges:
            if self.has_edge(*edge):
                self.remove_edge(*edge)

    def prune(self, cutoff: int = 2):
        """ Prunes the CAG by removing redundant paths. If there are multiple
        (directed) paths between two nodes, this function removes all but the
        longest paths. Subsequently, it restricts the graph to the largest
        connected component.

        Args:
            cutoff: The maximum path length to consider for finding redundant
            paths. Higher values of this parameter correspond to more
            aggressive pruning.
        """

        edges_to_keep = set()

        # Remove redundant paths.
        for node_pair in tqdm(list(permutations(self.nodes(), 2))):
            paths = [
                list(pairwise(path))
                for path in nx.all_simple_paths(self, *node_pair, cutoff)
            ]
            if len(paths) > 1:
                for path in paths:
                    if len(path) == 1:
                        self.delete_edge(*path[0])
                        if any(self.degree(n) == 0 for n in path[0]):
                            self.add_edge(*path[0])
                        break

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
                st.obj.db_refs["UN"][0] = (n2, st.obj.db_refs["UN"][0][1])

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
                st.subj.db_refs["UN"][0] = (n2, st.subj.db_refs["UN"][0][1])

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
        self, concept: str, depth: int = 1, reverse: bool = False
    ):
        """ Returns a new subgraph of the analysis graph for a single concept.

        Args:
            concept: The concept that the subgraph will be centered around.
            depth: The depth to which the depth-first search must be performed.
            flow: The direction of causal influence flow to examine. Setting
                  this to 'incoming' will search for upstream causal influences, and
                  setting it to 'outgoing' will search for downstream causal
                  influences.
        returns:
            AnalysisGraph
        """
        flow = "incoming"
        if flow == "incoming":
            rev = self.reverse()
        elif flow == "outgoing":
            rev = self
        else:
            raise ValueError("flow must be one of [incoming|outgoing]")

        nodeset = {concept}

        if reverse:
            func = self.predecessors
        else:
            func = self.successors
        for i in range(depth):
            nodeset.update(
                chain.from_iterable([list(func(n)) for n in nodeset])
            )

        return AnalysisGraph(self.subgraph(nodeset).copy())

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

    # ==========================================================================
    # Database-related code
    # ==========================================================================

    def to_sql(self, app=None, last_known_value_date: Optional[date] = None):
        """ Inserts the model into the SQLite3 database associated with Delphi,
        for use with the ICM REST API. """

        from delphi.icm_api import create_app, db

        self.assign_uuids_to_nodes_and_edges()
        icm_metadata = ICMMetadata(
            id=self.id,
            created=self.dateCreated,
            estimatedNumberOfPrimitives=len(self.nodes) + len(self.edges),
            createdByUser_id=1,
            lastAccessedByUser_id=1,
            lastUpdatedByUser_id=1,
        )
        if last_known_value_date is None:
            today = date.today().isoformat()
        else:
            today = last_known_value_date.isoformat()
        default_latent_var_value = 1.0
        causal_primitives = []
        nodeset = {n.split("/")[-1] for n in self.nodes}
        simplified_labels = len(nodeset) == len(self)
        for n in self.nodes(data=True):
            n[1]["rv"] = LatentVar(n[0])
            n[1]["update_function"] = self.default_update_function
            rv = n[1]["rv"]
            rv.dataset = [default_latent_var_value for _ in range(self.res)]

            # if n[1].get("indicators") is not None:
                # for ind in n[1]["indicators"].values():
                    # ind.dataset = np.ones(self.res) * ind.mean

            causal_variable = CausalVariable(
                id=n[1]["id"],
                model_id=self.id,
                units="",
                namespaces={},
                auxiliaryProperties=[],
                label=n[0].split("/")[-1].replace("_", " ").capitalize() if simplified_labels else n[0],
                description=n[0],
                lastUpdated=today,
                confidence=1.0,
                lastKnownValue={
                    "active": "ACTIVE",
                    "trend": None,
                    "time": today,
                    "value": {
                        "baseType": "FloatValue",
                        "value": n[1]["rv"].dataset[0],
                    },
                },
                range={
                    "baseType": "FloatRange",
                    "range": {"min": 0, "max": 5, "step": 0.1},
                },
            )
            causal_primitives.append(causal_variable)

        max_evidences = max(
            [
                sum([len(s.evidence) for s in e[2]["InfluenceStatements"]])
                for e in self.edges(data=True)
            ]
        )
        max_mean_betas = max(
            [abs(np.median(e[2]["βs"])) for e in self.edges(data=True)]
        )
        for e in self.edges(data=True):
            causal_relationship_id = e[2]["id"]
            causal_relationship = CausalRelationship(
                id=e[2]["id"],
                namespaces={},
                source={
                    "id": self.nodes[e[0]]["id"],
                    "baseType": "CausalVariable",
                },
                target={
                    "id": self.nodes[e[1]]["id"],
                    "baseType": "CausalVariable",
                },
                model_id=self.id,
                auxiliaryProperties=[],
                lastUpdated=today,
                types=["causal"],
                description="",
                confidence=np.mean(
                    [s.belief for s in e[2]["InfluenceStatements"]]
                ),
                label="",
                strength=abs(np.median(e[2]["βs"]) / max_mean_betas),
                reinforcement=(
                    True
                    if np.mean(
                        [
                            stmt.subj_delta["polarity"]
                            * stmt.obj_delta["polarity"]
                            for stmt in e[2]["InfluenceStatements"]
                        ]
                    )
                    > 0
                    else False
                ),
            )
            causal_primitives.append(causal_relationship)
        evidences = []
        for edge in self.edges(data=True):
            for stmt in edge[2]["InfluenceStatements"]:
                for ev in stmt.evidence:
                    evidence = Evidence(
                        id=str(uuid4()),
                        causalrelationship_id=edge[2]["id"],
                        # TODO - make source and target appear in CauseEx HMI
                        description=(ev.text),
                    )
                    evidences.append(evidence)

        if app is None:
            app = create_app()

        with app.app_context():
            db.drop_all()
            db.create_all()
            db.session.add(icm_metadata)
            db.session.add(DelphiModel(id=self.id, model=self))
            for causal_primitive in causal_primitives:
                db.session.add(causal_primitive)
            for evidence in evidences:
                db.session.add(evidence)
            db.session.commit()
