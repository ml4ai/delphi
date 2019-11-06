import os
import json
import pickle
import random
from math import exp, log, pi
from datetime import date
from functools import partial
from itertools import permutations, cycle, chain
from typing import Dict, Optional, Union, Callable, Tuple, List, Iterable
from uuid import uuid4
import networkx as nx
import numpy as np
import warnings
from scipy.stats import gaussian_kde
import pandas as pd
from indra.statements import Influence, Concept, Event, QualitativeDelta
from indra.statements import Evidence as INDRAEvidence
from .random_variables import LatentVar, Indicator
from .utils.fp import flatMap, take, ltake, lmap, pairwise, iterate, exists
from .utils.indra import (
    get_statements_from_json_list,
    get_statements_from_json_file,
    nameTuple,
)
from .db import engine
from .assembly import (
    deltas,
    constructConditionalPDF,
    get_respdevs,
    get_indicator_value,
)
from future.utils import lzip
from tqdm import tqdm
from .apps.rest_api.models import (
    Evidence,
    ICMMetadata,
    CausalVariable,
    CausalRelationship,
    DelphiModel,
)
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from delphi.utils.misc import _insert_line_breaks


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
        self.name: str = (
            "Linear Dynamical System with Stochastic Transition Model"
        )
        self.res: int = 100
        self.transition_matrix_collection: List[pd.DataFrame] = []
        self.latent_state_sequences = None
        self.log_prior = None
        self.log_likelihood = None

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
    def from_statements(
        cls, sts: List[Influence], assign_default_polarities: bool = True
    ):
        """ Construct an AnalysisGraph object from a list of INDRA statements.
        Unknown polarities are set to positive by default.

        Args:
            sts: A list of INDRA Statements

        Returns:
            An AnalysisGraph instance constructed from a list of INDRA
            statements.
        """

        _dict = {}
        for s in sts:
            if assign_default_polarities:
                for delta in deltas(s):
                    if delta.polarity is None:
                        delta.polarity = 1
            concepts = nameTuple(s)

            # Excluding self-loops for now:
            if concepts[0] != concepts[1]:
                if all(map(exists, (delta.polarity for delta in deltas(s)))):
                    if concepts in _dict:
                        _dict[concepts].append(s)
                    else:
                        _dict[concepts] = [s]

        edges = [
            (*concepts, {"InfluenceStatements": statements})
            for concepts, statements in _dict.items()
        ]
        return cls(edges)

    @classmethod
    def from_text(cls, text: str, webservice=None):
        """ Construct an AnalysisGraph object from text, using Eidos to perform
        machine reading.

        Args:
            text: Input text to be processed by Eidos.
            webservice: URL for Eidos webservice, either the INDRA web service,
                or the instance of Eidos running locally on your computer (e.g.
                http://localhost:9000.
        """
        from indra.sources.eidos import process_text
        eidosProcessor = process_text(text, webservice=webservice)
        eidosProcessor.extract_causal_relations()
        return cls.from_statements([stmt for stmt in eidosProcessor.statements
            if isinstance(stmt, Influence)])

    @classmethod
    def from_json_serialized_statements_list(
        cls, json_serialized_list: List[Dict]
    ):
        """ Construct an AnalysisGraph object from a list of JSON serialized
        INDRA statements.

        Args:
            json_serialized_list: A list of JSON-serializable dicts
                representing INDRA statements.
        """
        return cls.from_statements(
            get_statements_from_json_list(json_serialized_list)
        )

    @classmethod
    def from_json_serialized_statements_file(cls, file: str):
        """ Construct an AnalysisGraph object from a file containing
        JSON-serialized INDRA statements.

        Args:
            file: Path to a file containing JSON-serialized INDRA statements.
        """
        return cls.from_statements(get_statements_from_json_file(file))

    @classmethod
    def from_uncharted_json_file(cls, file: str):
        """ Construct an AnalysisGraph object from a file containing INDRA
        statements serialized exported by Uncharted's CauseMos webapp.

        Args:
            file: Path to a file containing JSON-serialized INDRA statements
            from Uncharted's CauseMos HMI.
        """
        with open(file, "r") as f:
            _dict = json.load(f)
        return cls.from_uncharted_json_serialized_dict(_dict)

    @classmethod
    def from_uncharted_json_serialized_dict(
        cls, _dict, minimum_evidence_pieces_required: int = 1
    ):
        """ Construct an AnalysisGraph object from a dict of INDRA statements
        exported by Uncharted's CauseMos webapp.

        Args:
            _dict: A dict of INDRA statements exported by Uncharted's CauseMos
                HMI.
            minimum_evidence_pieces_required: The minimum number of evidence
                pieces required to consider a statement for assembly.
        """
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
                        Event(
                            Concept(subj_name, db_refs=subj["db_refs"]),
                            delta=QualitativeDelta(
                                s["subj_delta"]["polarity"],
                                s["subj_delta"]["adjectives"],
                            ),
                        ),
                        Event(
                            Concept(obj_name, db_refs=obj["db_refs"]),
                            delta=QualitativeDelta(
                                s["obj_delta"]["polarity"],
                                s["obj_delta"]["adjectives"],
                            ),
                        ),
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

        func_dict = {
            "mean": np.mean,
            "median": np.median,
            "max": max,
            "min": min,
            "raw": lambda x: x,
        }

        for concept, indicator in _dict["conceptIndicators"].items():
            indicator_source, indicator_name = (
                indicator["name"].split("/")[0],
                "/".join(indicator["name"].split("/")[1:]),
            )

            G.nodes[concept]["indicators"] = {
                indicator_name: Indicator(indicator_name, indicator_source)
            }
            values = [x["value"] for x in indicator["values"]]
            indicator["mean"] = func_dict[indicator["func"]](values)
            # indicator.source = indicator["source"]

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
    # ----------------------
    #
    # This section contains code for sampling and Bayesian inference.
    # ==========================================================================

    def sample_from_prior(self):
        """ Sample elements of the stochastic transition matrix from the prior
        distribution, based on gradable adjectives. """

        # Add probability distribution functions constructed from gradable
        # adjective data to the edges of the analysis graph data structure.

        df = pd.read_sql_table("gradableAdjectiveData", con=engine)
        gb = df.groupby("adjective")

        rs = gaussian_kde(
            flatMap(
                lambda g: gaussian_kde(get_respdevs(g[1]))
                .resample(self.res)
                .tolist(),
                gb,
            )
        ).resample(self.res)

        for edge in self.edges(data=True):
            edge[2]["ConditionalProbability"] = constructConditionalPDF(
                gb, rs, edge
            )
            edge[2]["βs"] = np.tan(
                edge[2]["ConditionalProbability"].resample(self.res)
            )

        # Sample a collection of transition matrices from the prior.

        node_pairs = list(permutations(self.nodes(), 2))

        # simple_path_dict caches the results of the graph traversal that finds
        # simple paths between pairs of nodes, so that it doesn't have to be
        # executed for every sampled transition matrix.

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

        return self.transition_matrix_collection


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

        xs = [
            self.transition_matrix_collection[i].loc[n[0]].values
            @ self.s0[i].values
            for i in range(self.res)
        ]
        return xs

    def initialize(self, initialize_indicators=True):
        """ Initialize the executable AnalysisGraph with a config file.

        Args:
            initialize_indicators: Boolean flag that sets whether indicators
            are initialized as well.

        Returns:
            None
        """
        self.t = 0.0

        # Create self.res copies of the initial latent state vector
        self.s0 = [
            self.construct_default_initial_state() for _ in range(self.res)
        ]

        # Create a 'reference copy' of the initial latent state vector
        self.s0_original = self.s0[0].copy(deep=True)

        for n in self.nodes(data=True):
            rv = LatentVar(n[0])
            n[1]["rv"] = rv
            n[1]["update_function"] = self.default_update_function
            rv.dataset = [1.0 for _ in range(self.res)]
            rv.partial_t = self.s0[0][f"∂({n[0]})/∂t"]
            if initialize_indicators:
                for indicator in n[1]["indicators"].values():
                    indicator.samples = np.random.normal(
                        indicator.mean * np.array(n[1]["rv"].dataset),
                        scale=0.01,
                    )

    def update(
        self,
        τ: float = 1.0,
        update_indicators=True,
        dampen=False,
        set_delta: float = None,
    ):
        """ Advance the model by one time step. set_delta is currently just a
        placeholder for a future feature.
        """

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
                        indicator.mean * np.array(n[1]["rv"].dataset),
                        scale=0.01,
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

    def to_pickle(self, filename: str = "delphi_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    # ==========================================================================
    # Model parameterization
    # ==========================================================================

    def map_concepts_to_indicators(
        self, n: int = 1, min_temporal_res: Optional[str] = None
    ):
        """ Map each concept node in the AnalysisGraph instance to one or more
        tangible quantities, known as 'indicators'.

        Args:
            n: Number of matches to keep
            min_temporal_res: Minimum temporal resolution that the indicators
            must have data for.
        """

        for node in self.nodes(data=True):
            query_parts = [
                "select Indicator from concept_to_indicator_mapping",
                f"where `Concept` like '{node[0]}'",
            ]

            # TODO May need to delve into SQL/database stuff a bit more deeply
            # for this. Foreign keys perhaps?

            query = "  ".join(query_parts)
            results = engine.execute(query)

            if min_temporal_res is not None:
                if min_temporal_res not in ["month"]:
                    raise ValueError("min_temporal_res must be 'month'")

                vars_with_required_temporal_resolution = [
                    r[0]
                    for r in engine.execute(
                        "select distinct `Variable` from indicator where "
                        f"`{min_temporal_res.capitalize()}` is not null"
                    )
                ]
                results = [
                    r
                    for r in results
                    if r[0] in vars_with_required_temporal_resolution
                ]

            node[1]["indicators"] = {
                x: Indicator(x, "MITRE12")
                for x in [r[0] for r in take(n, results)]
            }

    def parameterize(
        self,
        country: Optional[str] = "Ethiopia",
        state: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        units: Optional[dict] = None,
        fallback_aggaxes: List[str] = ["year", "month"],
        aggfunc: Callable = np.mean,
    ):
        """ Parameterize the analysis graph.

        Args:
            country
            year
            month
            units: Takes dictionary object with indictor variables as keys the
            desired units as values
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

        if units is not None:
            for n in self.nodes(data=True):
                for indicator in n[1]["indicators"].values():
                    if indicator.name in units:
                        unit = units[indicator.name]
                    else:
                        unit = None
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
        else:
            unit = None
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
                    st.obj.delta.polarity = -st.obj.delta.polarity
                st.obj.db_refs["WM"][0] = (n2, st.obj.db_refs["WM"][0][1])

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
                    st.subj.delta.polarity = -st.subj.delta.polarity
                st.subj.db_refs["WM"][0] = (n2, st.subj.db_refs["WM"][0][1])

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

            reverse: Sets the direction of causal influence flow to examine.
                Setting this to False (default) will search for upstream causal
                influences, and setting it to True will search for
                downstream causal influences.

        Returns:
            AnalysisGraph
        """

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

        from delphi.apps.rest_api import create_app, db

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

            causal_variable = CausalVariable(
                id=n[1]["id"],
                model_id=self.id,
                units="",
                namespaces={},
                auxiliaryProperties=[],
                label=n[0].split("/")[-1].replace("_", " ").capitalize()
                if simplified_labels
                else n[0],
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

        max_mean_betas = max(
            [abs(np.median(e[2]["βs"])) for e in self.edges(data=True)]
        )
        for e in self.edges(data=True):
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
                            stmt.subj.delta.polarity * stmt.obj.delta.polarity
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

        # Tag this model for displaying in the CauseWorks interface
        self.tag_for_CX = True
        with app.app_context():
            db.create_all()
            db.session.add(icm_metadata)
            db.session.add(DelphiModel(id=self.id, model=self))
            for causal_primitive in causal_primitives:
                db.session.add(causal_primitive)
            for evidence in evidences:
                db.session.add(evidence)
            db.session.commit()

    def to_agraph(
        self,
        indicators: bool = False,
        indicator_values: bool = False,
        nodes_to_highlight: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        """ Exports the CAG as a pygraphviz AGraph for visualization.

        Args:
            indicators: Whether to display indicators in the AGraph
            indicator_values: Whether to display indicator values in the AGraph
            nodes_to_highlight: Nodes to highlight in the AGraph.
        Returns:
            A PyGraphviz AGraph representation of the AnalysisGraph.
        """

        from delphi.utils.misc import choose_font

        FONT = choose_font()
        A = nx.nx_agraph.to_agraph(self)
        A.graph_attr.update(
            {
                "dpi": 227,
                "fontsize": 20,
                "rankdir": kwargs.get("rankdir", "TB"),
                "fontname": FONT,
                "overlap": "scale",
                "splines": True,
            }
        )

        A.node_attr.update(
            {
                "shape": "rectangle",
                "color": "black",
                # "color": "#650021",
                "style": "rounded",
                "fontname": FONT,
            }
        )

        nodes_with_indicators = [
            n
            for n in self.nodes(data=True)
            if n[1].get("indicators") is not None
        ]

        n_max = max(
            [
                sum([len(s.evidence) for s in e[2]["InfluenceStatements"]])
                for e in self.edges(data=True)
            ]
        )

        nodeset = {n.split("/")[-1] for n in self.nodes}

        simplified_labels = len(nodeset) == len(self)
        color_str = "#650021"
        for n in self.nodes(data=True):
            if kwargs.get("values"):
                node_label = (
                    n[0].capitalize().replace("_", " ")
                    + " ("
                    + str(np.mean(n[1]["rv"].dataset))
                    + ")"
                )
            else:
                node_label = (
                    n[0].split("/")[-1].replace("_", " ").capitalize()
                    if simplified_labels
                    else n[0]
                )
            A.add_node(n[0], label=node_label)

        if list(self.edges(data=True))[0][2].get("βs") is not None:
            max_median_betas = max(
                [abs(np.median(e[2]["βs"])) for e in self.edges(data=True)]
            )

        for e in self.edges(data=True):
            # Calculate reinforcement (ad-hoc!)

            sts = e[2]["InfluenceStatements"]
            total_evidence_pieces = sum([len(s.evidence) for s in sts])
            reinforcement = (
                sum(
                    [
                        stmt.overall_polarity() * len(stmt.evidence)
                        for stmt in sts
                    ]
                )
                / total_evidence_pieces
            )
            opacity = total_evidence_pieces / n_max
            h = (opacity * 255).hex()

            if list(self.edges(data=True))[0][2].get("βs") is not None:
                penwidth = 3 * abs(np.median(e[2]["βs"]) / max_median_betas)
                cmap = cm.Greens if reinforcement > 0 else cm.Reds
                c_str = (
                    matplotlib.colors.rgb2hex(cmap(abs(reinforcement)))
                    + h[4:6]
                )
            else:
                penwidth = 1
                c_str = "black"

            A.add_edge(e[0], e[1], color=c_str, penwidth=penwidth)

        # Drawing indicator variables

        if indicators:
            for n in nodes_with_indicators:
                for indicator_name, ind in n[1]["indicators"].items():
                    node_label = _insert_line_breaks(
                        ind.name.replace("_", " "), 30
                    )
                    if indicator_values:
                        if ind.unit is not None:
                            units = f" {ind.unit}"
                        else:
                            units = ""

                        if ind.mean is not None:
                            ind_value = "{:.2f}".format(ind.mean)
                            node_label = (
                                f"{node_label}\n{ind_value} {ind.unit}"
                                f"\nSource: {ind.source}"
                                f"\nAggregation axes: {ind.aggaxes}"
                                f"\nAggregation method: {ind.aggregation_method}"
                            )

                    A.add_node(
                        indicator_name,
                        style="rounded, filled",
                        fillcolor="lightblue",
                        label=node_label,
                    )
                    A.add_edge(n[0], indicator_name, color="royalblue4")

        nodes_to_highlight = kwargs.get("nodes_to_highlight")
        if nodes_to_highlight is not None:
            if isinstance(nodes_to_highlight, list):
                for n in nodes_to_highlight:
                    if n in A.nodes():
                        A.add_node(n, fontcolor="royalblue")
            elif isinstance(nodes_to_highlight, str):
                if nodes_to_highlight in A.nodes():
                    A.add_node(nodes_to_highlight, fontcolor="royalblue")
            else:
                pass

        if kwargs.get("graph_label") is not None:
            A.graph_attr["label"] = kwargs["graph_label"]

        return A
