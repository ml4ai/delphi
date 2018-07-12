from typing import Dict, List, Optional, Union, Callable
import json
import pickle
from dataclasses import dataclass
import networkx as nx
import pandas as pd
import numpy as np

from .assembly import (
    get_data,
    get_respdevs,
    construct_concept_to_indicator_mapping,
    get_indicators,
    get_indicator_value,
)

from future.utils import lmap, lzip
from delphi.assembly import (
    constructConditionalPDF,
    get_concepts,
    make_edge,
    get_valid_statements_for_modeling,
)

from .export import (
    _process_datetime,
    _get_dtype,
    _get_units,
    _export_edge,
)

from .jupyter_tools import (
    print_full_edge_provenance,
    create_statement_inspection_table,
)
from .utils import flatMap, iterate, take, ltake, _insert_line_breaks, compose
from .paths import adjectiveData, south_sudan_data
from datetime import datetime
from scipy.stats import gaussian_kde
from itertools import chain, permutations, cycle
from indra.statements import Influence
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
from functools import partial

set_matplotlib_formats("retina")
import seaborn as sns

plt.style.use("ggplot")
from tqdm import tqdm, trange


class LatentState(object):
    def __init__(self, states: List[pd.Series]):
        self.dataset = states


class ObservedState(object):
    def __init__(self, dataset=None):
        if dataset is not None:
            self.dataset = dataset


class AnalysisGraph(nx.DiGraph):
    """ The primary data structure for Delphi """

    def __init__(self, *args, **kwargs):
        """ Default constructor, accepts a list of edge tuples. """
        super().__init__(*args, **kwargs)
        self.t = 0.0
        self.Δt = 1.0
        self.time_unit = "Placeholder time unit"
        self.latent_state_components = self.get_latent_state_components()
        self.data = None

    # ==========================================================================
    # Creation
    # ==========================================================================

    @classmethod
    def from_statements(cls, sts):
        sts = get_valid_statements_for_modeling(sts)
        node_permutations = permutations(get_concepts(sts), 2)
        edges = [
            e
            for e in [make_edge(sts, p) for p in node_permutations]
            if len(e[2]["InfluenceStatements"]) != 0
        ]

        return cls(edges)

    @staticmethod
    def from_pickle(pickle_file):
        with open(pickle_file, "rb") as f:
            G = pickle.load(f)

        return G

    # ==========================================================================
    # Subgraphs
    # ==========================================================================

    def get_subgraph_for_concept(self, concept: str, depth_limit=None):
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

    # ==========================================================================
    # Inspection
    # ==========================================================================

    def _get_edge_sentences(self, source: str, target: str) -> List[str]:
        """ Return the sentences that led to the construction of a specified edge.

        Args:
            source: The source of the edge.
            target: The target of the edge.
            cag: The analysis graph.
        """

        return chain.from_iterable(
            [
                [repr(e.text) for e in s.evidence]
                for s in self.edges[source, target]["InfluenceStatements"]
            ]
        )

    def inspect_edge(self, source, target):
        """ 'Drill down' into an edge in the analysis graph and inspect its
        provenance. This function prints the provenance."""
        return create_statement_inspection_table(
            self[source][target]["InfluenceStatements"]
        )

    # ==========================================================================
    # Manipulation
    # ==========================================================================

    @property
    def statements(self):
        chainMap = compose(chain.from_iterable, map)
        sts = chainMap(
            lambda e: e[2]["InfluenceStatements"], self.edges(data=True)
        )
        return sorted(
            sts,
            key=lambda s: (
                s.subj.db_refs["UN"][0][0].split("/")[-1]
                + s.obj.db_refs["UN"][0][0].split("/")[-1]
            ),
        )

    def merge_nodes(self, n1, n2, same_polarity=True):
        """ Merge node n1 into n2, where n1 and n2 have opposite polarities. """

        for p in self.predecessors(n1):
            for st in self[p][n1]["InfluenceStatements"]:
                if not same_polarity:
                    st.obj_delta["polarity"] = -st.obj_delta["polarity"]
                st.obj.db_refs["UN"][0] = (
                    "/".join(st.obj.db_refs["UN"][0][0].split("/")[:-1] + [n2]),
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
    # Quantification
    # ==========================================================================

    def map_concepts_to_indicators(self, n: int = 1, manual_mapping=None):
        """ Add indicators to the analysis graph.

        Args:
            n
        """
        mapping = construct_concept_to_indicator_mapping(n=n)

        for n in self.nodes(data=True):
            n[1]["indicators"] = get_indicators(
                n[0].lower().replace(" ", "_"), mapping
            )
            if manual_mapping is not None:
                if n[0] in manual_mapping:
                    n[1]["indicators"] = manual_mapping[n[0]]

    def parameterize(self, time: datetime, data: Optional[pd.DataFrame] = None):
        """ Parameterize the analysis graph.

        Args:
            time
            data
        """

        if data is not None:
            self.data = data
        else:
            if self.data is None:
                self.data = get_data(south_sudan_data)
            else:
                pass

        nodes_with_indicators = [
            n for n in self.nodes(data=True) if n[1]["indicators"] is not None
        ]

        for n in nodes_with_indicators:
            for indicator in n[1]["indicators"]:
                indicator.value, indicator.unit = get_indicator_value(
                    indicator, time, self.data
                )
                indicator.time = time
                if not indicator.value is None:
                    indicator.stdev = 0.1 * abs(indicator.value)
            n[1]["indicators"] = [
                ind for ind in n[1]["indicators"] if ind.value is not None
            ]

    def infer_transition_model(
        self, adjective_data: str = None, res: int = 100
    ):
        """ Add probability distribution functions constructed from gradable
        adjective data to the edges of the analysis graph data structure.

        Args:
            self
            adjective_data
            res
        """

        self.res = res
        if adjective_data is None:
            adjective_data = adjectiveData

        gb = (pd.read_csv(adjectiveData, delim_whitespace=True)
                .groupby("adjective"))
        rs = (
            gaussian_kde(
                flatMap(
                    lambda g: gaussian_kde(get_respdevs(g[1]))
                    .resample(20)[0]
                    .tolist(),
                    gb,
                )
            )
            .resample(100)[0]
            .tolist()
        )

        for e in self.edges(data=True):
            e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)

        self.transition_functions = [
            self.sample_transition_function() for _ in range(res)
        ]

    # ==========================================================================
    # Execution
    # ==========================================================================

    def initialize_transition_matrix(self) -> pd.DataFrame:
        cs = self.get_latent_state_components()
        A = pd.DataFrame(np.identity(len(cs)), cs, cs)
        for c in cs[::2]:
            A[f"∂({c})/∂t"][f"{c}"] = self.Δt
        return A


    def transition_function(self, A, s):
        return pd.Series(A.values @ s.values, index=self.latent_state_components)

    def sample_transition_function(self) -> Callable:
        A = self.initialize_transition_matrix()

        for e in self.edges(data=True):
            if "ConditionalProbability" in e[2].keys():
                β = np.tan(e[2]["ConditionalProbability"].resample(1)[0][0])
                A[f"∂({e[0]})/∂t"][f"∂({e[1]})/∂t"] = β * self.Δt


        return partial(self.transition_function, A)

    def emission_function(self, latent_state):
        latent_state_components = self.get_latent_state_components()
        observed_state = []
        for i, s in enumerate(latent_state_components):
            if i % 2 == 0:
                if self.node[s].get("indicators") is not None:
                    for ind in self.node[s]["indicators"]:
                        new_value = np.random.normal(
                            latent_state[i] * ind.value, ind.stdev
                        )
                        observed_state.append((ind.name, new_value))
                else:
                    o = np.random.normal(latent_state[i], 0.1)
                    observed_state.append((s, o))

        series = pd.Series({k: v for k, v in observed_state})
        return series

    def get_latent_state_components(self) -> List[str]:
        return flatMap(lambda a: (a, f"∂({a})/∂t"), self.nodes())

    def _write_latent_state(self, f):
        for i, s in enumerate(self.latent_state.dataset):
            f.write(str(i)+','+str(self.get_current_time())+',')
            f.write(','.join([str(v) for v in s.values[::2]])+'\n')

    def _write_sequences_to_file(self, seqs, output_filename: str) -> None:
        with open(output_filename, "w") as f:
            f.write(
                ",".join(
                    [
                        "seq_no",
                        "time_slice",
                        *self.get_latent_state_components()[::2],
                    ]
                )
                + "\n"
            )
            for seq_no, seq in enumerate(seqs):
                for time_slice, latent_state in enumerate(seq):
                    vs = ",".join([str(x) for x in latent_state[::2]])
                    f.write(",".join([str(seq_no), str(time_slice), vs]) + "\n")


    def construct_default_initial_state(self) -> pd.Series:
        return pd.Series(
            ltake(len(self.latent_state_components), cycle([1.0, 0.0])),
            self.latent_state_components,
        )

    # ==========================================================================
    # Visualization
    # ==========================================================================

    def visualize(self, *args, **kwargs):
        """ Visualize the analysis graph in a Jupyter notebook cell. """

        from IPython.core.display import Image
        return Image(
            self.to_agraph(self, *args, **kwargs).draw(
                format="png", prog=kwargs.get("prog", "dot")
            ),
            retina=True
        )


    def plot_distribution_of_latent_variable(
        self, latent_variable, ax, xlim = None, **kwargs
    ):
        displayName = kwargs.get('displayName',
                _insert_line_breaks(latent_variable, 30))
        vals = [s[latent_variable] for s in self.latent_state.dataset]
        if xlim is not None:
            ax.set_xlim(*xlim)
            vals = [v for v in vals if ((v > xlim[0]) and (v < xlim[1]))]
        sns.distplot(vals, ax=ax, kde=kwargs.get('kde', True), norm_hist=True)
        ax.set_xlabel(displayName)
        ax.set_ylabel(_insert_line_breaks(f"p({displayName})"))


    def plot_distribution_of_observed_variable(
        self, observed_variable, ax, xlim=None, **kwargs
    ):
        displayName = kwargs.get('displayName',
                _insert_line_breaks(observed_variable, 30))

        vals = [s[observed_variable] for s in self.observed_state.dataset]
        if xlim is not None:
            ax.set_xlim(*xlim)
            vals = [v for v in vals if (v > xlim[0]) and (v < xlim[1])]
        plt.style.use("ggplot")
        sns.distplot(vals, ax=ax, kde=kwargs.get('kde', True), norm_hist=True)
        ax.set_xlabel(displayName)
        ax.set_ylabel(_insert_line_breaks(f"p({displayName})"))

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
            node_dict["indicators"] = [
                _process_datetime(ind.__dict__) for ind in n[1]["indicators"]
            ]
        else:
            node_dict["indicators"] = None

        return node_dict


    def export(
        self,
        format="full",
        json_file="delphi_cag.json",
        pickle_file="delphi_cag.pkl",
        variables_file="variables.csv",
    ):
        """ Export the model in various formats. """

        if format == "full":
            self._to_json(json_file)
            self._pickle(pickle_file)
            self.export_default_initial_values(variables_file)

        if format == "agraph":
            return self.to_graph()

        if format == "json":
            self._to_json(json_file)

    def _pickle(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def _to_json(self, filename: str):
        """ Export the CAG to JSON """
        with open(filename, "w") as f:
            json.dump(
                {
                    "name": "Linear Dynamical System with Stochastic Transition Model",
                    "dateCreated": str(datetime.now()),
                    "variables": lmap(self.export_node, self.nodes(data=True)),
                    "timeStep": str(self.Δt),
                    "edge_data": lmap(_export_edge, self.edges(data=True)),
                },
                f,
                indent=2,
            )

    def export_default_initial_values(self, variables_file: str):
        s0 = self.construct_default_initial_state()
        s0.to_csv(variables_file, index_label="variable")

    # ==========================================================================
    # Basic Modeling Interface (BMI)
    # ==========================================================================

    def initialize(self, filename=None):
        if filename is not None:
            self.s0 = pd.read_csv(
                filename, index_col=0, header=None, error_bad_lines=False
            )[1]
        else:
            self.s0 = self.construct_default_initial_state()

        self.latent_state = LatentState([self.s0 for _ in range(self.res)])
        self.observed_state = ObservedState(
            [self.emission_function(s) for s in self.latent_state.dataset]
        )

    def update(self):
        self.latent_state.dataset = [
            f(s)
            for f, s in lzip(
                self.transition_functions, self.latent_state.dataset
            )
        ]

        self.observed_state.dataset = [
            self.emission_function(s) for s in self.latent_state.dataset
        ]
        self.t += self.Δt

    def get_input_var_names():
        """ Returns the input variable names """
        return self.get_latent_state_components()

    def get_output_var_names():
        """ Returns the output variable names. """
        return self.get_latent_state_components()

    def get_time_step(self):
        """ Returns the time step size """
        return self.Δt

    def get_time_units(self):
        """ Returns the time unit. """
        return self.time_unit

    def get_current_time(self):
        """ Returns the current time in the execution of the model. """
        return self.t

