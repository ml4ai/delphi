from typing import Dict, List, Optional, Union, Callable, Tuple
import random
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
    get_valid_statements_for_modeling,
    nameTuple,
)

from .export import (_process_datetime, _get_dtype, _get_units, _export_edge,
        to_agraph)

from .jupyter_tools import (
    print_full_edge_provenance,
    create_statement_inspection_table,
)
from .utils import flatMap, iterate, take, ltake, _insert_line_breaks, compose
from .types import RV, LatentVar, Indicator
from .paths import adjectiveData, south_sudan_data
from datetime import datetime
from scipy.stats import gaussian_kde
from itertools import chain, permutations, cycle
from indra.statements import Influence
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
from IPython.display import set_matplotlib_formats
from functools import partial

set_matplotlib_formats("retina")
plt.style.use("ggplot")


def make_edge(
    sts: List[Influence], p: Tuple[str, str]
) -> Tuple[str, str, Dict[str, List[Influence]]]:
    edge = (
        p[0],
        p[1],
        {
            "InfluenceStatements": [
                s for s in sts if (p[0], p[1]) == nameTuple(s)
            ]
        },
    )
    return edge


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
    def from_statements(cls, sts: List[Influence]):
        """ Construct an AnalysisGraph object from a list of INDRA statements. """
        sts = get_valid_statements_for_modeling(sts)
        node_permutations = permutations(get_concepts(sts), 2)
        edges = [
            e
            for e in [make_edge(sts, p) for p in node_permutations]
            if len(e[2]["InfluenceStatements"]) != 0
        ]

        return cls(edges)

    @staticmethod
    def from_pickle(pkl_file: str):
        """ Load an AnalysisGraph object from a pickle file. """
        with open(pkl_file, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, cls):
            raise TypeError(f"The pickled object in {pkl_file} is not an instance of AnalysisGraph")
        else:
            return G


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
                indicator.mean, indicator.unit = get_indicator_value(
                    indicator, time, self.data
                )
                indicator.time = time
                if not indicator.mean is None:
                    indicator.stdev = 0.1 * abs(indicator.mean)
            n[1]["indicators"] = [
                ind for ind in n[1]["indicators"] if ind.mean is not None
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

        gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby(
            "adjective"
        )
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
            e[2]["betas"] = np.tan(
                e[2]["ConditionalProbability"].resample(self.res)[0]
            )


    # ==========================================================================
    # Execution
    # ==========================================================================


    def emission_function(self, s_i, mu_ij, sigma_ij):
        return np.random.normal(s_i*mu_ij, sigma_ij)

    def get_latent_state_components(self) -> List[str]:
        return flatMap(lambda a: (a, f"∂({a})/∂t"), self.nodes())

    def _write_latent_state(self, f):
        for i, s in enumerate(self.latent_state.dataset):
            f.write(str(i) + "," + str(self.get_current_time()) + ",")
            f.write(",".join([str(v) for v in s.values[::2]]) + "\n")

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

    def _repr_png_(self, *args, **kwargs):
        return to_agraph(self, *args, **kwargs).draw(
                format="png", prog=kwargs.get("prog", "dot")
            )

    def visualize(self, *args, **kwargs):
        """ Visualize the analysis graph in a Jupyter notebook cell. """
        from IPython.core.display import Image

        return Image(
            to_agraph(self, *args, **kwargs).draw(
                format="png", prog=kwargs.get("prog", "dot")
            ),
            retina=True,
        )

    def plot_distribution_of_latent_variable(
        self, latent_variable, ax, xlim=None, **kwargs
    ):
        displayName = kwargs.get(
            "displayName", _insert_line_breaks(latent_variable, 30)
        )
        vals = [s[latent_variable] for s in self.latent_state.dataset]
        if xlim is not None:
            ax.set_xlim(*xlim)
            vals = [v for v in vals if ((v > xlim[0]) and (v < xlim[1]))]
        sns.distplot(vals, ax=ax, kde=kwargs.get("kde", True), norm_hist=True)
        ax.set_xlabel(displayName)
        ax.set_ylabel(_insert_line_breaks(f"p({displayName})"))

    def plot_distribution_of_observed_variable(
        self, observed_variable, ax, xlim=None, **kwargs
    ):
        displayName = kwargs.get(
            "displayName", _insert_line_breaks(observed_variable, 30)
        )

        vals = [s[observed_variable] for s in self.observed_state.dataset]
        if xlim is not None:
            ax.set_xlim(*xlim)
            vals = [v for v in vals if (v > xlim[0]) and (v < xlim[1])]
        plt.style.use("ggplot")
        sns.distplot(vals, ax=ax, kde=kwargs.get("kde", True), norm_hist=True)
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
            return to_agraph(self)

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

    def initialize(self, cfg: str = None):
        """ Initialize the executable AnalysisGraph with a config file. """
        if cfg is not None:
            self.s0 = pd.read_csv(
                cfg, index_col=0, header=None, error_bad_lines=False
            )[1]
            for n in self.nodes(data=True):
                n[1]["rv"] = LatentVar(n[0])
                n[1]["update_function"] = self.default_update_function
                node = n[1]["rv"]
                node.dataset = [self.s0[n[0]] for _ in range(self.res)]
                node.partial_t = self.s0[f'∂({n[0]})/∂t']
                if n[1].get('indicators') is not None:
                    for ind in n[1]['indicators']:
                        ind.dataset = np.ones(self.res)*ind.mean

    def default_update_function(self, n):
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

    def update(self):
        """ Advance the model by one time step. """

        next_state = {}

        for n in self.nodes(data=True):
            next_state[n[0]] = n[1]['update_function'](n)

        for n in self.nodes(data=True):
            n[1]["rv"].dataset = next_state[n[0]]
            if n[1].get('indicators') is not None:
                ind = n[1]['indicators'][0]
                ind.dataset = [
                        self.emission_function(x, ind.mean, ind.stdev)
                        for x in n[1]['rv'].dataset
                    ]

        self.t += self.Δt

    def update_until(self, t_final):
        """ Updates the model to a particular time t_final """
        while self.t < t_final:
            self.update()

    def finalize(self):
        pass

    # Model information

    def get_component_name():
        """ Return the name of the model. """
        return "DelphiModel"

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
