#!/usr/bin/env python

import os
import sys
import pickle
from pathlib import Path
from glob import glob
from datetime import datetime
from typing import Any
from pandas import read_csv
from functools import partial
from delphi.utils import ltake
from argparse import (
    ArgumentParser,
    ArgumentTypeError,
    FileType,
    ArgumentDefaultsHelpFormatter,
)
from delphi.api import *

def create_model(args):
    from delphi.api import parameterize
    from delphi.export import to_json
    from delphi.assembly import get_faostat_wdi_data, assemble_model

    with open(args.indra_statements, "rb") as f:
        sts = pickle.load(f)
        sts = get_valid_statements(sts)
        time = datetime(2012, 1,1)
        df = get_faostat_wdi_data("data/south_sudan_data.csv")
        cag = create_qualitative_analysis_graph(sts)
        cag = add_transition_model(cag, args.adjective_data)
        cag = add_indicators(cag)
        cag = parameterize(cag, time, df)
    return cag


def execute_model(args):
    from delphi.core import (
        load_model,
        construct_default_initial_state,
        get_latent_state_components,
        sample_sequences,
        write_sequences_to_file,
    )

    CAG = load_model(args.input_dressed_cag)

    s0 = read_csv(args.input_variables_path, index_col=0, header=None, error_bad_lines=False)[1]

    write_sequences_to_file(
        CAG,
        sample_sequences(CAG, s0, args.steps, args.samples, args.dt),
        args.output_sequences,
    )


def positive_real(arg, x):
    try:
        val = float(x)
    except ValueError:
        raise ArgumentTypeError(
            f"{arg} should be a positive real number (you entered {x})."
        )

    if not val > 0.:
        raise ArgumentTypeError(
            f"{arg} should be a positive real number (you entered {x})."
        )
    return x


def positive_int(arg, x):
    try:
        val = int(x)
    except ValueError:
        raise ArgumentTypeError(
            f"{arg} should be a positive integer (you entered {x})."
        )

    if not val > 0:
        raise ArgumentTypeError(
            f"{arg} should be a positive integer (you entered {x})."
        )

    return val


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Dynamic Bayes Net Executable Model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    def add_arg(arg: str, help: str, type: Any, default: Any) -> None:
        parser.add_argument("--" + arg, help=help, type=type, default=default)

    parser.add_argument(
        "--create_model",
        help="Export the dressed CAG as a "
        "pickled object, the variables with default initial values as a CSV"
        " file, and the link structure of the CAG as a JSON file.",
        action="store_true",
    )

    parser.add_argument(
        "--create_cra_cag",
        help="Export CAG in JSON format for " "Charles River Analytics",
        action="store_true",
    )

    add_arg(
        "indra_statements",
        "Pickle file containing INDRA statements",
        str,
        Path(__file__).parents[0] / "data" / "curated_statements.pkl",
    )

    add_arg(
        "adjective_data",
        "Path to the gradable adjective data file.",
        str,
        Path(__file__).parents[0] / "data" / "adjectiveData.tsv",
    )

    parser.add_argument(
        "--execute_model",
        help="Execute DBN and sample time " "evolution sequences",
        action="store_true",
    )

    add_arg("dt", "Time step size", partial(positive_real, "dt"), 1.0)

    add_arg(
        "n_statements",
        "Number of INDRA statements to take from the "
        "pickled object containing them",
        partial(positive_int, "n_statements"),
        5,
    )

    add_arg(
        "steps",
        "Number of time steps to take",
        partial(positive_int, "steps"),
        5,
    )

    add_arg(
        "samples",
        "Number of sequences to sample",
        partial(positive_int, "samples"),
        100,
    )

    add_arg(
        "output_sequences",
        "Output file containing sampled sequences",
        str,
        "dbn_sampled_sequences.csv",
    )

    add_arg(
        "output_cag_json",
        "Path to the output CAG JSON file",
        str,
        "delphi_cag.json",
    )

    add_arg(
        "input_variables_path",
        "Path to the variables of the input dressed cag",
        str,
        "variables.csv",
    )

    add_arg(
        "output_variables_path",
        "Path to the variables of the output dressed cag",
        str,
        "variables.csv",
    )

    add_arg(
        "input_dressed_cag",
        "Path to the input dressed cag",
        str,
        "dressed_CAG.pkl",
    )

    add_arg(
        "output_dressed_cag",
        "Path to the output dressed cag",
        str,
        "dressed_CAG.pkl",
    )

    add_arg(
        "concept_to_indicator_mapping",
        "Path to the YAML file containing the mapping between concepts and indicators.",
        str,
        Path(__file__).parents[0] / "data" / "concept_to_indicator_mapping.yml",
    )
    add_arg(
        "south_sudan_data",
        "Path to the file containing the data for FAO and WDI indicators for South Sudan",
        str,
        Path(__file__).parents[0] / "data" / "south_sudan_data.csv",
    )

    add_arg(
        "year", "Year to get the indicator variable values for", str, "2012"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    if args.create_model:
        cag = create_model(args)
        with open(args.output_dressed_cag, "wb") as f:
            pickle.dump(cag, f)
        to_json(cag)

    if args.execute_model:
        execute_model(args)
