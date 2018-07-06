#!/usr/bin/env python

import os
import sys
import pickle
from pathlib import Path
from typing import Any
from functools import partial
from argparse import (
    ArgumentParser,
    ArgumentTypeError,
    FileType,
    ArgumentDefaultsHelpFormatter,
)
import delphi
import delphi.core


def create_model(args):
    from delphi.export import to_json
    from delphi.assembly import get_faostat_wdi_data
    from delphi.api import (
        parameterize,
        create_qualitative_analysis_graph,
        add_transition_model,
        add_indicators,
        get_valid_statements_for_modeling
    )
    from delphi.utils import rcompose
    from datetime import datetime

    parameterize_for_south_sudan = partial(
        parameterize,
        datetime(2012, 1, 1),
        get_faostat_wdi_data(Path(__file__).parents[0] + "/data/south_sudan_data.csv"),
    )

    create_cag = rcompose(
        create_qualitative_analysis_graph,
        partial(add_transition_model, args.adjective_data),
        partial(add_indicators, 1),
        parameterize_for_south_sudan,
    )

    with open(args.indra_statements, "rb") as f:
        sts = get_valid_statements_for_modeling(pickle.load(f))

    cag = create_cag(sts)

    return cag


def export_model(cag, args):
    import pickle
    from delphi.export import to_json, export_default_variables

    with open(args.output_dressed_cag, "wb") as f:
        pickle.dump(cag, f)

    to_json(cag, args.output_cag_json)
    export_default_variables(cag, args)


def execute_model(args):
    from delphi.api import load
    from pandas import read_csv
    from delphi.core import (
        sample_sequence_of_latent_states,
        write_sequences_to_file,
    )

    CAG = load(args.input_dressed_cag)
    s0 = read_csv(
        args.input_variables_path,
        index_col=0,
        header=None,
        error_bad_lines=False,
    )[1]
    latent_states = [
        sample_sequence_of_latent_states(CAG, s0, args.steps, args.dt)
        for n in range(args.samples)
    ]
    write_sequences_to_file(CAG, latent_states, args.output_sequences)



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

    parser.add_subparsers()
    def add_arg(long_arg, help, argtype, default):
        parser.add_argument(
            "--" + long_arg, help=help, type=argtype, default=default
        )

    def add_flag(short_arg: str, long_arg: str, help: str):
        parser.add_argument(
            "-" + short_arg, "--" + long_arg, help=help, action="store_true"
        )

    add_flag("c", "create", "Create model")
    add_flag("x", "execute", "Execute model")

    data_dir = Path(__file__).parents[0] / "data"

    add_arg(
        "indra_statements",
        "Pickle file containing INDRA statements",
        str,
        data_dir / "curated_statements.pkl",
    )

    add_arg(
        "adjective_data",
        "Path to the gradable adjective data file.",
        str,
        data_dir / "adjectiveData.tsv",
    )

    add_arg("dt", "Time step size", partial(positive_real, "dt"), 1.0)
    add_arg("steps", "Number of time steps to take", partial(positive_int, "steps"), 5,)
    add_arg("samples", "Number of sequences to sample", partial(positive_int, "samples"), 100,)
    add_arg("output_sequences", "Output file containing sampled sequences", str, "dbn_sampled_sequences.csv",)
    add_arg("output_cag_json", "Path to the output CAG JSON file", str, "delphi_cag.json",)
    add_arg("input_variables_path", "Path to the variables of the input dressed cag", str, "variables.csv",)
    add_arg("output_variables_path", "Path to the variables of the output dressed cag", str, "variables.csv",)
    add_arg("input_dressed_cag", "Path to the input dressed cag", str, "dressed_CAG.pkl",)
    add_arg("output_dressed_cag", "Path to the output dressed cag", str, "dressed_CAG.pkl",)

    add_arg(
        "concept_to_indicator_mapping",
        "Path to the YAML file containing the mapping between concepts and indicators.",
        str,
        data_dir / "concept_to_indicator_mapping.yml",
    )

    add_arg(
        "south_sudan_data",
        "Path to the file containing the data for FAO and WDI indicators for South Sudan",
        str,
        data_dir / "south_sudan_data.csv",
    )

    add_arg(
        "year", "Year to get the indicator variable values for", str, "2012"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    if args.create:
        cag = create_model(args)
        export_model(cag, args)

    if args.execute:
        execute_model(args)
