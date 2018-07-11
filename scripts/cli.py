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
    from delphi.assembly import get_data
    from delphi.api import (
        create_qualitative_analysis_graph,
        get_valid_statements_for_modeling,
    )
    from datetime import datetime

    with open(args.indra_statements, "rb") as f:
        sts = get_valid_statements_for_modeling(pickle.load(f))

    G = create_qualitative_analysis_graph(sts)
    G.add_transition_model(args.adjective_data)
    G.add_indicators()
    G.parameterize(datetime(args.year, 1, 1), get_faostat_wdi_data(args.data))

    return G


def execute_model(args):
    from delphi.api import load
    from pandas import read_csv

    G = load(args.input_dressed_cag)
    G.initialize(args.input_variables_path)
    latent_states = [
        G.sample_sequence_of_latent_states(s0, args.steps, args.dt)
        for n in range(args.samples)
    ]
    G._write_sequences_to_file(latent_states, args.output_sequences)


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
    data_dir = Path(__file__).parents[0] / "data"

    parser = ArgumentParser(
        description="Dynamic Bayes Net Executable Model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )


    def add_flag(short_arg: str, long_arg: str, help: str):
        parser.add_argument(
            "-" + short_arg, "--" + long_arg, help=help, action="store_true"
        )

    subparsers = parser.add_subparsers()

    parser_create = subparsers.add_parser("create", help="Model creation")
    parser_execute = subparsers.add_parser("execute", help="Model execution")

    #==========================================================================
    # Model creation options
    #==========================================================================

    parser_create.add_argument(
        "--output_dressed_cag",
        help="Path to the output dressed cag",
        type=str,
        default="dressed_CAG.pkl",
    )

    parser_create.add_argument(
        "--indra_statements",
        type=str,
        help="Pickle file containing INDRA statements",
        default=data_dir / "curated_statements.pkl",
    )

    parser_create.add_argument(
        "--adjective_data",
        help="Path to the gradable adjective data file.",
        type=str,
        default=data_dir / "adjectiveData.tsv",
    )

    parser_create.add_argument(
        "--dt",
        help="Time step size",
        type=partial(positive_real, "dt"),
        default=1.0,
    )

    parser_create.add_argument(
        "--output_variables",
        help="Path to the variables of the output dressed cag",
        type=str,
        default="variables.csv",
    )

    parser_create.add_argument(
        "--concept_to_indicator_mapping",
        help = "Path to the file containing the mapping between concepts and indicators.",
        type=str,
        default=data_dir / "concept_to_indicator_mapping.yml",
    )

    parser_create.add_argument(
        "--data",
        help="Path to the file containing the data for FAO and WDI indicators for South Sudan",
        type=str,
        default=data_dir / "south_sudan_data.csv",
    )

    parser_create.add_argument(
        "--year",
        help="Year to get the indicator variable values for",
        type=int,
        default=2012,
    )

    #==========================================================================
    # Model execution options
    #==========================================================================

    parser_execute.add_argument(
        "input_dressed_cag",
        help="Path to the input dressed cag",
        type=str,
        default="dressed_CAG.pkl",
    )

    parser_execute.add_argument(
        "--steps",
        help="Number of time steps to take",
        type=partial(positive_int, "steps"),
        default=5,
    )

    parser_execute.add_argument(
        "--samples",
        help="Number of sequences to sample",
        type=partial(positive_int, "samples"),
        default=100,
    )

    parser_execute.add_argument(
        "--output_sequences",
        help="Output file containing sampled sequences",
        type=str,
        default="dbn_sampled_sequences.csv",
    )

    parser_execute.add_argument(
        "--output_cag_json",
        help="Path to the output CAG JSON file",
        type=str,
        default="delphi_cag.json",
    )

    parser_execute.add_argument(
        "--input_variables",
        help="Path to the variables of the input dressed cag",
        type=str,
        default="variables.csv",
    )


    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    if args.create:
        G = create_model(args)
        G.export(
            format="full",
            json_file=args.output_cag_json,
            pickle_file=args.output_dressed_cag,
            variables_file=args.output_variables_path,
        )

    if args.execute:
        execute_model(args)
