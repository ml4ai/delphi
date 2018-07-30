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


def create(args):
    print("Creating model")
    from .assembly import get_data
    from .parameterization import parameterize
    from .quantification import map_concepts_to_indicators
    from datetime import datetime
    from .AnalysisGraph import AnalysisGraph
    from .export import export

    with open(args.indra_statements, "rb") as f:
        sts = pickle.load(f)

    G = AnalysisGraph.from_statements(sts)
    G.infer_transition_model(args.adjective_data)
    G = map_concepts_to_indicators(G, 2)
    G = parameterize(G, datetime(args.year, 1, 1), get_data(args.data))
    export(
        G,
        format="full",
        json_file=args.output_cag_json,
        pickle_file=args.output_dressed_cag,
        variables_file=args.output_variables,
    )


def execute(args):
    from pandas import read_csv
    from .AnalysisGraph import AnalysisGraph
    from .execution import _write_latent_state, get_latent_state_components
    from .bmi import initialize, update

    print("Executing model")
    G = AnalysisGraph.from_pickle(args.input_dressed_cag)

    initialize(G, args.input_variables)
    with open(args.output_sequences, "w") as f:
        f.write(
            ",".join(
                ["seq_no", "variable"]+[f"sample_{str(i)}" for i in range(1,G.res+1)]
            )
            + "\n"
        )

        for t in range(args.steps):
            update(G)
            _write_latent_state(G, f)


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


def main():
    data_dir = Path(__file__).parents[1] / "data"

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
    parser_create.set_defaults(func=create)

    parser_execute = subparsers.add_parser("execute", help="Model execution")
    parser_execute.set_defaults(func=execute)

    # ==========================================================================
    # Model creation options
    # ==========================================================================

    parser_create.add_argument(
        "--output_dressed_cag",
        help="Path to the output dressed cag",
        type=str,
        default="delphi_cag.pkl",
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
        help="Path to the file containing the mapping between concepts and indicators.",
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

    # ==========================================================================
    # Model execution options
    # ==========================================================================

    parser_execute.add_argument(
        "--input_dressed_cag",
        help="Path to the input dressed cag",
        type=str,
        default="delphi_cag.pkl",
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

    parser_create.add_argument(
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

    args.func(args)
