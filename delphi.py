#!/usr/bin/env python

import os
import sys
import pickle
from glob import glob
from typing import Any
from pandas import read_csv
from functools import partial
from delphi.utils import ltake
from argparse import (ArgumentParser, ArgumentTypeError, FileType,
                      ArgumentDefaultsHelpFormatter)

def create_model(args):
    from delphi.core import (isSimulable, add_conditional_probabilities,
                             construct_CAG_skeleton, export_to_ISI)

    with open(args.indra_statements, 'rb') as f:
        export_to_ISI(add_conditional_probabilities(construct_CAG_skeleton(
            ltake(5, filter(isSimulable, pickle.load(f))))))


def execute_model(args):
    from delphi.core import (load_model, construct_default_initial_state,
                             get_latent_state_components, sample_sequences,
                             write_sequences_to_file)

    CAG = load_model(args.dressed_cag)

    if not args.init_values:
        s0 = construct_default_initial_state(get_latent_state_components(CAG))
    else:
        s0 = read_csv(args.init_values, index_col=0, header=None)[1]

    write_sequences_to_file(CAG, sample_sequences(CAG, s0, args.steps,
        args.samples, args.dt), args.output)


def positive_real(arg, x):
    try:
        val = float(x)
    except ValueError:
        raise ArgumentTypeError(
                f"{arg} should be a positive real number (you entered {x}).")

    if not val > 0.:
        raise ArgumentTypeError(
                f"{arg} should be a positive real number (you entered {x}).")
    return x


def positive_int(arg, x):
    try:
        val = int(x)
    except ValueError:
        raise ArgumentTypeError(
                f"{arg} should be a positive integer (you entered {x}).")

    if not val > 0:
        raise ArgumentTypeError(
                f"{arg} should be a positive integer (you entered {x}).")

    return val


if __name__ == '__main__':
    parser = ArgumentParser(description='Dynamic Bayes Net Executable Model',
            formatter_class=ArgumentDefaultsHelpFormatter)

    def add_arg(arg: str, help: str, type: Any, default: Any) -> None:
        parser.add_argument('--'+arg, help=help, type=type, default=default)

    parser.add_argument('--execute_model', help='Execute DBN and sample time '
            'evolution sequences', action="store_true")

    parser.add_argument('--create_model', help='Export the dressed CAG as a '
            'pickled object, the variables with default initial values as a CSV'
            ' file, and the link structure of the CAG as a JSON file.',
            action="store_true")

    add_arg('indra_statements', 'Pickle file containing INDRA statements', str,
            'delphi/data/eval_indra_assembled.pkl')

    add_arg('dt', 'Time step size', partial(positive_real, 'dt'), 1.0)

    add_arg('steps', "Number of time stepsto take",
            partial(positive_int, 'steps'), 5)

    add_arg('samples', "Number of sequences to sample",
            partial(positive_int, 'samples'), 100)

    add_arg('init_values', "CSV file containing initial values of variables",
            FileType('r'), 'variables.csv')

    add_arg('dressed_cag', 'Pickle file containing the dressed CAG',
            FileType('rb'), 'dressedCAG.pkl')

    add_arg('output', 'Output file containing sampled sequences',
            FileType('w', encoding='UTF-8'), 'output.csv')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    if args.create_model:
        create_model(args)

    if args.execute_model:
        execute_model(args)
