#!/usr/bin/env python

import os
import pickle
from glob import glob
from typing import Any
from pandas import read_csv
from functools import partial
from argparse import ArgumentParser, ArgumentTypeError

def assemble_and_export_model():
    from delphi.core import (isSimulable, add_conditional_probabilities,
                             construct_CAG_skeleton, export_to_ISI)
    with open('eval_indra_assembled.pkl', 'rb') as f:
        sts = ltake(2, filter(isSimulable, pickle.load(f)))
    CAG = add_conditional_probabilities(construct_CAG_skeleton(sts))
    export_to_ISI(CAG)

def execute_model(args):
    from delphi.core import (load_model, construct_default_initial_state,
                             get_latent_state_components, sample_sequences,
                             write_sequences_to_file)

    CAG = load_model(args.dressed_cag)

    if not args.init_values:
        s0 = construct_default_initial_state(get_latent_state_components(CAG))
    else:
        s0 = read_csv(args.init_values, index_col=0, header=None)[1]

    if args.execute_model:
        seqs = sample_sequences(CAG, s0, args.steps, args.samples, args.dt)
        write_sequences_to_file(CAG, seqs)

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
    parser = ArgumentParser(description='Dynamic Bayes Net Executable Model')

    def add_arg(arg: str, help: str, type: Any, default: Any) -> None:
        parser.add_argument(arg, help=help, type=type, default=default)

    parser.add_argument('--execute_model',
            help='Execute DBN and sample time evolution sequences',
            action="store_true")
    parser.add_argument('--create_model',
            help='Export the dressed CAG as a pickled object, the variables with
            'default initial values as a csv file, and the link structure of the
            'CAG as a json file.', action="store_true")

    add_arg('--dt', 'Time step size', partial(positive_real, 'dt'), 1.0)
    add_arg('--steps', "Number of time stepsto take",
            partial(positive_int, 'steps'), 10)
    add_arg('--samples', "Number of sequences to sample",
            partial(positive_int, 'samples'), 100)
    add_arg('--init_values',
            "CSV file containing initial values of variables", str, None)
    add_arg('--dressed_cag', 'Pickle file containing the dressed CAG', str,
            'dressedCAG.pkl')

    args = parser.parse_args()

    if args.execute_model:
        execute_model(args)
