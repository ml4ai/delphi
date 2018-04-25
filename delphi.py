#!/usr/bin/env python

import os
from argparse import ArgumentParser, ArgumentTypeError
from future.utils import lfilter
from glob import glob
import pickle
from functools import partial
from pandas import read_csv
from delphi.core import *

def test_export():
    with open('eval_indra_assembled.pkl', 'rb') as f:
        sts = ltake(2, filter(isSimulable, pickle.load(f)))
    CAG = add_conditional_probabilities(construct_CAG_skeleton(sts))
    export_to_ISI(CAG)

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

    add_arg('--dt', 'Time step size', partial(positive_real, 'dt'), 1.0)
    add_arg('--steps', "Number of time stepsto take",
            partial(positive_int, 'steps'), 10)
    add_arg('--samples', "Number of sequences to sample",
            partial(positive_int, 'samples'), 100)

    add_arg('--init_values',
            "CSV file containing initial values of variables", str, None)

    add_arg('--cag', 'Pickle file containing the dressed CAG', str, 'dressedCAG.pkl')

    args = parser.parse_args()

    CAG = load_model('dressedCAG.pkl')

    if not args.init_values:
        s0 = construct_default_initial_state(get_latent_state_components(CAG))
    else:
        s0 = read_csv(args.init_values, index_col=0, header=None)[1]

    seqs = sample_sequences(CAG, s0, args.steps, args.samples, args.dt)
    write_sequences_to_file(CAG, seqs)
