#!/usr/bin/env python

import os
import argparse
from future.utils import lfilter
from glob import glob
import pickle
from delphi.core import *

def main():
    with open('eval_indra_assembled.pkl', 'rb') as f:
        sts = ltake(10, filter(isSimulable, pickle.load(f)))
    CAG = add_conditional_probabilities(construct_CAG_skeleton(sts))
    export_model_to_json(CAG)
    # with open('CAG.pkl', 'wb') as f:
        # export_model(CAG, f)
    # with open('CAG.pkl', 'rb') as f:
        # CAG = load_model(f)
    # s0 = construct_default_initial_state(get_latent_state_components(CAG))
    # seqs = sample_sequences(CAG, s0, 2, 1, args.dt)
    # print(seqs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Dynamic Bayes Net Executable Model')
    parser.add_argument('dt', metavar='Δt', help="Time step size", type=float)
    args = parser.parse_args()
    if args.dt <= 0.:
        print("Time step should be a positive real number.")
    else:
        main()

