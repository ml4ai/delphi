#!/usr/bin/env python

import os
import argparse
from future.utils import lfilter
from delphi.core import *
from glob import glob
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Bayes Net Executable Model')
    parser.add_argument('--dt', metavar='Δt', help="Time step")
    parser.parse_args()
    # with open('eval_indra_assembled.pkl', 'rb') as f:
        # sts = ltake(10, filter(isSimulable, pickle.load(f)))
    # CAG = add_conditional_probabilities(construct_CAG_skeleton(sts))
    # export_model(CAG)
    CAG = load_model('CAG.pkl')
    s0 = construct_default_initial_state(get_latent_state_components(CAG))
    A=sample_transition_matrix(CAG)
    seqs = sample_sequences(CAG, s0, 2, 1)
    print(seqs)
