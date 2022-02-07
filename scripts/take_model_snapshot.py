"""
Takes a snapshot of a how Delphi works at a particular point of its implementation
A snapshot taken before and a snapshot taken after a code change such as an
optimization or refactoring could be used to verify whether the accuracy of the
code got changed as a byproduct of the changes made to the code. If there is no
change at all, the before and after snaps should be identical.

Use the snap_diff.py script to compare before and after snaps.
"""

from delphi.cpp.DelphiPython import AnalysisGraph, InitialBeta, InitialDerivative
import pandas as pd
import datetime
import argparse

# _______________________________________________________________________
# Configuration
# For the snaps to be comparable, the configuration should be the same for before and after snaps
# Total number of runs = Number of seeds X Number of models
# Add enough distinct seeds to initialize the random number generator to sample different runs
# More seeds the better. But it would take longer to complete. We suggest 20 as a start.
seeds = [1, 14, 27, 5, 2020, 66, 2001, 38, 20211013, 81, 59, 100, 325, 774, 15, 92, 204, 571, 999, 76]

# Add enough different models to sample runs.
# Better to have models with different number of nodes, edges and different topologies.
# Should have at least one model.
# The experiment json is for possible future extensions of the snapshot script.
# At the moment it could be left as an empty string.
models = [
    ("../tests/data/delphi/create_model_rain--temperature.json",
     "../tests/data/delphi/experiments_rain--temperature--yield.json")
]

burn = 10000
res = 200
kde_kernels = 1000
use_continuous = False
initial_beta = InitialBeta.ZERO  # ONE, HALF, MEAN, MEDIAN, PRIOR, RANDOM
initial_derivative = InitialDerivative.DERI_ZERO  # DERI_PRIOR
belief_score_cutoff = 0
grounding_score_cutoff = 0
# End Configuration
# _______________________________________________________________________

parser = argparse.ArgumentParser(description='Take Delphi Snapshot')
parser.add_argument('-o', default='', type=str, metavar='Output file name suffix',
                    help='If provided, this is added as a suffix to the output file name, \
                     which would be: snap_<suffix>.csv. Otherwise, the output file name \
                     would be: snap_<current date & time>.csv. We suggest using "before" \
                     and "after" as suffixes for before and after snapshots respectively.')

args = parser.parse_args()
suffix = args.o

snaps = []

for model_id, model in enumerate(models):
    for seed in seeds:
        create_model, crete_experiment = model
        print('\n', model_id, seed, '\n')
        G = AnalysisGraph.from_causemos_json_file(filename=create_model,
                                                  belief_score_cutoff=belief_score_cutoff,
                                                  grounding_score_cutoff=grounding_score_cutoff,
                                                  kde_kernels=kde_kernels)
        G.set_random_seed(seed)
        G.run_train_model("take_model_snapshot",
                          res=res,
                          burn=burn,
                          initial_beta=initial_beta,
                          initial_derivative=initial_derivative,
                          use_continuous=use_continuous)
        MAP_ll = G.get_MAP_log_likelihood()

        '''
        Possible extensions
        1. Take snaps of the sampled parameter distributions
        model_state = G.get_complete_state()
        concept_indicators, edges, adjectives, polarities, edge_data, derivatives, \
        data_range, data_set,pred_range, predictions, cis, log_likelihoods = model_state
        
        2. Perform some projections and take snaps of the projections
            (and sampled parameter distributions)
        G.run_causemos_projection_experiment_from_json_file(filename=crete_experiment)
        model_state = G.get_complete_state()
        concept_indicators, edges, adjectives, polarities, edge_data, derivatives, \
        data_range, data_set,pred_range, predictions, cis, log_likelihoods = model_state
        '''

        snaps.append(
            {'Model': model_id,
             'Seed': seed,
             'MAP_ll': MAP_ll}
        )

columns = ('Model', 'Seed', 'MAP_ll')
snaps_df = pd.DataFrame(snaps)

output_file_name = f'snap_{suffix if suffix else datetime.datetime.now()}.csv' \
    .replace(' ', '_').replace(':', '-')
snaps_df.to_csv(output_file_name, index=False, columns=columns)
