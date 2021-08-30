#include "AnalysisGraph.hpp"
#include "Timer.hpp"
#include "CSVWriter.hpp"
#include "dbg.h"
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;

int main(int argc, char* argv[]) {
  std::pair<std::vector<std::string>, std::vector<long>> durations;
  CSVWriter writer("timing.csv");
  vector<string> headings = {"Runs", "Nodes", "Edges", "Create", "Train", "Predict"};
  writer.write_row(headings.begin(), headings.end());

  int min_nodes = 2;
  int max_nodes = 20;
  int max_extra_edges = 0;
  int num_repeats = 10;

  int num_obs = 48;
  int pred_timesteps = 24;
  int burn = 10000;
  int res = 1000;
  int kde_kernels = 1000;
  double noise_variance = 16;

  for (int nodes = min_nodes; nodes <= max_nodes; ++nodes) {
    for (int extra_edges = 0; extra_edges <= max_extra_edges; ++extra_edges) {
      for (int run = 1; run <= num_repeats; ++run) {
        AnalysisGraph G1;

        durations.first.clear();
        durations.second.clear();
        durations.first = {"Runs", "Nodes", "Edges"};
        durations.second = {run, nodes, nodes - 1 + extra_edges};

        {
          Timer t = Timer("create", durations);
          G1 = AnalysisGraph::generate_random_CAG(nodes, extra_edges);
          G1.generate_synthetic_data(num_obs,
                                     noise_variance,
                                     kde_kernels,
                                     InitialBeta::PRIOR,
                                     InitialDerivative::DERI_PRIOR,
                                     false);
        }
        {
          Timer t = Timer("train", durations);
          G1.run_train_model(res,
                             burn,
                             InitialBeta::ZERO,
                             InitialDerivative::DERI_ZERO,
                             false);
        }
        {
          Timer t = Timer("predict", durations);
          G1.generate_prediction(1, pred_timesteps);
        }
        writer.write_row(durations.second.begin(), durations.second.end());
        //dbg(durations.first);
        //cout << endl;
        //dbg(durations.second);
        //cout << endl << run << "  |  " << nodes << "  |  " << nodes - 1 + extra_edges << endl << "---------------------\n";
      }
    }
  }
  return(0);
  }
