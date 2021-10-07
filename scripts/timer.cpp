#include "AnalysisGraph.hpp"
#include "Timer.hpp"
#include "CSVWriter.hpp"
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;

int main(int argc, char* argv[]) {
  variables_map vm;
  bool check = false;

  try {
    options_description desc{"Options"};
    desc.add_options()
    ("help,h", "Help screen")
    ("min-nodes,n", value<int>()->default_value(2), "Minimum number of nodes")
    ("max-nodes,x", value<int>()->default_value(3), "Maximum number of nodes")
    ("multiplicative-factor,m", value<double>()->default_value(1.0f), "number of nodes in the next graph = number of nodes in the current graph * mf + af")
    ("additive-factor,a", value<int>()->default_value(1), "number of nodes in the next graph = number of nodes in the current graph * mf + af")
    ("frac-extra-edges,e", value<double>()->default_value(0.0f), "The proportion of extra edges to be added. Zero means the minimum graph, which is a tree. One means a complete graph.")
    ("num-repeats,i", value<int>()->default_value(2), "Number of times to repeat the experiment for a particular size graph.")
    ("num-obs,d", value<int>()->default_value(48), "Number of observations (training data points) to generate for each node.")
    ("pred-timesteps,p", value<int>()->default_value(24), "Number of time steps to predict.")
    ("burn,b", value<int>()->default_value(10000), "Number of samples to throw away before starting to retain samples.")
    ("res,r", value<int>()->default_value(1000), "Number of samples to retain.")
    ("kernels,k", value<int>()->default_value(1000), "The number of KDE kernels to be used when creating prior distributions for betas.")
    ("noise-variance,v", value<int>()->default_value(16), "Variance of the emission distribution when generating data.")
    ("output-file,o", value<string>()->default_value("timing"), "Output file name. Creates if not exist. Appends otherwise.")
    ("check-config,c", bool_switch(&check), "Check the timing configuration.");

    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << desc << '\n';
      return(0);
    }
  }
  catch (const error &ex)
  {
    std::cerr << ex.what() << '\n';
  }

  int min_nodes = vm["min-nodes"].as<int>();
  int max_nodes = vm["max-nodes"].as<int>();
  double frac_extra_edges = vm["frac-extra-edges"].as<double>();
  double multiplicative_factor = vm["multiplicative-factor"].as<double>();
  int additive_factor = vm["additive-factor"].as<int>();
  int num_repeats = vm["num-repeats"].as<int>();

  int num_obs = vm["num-obs"].as<int>();
  int pred_timesteps = vm["pred-timesteps"].as<int>();
  int burn = vm["burn"].as<int>();
  int res = vm["res"].as<int>();
  int kde_kernels = vm["kernels"].as<int>();
  double noise_variance = vm["noise-variance"].as<int>();
  string output_file = vm["output-file"].as<string>() + "_" + to_string(min_nodes) + "-" + to_string(max_nodes) + ".csv";
  cout << "The output is stored in: " << output_file << endl;

  std::pair<std::vector<std::string>, std::vector<long>> durations;
  CSVWriter writer(output_file);
  vector<string> headings = {"Runs", "Nodes", "Edges", "Create", "Train", "Predict"};
  writer.write_row(headings.begin(), headings.end());

  for (int run = 1; run <= num_repeats; ++run) {
    cout << "\n\nRun: " << run << "\n";
    for (int nodes = min_nodes; nodes <= max_nodes; nodes = (nodes < 16? nodes + 1 : lround(nodes * multiplicative_factor + additive_factor))) {
      int max_extra_edges = nodes - 1; //ceil((nodes - 1) * (nodes - 1) * frac_extra_edges);
      for (int extra_edges = 0; extra_edges <= max_extra_edges; extra_edges = (extra_edges < 16? extra_edges + 1 : extra_edges * 2)) {
        cout << "\n\tNodes: " << nodes << "  \tExtra edges: " << extra_edges << "\n";
        if (check) {
          continue;
        }
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
      }
    }
  }
  return(0);
}
