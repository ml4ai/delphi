#include "AnalysisGraph.hpp"
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
    ("min-nodes,n", value<int>()->default_value(5), "Minimum number of nodes")
    ("max-nodes,x", value<int>()->default_value(9), "Maximum number of nodes")
    ("num-repeats,i", value<int>()->default_value(2), "Number of times to repeat the experiment for a particular size graph.")
    ("num-obs,d", value<int>()->default_value(48), "Number of observations (training data points) to generate for each node.")
    ("burn,b", value<int>()->default_value(10000), "Number of samples to throw away before starting to retain samples.")
    ("res,r", value<int>()->default_value(1000), "Number of samples to retain.")
    ("kernels,k", value<int>()->default_value(1000), "The number of KDE kernels to be used when creating prior distributions for betas.")
    ("noise-variance,v", value<int>()->default_value(16), "Variance of the emission distribution when generating data.")
    ("output-file,o", value<string>()->default_value("fitness"), "Output file name. Creates if not exist. Appends otherwise.")
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
  int num_repeats = vm["num-repeats"].as<int>();

  int num_obs = vm["num-obs"].as<int>();
  int burn = vm["burn"].as<int>();
  int res = vm["res"].as<int>();
  int kde_kernels = vm["kernels"].as<int>();
  double noise_variance = vm["noise-variance"].as<int>();
  string output_file_prefix = vm["output-file"].as<string>() + "_";

  string output_file = vm["output-file"].as<string>() + "_RMSEs_" +
                       to_string(min_nodes) + "-" +
                       to_string(max_nodes) + "_" +
                       delphi::utils::get_timestamp() + ".csv";

  /*
  vector<double> row;
  CSVWriter writer(output_file);
  vector<string> headings = {"Seed", "Theta Sampling Probability", "RMSE"};
  writer.write_row(headings.begin(), headings.end());
   */

  int node_jump = 4;
  vector<AnalysisGraph> ags;
  vector<int> seeds = {1, 14, 27, 5, 2020};
  vector<double> theta_probs = {0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01};

  for (int run = 1; run <= num_repeats; ++run) {
    cout << "\n\nRun: " << run << "\n";
    for (int nodes = min_nodes; nodes <= max_nodes; nodes += node_jump) {
      int max_extra_edges = nodes - 1;
      int edge_jump = int(max_extra_edges / node_jump);
      for (int extra_edges = 0; extra_edges <= max_extra_edges; extra_edges += edge_jump) {
        cout << "\n\tNodes: " << nodes << "  \tExtra edges: " << extra_edges << "\n";
        if (check) {
          continue;
        }
        AnalysisGraph G;

        G = AnalysisGraph::generate_random_CAG(nodes, extra_edges);
        G.generate_synthetic_data(num_obs,
                                   noise_variance,
                                   kde_kernels,
                         InitialBeta::PRIOR,
                     InitialDerivative::DERI_PRIOR,
                     false);
        G.to_png(output_file_prefix + fmt::to_string(nodes) + "_"
                 + fmt::to_string(extra_edges + nodes - 1) + "_"
                 + fmt::to_string(run) + ".png",
                 false, 1, "", "TB", false);
        ags.push_back(G);
      }
    }
  }

  //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // To debug delphi being unable to open the delphi.db
  /*
  double theta_prob = 0.3;
  int seed = 14;
  int cag_id = 30;
  AnalysisGraph G = ags[cag_id];
  cout << "\nCag ID: " << cag_id << endl;
  cout << "\tNodes: " << G.num_vertices() << endl;
  cout << "\tEdges: " << G.num_edges() << endl;
  cout << "\tTheta prob: " << theta_prob << endl;
  cout << "\tSeed: " << seed << endl;
  G.set_random_seed(seed);
  G.run_train_model(res,
                    burn,
                    InitialBeta::ZERO,
                    InitialDerivative::DERI_ZERO,
                    theta_prob);
  pair<int, double> MAP_squared_error =
      G.assess_model_fit(output_file_prefix, cag_id, seed);
  return(0);
   */
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  int model_no = 0;

  for (double theta_prob : theta_probs) {

      int tot_parameters_all_seeds = 0;
      double tot_MAP_squared_error_all_seeds = 0.0;

      for (int seed : seeds) {

          int tot_parameters = 0;
          double tot_MAP_squared_error = 0.0;

          for (int cag_id = 0; cag_id < ags.size(); cag_id++) {
              AnalysisGraph G = ags[cag_id];
              cout << "\n\nCag ID: " << cag_id << "/" << ags.size() << endl;
              cout << "\tTraining model: " << model_no++ << endl;
              cout << "\tNodes: " << G.num_vertices() << endl;
              cout << "\tEdges: " << G.num_edges() << endl;
              cout << "\tTheta prob: " << theta_prob << endl;
              cout << "\tSeed: " << seed << endl;
              G.set_random_seed(seed);
              G.debug();
              /*
              G.run_train_model(res,
                                burn,
                                InitialBeta::ZERO,
                                InitialDerivative::DERI_ZERO,
                                theta_prob);
              pair<int, double> MAP_squared_error =
                  G.assess_model_fit(output_file_prefix, cag_id, seed);
              tot_parameters += MAP_squared_error.first;
              tot_MAP_squared_error += MAP_squared_error.second;
               */
              //cout << MAP_squared_error.first << ", " << MAP_squared_error.second << endl;
          }
          /*
          tot_parameters_all_seeds += tot_parameters;
          tot_MAP_squared_error_all_seeds += tot_MAP_squared_error;
          double grand_MAP_rmse = sqrt(tot_MAP_squared_error / tot_parameters);
          //cout << tot_parameters << ", " << tot_MAP_squared_error << ", " << grand_MAP_rmse << endl;
          row.clear();
          row = {(double)seed, theta_prob, grand_MAP_rmse};
          writer.write_row(row.begin(), row.end());
           */
      }
      /*
      double grand_MAP_rmse_all_seeds = sqrt(tot_MAP_squared_error_all_seeds / tot_parameters_all_seeds);
      row.clear();
      row = {0, theta_prob, grand_MAP_rmse_all_seeds};
      writer.write_row(row.begin(), row.end());
       */
  }
  return(0);
}
