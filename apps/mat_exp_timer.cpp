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

    vector<double> unique_gaps = {1, 2, 5};

    int node_jump = 4;
    vector<AnalysisGraph> ags;

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
//                G.to_png(output_file_prefix + fmt::to_string(nodes) + "_"
//                             + fmt::to_string(extra_edges + nodes - 1) + "_"
//                             + fmt::to_string(run) + ".png",
//                         false, 1, "", "TB", false);
                ags.push_back(G);
            }
        }
    }

    for (int cag_id = 0; cag_id < ags.size(); cag_id++) {
        for (bool multi_threaded : {true, false}) {
            for (vector<double> unique_gaps : vector<vector<double>>({{1}, {1, 2, 5}})) {
                cout << cag_id << "/" << ags.size() << " - "
                     << (multi_threaded ? "mt\n" : "st\n");
                string output_file_prefix_2 = output_file_prefix + "--" + to_string(unique_gaps.size()) + "--";
                AnalysisGraph G = ags[cag_id];
                G.profile_matrix_exponential(cag_id,
                                             output_file_prefix_2,
                                             unique_gaps,
                                             res,
                                             multi_threaded);
            }
        }
    }
    return(0);
}