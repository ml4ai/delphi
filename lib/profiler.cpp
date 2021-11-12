#include "AnalysisGraph.hpp"
#include <tqdm.hpp>
#include "utils.hpp"
#include "Timer.hpp"
#include "CSVWriter.hpp"

using namespace std;
using tq::trange;

void AnalysisGraph::initialize_profiler(int res,
                                        int kde_kernels,
                                        InitialBeta initial_beta,
                                        InitialDerivative initial_derivative,
                                        bool use_continuous) {
    unordered_set<int> train_vertices =
        unordered_set<int>
        (this->node_indices().begin(), this->node_indices().end());

    this->concept_sample_pool.clear();
    for (int vert : train_vertices) {
      if (this->head_nodes.find(vert) == this->head_nodes.end()) {
        this->concept_sample_pool.push_back(vert);
      }
    }

    this->n_kde_kernels = kde_kernels;

    this->initialize_parameters(res, initial_beta, initial_derivative,
                            false, use_continuous);
}

void AnalysisGraph::profile_mcmc(int run, string file_name_prefix) {
    this->n_timesteps = this->observed_state_sequence.size();
    int n_nodes = this->num_nodes();
    int n_edges = this->num_edges();

    pair<std::vector<std::string>, std::vector<long>> durations;

    string filename = file_name_prefix + "_" +
                      to_string(n_nodes) + "-" +
                      to_string(n_edges) + "_" +
                      to_string(run) + "_" +
                      delphi::utils::get_timestamp() + ".csv";
    CSVWriter writer(filename);
    vector<string> headings = {"Run", "Nodes", "Edges", "KDE Kernels", "Wall Clock Time (ns)", "CPU Time (ns)", "Sample Type"};
    writer.write_row(headings.begin(), headings.end());
    cout << filename << endl;

    cout << "\nProfiling the MCMC\n";
    cout << "\nSampling " << this->res << " samples from posterior..." << endl;
    for (int i : trange(this->res)) {
        {
            durations.first.clear();
            durations.second.clear();
            durations.first = {"Run", "Nodes", "Edges", "KDE Kernels"};
            durations.second = {run,n_nodes, n_edges, long(this->n_kde_kernels)};
            Timer t = Timer("MCMC", durations);

            this->sample_from_posterior();
        }

        durations.first.push_back("Sample Type");
        durations.second.push_back(this->coin_flip < this->coin_flip_thresh ? 1
                                                                            : 0);
        writer.write_row(durations.second.begin(), durations.second.end());

        this->transition_matrix_collection[i] = this->A_original;
        this->initial_latent_state_collection[i] = this->s0;
    }
    cout << endl;

    this->trained = true;
    RNG::release_instance();
}


void AnalysisGraph::profile_kde(int run, string file_name_prefix) {
    this->initialize_random_number_generator();
    int n_nodes = this->num_nodes();
    int n_edges = this->num_edges();

    pair<std::vector<std::string>, std::vector<long>> durations;

    string filename = file_name_prefix + "_" +
                      to_string(n_nodes) + "-" +
                      to_string(n_edges) + "_" +
                      to_string(run) + "_" +
                      delphi::utils::get_timestamp() + ".csv";
    CSVWriter writer(filename);
    vector<string> headings = {"Run", "Nodes", "Edges", "KDE Kernels", "Wall Clock Time (ns)", "CPU Time (ns)", "Sample Type"};
    writer.write_row(headings.begin(), headings.end());
    cout << filename << endl;

    int num_iterations = int(this->res / 2);

    cout << "\nProfiling the KDE\n";
    cout << "\nRunning KDE for " << num_iterations << " times..." << endl;
    for (int i : trange(num_iterations)) {
        // Randomly pick an edge ≡ θ
        boost::iterator_range edge_it = this->edges();

        vector<EdgeDescriptor> e(1);
        sample(
            edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator);

        // Perturb the θ
        this->graph[e[0]].theta += this->norm_dist(this->rand_num_generator);
        KDE& kde = this->graph[e[0]].kde;

        {
            durations.first.clear();
            durations.second.clear();
            durations.first = {"Run", "Nodes", "Edges", "KDE Kernels"};
            durations.second = {run,n_nodes, n_edges, long(this->n_kde_kernels)};
            Timer t = Timer("KDE", durations);

            kde.logpdf(this->graph[e[0]].theta);
        }

        durations.first.push_back("Sample Type");
        durations.second.push_back(10);

        writer.write_row(durations.second.begin(), durations.second.end());
    }
    cout << endl;
    RNG::release_instance();
}


void AnalysisGraph::profile_prediction(int run, int pred_timesteps, string file_name_prefix) {
    if (!this->trained) {
      cout << "\n\n\t****ERROR: Untrained model! Cannot generate projections...\n\n";
      return;
    }

    int n_nodes = this->num_nodes();
    int n_edges = this->num_edges();

    pair<std::vector<std::string>, std::vector<long>> durations;

    string filename = file_name_prefix + "_" +
                      to_string(n_nodes) + "-" +
                      to_string(n_edges) + "_" +
                      to_string(run) + "_" +
                      delphi::utils::get_timestamp() + ".csv";
    CSVWriter writer(filename);
    vector<string> headings = {"Run", "Nodes", "Edges", "Timesteps", "Wall Clock Time (ns)", "CPU Time (ns)"};
    writer.write_row(headings.begin(), headings.end());
    cout << filename << endl;

    cout << "\nProfiling prediction\n";
    cout << "\nRunning prediction for " << pred_timesteps + 1 << " times..." << endl;

    {
        durations.first.clear();
        durations.second.clear();
        durations.first = {"Run", "Nodes", "Edges", "Timesteps"};
        durations.second = {run,n_nodes, n_edges, pred_timesteps + 1};
        Timer t = Timer("Prediction", durations);

        this->generate_prediction(1, pred_timesteps);
    }

    writer.write_row(durations.second.begin(), durations.second.end());
    cout << endl;
}

