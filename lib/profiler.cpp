#include "AnalysisGraph.hpp"
#include <tqdm.hpp>
#include "utils.hpp"
#include "Timer.hpp"
#include "CSVWriter.hpp"
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <future>

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

    this->edge_sample_pool.clear();
    for (EdgeDescriptor ed : this->edges()) {
        if (!this->graph[ed].is_frozen()) {
            this->edge_sample_pool.push_back(ed);
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

    pair<std::vector<std::string>, std::vector<long>> durations_kde;
    pair<std::vector<std::string>, std::vector<long>> durations_me;
    pair<std::vector<std::string>, std::vector<long>> durations_uptm;

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
        this->graph[e[0]].set_theta(this->graph[e[0]].get_theta() + this->norm_dist(this->rand_num_generator));
        KDE& kde = this->graph[e[0]].kde;

        {
          durations_uptm.first.clear();
          durations_uptm.second.clear();
          durations_uptm.first = {"Run", "Nodes", "Edges", "KDE Kernels"};
          durations_uptm.second = {run,n_nodes, n_edges, long(this->n_kde_kernels)};
          Timer t = Timer("UPTM", durations_uptm);

          this->update_transition_matrix_cells(e[0]);
        }

        durations_uptm.first.push_back("Sample Type");
        durations_uptm.second.push_back(12);

        writer.write_row(durations_uptm.second.begin(), durations_uptm.second.end());

        {
            durations_kde.first.clear();
            durations_kde.second.clear();
            durations_kde.first = {"Run", "Nodes", "Edges", "KDE Kernels"};
            durations_kde.second = {run,n_nodes, n_edges, long(this->n_kde_kernels)};
            Timer t = Timer("KDE", durations_kde);

            this->graph[e[0]].compute_logpdf_theta();
        }

        durations_kde.first.push_back("Sample Type");
        durations_kde.second.push_back(10);

        writer.write_row(durations_kde.second.begin(), durations_kde.second.end());

        {
            durations_me.first.clear();
            durations_me.second.clear();
            durations_me.first = {"Run", "Nodes", "Edges", "KDE Kernels"};
            durations_me.second = {run,n_nodes, n_edges, long(this->n_kde_kernels)};
            Timer t = Timer("ME", durations_me);

            #ifdef _OPENMP
                this->e_A_ts.clear();
                #pragma omp parallel
                {
                    unordered_map<double, Eigen::MatrixXd> partial_e_A_ts;
                    for (int i = 0; i < this->observation_timestep_unique_gaps.size();
                         i++) {
                        int gap = this->observation_timestep_unique_gaps[i];
                        partial_e_A_ts[gap] = (this->A_original * gap).exp();
                    }
                    #pragma omp critical
                    this->e_A_ts.merge(partial_e_A_ts);
                    #pragma omp barrier
                }
            #else
                for (auto [gap, mat] : this->e_A_ts) {
                    this->e_A_ts[gap] = (this->A_original * gap).exp();
                }
            #endif
        }

        durations_me.first.push_back("Sample Type");
        durations_me.second.push_back(11);

        writer.write_row(durations_me.second.begin(), durations_me.second.end());
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

mutex g_display_mutex;
static Eigen::MatrixXd compute_matrix_exponential(const Eigen::Ref<const Eigen::MatrixXd> A, double gap) {
    thread::id this_id = std::this_thread::get_id();
    g_display_mutex.lock();
    cout << "\nthread: " << this_id << endl;
    g_display_mutex.unlock();

    return (A * gap).exp();
}


void AnalysisGraph::profile_matrix_exponential(int run, std::string file_name_prefix,
                                               std::vector<double> unique_gaps,
                                               int repeat,
                                               bool multi_threaded) {
    this->initialize_random_number_generator();
    int n_nodes = this->num_nodes();
    int n_edges = this->num_edges();

    pair<std::vector<std::string>, std::vector<long>> durations_me;

    string filename = file_name_prefix +
                      (multi_threaded ? "mt_" : "st_") +
                      to_string(n_nodes) + "-" +
                      to_string(n_edges) + "_" +
                      to_string(run) + "_" +
                      delphi::utils::get_timestamp() + ".csv";
    CSVWriter writer(filename);
    vector<string> headings = {"Run", "Nodes", "Edges",
                               "KDE Kernels", "Wall Clock Time (ns)",
                               "CPU Time (ns)", "Sample Type"};
    writer.write_row(headings.begin(), headings.end());
    cout << filename << endl;

    vector<future<Eigen::MatrixXd>> matrix_exponentials(unique_gaps.size());

    if (multi_threaded) {
        thread::id this_id = std::this_thread::get_id();
        cout << "\nmain thread: " << this_id << endl;
        cout << "\nmax hardware threads: " << thread::hardware_concurrency() << endl;
    }

    this->observation_timestep_unique_gaps = unique_gaps;

    this->res = repeat;

    cout << "\nProfiling Matrix Exponential\n";
    cout << "\nRunning Matrix Exponential for " << this->res << " times..." << endl;
    for (int i : trange(this->res)) {
        // Randomly pick an edge ≡ θ
        boost::iterator_range edge_it = this->edges();

        vector<EdgeDescriptor> e(1);
        sample(
            edge_it.begin(), edge_it.end(), e.begin(), 1, this->rand_num_generator);

        // Perturb the θ
        this->graph[e[0]].set_theta(this->graph[e[0]].get_theta() +
                                    this->norm_dist(this->rand_num_generator) / 10);
        KDE& kde = this->graph[e[0]].kde;

        this->update_transition_matrix_cells(e[0]);
        this->graph[e[0]].compute_logpdf_theta();

        durations_me.first.clear();
        durations_me.second.clear();
        durations_me.first = {"Run", "Nodes", "Edges", "KDE Kernels"};
        durations_me.second = {run,n_nodes, n_edges, long(this->n_kde_kernels)};
        {
            Timer t = Timer("ME", durations_me);

            if (multi_threaded) {
                /*
                this->e_A_ts.clear();
                #pragma omp parallel
                {
                    unordered_map<double, Eigen::MatrixXd> partial_e_A_ts;
                    for (int i = 0;
                         i < this->observation_timestep_unique_gaps.size();
                         i++) {
                        int gap = this->observation_timestep_unique_gaps[i];
                        partial_e_A_ts[gap] = (this->A_original * gap).exp();
                    }
                    #pragma omp critical
                    //dumb++;
                    this->e_A_ts.merge(partial_e_A_ts);
                    #pragma omp barrier
                }
                 */
                for (int i = 0;
                     i < this->observation_timestep_unique_gaps.size();
                     i++) {
                    int gap = this->observation_timestep_unique_gaps[i];
                    matrix_exponentials[i] = async(launch::async,// | launch::deferred,
                                                   compute_matrix_exponential,
                                                   A_original, gap);
                }
                for (int i = 0;
                     i < this->observation_timestep_unique_gaps.size();
                     i++) {
                    int gap = this->observation_timestep_unique_gaps[i];
                    this->e_A_ts[gap] = matrix_exponentials[i].get();
                }
            }
            else {
                for (double gap : this->observation_timestep_unique_gaps) {
                    this->e_A_ts[gap] = (this->A_original * gap).exp();
                }
            }
        }

        /*
        for (double gap :unique_gaps) {
            cout << "\ngap: " << gap << endl;
            cout << this->e_A_ts[gap] << endl;
        }
         */

        durations_me.first.push_back("Sample Type");
        durations_me.second.push_back(multi_threaded ? 13
                                                     : 11);

        writer.write_row(durations_me.second.begin(), durations_me.second.end());
    }
    cout << endl;
    RNG::release_instance();
    cout << "\nmax hardware threads: " << thread::hardware_concurrency() << endl;
}

