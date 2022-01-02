#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptors.hpp>
#include "CSVWriter.hpp"

using namespace std;
using namespace delphi::utils;
using Eigen::VectorXd;
using fmt::print;


/*
 ============================================================================
 Private: Synthetic Data Experiment
 ============================================================================
*/

/*
 ============================================================================
 Public: Synthetic data experiment
 ============================================================================
*/

AnalysisGraph AnalysisGraph::generate_random_CAG(unsigned int num_nodes,
                                                 unsigned int num_extra_edges) {
  AnalysisGraph G;

  G.initialize_random_number_generator();

  // Get all adjectives.
  vector<string> adjectives;
  AdjectiveResponseMap adjective_response_map =
      G.construct_adjective_response_map(G.rand_num_generator, G.uni_dist, G.norm_dist, 100);
  boost::range::copy(adjective_response_map | boost::adaptors::map_keys,
              back_inserter(adjectives));

  vector<int> cag_nodes = {0};
  vector<int> rand_node(2);
  vector<string> rand_adjectives(2);
  int polarity = 0;
  string source = "";
  string target = "";
  int src_idx = 0;

  for (rand_node[1] = 1; rand_node[1] < num_nodes; rand_node[1]++) {
    sample(cag_nodes.begin(), cag_nodes.end(), rand_node.begin(), 1, G.rand_num_generator);
    cag_nodes.push_back(rand_node[1]);

    src_idx = G.uni_dist(G.rand_num_generator) < 0.5? 0 : 1;
    source = to_string(rand_node[src_idx]);
    target = to_string(rand_node[1 - src_idx]);

    sample(adjectives.begin(), adjectives.end(), rand_adjectives.begin(), 2, G.rand_num_generator);
    polarity = G.uni_dist(G.rand_num_generator) < 0.5 ? 1 : -1;

    auto causal_fragment =
        CausalFragment({rand_adjectives[0], 1, source},
                       {rand_adjectives[1], polarity, target});
    G.add_edge(causal_fragment);
  }

  num_extra_edges = min(num_extra_edges, (num_nodes - 1) * (num_nodes - 1));

  pair<EdgeDescriptor, bool> edge;

  for (int _ = 0; _ < num_extra_edges; _++) {
    edge.second = true;
    while (edge.second) {
      sample(cag_nodes.begin(), cag_nodes.end(), rand_node.begin(), 2, G.rand_num_generator);
      src_idx = G.uni_dist(G.rand_num_generator) < 0.5? 0 : 1;
      source = to_string(rand_node[src_idx]);
      target = to_string(rand_node[1 - src_idx]);
      edge = boost::edge(G.get_vertex_id(source),
                   G.get_vertex_id(target), G.graph);
    }

    sample(adjectives.begin(), adjectives.end(), rand_adjectives.begin(), 2, G.rand_num_generator);
    polarity = G.uni_dist(G.rand_num_generator) < 0.5 ? 1 : -1;

    auto causal_fragment =
        CausalFragment({rand_adjectives[0], 1, source},
                       {rand_adjectives[1], polarity, target});
    G.add_edge(causal_fragment);
  }

  RNG::release_instance();
  return G;
}

/**
 * TODO: This is very similar to initialize_parameters() method defined in
 * parameter_initialization.cpp. Might be able to merge the two
 * @param kde_kernels Number of KDE kernels to use when constructing beta prior distributions
 * @param initial_beta How to initialize betas
 * @param initial_derivative How to initialize derivatives
 * @param use_continuous Whether to use matrix exponential or not
 */
void AnalysisGraph::initialize_random_CAG(unsigned int num_obs,
                                          unsigned int kde_kernels,
                                          InitialBeta initial_beta,
                                          InitialDerivative initial_derivative,
                                          bool use_continuous) {
  this->initialize_random_number_generator();
  this->set_default_initial_state(initial_derivative, true);
  this->n_kde_kernels = kde_kernels;
  this->construct_theta_pdfs();
  this->init_betas_to(initial_beta, true);
  this->pred_timesteps = num_obs + 1;
  this->continuous = use_continuous;
  this->find_all_paths();
  this->set_transition_matrix_from_betas();
  this->transition_matrix_collection.clear();
  this->initial_latent_state_collection.clear();
  // TODO: We are using this variable for two different purposes.
  // create another variable.
  this->res = 1;
  this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);
  this->initial_latent_state_collection = vector<Eigen::VectorXd>(this->res);
  this->transition_matrix_collection[0] = this->A_original;
  this->initial_latent_state_collection[0] = this->s0;
}


void AnalysisGraph::generate_synthetic_data(unsigned int num_obs,
                                            double noise_variance,
                                            unsigned int kde_kernels,
                                            InitialBeta initial_beta,
                                            InitialDerivative initial_derivative,
                                            bool use_continuous) {
  this->initialize_random_CAG(num_obs, kde_kernels, initial_beta, initial_derivative, use_continuous);
  vector<int> periods = {2, 3, 4, 6, 12};
  vector<int> rand_period(1);
  uniform_real_distribution<double> centers_dist(-100, 100);
  int max_samples_per_period = 12;
  for (int v : this->head_nodes) {
    Node& n = (*this)[v];
    sample(periods.begin(), periods.end(), rand_period.begin(), 1, this->rand_num_generator);
    n.period = rand_period[0];
    int gap_size = max_samples_per_period / n.period;
    int offset = 0; // 0 <= offset < period
    vector<int> filled_months;
    for (int p = 0; p < n.period; p++) {
      double center = centers_dist(this->rand_num_generator);
      double spread = this->norm_dist(this->rand_num_generator) * 5;
      n.centers.push_back(center);
      n.spreads.push_back(spread);
      int month = offset + gap_size * p;
      n.generated_monthly_latent_centers_for_a_year[month] = center;
      n.generated_monthly_latent_spreads_for_a_year[month] = spread;
      filled_months.push_back(month);
    }
    this->interpolate_missing_months(filled_months, n);
  }

  for (int v = 0; v < this->num_vertices(); v++) {
    Node& n = (*this)[v];
    n.add_indicator("ind_" + n.name, "synthetic");
    n.mean = centers_dist(this->rand_num_generator);
    while (n.mean == 0) {
      n.mean = centers_dist(this->rand_num_generator);
    }
  }

  this->generate_latent_state_sequences(0);
  this->generate_observed_state_sequences();

  this->observed_state_sequence.clear();
  this->observation_timestep_gaps.clear();
  this->n_timesteps = num_obs;

  // Access (concept is a vertex in the CAG)
  // [ timestep ][ concept ][ indicator ][ observation ]
  this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);
  this->observation_timestep_gaps = vector<double>(this->n_timesteps, 1);
  this->observation_timestep_gaps[0] = 0;

  int num_verts = this->num_vertices();
  // Fill in observed state sequence
  // NOTE: This code is very similar to the implementations in
  // set_observed_state_sequence_from_data and get_observed_state_from_data
  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->observed_state_sequence[ts] = vector<vector<vector<double>>>(num_verts);

    for (int v = 0; v < num_verts; v++) {
      Node& n = (*this)[v];
      this->observed_state_sequence[ts][v] = vector<vector<double>>(n.indicators.size());

      for (int i = 0; i < n.indicators.size(); i++) {
        this->observed_state_sequence[ts][v][i] = vector<double>();
        this->observed_state_sequence[ts][v][i].push_back(
            this->predicted_observed_state_sequences[0][ts + 1][v][i] +
            noise_variance * this->norm_dist(this->rand_num_generator));
      }
    }
  }
  RNG::release_instance();
}

void AnalysisGraph::interpolate_missing_months(vector<int> &filled_months, Node &n) {
  sort(filled_months.begin(), filled_months.end());

  // Interpolate values for the missing months
  if (filled_months.size() > 1) {
    for (int i = 0; i < filled_months.size(); i++) {
      int month_start = filled_months[i];
      int month_end = filled_months[(i + 1) % filled_months.size()];

      int num_missing_months = 0;
      if (month_end > month_start) {
        num_missing_months = month_end - month_start - 1;
      }
      else {
        num_missing_months = (11 - month_start) + month_end;
      }

      for (int month_missing = 1;
      month_missing <= num_missing_months;
      month_missing++) {
        n.generated_monthly_latent_centers_for_a_year
        [(month_start + month_missing) % 12] =
            ((num_missing_months - month_missing + 1) *
            n.generated_monthly_latent_centers_for_a_year
            [month_start] +
            (month_missing)*n
            .generated_monthly_latent_centers_for_a_year
            [month_end]) /
            (num_missing_months + 1);

        n.generated_monthly_latent_spreads_for_a_year
        [(month_start + month_missing) % 12] =
            ((num_missing_months - month_missing + 1) *
            n.generated_monthly_latent_spreads_for_a_year
            [month_start] +
            (month_missing)*n
            .generated_monthly_latent_spreads_for_a_year
            [month_end]) /
            (num_missing_months + 1);
      }
    }
  } else if (filled_months.size() == 1) {
    for (int month = 0; month < n.generated_monthly_latent_centers_for_a_year.size(); month++) {
      n.generated_monthly_latent_centers_for_a_year[month] =
          n.generated_monthly_latent_centers_for_a_year
          [filled_months[0]];
      n.generated_monthly_latent_spreads_for_a_year[month] =
          n.generated_monthly_latent_spreads_for_a_year
          [filled_months[0]];
    }
  }
}

pair<int, double> AnalysisGraph::assess_model_fit(string output_file_prefix,
                                                  int cag_id,
                                                  int seed) {
    if (!this->trained) {
        cout << "WARNING: The model is not trained. "
                "Cannot assess the quality of the fit\n";
        return make_pair(0, 0.0);
    }

    // Calculate the MAP sample fit
    Eigen::MatrixXd& A_MAP = this->transition_matrix_collection
                                [this->MAP_sample_number];
    vector<double> theta_errors(this->res);
    double tot_MAP_squared_error_theta = 0.0;
    int n_nodes = this->num_nodes();
    int n_edges = this->num_edges();
    CSVWriter fitness_writer = CSVWriter(output_file_prefix +
                                         to_string(cag_id) + "_" +
                                         to_string(n_nodes) + "-" +
                                         to_string(n_edges) + "_" +
                                         to_string(seed) + "_" +
                                         to_string(this->coin_flip_thresh) +
                                         "_model_fit_"
                                         + delphi::utils::get_timestamp() +
                                         ".csv");
    vector<string> headings = {"Run", "Nodes", "Edges", "Source", "Target",
                               "True Theta", "MAP Theta", "True - MAP",
                               "Edge RMSE", "Edge Error Mean",
                               "Edge Error Std"};
    fitness_writer.write_row(headings.begin(), headings.end());
    vector<string> row;

    for (EdgeDescriptor ed : this->edges()) {
        string source_name = (*this)[boost::source(ed, this->graph)].name;
        string target_name = (*this)[boost::target(ed, this->graph)].name;
        int source_id = this->name_to_vertex[source_name];
        int target_id = this->name_to_vertex[target_name];

        double theta_MAP = atan(A_MAP(target_id * 2, source_id * 2 + 1));
        theta_MAP = theta_MAP < 0 ? M_PI + theta_MAP : theta_MAP;

        double theta_gt = this->graph[ed].get_theta_gt();

        double theta_error_MAP = theta_gt - theta_MAP;
        tot_MAP_squared_error_theta += theta_error_MAP * theta_error_MAP;

        for (int i = 0; i < this->res; i++) {
            theta_errors[i] = theta_gt - this->graph[ed].sampled_thetas[i];
        }

        double rmse_edge = sqrt(inner_product(theta_errors.begin(),
                                              theta_errors.end(),
                                              theta_errors.begin(), 0.0)
                                / this->res);
        double mean_error_edge = accumulate(theta_errors.begin(),
                                            theta_errors.end(), 0.0)
                                 / theta_errors.size();
        double tot_error_diff_edge = 0.0;
        std::for_each (
            theta_errors.begin(), theta_errors.end(),
                      [&](const double err) {
                                double error_diff = err - mean_error_edge;
                                tot_error_diff_edge += error_diff * error_diff;
                                            });

        double std_error_edge = sqrt(tot_error_diff_edge / (this->res - 1));

        row.clear();
        row = {to_string(cag_id), to_string(n_nodes), to_string(n_edges),
               source_name, target_name, to_string(theta_gt),
               to_string(theta_MAP), to_string(theta_error_MAP),
               to_string(rmse_edge), to_string(mean_error_edge),
               to_string(std_error_edge)};
        fitness_writer.write_row(row.begin(), row.end());
    }

    double rmse_MAP = sqrt(tot_MAP_squared_error_theta / n_edges);
    row.clear();
    row = {to_string(cag_id), to_string(n_nodes), to_string(n_edges),
           "MAP RMSE", "", "", "", "", to_string(rmse_MAP)};
    fitness_writer.write_row(row.begin(), row.end());
    headings = {"Run", "Nodes", "Edges", "Node", "True Derivative",
                "MAP Derivative", "True - MAP", "Node RMSE", "Node Error Mean",
                "Node Error Std"};
    row.clear();
    row = {""};
    fitness_writer.write_row(row.begin(), row.end()); // Blank row
    fitness_writer.write_row(headings.begin(), headings.end());

    Eigen::VectorXd s0_MAP = this->initial_latent_state_collection
                                 [this->MAP_sample_number];
    double tot_MAP_squared_error_deri = 0.0;
    vector<double> deri_errors(this->res);

    for (int node_id : this->body_nodes) {
        int deri_idx = 2 * node_id + 1;
        double deri_MAP = s0_MAP(deri_idx);
        double deri_gt = this->s0_gt(deri_idx);
        double deri_error_MAP = deri_gt - deri_MAP;
        tot_MAP_squared_error_deri += deri_error_MAP * deri_error_MAP;

        for (int i = 0; i < this->res; i++) {
            deri_errors[i] = deri_gt - this->initial_latent_state_collection
                                            [i](deri_idx);
        }

        double rmse_deri = sqrt(inner_product(deri_errors.begin(),
                                              deri_errors.end(),
                                              deri_errors.begin(), 0.0)
                                / this->res);
        double mean_error_deri = accumulate(deri_errors.begin(),
                                            deri_errors.end(), 0.0)
                                 / this->res;
        double tot_error_diff_deri = 0.0;
        std::for_each (
            deri_errors.begin(), deri_errors.end(),
            [&](const double err) {
                double error_diff = err - mean_error_deri;
                tot_error_diff_deri += error_diff * error_diff;
            });

        double std_error_deri = this->res > 1
                                ? sqrt(tot_error_diff_deri / (this->res - 1))
                                : 0;

        Node& n = (*this)[node_id];

        row.clear();
        row = {to_string(cag_id), to_string(n_nodes), to_string(n_edges),
               n.name, to_string(deri_gt),
               to_string(deri_MAP), to_string(deri_error_MAP),
               to_string(rmse_deri), to_string(mean_error_deri),
               to_string(std_error_deri)};
        fitness_writer.write_row(row.begin(), row.end());
    }

    rmse_MAP = sqrt(tot_MAP_squared_error_deri / this->body_nodes.size());
    row.clear();
    row = {to_string(cag_id), to_string(n_nodes), to_string(n_edges),
           "MAP RMSE", "", "", "", to_string(rmse_MAP)};
    fitness_writer.write_row(row.begin(), row.end());

    int tot_parameters = this->body_nodes.size() + n_edges;
    double tot_MAP_squared_error_all = tot_MAP_squared_error_deri +
                                        tot_MAP_squared_error_theta;
    double rmse_MAP_all = sqrt(tot_MAP_squared_error_all / tot_parameters);
    row.clear();
    row = {to_string(cag_id), to_string(n_nodes), to_string(n_edges),
           "MAP RMSE All", "", "", "", to_string(rmse_MAP_all)};
    fitness_writer.write_row(row.begin(), row.end());

    return make_pair(this->body_nodes.size() + n_edges,
                     tot_MAP_squared_error_deri + tot_MAP_squared_error_theta);
}

