#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include <sqlite3.h>

using namespace std;
using namespace delphi::utils;

/*
============================================================================
Private: Initializing model parameters
============================================================================
*/

/**
 * Initialize all the parameters and hyper-parameters of the Delphi model.
 */
void AnalysisGraph::initialize_parameters(int res,
                                          InitialBeta initial_beta,
                                          bool use_heuristic,
                                          bool use_continuous) {
    this->initialize_random_number_generator();
    this->uni_disc_dist = uniform_int_distribution<int>(0, this->num_nodes() - 1);

    this->res = res;
    this->continuous = use_continuous;
    this->data_heuristic = use_heuristic;

    this->find_all_paths();

    if (!causemos_call) {
        // θ pdfs are generated during the create-model call and they are
        // serialized into json. So we do not need to re-generate them now,
        // which is a bit time consuming task.
        // We had to generate θ pdfs while create-model to generate the
        // appropriate model-creation response, which requires edge weights.
        this->construct_theta_pdfs();
    }
    this->init_betas_to(initial_beta);

    this->set_indicator_means_and_standard_deviations();
    this->set_transition_matrix_from_betas();
    this->set_default_initial_state();
    this->set_log_likelihood();

    this->transition_matrix_collection.clear();
    this->initial_latent_state_collection.clear();

    this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);
    this->initial_latent_state_collection = vector<Eigen::VectorXd>(this->res);
}

void AnalysisGraph::init_betas_to(InitialBeta ib) {
  switch (ib) {
  // Initialize the initial β for this edge
  // Note: I am repeating the loop within each case for efficiency.
  // If we embed the switch within the for loop, there will be less code
  // but we will evaluate the switch for each iteration through the loop
  case InitialBeta::ZERO:
    for (EdgeDescriptor e : this->edges()) {
      // β = tan(0.0) = 0
      graph[e].theta = 0.0;
    }
    break;
  case InitialBeta::ONE:
    for (EdgeDescriptor e : this->edges()) {
      // θ = atan(1) = Π/4
      // β = tan(atan(1)) = 1
      graph[e].theta = std::atan(1);
    }
    break;
  case InitialBeta::HALF:
    for (EdgeDescriptor e : this->edges()) {
      // β = tan(atan(0.5)) = 0.5
      graph[e].theta = std::atan(0.5);
    }
    break;
  case InitialBeta::MEAN:
    for (EdgeDescriptor e : this->edges()) {
      graph[e].theta = graph[e].kde.mu;
    }
    break;
  case InitialBeta::RANDOM:
    for (EdgeDescriptor e : this->edges()) {
      // this->uni_dist() gives a random number in range [0, 1]
      // Multiplying by 2 scales the range to [0, 2]
      // Subtracting 1 moves the range to [-1, 1]
      graph[e].theta = this->uni_dist(this->rand_num_generator) * 2 - 1;
    }
    break;
  }
}

void AnalysisGraph::set_indicator_means_and_standard_deviations() {
  int num_verts = this->num_vertices();
  vector<double> mean_sequence;
  vector<double> std_sequence;
  vector<int> ts_sequence;

  for (int v = 0; v < num_verts; v++) {
      Node &n = (*this)[v];

      for (int i = 0; i < n.indicators.size(); i++) {
          Indicator &ind = n.indicators[i];
          double mean = 0.0001;

          // The (v, i) pair uniquely identifies an indicator. It is the i th
          // indicator of v th concept.
          // For each indicator, there could be multiple observations per
          // time step. Aggregate those observations to create a single
          // observation time series for the indicator (v, i).
          // We also ignore the missing values.
          mean_sequence.clear();
          std_sequence.clear();
          ts_sequence.clear();
          for (int ts = 0; ts < this->n_timesteps; ts++) {
              vector<double> &obs_at_ts = this->observed_state_sequence[ts][v][i];

              if (!obs_at_ts.empty()) {
                  // There is an observation for indicator (v, i) at ts.
                  ts_sequence.push_back(ts);
                  mean_sequence.push_back(delphi::utils::mean(obs_at_ts));
                  std_sequence.push_back(delphi::utils::standard_deviation(mean_sequence.back(), obs_at_ts));
              } //else {
                // This is a missing observation
                //}
          }

          // Set the indicator standard deviation
          // NOTE: This is what we have been doing earlier for delphi local
          // run:
          //stdev = 0.1 * abs(indicator.get_mean());
          //stdev = stdev == 0 ? 1 : stdev;
          //indicator.set_stdev(stdev);
          // Surprisingly, I did not see indicator standard deviation being set
          // when delphi is called from CauseMos HMI and in the Indicator class
          // the default standard deviation was set to be 0. This could be an
          // omission. I updated Indicator class so that the default indicator
          // standard deviation is 1.
          double max_std = ranges::max(std_sequence);
          if(!isnan(max_std)) {
              // For indicator (v, i), at least one time step had
              // more than one observation.
              // We can use that to assign the indicator standard deviation.
              // TODO: Is that a good idea?
              ind.set_stdev(max_std);
          }   // else {
              // All the time steps had either 0 or 1 observation.
              // In this case, indicator standard deviation defaults to 1
              // TODO: Is this a good idea?
              //}

          // Set the indicator mean
          string aggregation_method = ind.get_aggregation_method();

          // TODO: Instead of comparing text, it would be better to define an
          // enumerated type say AggMethod and use it. Such an enumerated type
          // needs to be shared between AnalysisGraph and Indicator classes.
          if (aggregation_method.compare("first") == 0) {
              if (ts_sequence[0] == 0) {
                  // The first observation is not missing
                  mean = mean_sequence[0];
              } else {
                  // First observation is missing
                  // TODO: Decide what to do
                  // I feel getting a decaying weighted average of existing
                  // observations with earlier the observation, higher the
                  // weight is a good idea. TODO: If so how to calculate
                  // weights? We need i < j => w_i > w_j and sum of w's = 1.
                  // For the moment as a placeholder, average of the earliest
                  // available observation is used to set the indicator mean.
                  mean = mean_sequence[0];
              }
          }
          else if (aggregation_method.compare("last") == 0) {
              int last_obs_idx = this->n_timesteps - 1;
              if (ts_sequence.back() == last_obs_idx) {
                  // The first observation is not missing
                  mean = mean_sequence.back();
              } else {
                  // First observation is missing
                  // TODO: Similar to "first" decide what to do.
                  // For the moment the observation closest to the final
                  // time step is used to set the indicator mean.
                  ind.set_mean(mean_sequence.back());
              }
          }
          else if (aggregation_method.compare("min") == 0) {
              mean = ranges::min(mean_sequence);
          }
          else if (aggregation_method.compare("max") == 0) {
              mean = ranges::max(mean_sequence);
          }
          else if (aggregation_method.compare("mean") == 0) {
              mean = delphi::utils::mean(mean_sequence);
          }
          else if (aggregation_method.compare("median") == 0) {
              mean = delphi::utils::median(mean_sequence);
          }
          if (mean != 0) {
            ind.set_mean(mean);
          } else {
            // To avoid division by zero error later on
            ind.set_mean(0.0001);
          }
      }
  }
}

/**
 * This is a helper function used by construct_theta_pdfs()
 */
AdjectiveResponseMap construct_adjective_response_map(
    mt19937 gen,
    uniform_real_distribution<double>& uni_dist,
    normal_distribution<double>& norm_dist,
    size_t n_kernels
    ) {
  sqlite3* db = nullptr;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);

  if (rc == 1)
    throw "Could not open db\n";

  sqlite3_stmt* stmt = nullptr;
  const char* query = "select * from gradableAdjectiveData";
  rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);

  AdjectiveResponseMap adjective_response_map;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string adjective =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
    double response = sqlite3_column_double(stmt, 6);
    if (!in(adjective_response_map, adjective)) {
      adjective_response_map[adjective] = {response};
    }
    else {
      adjective_response_map[adjective].push_back(response);
    }
  }

  for (auto& [k, v] : adjective_response_map) {
    v = KDE(v).resample(n_kernels, gen, uni_dist, norm_dist);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
  return adjective_response_map;
}

void AnalysisGraph::construct_theta_pdfs() {

  // The choice of sigma_X and sigma_Y is somewhat arbitrary here - we need to
  // come up with a principled way to select this value, or infer it from data.
  double sigma_X = 1.0;
  double sigma_Y = 1.0;
  AdjectiveResponseMap adjective_response_map =
      construct_adjective_response_map(
          this->rand_num_generator, this->uni_dist, this->norm_dist, this->res);
  vector<double> marginalized_responses;
  for (auto [adjective, responses] : adjective_response_map) {
    for (auto response : responses) {
      marginalized_responses.push_back(response);
    }
  }

  marginalized_responses = KDE(marginalized_responses)
                               .resample(this->res,
                                         this->rand_num_generator,
                                         this->uni_dist,
                                         this->norm_dist);

  for (auto e : this->edges()) {
    vector<double> all_thetas = {};

    for (Statement stmt : this->graph[e].evidence) {
      Event subject = stmt.subject;
      Event object = stmt.object;

      string subj_adjective = subject.adjective;
      string obj_adjective = object.adjective;

      auto subj_responses = lmap(
          [&](auto x) { return x * subject.polarity; },
          get(adjective_response_map, subj_adjective, marginalized_responses));

      auto obj_responses = lmap(
          [&](auto x) { return x * object.polarity; },
          get(adjective_response_map, obj_adjective, marginalized_responses));

      for (auto [x, y] : ranges::views::cartesian_product(subj_responses, obj_responses)) {
        all_thetas.push_back(atan2(sigma_Y * y, sigma_X * x));
      }
    }

    this->graph[e].kde = KDE(all_thetas);
  }
}
