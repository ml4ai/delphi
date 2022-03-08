#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include <sqlite3.h>
#include <limits.h>

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
                                          InitialDerivative initial_derivative,
                                          bool use_heuristic,
                                          bool use_continuous) {
    this->initialize_random_number_generator();
    this->uni_disc_dist = uniform_int_distribution<int>
                                (0, this->concept_sample_pool.size() - 1);
    this->uni_disc_dist_edge = uniform_int_distribution<int>
                                (0, this->edge_sample_pool.size() - 1);

    this->res = res;
    this->continuous = use_continuous;
    this->data_heuristic = use_heuristic;

    this->find_all_paths();
    this->construct_theta_pdfs();
    this->init_betas_to(initial_beta);
    this->set_indicator_means_and_standard_deviations();
    this->assemble_base_LDS(initial_derivative);
//    this->set_transition_matrix_from_betas();
    #ifdef MULTI_THREADING
        this->compute_multiple_matrix_exponentials_parallelly();
    #endif
//    this->derivative_prior_variance = 0.1;
//    this->set_default_initial_state(initial_derivative);
    //this->generate_head_node_latent_sequences(-1, this->n_timesteps);
    if (this->head_node_model == HeadNodeModel::HNM_NAIVE) {
        this->generate_head_node_latent_sequences(
            -1,
            accumulate(this->modeling_timestep_gaps.begin() + 1,
                       this->modeling_timestep_gaps.end(),
                       0) +
                1);
    }
    this->set_log_likelihood();
    this->log_likelihood_MAP = -(DBL_MAX - 1);

    this->transition_matrix_collection.clear();
    this->initial_latent_state_collection.clear();
    //this->latent_mean_collection.clear();
    //this->latent_std_collection.clear();
    //this->latent_mean_std_collection.clear();

    this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);
    this->initial_latent_state_collection = vector<Eigen::VectorXd>(this->res);
    //this->latent_mean_collection = vector<vector<double>>(this->res);
    //this->latent_std_collection = vector<vector<double>>(this->res);
    //this->latent_mean_std_collection = vector<vector<
    //                      unordered_map<int, pair<double, double>>>>(this->res);
}

void AnalysisGraph::init_betas_to(InitialBeta ib) {
  if (this->MAP_sample_number > -1) {
    // Warm start using the MAP estimate of the previous training run
    for (EdgeDescriptor e : this->edges()) {
      this->graph[e].set_theta(this->graph[e].sampled_thetas[this->MAP_sample_number]);
      this->graph[e].compute_logpdf_theta();
    }
    return;
  }

  switch (ib) {
  // Initialize the initial β for this edge
  // Note: I am repeating the loop within each case for efficiency.
  // If we embed the switch within the for loop, there will be less code
  // but we will evaluate the switch for each iteration through the loop
  case InitialBeta::ZERO:
    for (EdgeDescriptor e : this->edges()) {
      // β = tan(0.0) = 0
      this->graph[e].set_theta(0.0);
      this->graph[e].compute_logpdf_theta();
    }
    break;
  case InitialBeta::ONE:
    for (EdgeDescriptor e : this->edges()) {
      // θ = atan(1) = Π/4
      // β = tan(atan(1)) = 1
      this->graph[e].set_theta(std::atan(1));
      this->graph[e].compute_logpdf_theta();
    }
    break;
  case InitialBeta::HALF:
    for (EdgeDescriptor e : this->edges()) {
      // β = tan(atan(0.5)) = 0.5
      this->graph[e].set_theta(std::atan(0.5));
      this->graph[e].compute_logpdf_theta();
    }
    break;
  case InitialBeta::MEAN:
    for (EdgeDescriptor e : this->edges()) {
      this->graph[e].set_theta(this->graph[e].kde.mu);
      this->graph[e].compute_logpdf_theta();
    }
    break;
  case InitialBeta::MEDIAN:
      for (EdgeDescriptor e : this->edges()) {
        this->graph[e].set_theta(median(this->graph[e].kde.dataset));
        this->graph[e].compute_logpdf_theta();
      }
      break;
    case InitialBeta::PRIOR:
      for (EdgeDescriptor e : this->edges()) {
        this->graph[e].set_theta(this->graph[e].kde.most_probable_theta);
        this->graph[e].compute_logpdf_theta();
      }
      break;
  case InitialBeta::RANDOM:
    for (EdgeDescriptor e : this->edges()) {
      // this->uni_dist() gives a random number in range [0, 1]
      // Multiplying by M_PI scales the range to [0, M_PI]
      this->graph[e].set_theta(this->uni_dist(this->rand_num_generator) * M_PI);
      this->graph[e].compute_logpdf_theta();
    }
    break;
  }
}

void AnalysisGraph::set_indicator_means_and_standard_deviations() {
  int num_verts = this->num_vertices();
  vector<double> mean_sequence;
  vector<double> std_sequence;
  vector<int> ts_sequence;

  if (this->observed_state_sequence.empty()) {
    return;
  }

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
              if (this->observed_state_sequence[ts][v].empty()) {
                  // This concept has no indicator specified in the create-model
                  // call
                  continue;
              }
              vector<double> &obs_at_ts = this->observed_state_sequence[ts][v][i];

              if (!obs_at_ts.empty()) {
                  // There is an observation for indicator (v, i) at ts.
                  ts_sequence.push_back(ts);
                  mean_sequence.push_back(delphi::utils::mean(obs_at_ts));
                  std_sequence.push_back(delphi::utils::standard_deviation(mean_sequence.back(), obs_at_ts));

                  if (i == 0 && n.period > 1 &&
                      this->head_nodes.find(v) != this->head_nodes.end()) {
                      // Only use observations of the 1st indicator attached to
                      // each concept. Seasonal Fourier model is applied only to
                      // head nodes with period > 1.

                      // For all the concepts ts = 0 is the start
                      int partition = ts % n.period;

                      // Note: if there are multiple observations for some
                      // time steps, the length of time step vector and the
                      // observation vector will be different.
                      n.partitioned_data[partition].first.push_back(ts);

                      // Partition the raw observations for now.
                      // Since we fit the seasonal Fourier model for the latent
                      // variables, we have to divide these raw observations by
                      // the scaling factor (for now we are using the mean) of
                      // the first indicator, which we do not know at the moment
                      // (we are in the process of computing the mean) before
                      // fitting the fourier curve.
                      n.partitioned_data[partition].second.insert(
                                     n.partitioned_data[partition].second.end(),
                                     obs_at_ts.begin(),
                                     obs_at_ts.end());
                      n.tot_observations += obs_at_ts.size();
                  }
              } //else {
                // This is a missing observation
                //}
          }

          if (mean_sequence.empty()) {
              continue;
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

          // TODO: Just for debugging delete
          //mean = 1;

          if (mean != 0) {
              ind.set_mean(mean);
          } else {
              // To avoid division by zero error later on
              ind.set_mean(0.0001);
          }

          // Set mean and standard deviation of the concept based on the mean
          // and the standard deviation of the first indicator attached to it.
          if (i == 0) {
              if (n.has_min) {
                  if (n.indicators[0].mean > 0) {
                      n.min_val = n.min_val_obs / n.indicators[0].mean;
                  } else if (n.indicators[0].mean < 0) {
                      n.max_val = n.min_val_obs / n.indicators[0].mean;
                  }
              }
              if (n.has_max) {
                  if (n.indicators[0].mean > 0) {
                      n.max_val = n.max_val_obs / n.indicators[0].mean;
                  } else if (n.indicators[0].mean < 0) {
                      n.min_val = n.max_val_obs / n.indicators[0].mean;
                  }
              }

              for (auto& [parititon, obs_in_parititon]: n.partitioned_data) {
                  transform(obs_in_parititon.second.begin(), obs_in_parititon.second.end(),
                            obs_in_parititon.second.begin(),
                            [&](double obs){return obs / n.indicators[0].mean;});
              }

              transform(mean_sequence.begin(), mean_sequence.end(),
                        mean_sequence.begin(),
                      [&](double obs_mean){return obs_mean / n.indicators[0].mean;});

              // Computing the midpoints
              if (this->head_node_model == HNM_FOURIER && n.period > 1 &&
                  this->head_nodes.find(v) != this->head_nodes.end()) {
                  // Midpoints are only needed for head nodes with period > 1 to
                  // fit the Fourier decomposition based seasonal model.
                  this->linear_interpolate_between_bin_midpoints(v, ts_sequence,
                                                                 mean_sequence);
              }

              /*
              for (int ts = 0; ts < ts_sequence.size(); ts++) {
                  // TODO: I feel that this partitioning is worng. Should be corrected as:
                  // First we convert the observation time steps for an indicator into a
                  // zero based contiguous sequence and then take the modules
                  // int((ts_sequence[ts] - ts_sequence[0]) * n.period / 12) % n.period
                  //int partition = ts_sequence[ts] % n.period;
                  //int partition = int((ts_sequence[ts] - ts_sequence[0]) * n.period / 12) % n.period;
                  int partition = this->observation_timesteps_sorted[ts] % n.period;
                  n.partitioned_data[partition].first.push_back(ts_sequence[ts]);
                  n.partitioned_data[partition].second.push_back(mean_sequence[ts]);
              }
              */

              double center;
              vector<int> filled_observation_timesteps_within_a_period;
              n.centers = vector<double>(n.period + 1);
              n.spreads = vector<double>(n.period);
              n.generated_latent_centers_for_a_period = std::vector<double>(n.period, 0);
              n.generated_latent_spreads_for_a_period = std::vector<double>(n.period, 0);
              for (const auto & [ partition, data ] : n.partitioned_data) {
                  if (n.center_measure.compare("mean") == 0) {
                      center = delphi::utils::mean(data.second);
                  } else {
                      center = delphi::utils::median(data.second);
                  }
                  n.centers[partition] = center;

                  double spread = 0;
                  if (data.second.size() > 1) {
                      if (n.center_measure.compare("mean") == 0) {
                          spread = delphi::utils::standard_deviation(center,
                                                                     data.second);
                      } else {
                          spread = delphi::utils::median_absolute_deviation(center,
                                                                            data.second);
                      }
                  }
                  n.spreads[partition] = spread;

                  if (!data.first.empty()) {
                      //int observation_timestep_within_a_period =
                      //    this->observation_timesteps_sorted[data.first[0]] %
                      //    n.period;
                      n.generated_latent_centers_for_a_period[partition]
                          = center;
                      n.generated_latent_spreads_for_a_period[partition]
                          = spread;
                      filled_observation_timesteps_within_a_period.push_back(
                          partition);
                  }
              }

              sort(filled_observation_timesteps_within_a_period.begin(),
                   filled_observation_timesteps_within_a_period.end());

              // Interpolate values for the missing months
              if (filled_observation_timesteps_within_a_period.size() > 1) {
                  for (int i = 0; i < filled_observation_timesteps_within_a_period.size(); i++) {
                      int observation_timestep_within_a_period_start =
                          filled_observation_timesteps_within_a_period[i];
                      int observation_timestep_within_a_period_end =
                          filled_observation_timesteps_within_a_period
                              [(i + 1) %
                               filled_observation_timesteps_within_a_period
                                   .size()];

                      int num_missing_observation_timesteps = 0;
                      if (observation_timestep_within_a_period_end >
                          observation_timestep_within_a_period_start) {
                          num_missing_observation_timesteps =
                              observation_timestep_within_a_period_end -
                              observation_timestep_within_a_period_start - 1;
                      }
                      else {
                          num_missing_observation_timesteps = (n.period - 1 -
                               observation_timestep_within_a_period_start) +
                              observation_timestep_within_a_period_end;
                      }

                      for (int missing_observation_timestep = 1;
                           missing_observation_timestep <= num_missing_observation_timesteps;
                           missing_observation_timestep++) {
                          n.generated_latent_centers_for_a_period
                              [(observation_timestep_within_a_period_start +
                                missing_observation_timestep) % n.period] =
                              ((num_missing_observation_timesteps -
                                missing_observation_timestep + 1) *
                                   n.generated_latent_centers_for_a_period
                                       [observation_timestep_within_a_period_start] +
                               (missing_observation_timestep) *
                                   n.generated_latent_centers_for_a_period
                                       [observation_timestep_within_a_period_end]) /
                              (num_missing_observation_timesteps + 1);

                          n.generated_latent_spreads_for_a_period
                              [(observation_timestep_within_a_period_start +
                                missing_observation_timestep) % n.period] =
                              ((num_missing_observation_timesteps -
                                missing_observation_timestep + 1) *
                               n.generated_latent_spreads_for_a_period
                                       [observation_timestep_within_a_period_start] +
                               (missing_observation_timestep) *
                               n.generated_latent_spreads_for_a_period
                                       [observation_timestep_within_a_period_end]) /
                              (num_missing_observation_timesteps + 1);
                      }
                  }
              } else if (filled_observation_timesteps_within_a_period.size() == 1) {
                  for (int observation_timestep = 0;
                       observation_timestep < n.generated_latent_centers_for_a_period.size();
                       observation_timestep++) {
                      n.generated_latent_centers_for_a_period[observation_timestep] =
                          n.generated_latent_centers_for_a_period
                              [filled_observation_timesteps_within_a_period[0]];
                      n.generated_latent_spreads_for_a_period[observation_timestep] =
                          n.generated_latent_spreads_for_a_period
                              [filled_observation_timesteps_within_a_period[0]];
                 }
              }

              n.changes = vector<double>(n.centers.size(), 0.0);
              n.centers[n.period] = n.centers[0];
              if (n.model.compare("center") != 0) {
                  // model == absolute_change
                  adjacent_difference(
                      n.centers.begin(), n.centers.end(), n.changes.begin());
                  if (n.model.compare("relative_change") == 0) {
                      transform(n.centers.begin(),
                                n.centers.end() - 1,
                                n.changes.begin() + 1,
                                n.changes.begin() + 1,
                                [&](double start_value, double abs_change) {
                                  return abs_change / (start_value + 1);
                                });
                  }
              }


              // Experiment: First calculate adjacent changes, then partition changes
              // and compute the center of each changes partition
              /*
              // Absolute changes
              vector<double> absolute_change = vector<double>(mean_sequence.size());
              adjacent_difference(mean_sequence.begin(),
                                  mean_sequence.end(),
                                  absolute_change.begin());

              // Relative changes
              vector<double> relative_change = vector<double>(mean_sequence.size() - 1);
              transform(mean_sequence.begin(), mean_sequence.end() - 1,
                        absolute_change.begin() + 1, relative_change.begin(),
                        [&](double start_value, double abs_change){return abs_change / (start_value + 1);});

              // Partition changes
              for (int ts = 0; ts < relative_change.size(); ts++) {
                  int partition = ts % n.period;
                  n.partitioned_absolute_change[partition].first.push_back(ts_sequence[ts]);
                  n.partitioned_absolute_change[partition].second.push_back(absolute_change[ts + 1]);

                  n.partitioned_relative_change[partition].first.push_back(ts_sequence[ts]);
                  n.partitioned_relative_change[partition].second.push_back(relative_change[ts]);
              }

              // Compute partition centers
              //n.changes = vector<double>(n.period + 1);
              for (const auto & [ partition, data ] : n.partitioned_absolute_change) {
                  double partition_median = delphi::utils::median(data.second);
                  n.changes[partition + 1] = partition_median;
              }
              //for (const auto & [ partition, data ] : n.partitioned_relative_change) {
              //    double partition_median = delphi::utils::median(data.second);
              //    n.changes[partition + 1] = partition_median;
              //}

              // Experimenting with zero centering the centers
              //vector<double> only_changes = vector<double>(n.changes.begin() + 1, n.changes.end());
              //double change_mean = delphi::utils::mean(only_changes);
              //transform(n.changes.begin() + 1, n.changes.end(),
              //          n.changes.begin() + 1,
              //          [&](double val){return val - change_mean;});
              */

              n.mean = delphi::utils::mean(mean_sequence);

              if (mean_sequence.size() > 1) {
                  n.std = delphi::utils::standard_deviation(n.mean, mean_sequence);
              }
          }
      }
  }
}

void AnalysisGraph::construct_theta_pdfs() {

  // The choice of sigma_X and sigma_Y is somewhat arbitrary here - we need to
  // come up with a principled way to select this value, or infer it from data.
  double sigma_X = 1.0;
  double sigma_Y = 1.0;
  AdjectiveResponseMap adjective_response_map =
      this->construct_adjective_response_map(
          this->rand_num_generator, this->uni_dist, this->norm_dist, this->n_kde_kernels);
  vector<double> marginalized_responses;
  for (auto [adjective, responses] : adjective_response_map) {
    for (auto response : responses) {
      marginalized_responses.push_back(response);
    }
  }

  marginalized_responses = KDE(marginalized_responses)
                               .resample(this->n_kde_kernels,
                                         this->rand_num_generator,
                                         this->uni_dist,
                                         this->norm_dist);

  for (auto e : this->edges()) {
    if (!this->edge(e).kde.log_prior_hist.empty()) {
      continue;
    }

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
          all_thetas.push_back(atan((sigma_Y * y) / (sigma_X * x)));
      }
    }

    // TODO: Using n_kde_kernels is a quick hack. Should define another
    // variable like n_bins to make the number of bins independent from the
    // number of kernels
    this->graph[e].kde = KDE(all_thetas, this->n_kde_kernels);
//    this->graph[e].kde.dataset = all_thetas;
  }
}
