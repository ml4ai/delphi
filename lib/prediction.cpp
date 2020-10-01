#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/range/adaptors.hpp>

using namespace std;
using Eigen::VectorXd, Eigen::MatrixXd;
using fmt::print, fmt::format;
namespace rs = ranges;
using boost::adaptors::transformed;

/*
 ============================================================================
 Private: Prediction
 ============================================================================
*/

void AnalysisGraph::generate_latent_state_sequences(
    int prediction_timesteps,
    int initial_prediction_step,
    int total_timesteps,
    bool project) {
  this->n_timesteps = prediction_timesteps;

  // Allocate memory for prediction_latent_state_sequences
  this->predicted_latent_state_sequences.clear();
  this->predicted_latent_state_sequences = vector<vector<VectorXd>>(
      this->res,
      vector<VectorXd>(this->n_timesteps, VectorXd(this->num_vertices() * 2)));

  for (int samp = 0; samp < this->res; samp++) {
      // The sampled transition matrices would be either of matrix exponential
      // (continuous) version or discretized version depending on whether the
      // matrix exponential (continuous) version or the discretized transition
      // matrix version had been used to train the model. This allows us to
      // make the prediction routine common for both the versions, except for
      // the exponentiation of the matrix exponential (continuous) transition
      // matrices.
      MatrixXd A;
      if (this->continuous) {
          // Here Ac = this->transition_matrix_collection[samp] (continuous)
          // Computing the matrix exponential for a Δt time step.
          // By default we are using Δt = 1
          // A = e^{Ac * Δt)
          A = (this->transition_matrix_collection[samp] * this->delta_t).exp();
      } else {
          // Here A = Ad = this->transition_matrix_collection[samp] (discrete)
          A = this->transition_matrix_collection[samp];
      }

      if (project) {
          this->predicted_latent_state_sequences[samp][0] = this->s0;
      } else {
          // Perform inference based on the sampled initial latent states
          this->predicted_latent_state_sequences[samp][0] =
              this->initial_latent_state_collection[samp];

          // Apply constraints to latent state if any
          if (delphi::utils::in(this->latent_state_constraints, 0)) {
              this->perturb_predicted_latent_state_at(0, samp);
          }
      }

      for (int t = 1; t < this->n_timesteps; t++) {
          // When continuous: The standard matrix exponential equation is,
          //                        s_{t+Δt} = e^{Ac * Δt } * s_t
          //                  Since vector indexes are integral values, and in
          //                  the implementation s is the vector, to index into
          //                  the vector we uses consecutive integers. Thus in
          //                  the implementation, the matrix exponential
          //                  equation becomes,
          //                      s_{t+1} = e^{Ac * Δt } * s_t
          //                  What this equation says is that although vector
          //                  indexes advance by 1, predictions stored in two
          //                  adjacent vector cells need not be a single time
          //                  step apart. They are actually Δt time steps
          //                  apart.
          //                  The actual line of code represents,
          //                        s_t = e^{Ac * Δt } * s_{t-1}
          // When discrete  : s_t = Ad * s_{t-1}
          this->predicted_latent_state_sequences[samp][t] =
              A * this->predicted_latent_state_sequences[samp][t - 1];

          // Apply constraints to latent state if any
          if (delphi::utils::in(this->latent_state_constraints, t)) {
              this->perturb_predicted_latent_state_at(t, samp);
          }
      }
  }
}

void AnalysisGraph::perturb_predicted_latent_state_at(int timestep, int sample_number) {
    // Let vertices of the CAG be v = 0, 1, 2, 3, ...
    // Then,
    //    indexes 2*v keeps track of the state of each variable v
    //    indexes 2*v+1 keeps track of the state of ∂v/∂t
    for (auto constraint : this->latent_state_constraints.at(t)) {
        int node_id = constraint.first;
        double value = constraint.second;

        this->predicted_latent_state_sequences[sample_number][t](2 * node_id) = value;
    }
}

void AnalysisGraph::generate_observed_state_sequences() {
  using rs::to, rs::views::transform;

  // Allocate memory for observed_state_sequences
  this->predicted_observed_state_sequences.clear();
  this->predicted_observed_state_sequences =
      vector<PredictedObservedStateSequence>(
          this->res,
          PredictedObservedStateSequence(this->n_timesteps,
                                         vector<vector<double>>()));

  for (int samp = 0; samp < this->res; samp++) {
    vector<VectorXd>& sample = this->predicted_latent_state_sequences[samp];

    this->predicted_observed_state_sequences[samp] =
        sample | transform([this](VectorXd latent_state) {
          return this->sample_observed_state(latent_state);
        }) |
        to<vector>();
  }
}

FormattedPredictionResult AnalysisGraph::format_prediction_result() {
  // Access
  // [ sample ][ time_step ][ vertex_name ][ indicator_name ]
  auto result = FormattedPredictionResult(
      this->res,
      vector<unordered_map<string, unordered_map<string, double>>>(
          this->pred_timesteps));

  for (int samp = 0; samp < this->res; samp++) {
    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      for (auto [vert_name, vert_id] : this->name_to_vertex) {
        for (auto [ind_name, ind_id] : (*this)[vert_id].nameToIndexMap) {
          result[samp][ts][vert_name][ind_name] =
              this->predicted_observed_state_sequences[samp][ts][vert_id]
                                                      [ind_id];
        }
      }
    }
  }

  return result;
}

void AnalysisGraph::run_model(int start_year,
                              int start_month,
                              int end_year,
                              int end_month,
                              bool project) {
  if (!this->trained) {
    print("Passed untrained Causal Analysis Graph (CAG) Model. \n",
          "Try calling <CAG>.train_model(...) first!");
    throw "Model not yet trained";
  }

  // Check for sensible ranges.
  if (start_year < this->training_range.first.first ||
      (start_year == this->training_range.first.first &&
       start_month < this->training_range.first.second)) {
    print("The initial prediction date can't be before the "
         "initial training date. Defaulting initial prediction date "
         "to initial training date.");
    start_year = this->training_range.first.first;
    start_month = this->training_range.first.second;
  }

  /*
   *              total_timesteps
   *   ____________________________________________
   *  |                                            |
   *  v                                            v
   * start training                          end prediction
   *  |--------------------------------------------|
   *  :           |--------------------------------|
   *  :         start prediction                   :
   *  ^           ^                                ^
   *  |___________|________________________________|
   *      diff              pred_timesteps
   */
  int total_timesteps =
      this->calculate_num_timesteps(this->training_range.first.first,
                                    this->training_range.first.second,
                                    end_year,
                                    end_month);

  this->pred_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);

  int pred_init_timestep = total_timesteps - pred_timesteps;

  int year = start_year;
  int month = start_month;

  this->pred_range.clear();
  this->pred_range = vector<string>(this->pred_timesteps);

  for (int t = 0; t < this->pred_timesteps; t++) {
    this->pred_range[t] = to_string(year) + "-" + to_string(month);

    if (month == 12) {
      year++;
      month = 1;
    }
    else {
      month++;
    }
  }

  this->generate_latent_state_sequences(this->pred_timesteps, 0,
                                                    total_timesteps, project);
  this->generate_observed_state_sequences();
}

/*
 ============================================================================
 Public: Prediction
 ============================================================================
*/

Prediction AnalysisGraph::generate_prediction(int start_year,
                                              int start_month,
                                              int end_year,
                                              int end_month) {
  this->run_model(start_year, start_month, end_year, end_month);

  return make_tuple(
      this->training_range, this->pred_range, this->format_prediction_result());
}

vector<vector<double>> AnalysisGraph::prediction_to_array(string indicator) {
  int vert_id = -1;
  int ind_id = -1;

  auto result =
      vector<vector<double>>(this->res, vector<double>(this->pred_timesteps));

  // Find the vertex id the indicator is attached to and
  // the indicator id of it.
  // TODO: We can make this more efficient by making indicators_in_CAG
  // a map from indicator names to vertices they are attached to.
  // This is just a quick and dirty implementation
  for (auto [v_name, v_id] : this->name_to_vertex) {
    for (auto [i_name, i_id] : (*this)[v_id].nameToIndexMap) {
      if (indicator.compare(i_name) == 0) {
        vert_id = v_id;
        ind_id = i_id;
        goto indicator_found;
      }
    }
  }
  // Program will reach here only if the indicator is not found
  throw IndicatorNotFoundException(format(
      "AnalysisGraph::prediction_to_array - indicator \"{}\" not found!\n",
      indicator));

indicator_found:

  for (int samp = 0; samp < this->res; samp++) {
    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      result[samp][ts] =
          this->predicted_observed_state_sequences[samp][ts][vert_id][ind_id];
    }
  }

  return result;
}
