#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/range/adaptors.hpp>

using namespace std;
using Eigen::VectorXd, Eigen::MatrixXd;
using fmt::print, fmt::format;
namespace rs = ranges;
using boost::adaptors::transformed;

#include "dbg.h"
using fmt::print;
/*
 ============================================================================
 Private: Prediction
 ============================================================================
*/

void AnalysisGraph::generate_latent_state_sequences(
    int initial_prediction_step) {

  // Allocate memory for prediction_latent_state_sequences
  this->predicted_latent_state_sequences.clear();
  this->predicted_latent_state_sequences = vector<vector<VectorXd>>(
      this->res,
      vector<VectorXd>(this->pred_timesteps, VectorXd(this->num_vertices() * 2)));

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
          // Here A = Ac = this->transition_matrix_collection[samp] (continuous)

          // Evolving the system one time step before the
          // initial_prediction_step
          A = (this->transition_matrix_collection[samp] *
                  this->delta_t * (initial_prediction_step - 1)).exp();

          this->predicted_latent_state_sequences[samp][0] =
                               A * this->initial_latent_state_collection[samp];

          // After jumping to time step ips - 1, we take one step of length Δt
          // at a time.
          // So compute the transition matrix for a single step.
          // Computing the matrix exponential for a Δt time step.
          // By default we are using Δt = 1
          // A = e^{Ac * Δt)
          A = (this->transition_matrix_collection[samp] * this->delta_t).exp();

      } else {
          // Here A = Ad = this->transition_matrix_collection[samp] (discrete)
          // This is the discrete transition matrix to take a single step of
          // length Δt
          A = this->transition_matrix_collection[samp];

          // Evolving the system one time step before the
          // initial_prediction_step
          this->predicted_latent_state_sequences[samp][0] =
                              A.pow(initial_prediction_step - 1) *
                                  this->initial_latent_state_collection[samp];
      }

      // Clear out perpetual constraints residual from previous sample
      this->perpetual_constraints.clear();

      if (this->clamp_at_derivative) {
          // To clamp a latent state value to x_c at prediction step 1 via
          // clamping the derivative, we have to perturb the derivative at
          // prediction step 0, before evolving it to time prediction step 1.
          // So we have to look one time step ahead whether we have to clamp
          // at 1.
          //
          if (delphi::utils::in(this->one_off_constraints, 1)) {
              this->perturb_predicted_latent_state_at(1, samp);
          }
      }

      // Since we used ts = 0 for the initial_prediction_step - 1, this loop is
      // one ahead of the prediction time steps. That is, index 1 in the
      // prediction data structures is the 0th index for the requested
      // prediction sequence. Hence, when we are at time step ts, we should
      // check whether there are any constconstraints
        // time step index is 1
      for (int ts = 1; ts < this->pred_timesteps; ts++) {
          // When continuous: The standard matrix exponential equation is,
          //                        s_{t+Δt} = e^{Ac * Δt } * s_t
          //                  Since vector indices are integral values, and in
          //                  the implementation s is the vector, to index into
          //                  the vector we uses consecutive integers. Thus in
          //                  the implementation, the matrix exponential
          //                  equation becomes,
          //                      s_{t+1} = e^{Ac * Δt } * s_t
          //                  What this equation says is that although vector
          //                  indices advance by 1, the duration between two
          //                  predictions stored in two adjacent vector cells
          //                  need not be 1. The time duration is actually Δt.
          //                  The actual line of code represents,
          //                        s_t = e^{Ac * Δt } * s_{t-1}
          // When discrete  : s_t = Ad * s_{t-1}
          this->predicted_latent_state_sequences[samp][ts] =
              A * this->predicted_latent_state_sequences[samp][ts - 1];

          if (this->clamp_at_derivative ) {
              if (ts == this->rest_derivative_clamp_ts) {
                  // We have perturbed the derivative at ts - 1. If we do not
                  // take any action, that clamped derivative will be in effect
                  // until another clamping or end of prediction. This is where
                  // we take the necessary actions.

                  if (is_one_off_constraints) {
                      // We should revert the derivative to its original value
                      // at this time step so that clamping does not affect
                      // time step ts + 1.
                      for (auto constraint : this->one_off_constraints.at(ts)) {
                          int node_id = constraint.first;

                          this->predicted_latent_state_sequences[samp][ts]
                              (2 * node_id + 1) =
                              this->initial_latent_state_collection[samp]
                              (2 * node_id + 1);
                          dbg("Resetting");
                          dbg(ts);
                          dbg(this->predicted_latent_state_sequences[samp][ts](2 * node_id + 1));
                  }
                  } else {
                      for (auto [node_id, value]: this->one_off_constraints.at(ts)) {
                          this->predicted_latent_state_sequences[samp][ts]
                                                        (2 * node_id + 1) = 0;
                      }
                  }
              }

              // To clamp a latent state value to x_c at prediction step ts + 1
              // via clamping the derivative, we have to perturb the derivative
              // at prediction step ts, before evolving it to prediction time
              // step ts + 1. So we have to look one time step ahead whether we
              // have to clamp at ts + 1 and clamp the derivative now.
              //
              if (delphi::utils::in(this->one_off_constraints, ts + 1) ||
                      !this->perpetual_constraints.empty()) {
                  this->perturb_predicted_latent_state_at(ts + 1, samp);
              }
          } else {
              // Apply constraints to latent state if any
              // Logic of this condition:
              //    Initially perpetual_constraints = ∅
              //
              //    one_off_constraints.at(ts) = ∅ => Unconstrained ∀ ts
              //
              //    one_off_constraints.at(ts) ‡ ∅
              //        => ∃ some constraints (But we do not know what kind)
              //        => Perturb latent state
              //           Call perturb_predicted_latent_state_at()
              //           one_off_constraints == true
              //                => We are applying One-off constraints
              //           one_off_constraints == false
              //                => We are applying perpetual constraints
              //                => We add constraints to perpetual_constraints
              //                => perpetual_constraints ‡ ∅
              //                => The if condition is true ∀ subsequent time steps
              //                   after ts (the first time step s.t.
              //                   one_off_constraints.at(ts) ‡ ∅
              //                => Constraints are perpetual
              //
              if (delphi::utils::in(this->one_off_constraints, ts) ||
                      !this->perpetual_constraints.empty()) {
                  this->perturb_predicted_latent_state_at(ts, samp);
              }
          }
      }
  }
}

/*
 * Applying constraints (interventions) to latent states
 * Check the data structure definition to get more descriptions
 *
 * Clamping at time step ts for value v
 * Two methods work in tandem to achieve this. Let us label them as follows:
 *      glss  : generate_latent_state_sequences()
 *      ppls@ : perturb_predicted_latent_state_at()
 *                                 ┍━━━━━━━━━━━━━━━━━━━━━━━━┑
 *                                 │           How          ┃
 *                                 ┝━━━━━━━━━━┳━━━━━━━━━━━━━┫
 *                                 │ One-off  ┃  Perpetual  ┃
 * ━━━━━━┳━━━━━━━━━━━━┯━━━━━━━━━━━━┿━━━━━━━━━━┻━━━━━━━━━━━━━┫
 *       ┃            │ Clamp at   │           v - x_{ts-1} ┃
 *       ┃            │ (by ppls@) │   ts-1 to ──────────── ┃
 *       ┃            │            │               Δt       ┃
 *       ┃ Derivative ├┈┈┈┈┈┈┈┈┈┈┈┈┼┈┈┈┈┈┈┈┈┈┈┰┈┈┈┈┈┈┈┈┈┈┈┈┈┨
 * Where ┃            │ Reset at   │ ts to ẋ₀ ┃ ts to 0     ┃
 *       ┃            │ (by glss)  │ from S₀  ┃             ┃
 *       ┣━━━━━━━━━━━━┿━━━━━━━━━━━━┿━━━━━━━━━━╋━━━━━━━━━━━━━┫
 *       ┃ Value      │ Clamp at   │ ts to v  ┃ ∀ t≥ts to v ┃
 *       ┃            │ (by ppls@) │          ┃             ┃
 * ━━━━━━┻━━━━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━━━┻━━━━━━━━━━━━━┛
 */
void AnalysisGraph::perturb_predicted_latent_state_at(int timestep, int sample_number) {
    // Let vertices of the CAG be v = 0, 1, 2, 3, ...
    // Then,
    //    indices 2*v keeps track of the state of each variable v
    //    indices 2*v+1 keeps track of the state of ∂v/∂t

    if (this->clamp_at_derivative) {
        for (auto [node_id, value]: this->one_off_constraints.at(timestep)) {
            // To clamp the latent state value to x_c at time step t via
            // clamping the derivative, we have to clamp the derivative
            // appropriately at time step t-1.
            // Example:
            //      Say we want to clamp the latent state at t=6 to value
            //      x_c (i.e. we want x₆ = x_c. So we have to set the
            //      derivative at t=6-1=5, ẋ₅, as follows:
            //                  x_c - x₅
            //             ẋ₅ = --------- ........... (1)
            //                     Δt
            //             x₆ = x₅ + (ẋ₅ × Δt)
            //                = x_c
            //      Thus clamping ẋ₅ (at t = 5) as described in (1) gives
            //      us the desired clamping at t = 5 + 1 = 6
            dbg("Derivative");
            dbg(timestep);
            dbg(this->predicted_latent_state_sequences[sample_number][timestep - 1](2 * node_id + 1));
            double clamped_derivative = (value -
                    this->predicted_latent_state_sequences[sample_number]
                    [timestep - 1](2 * node_id)) / this->delta_t;

            this->predicted_latent_state_sequences[sample_number]
                [timestep - 1](2 * node_id + 1) = clamped_derivative;
            dbg(clamped_derivative);
        }

        // Clamping the derivative at t-1 changes the value at t.
        // According to our model, derivatives never chance. So if we
        // do not revert it, clamped derivative stays till another
        // clamping or the end of prediction. Since this is a one-off
        // clamping, we have to return the derivative back to its
        // original value at time step t, before we use it to evolve
        // time step t + 1. Thus we remember the time step at which we
        // have to perform this.
        this->rest_derivative_clamp_ts = timestep;

        return;
    }

    if (is_one_off_constraints) {

        for (auto [node_id, value]: this->one_off_constraints.at(timestep)) {
            this->predicted_latent_state_sequences[sample_number][timestep](2 * node_id) = value;
            dbg("value");
            dbg(value);
        }
    } else { // Perpetual constraints
        if (delphi::utils::in(this->one_off_constraints, timestep)) {
            // Update any previous perpetual constraints
            for (auto [node_id, value]: this->one_off_constraints.at(timestep)) {
                dbg("Perpetual");
                dbg(value);

                this->perpetual_constraints[node_id] = value;
            }
        }

        // Apply perpetual constraints
        for (auto [node_id, value]: this->perpetual_constraints) {

            this->predicted_latent_state_sequences[sample_number][timestep](2 * node_id) = value;
        }
    }
}

void AnalysisGraph::generate_observed_state_sequences() {
  using rs::to, rs::views::transform;

  // Allocate memory for observed_state_sequences
  this->predicted_observed_state_sequences.clear();
  this->predicted_observed_state_sequences =
      vector<PredictedObservedStateSequence>(
          this->res,
          PredictedObservedStateSequence(this->pred_timesteps,
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

  // NOTE: To facilitate clamping derivatives, we start prediction one time
  //       step before the requested prediction start time. We are omitting
  //       that additional time step from the results returned to the user.
  this->pred_timesteps--;

  // Access
  // [ sample ][ time_step ][ vertex_name ][ indicator_name ]
  auto result = FormattedPredictionResult(
      this->res,
      vector<unordered_map<string, unordered_map<string, double>>>(
          this->pred_timesteps));

  for (int samp = 0; samp < this->res; samp++) {
    // NOTE: To facilitate clamping derivatives, we start prediction one time
    //       step before the requested prediction start time. We are omitting
    //       that additional time step from the results returned to the user.
    for (int ts = 1; ts < this->pred_timesteps; ts++) {
      for (auto [vert_name, vert_id] : this->name_to_vertex) {
        for (auto [ind_name, ind_id] : (*this)[vert_id].nameToIndexMap) {
          result[samp][ts - 1][vert_name][ind_name] =
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

  // Check for sensible prediction time step ranges.
  // NOTE: To facilitate clamping derivatives, we start prediction one time
  //       step before the requested prediction start time. Therefor, the
  //       possible valid earliest prediction time step is one time step after
  //       the training start time.
  if (start_year < this->training_range.first.first ||
      (start_year == this->training_range.first.first &&
       start_month <= this->training_range.first.second)) {
    print("The initial prediction date can't be before the "
         "initial training date. Defaulting initial prediction date "
         "to initial training date.");
    start_month = this->training_range.first.second + 1;
    if (start_month == 13) {
        start_year = this->training_range.first.first + 1;
        start_month = 1;
    } else {
        start_year = this->training_range.first.first;
    }
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

  // NOTE: To facilitate clamping derivatives, we start prediction one time
  //       step before the requested prediction start time. When we are
  //       returning results, we have to remove the predictions at the 0th
  //       index of each predicted observed state sequence.
  //       Adding that additional time step.
  this->pred_timesteps++;

  this->generate_latent_state_sequences(pred_init_timestep);
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
