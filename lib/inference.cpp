#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include "spdlog/spdlog.h"

using namespace std;
using fmt::print, fmt::format;
using Eigen::VectorXd;
using spdlog::warn;

/*
 ============================================================================
 Private: Inference
 ============================================================================
*/

void AnalysisGraph::sample_predicted_latent_state_sequences(
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
    for (int t = 0; t < this->n_timesteps; t++) {
      const Eigen::MatrixXd& A_t =
          tuning_param * t * this->transition_matrix_collection[samp];
      if (project) {
        // Perform projection based on the perturbed initial latenet state s0
        this->predicted_latent_state_sequences[samp][t] = A_t.exp() * this->s0;
      }
      else {
        // Perform inference based on the sampled initial latent states
        const Eigen::VectorXd& s0_samp = this->initial_latent_state_collection[samp];
        this->predicted_latent_state_sequences[samp][t] = A_t.exp() * s0_samp;
      }
    }
  }
}

void AnalysisGraph::
    generate_predicted_observed_state_sequences_from_predicted_latent_state_sequences() {
  using ranges::to;
  using ranges::views::transform;

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
    warn("The initial prediction date can't be before the "
         "inital training date. Defaulting initial prediction date "
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

  this->sample_predicted_latent_state_sequences(
      this->pred_timesteps, 0, total_timesteps, project);
  this->generate_predicted_observed_state_sequences_from_predicted_latent_state_sequences();
}


/*
 ============================================================================
 Public: Inference
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

void AnalysisGraph::generate_projection(string json_projection) {
  auto json_data = nlohmann::json::parse(json_projection);

  auto start_time = json_data["startTime"];
  int start_year = start_time["year"].get<int>();
  int start_month = start_time["month"].get<int>();

  int time_steps = json_data["timeStepsInMonths"].get<int>();

  // This calculation adds one more month to the time steps
  // To avoid that we need to check some corner cases
  int end_year = start_year + (start_month + time_steps) / 12;
  int end_month = (start_month + time_steps) % 12;

  cout << start_year << endl;
  cout << start_month << endl;
  cout << time_steps << endl;
  cout << end_year << endl;
  cout << end_month << endl;

  // Create the perturbed initial latent state
  this->set_default_initial_state();

  auto perturbations =json_data["perturbations"];

  for(auto pert : perturbations) {
    string concept = pert["concept"].get<string>();
    double value = pert["value"].get<double>();
    cout << concept << ", " << value << endl;

    try {
      this->s0(2 * this->name_to_vertex.at(concept) + 1) = value;
    }
    catch (const out_of_range& oor) {
      cout << "Concept: " << concept << " Not present in the CAG. Cannot perturb\n";
    }
  }

  cout << this->s0 << endl;

  this->run_model(start_year, start_month, end_year, end_month, true);
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
