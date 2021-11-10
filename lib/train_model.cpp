#include "AnalysisGraph.hpp"
#include "TrainingStatus.hpp"
#include "data.hpp"
#include <tqdm.hpp>
#include <range/v3/all.hpp>
#include <nlohmann/json.hpp>


using namespace std;
using tq::trange;
using json = nlohmann::json;


void AnalysisGraph::train_model(int start_year,
                                int start_month,
                                int end_year,
                                int end_month,
                                int res,
                                int burn,
                                string country,
                                string state,
                                string county,
                                map<string, string> units,
                                InitialBeta initial_beta,
                                InitialDerivative initial_derivative,
                                bool use_heuristic,
                                bool use_continuous) {

  this->training_range = make_pair(make_pair(start_year, start_month),
                                   make_pair(  end_year,   end_month));
  this->n_timesteps = this->calculate_num_timesteps(start_year, start_month,
                                                      end_year,   end_month);

  this->observation_timestep_gaps.clear();
  this->observation_timestep_gaps = vector<double>(this->n_timesteps, 1.0);
  this->observation_timestep_gaps[0] = 0;

  if(this->n_timesteps > 0) {
      if (!synthetic_data_experiment && !causemos_call) {
          // Delphi is run locally using observation data from delphi.db
          // For a synthetic data experiment, the observed state sequence is
          // generated.
          // For a CauseMos call, the observation sequences are provided in the create
          // model JSON call and the observed state sequence is set in the method
          // AnalysisGraph::set_observed_state_sequence_from_json_data(), which is
          // defined in causemos_integration.cpp
          this->set_observed_state_sequence_from_data(country, state, county);
      }

      this->run_train_model(res, burn, initial_beta, initial_derivative,
                            use_heuristic, use_continuous);
  }
}


std::string progress_to_json(float progress) {

  json ret_exp;
  ret_exp["training_progress"] = progress;

  return ret_exp.dump();
}


bool AnalysisGraph::get_trained(){
  return this->trained;
}

float AnalysisGraph::get_training_progress(){
  return this->training_progress;
}

void AnalysisGraph::run_train_model(int res,
                                int burn,
                                InitialBeta initial_beta,
                                InitialDerivative initial_derivative,
                                bool use_heuristic,
                                bool use_continuous,
                                int train_start_timestep,
                                int train_timesteps,
                                unordered_map<string, int> concept_periods,
                                unordered_map<string, string> concept_center_measures,
                                unordered_map<string, string> concept_models,
                                unordered_map<string, double> concept_min_vals,
                                unordered_map<string, double> concept_max_vals,
                                unordered_map<string, function<double(unsigned int, double)>> ext_concepts
                                ) {

    cout << "train_model.cpp.run_train_model" << endl;

    TrainingStatus ts(this);

    float training_step = 1.0 / (res + burn);

    this->training_progress = 0;

    if (train_timesteps < 0) {
      this->n_timesteps = this->observed_state_sequence.size();
    }
    else {
      this->n_timesteps = train_timesteps;
    }

    unordered_set<int> train_vertices =
        unordered_set<int>
            (this->node_indices().begin(), this->node_indices().end());

    for (const auto & [ concept, deriv_func ] : ext_concepts) {
      try {
        int vert_id = this->name_to_vertex.at(concept);
        this->external_concepts[vert_id] = deriv_func;
        train_vertices.erase(vert_id);
      }
      catch (const std::out_of_range& oor) {
        cout << "\nERROR: train_model - Concept << concept << is not in CAG!\n";
      }
    }

    for (const auto & [ concept, period ] : concept_periods) {
      try {
        int vert_id = this->name_to_vertex.at(concept);
        Node &n = (*this)[vert_id];
        n.period = period;
      }
      catch (const std::out_of_range& oor) {
        cout << "\nERROR: train_model - Concept << concept << is not in CAG!\n";
      }
    }

    for (const auto & [ concept, center_measure ] : concept_center_measures) {
      try {
        int vert_id = this->name_to_vertex.at(concept);
        Node &n = (*this)[vert_id];
        n.center_measure = center_measure;
      }
      catch (const std::out_of_range& oor) {
        cout << "\nERROR: train_model - Concept << concept << is not in CAG!\n";
      }
    }

    for (const auto & [ concept, model ] : concept_models) {
      try {
        int vert_id = this->name_to_vertex.at(concept);
        Node &n = (*this)[vert_id];
        n.model = model;
      }
      catch (const std::out_of_range& oor) {
        cout << "\nERROR: train_model - Concept << concept << is not in CAG!\n";
      }
    }

    for (const auto & [ concept, min_val ] : concept_min_vals) {
      try {
        int vert_id = this->name_to_vertex.at(concept);
        Node &n = (*this)[vert_id];
        n.min_val = min_val;
        n.has_min = true;
      }
      catch (const std::out_of_range& oor) {
        cout << "\nERROR: train_model - Concept << concept << is not in CAG!\n";
      }
    }

    for (const auto & [ concept, max_val ] : concept_max_vals) {
      try {
        int vert_id = this->name_to_vertex.at(concept);
        Node &n = (*this)[vert_id];
        n.max_val = max_val;
        n.has_max = true;
      }
      catch (const std::out_of_range& oor) {
        cout << "\nERROR: train_model - Concept << concept << is not in CAG!\n";
      }
    }

    this->concept_sample_pool.clear();
//    this->concept_sample_pool = vector<unsigned int>(train_vertices.begin(),
//                                                     train_vertices.end());
    for (int vert : train_vertices) {
      if (this->head_nodes.find(vert) == this->head_nodes.end()) {
        this->concept_sample_pool.push_back(vert);
      }
    }

    this->initialize_parameters(res, initial_beta, initial_derivative,
                                use_heuristic, use_continuous);

    this->log_likelihoods.clear();
    this->log_likelihoods = vector<double>(burn + this->res, 0);
    this->MAP_sample_number = -1;

    cout << "\nBurning " << burn << " samples out..." << endl;
    for (int i : trange(burn)) {

      this->training_progress += training_step;

      this->sample_from_posterior();
      this->log_likelihoods[i] = this->log_likelihood;

      if (this->log_likelihood > this->log_likelihood_MAP) {
          this->log_likelihood_MAP = this->log_likelihood;
          this->transition_matrix_collection[this->res - 1] = this->A_original;
          this->initial_latent_state_collection[this->res - 1] = this->s0;
          this->log_likelihoods[burn + this->res - 1] = this->log_likelihood;
          this->MAP_sample_number = this->res - 1;
      }
    }

    //int num_verts = this->num_vertices();

    cout << "\nSampling " << this->res << " samples from posterior..." << endl;
    for (int i : trange(this->res - 1)) {

      this->training_progress += training_step;

      this->sample_from_posterior();
      this->transition_matrix_collection[i] = this->A_original;
      this->initial_latent_state_collection[i] = this->s0;

      if (this->log_likelihood > this->log_likelihood_MAP) {
        this->log_likelihood_MAP = this->log_likelihood;
        this->MAP_sample_number = i;
      }

      for (auto e : this->edges()) {
        this->graph[e].sampled_thetas.push_back(this->graph[e].theta);
      }

      this->log_likelihoods[burn + i] = this->log_likelihood;
      /*
      this->latent_mean_collection[i] = vector<double>(num_verts);
      this->latent_std_collection[i] = vector<double>(num_verts);
      this->latent_mean_std_collection[i] = vector<
                           unordered_map<int, pair<double, double>>>(num_verts);

      for (int v : this->node_indices()) {
        Node &n = (*this)[v];
        this->latent_mean_collection[i][v] = n.mean;
        this->latent_std_collection[i][v] = n.std;
        this->latent_mean_std_collection[i][v] = n.partition_mean_std;
      }
      */
    }

    if (this->MAP_sample_number < int(this->res) - 1) {
      this->sample_from_posterior();
      this->transition_matrix_collection[this->res - 1] = this->A_original;
      this->initial_latent_state_collection[this->res - 1] = this->s0;
      this->log_likelihoods[burn + this->res - 1] = this->log_likelihood;

      if ((this->log_likelihood > this->log_likelihood_MAP) or (this->log_likelihood_MAP == -1)) {
        this->log_likelihood_MAP = this->log_likelihood;
        this->MAP_sample_number = this->res - 1;
      }

      for (auto e : this->edges()) {
        this->graph[e].sampled_thetas.push_back(this->graph[e].theta);
      }

      this->log_likelihoods[burn + this->res - 1] = this->log_likelihood;
    } else {
      this->MAP_sample_number = this->res - 1;
    }

    this->trained = true;
    this->training_progress= 1.0;
    this->write_training_status_to_db();
    RNG::release_instance();
}

void AnalysisGraph::run_train_model_2(int res,
                                    int burn,
                                    InitialBeta initial_beta,
                                    InitialDerivative initial_derivative,
                                    bool use_heuristic,
                                    bool use_continuous
                                    ) {


    this->initialize_parameters(res, initial_beta, initial_derivative,
                              use_heuristic, use_continuous);

    cout << "\nBurning " << burn << " samples out..." << endl;
    for (int i : trange(burn)) {
        this->sample_from_posterior();
    }

    cout << "\nSampling " << this->res << " samples from posterior..." << endl;
    for (int i : trange(this->res)) {
        this->sample_from_posterior();
        this->transition_matrix_collection[i] = this->A_original;
        this->initial_latent_state_collection[i] = this->s0;

        for (auto e : this->edges()) {
          this->graph[e].sampled_thetas.push_back(this->graph[e].theta);
        }
    }

    this->trained = true;
    RNG::release_instance();
}


/*
 ============================================================================
 Private: Get Training Data Sequence
 ============================================================================
*/

void AnalysisGraph::set_observed_state_sequence_from_data(string country,
                                                          string state,
                                                          string county) {
  this->observed_state_sequence.clear();

  // Access (concept is a vertex in the CAG)
  // [ timestep ][ concept ][ indicator ][ observation ]
  this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

  int year = this->training_range.first.first;
  int month = this->training_range.first.second;

  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->observed_state_sequence[ts] =
        get_observed_state_from_data(year, month, country, state, county);

    if (month == 12) {
      year++;
      month = 1;
    }
    else {
      month++;
    }
  }
}

vector<vector<vector<double>>> AnalysisGraph::get_observed_state_from_data(
    int year, int month, string country, string state, string county) {
  using ranges::to;
  using ranges::views::transform;

  int num_verts = this->num_vertices();

  // Access (concept is a vertex in the CAG)
  // [ concept ][ indicator ][ observation ]
  vector<vector<vector<double>>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;

    for (auto& ind : indicators) {
      vector<double> vals = get_observations_for(ind.get_name(),
                                                 country,
                                                 state,
                                                 county,
                                                 year,
                                                 month,
                                                 ind.get_unit(),
                                                 this->data_heuristic);

      observed_state[v].push_back(vals);
    }
  }

  return observed_state;
}
