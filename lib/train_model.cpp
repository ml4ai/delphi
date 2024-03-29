#include "AnalysisGraph.hpp"
#include "ModelStatus.hpp"
#include "data.hpp"
#include "TrainingStopper.hpp"
#include "Logger.hpp"
#include <tqdm.hpp>
#include <range/v3/all.hpp>
#include <nlohmann/json.hpp>


#ifdef TIME
  #include "utils.hpp"
  #include "Timer.hpp"
//  #include "CSVWriter.hpp"
#endif

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

  this->modeling_timestep_gaps.clear();
  this->modeling_timestep_gaps = vector<double>(this->n_timesteps, 1.0);
  this->modeling_timestep_gaps[0] = 0;

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

      this->run_train_model(res, burn, HeadNodeModel::HNM_NAIVE, initial_beta,
                            initial_derivative, use_heuristic, use_continuous);
  }
}


void AnalysisGraph::run_train_model(int res,
                                int burn,
                                HeadNodeModel head_node_model,
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
                                unordered_map<string, function<double(unsigned int, double)>> ext_concepts) {

    double training_step = 0.99 / (res + burn);

    TrainingStopper training_stopper;

    Logger logger;
    logger.info("AnalysisGraph::run_train_model");

    ModelStatus ms(this->id);

    ms.enter_working_state();

    this->trained = false;

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

    this->head_node_model = head_node_model;

    this->concept_sample_pool.clear();

    for (int vert : train_vertices) {
      Node& n = (*this)[vert];
      if (this->head_nodes.find(vert) == this->head_nodes.end()) {
        this->concept_sample_pool.push_back(vert);
      } else if (n.period == 1) {
          // Head nodes with period > 1 are modeled using the seasonality
          // period == 1 => this is not a seasonal node
          // To prevent this from modeled as seasonal,
          // remove it from head nodes
          // TODO: There is a terminology confusion since this is still a
          // head node, but we are removing it from head nodes to
          // prevent it from being modeled seasonally.
          this->concept_sample_pool.push_back(vert);
          this->head_nodes.erase(vert);
          this->body_nodes.insert(vert);
      }
    }

    this->edge_sample_pool.clear();
    for (EdgeDescriptor ed : this->edges()) {
        this->graph[ed].sampled_thetas.clear();
        if (!this->graph[ed].is_frozen()) {
            this->edge_sample_pool.push_back(ed);
        }
    }

    this->initialize_parameters(res, initial_beta, initial_derivative,
                                use_heuristic, use_continuous);

    this->log_likelihoods.clear();
    this->log_likelihoods = vector<double>(burn + this->res, 0);
    this->MAP_sample_number = -1;

    #ifdef TIME
        this->create_mcmc_part_timing_file();
//      int n_nodes = this->num_nodes();
//      int n_edges = this->num_nodes();
//      string filename = string("mcmc_timing_embeded_") +
//                        to_string(n_nodes) + "-" +
//                        to_string(n_edges) + "_" +
//                        delphi::utils::get_timestamp() + ".csv";
//      this->writer = CSVWriter(filename);
//      vector<string> headings = {"Nodes", "Edges", "Wall Clock Time (ns)", "CPU Time (ns)", "Sample Type"};
//      writer.write_row(headings.begin(), headings.end());
//      cout << filename << endl;
    #endif

    string text = "Burning " + to_string(burn) + " samples out...";
    logger.info(" " + text);
    logger.info(" #    log_likelihood");
//    cout << "\n" << text << endl;
    for (int i : trange(burn)) {
      ms.increment_progress(training_step);
      {
          #ifdef TIME
//            durations.first.clear();
            durations.second.clear();
            durations.second.push_back(this->timing_run_number);
//            durations.first.push_back("Nodes");
            durations.second.push_back(this->num_nodes());
//            durations.first.push_back("Edges");
            durations.second.push_back(this->num_nodes());
            Timer t = Timer("train", durations);
          #endif
          this->sample_from_posterior();
      }
      #ifdef TIME
//        durations.first.push_back("sample type");
        durations.second.push_back(this->coin_flip < this->coin_flip_thresh? 1 : 0);
        writer.write_row(durations.second.begin(), durations.second.end());
      #endif

      this->log_likelihoods[i] = this->log_likelihood;
      char buf[200];
      sprintf(buf, "%4d %.10f", i, this->log_likelihood);
      logger.info(" " + string(buf));

      if (this->log_likelihood > this->log_likelihood_MAP) {
          this->log_likelihood_MAP = this->log_likelihood;
          this->transition_matrix_collection[this->res - 1] = this->A_original;
          this->initial_latent_state_collection[this->res - 1] = this->s0;
          this->log_likelihoods[burn + this->res - 1] = this->log_likelihood;
          this->MAP_sample_number = this->res - 1;
      }

      if(training_stopper.stop_training(this->log_likelihoods, i)) {
	string text = "Model training stopped early at sample " + to_string(i);
	logger.info(" " + text);
        cout << text << endl;
        break;
      }
    }

    cout << "\nSampling " << this->res << " samples from posterior..." << endl;
    for (int i : trange(this->res - 1)) {
      {
        ms.increment_progress(training_step);

        #ifdef TIME
//                durations.first.clear();
                durations.second.clear();
                durations.second.push_back(this->timing_run_number);
//                durations.first.push_back("Nodes");
                durations.second.push_back(this->num_nodes());
//                durations.first.push_back("Edges");
                durations.second.push_back(this->num_edges());
                Timer t = Timer("Train", durations);
        #endif
        this->sample_from_posterior();
      }
      #ifdef TIME
//            durations.first.push_back("Sample Type");
            durations.second.push_back(this->coin_flip < this->coin_flip_thresh? 1 : 0);
            writer.write_row(durations.second.begin(), durations.second.end());
      #endif

      this->transition_matrix_collection[i] = this->A_original;
      this->initial_latent_state_collection[i] = this->s0;

      if (this->log_likelihood > this->log_likelihood_MAP) {
        this->log_likelihood_MAP = this->log_likelihood;
        this->MAP_sample_number = i;
      }

      for (auto e : this->edges()) {
        this->graph[e].sampled_thetas.push_back(this->graph[e].get_theta());
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

    if (this->MAP_sample_number < int(this->res)) {
      this->sample_from_posterior();
      this->transition_matrix_collection[this->res - 1] = this->A_original;
      this->initial_latent_state_collection[this->res - 1] = this->s0;
      this->log_likelihoods[burn + this->res - 1] = this->log_likelihood;

      if ((this->log_likelihood > this->log_likelihood_MAP) or (this->log_likelihood_MAP == -1)) {
        this->log_likelihood_MAP = this->log_likelihood;
        this->MAP_sample_number = this->res - 1;
      }

      for (auto e : this->edges()) {
        this->graph[e].sampled_thetas.push_back(this->graph[e].get_theta());
      }

      this->log_likelihoods[burn + this->res - 1] = this->log_likelihood;
    } else {
      this->MAP_sample_number = this->res - 1;
    }

    this->trained = true;
    ms.enter_writing_state();
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
          this->graph[e].sampled_thetas.push_back(this->graph[e].get_theta());
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
