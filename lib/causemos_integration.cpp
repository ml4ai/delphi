// CauseMos integration methods

#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include <fstream>
#include <range/v3/all.hpp>
#include <time.h>
#include <limits.h>

using namespace std;
using namespace delphi::utils;
using namespace fmt::literals;

/*
============================================================================
Private: Integration with Uncharted's CauseMos interface
============================================================================
*/

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                         create-model (private)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/** Extracts concept to indicator mapping and the indicator observation
 * sequences from the create model JSON input received from the CauseMose
 * HMI.
 *
 * Check method declaration in AnalysisGraph.hpp for a detailed comment.
 */
void AnalysisGraph::extract_concept_indicator_mapping_and_observations_from_json(
        const nlohmann::json &json_indicators,
        ConceptIndicatorData &concept_indicator_data,
        ConceptIndicatorEpochs &concept_indicator_epochs) {

    int num_verts = this->num_vertices();

    this->train_start_epoch = LONG_MAX;
    this->train_end_epoch = 0;

    long epoch;

    for (int v = 0; v < num_verts; v++) {
        Node& n = (*this)[v];
        // At the moment we are only attaching one indicator per node
        // when Analysis graph is called through CauseMos
        string indicator_name = "Qualitative measure of {}"_format(n.name);
        string indicator_source = "Delphi";

        if (json_indicators[n.name].is_null()) {
            // In this case we do not have any observation data to train the model
            this->set_indicator(n.name, indicator_name, indicator_source);
            n.get_indicator(indicator_name).set_mean(1.0);
            continue;
        }

        // TODO: At the moment the json file specifies one indicator per
        // concept. At a later time if we update the json file to specify
        // multiple indicators per concept, this is one place we have to
        // update. Instead of a single indicator, we might get a list of
        // indicators here. Rest of the code would have to be updated
        // according to whatever the updated json file format we come up with.
        auto indicator = json_indicators[n.name];

        if (!indicator["source"].is_null()) {
            indicator_source = indicator["source"].get<string>();
        }

        if (indicator["name"].is_null()) {
            // This case there could be observations. However we are discarding
            // them. Why?
            this->set_indicator(n.name, indicator_name, indicator_source);
            n.get_indicator(indicator_name).set_mean(1.0);
            continue;
        }

        indicator_name = indicator["name"].get<string>();

        int ind_idx = this->set_indicator(n.name, indicator_name, indicator_source);

        if (ind_idx == -1) {
            // This is a duplicate indicator, and will not be added again.
            // TODO: Decide how to handle this situation and inform the user at
            // the CauseMos HMI end.
            continue;
        }

        // [ epoch --→ observation ]
        multimap<long, double> indicator_data;

        // Accumulate which epochs data points are available for
        // The idea is to use this to assess the frequency of the data.
        set<long> epochs;

        for (auto& data_point : indicator["values"]) {
            if (data_point["value"].is_null()) {
                // This is a missing data point
                continue;
            }

            double observation = data_point["value"].get<double>();

            if (data_point["timestamp"].is_null()) {continue;}

            // Note: HMI sends epochs as unix epoch * 1000
            epoch = data_point["timestamp"].get<long>();

            // Keep track of multiple observations for each epoch
            indicator_data.insert(make_pair(epoch, observation));

            // Record the epochs where observations are available for this
            // indicator. This data is used to assess the observation
            // frequency.
            epochs.insert(epoch);

            // Find the start epoch of observations. When observation
            // sequences are not aligned:
            // start epoch => earliest observation among all the
            // observation sequences.
            if (this->train_start_epoch > epoch) {
                this->train_start_epoch = epoch;
            }

            // Find the end epoch of observations. When observation
            // sequences are not aligned:
            // end epoch => latest observation among all the observation
            // sequences.
            if (this->train_end_epoch < epoch) {
                this->train_end_epoch = epoch;
            }
        }

        // Add this indicator observations to the concept. The data structure
        // has provision to assign multiple indicator observation sequences for
        // a single concept.
        concept_indicator_data[v].push_back(indicator_data);

        // To assess the frequency of the data
        vector<long> epoch_sorted;
        for (long epo : epochs) {
            epoch_sorted.push_back(epo);
        }
        sort(epoch_sorted.begin(), epoch_sorted.end());
        concept_indicator_epochs[v] = epoch_sorted;
    }
}

/** Infer the least common observation frequency for all the
 * observation sequences so that they are time aligned starting from the
 * train_start_epoch.
 * The advantage of modeling at the least common observation frequency is less
 * missing data points.
 *
 * Check method declaration in AnalysisGraph.hpp for a detailed comment.
 */
void AnalysisGraph::infer_modeling_frequency(
                        const ConceptIndicatorEpochs &concept_indicator_epochs,
                        int &shortest_gap,
                        int &longest_gap,
                        int &frequent_gap,
                        int &highest_frequency) {

    unordered_map<long, int> gap_frequencies;
    unordered_map<long, int>::iterator itr;
    unordered_set<long> epochs_all;

    for (vector<long> ind_epochs : concept_indicator_epochs) {
        vector<long> gaps = vector<long>(ind_epochs.size());

        adjacent_difference (ind_epochs.begin(), ind_epochs.end(), gaps.begin());
        epochs_all.insert(ind_epochs.begin(), ind_epochs.end());

        // Compute number of epochs between data points
        for (int gap = 1; gap < gaps.size(); gap++) {
            // Check whether two adjacent data points with the same gap of
            // epochs in between is already found.
            itr = gap_frequencies.find(gaps[gap]);

            if (itr != gap_frequencies.end()) {
                // There were previous adjacent pairs of data points with
                // gap epochs in between. Now we have found one more
                // so increase the number of data points at this frequency.
                itr->second++;
            } else {
                // This is the first data point that is gap epochs
                // away from its previous data point. Start recording this new
                // frequency.
                gap_frequencies.insert(make_pair(gaps[gap], 1));
            }
        }
    }

    // Find the smallest gap and most frequent gap
    shortest_gap = INT_MAX;
    longest_gap = 0;
    frequent_gap = 0;
    highest_frequency = 0;

    // Determine the modeling frequency of data, which is the greatest common
    // divisor of gaps between observations.
    vector<long> epochs_sorted = vector<long>(epochs_all.begin(), epochs_all.end());
    vector<long> gaps_all = vector<long> (epochs_sorted.size());
    sort(epochs_sorted.begin(), epochs_sorted.end());
    adjacent_difference (epochs_sorted.begin(), epochs_sorted.end(), gaps_all.begin());

    itr = gap_frequencies.begin();
    this->modeling_frequency = gaps_all[1];

    for(int i = 2; i < gaps_all.size(); i++){
        this->modeling_frequency = gcd(this->modeling_frequency, gaps_all[i]);
    }

    // Epochs statistics for individual indicator time series for debugging purposes.
    for(auto const& [gap, freq] : gap_frequencies){
        if (shortest_gap > gap) {
            shortest_gap = gap;
        }

        if (longest_gap < gap) {
            longest_gap = gap;
        }

        if (highest_frequency < freq) {
            highest_frequency = freq;
            frequent_gap = gap;
        }
    }
}

/**
 * Set the observed state sequence from the create model JSON input received
 * from the HMI.
 * The start_year, start_month, end_year, and end_month are inferred from the
 * observation sequences for indicators provided in the JSON input.
 * The sequence includes both ends of the range.
 */
void
AnalysisGraph::set_observed_state_sequence_from_json_dict(
        const nlohmann::json &json_indicators) {

    int num_verts = this->num_vertices();

    // This is a multimap to keep provision to have multiple observations per
    // time point per indicator.
    // Access (concept is a vertex in the CAG)
    // [ concept ][ indicator ][ epoch --→ observation ]
    ConceptIndicatorData concept_indicator_data(num_verts);

    // Keeps the sequence of epochs for which data points are available
    // Data points are sorted according to epochs
    // Access:
    // [ concept ][ indicator ][epoch]
    ConceptIndicatorEpochs concept_indicator_epochs(num_verts);

    this->extract_concept_indicator_mapping_and_observations_from_json(
            json_indicators, concept_indicator_data, concept_indicator_epochs);

    // Decide the data frequency.
    int shortest_gap = INT_MAX;
    int longest_gap = 0;
    int frequent_gap = 0;
    int highest_frequency = 0;

    this->n_timesteps = 0;

    if (this->train_start_epoch <= this->train_end_epoch) {
        // Some training data has been provided
        this->infer_modeling_frequency(concept_indicator_epochs,
                    shortest_gap, longest_gap, frequent_gap, highest_frequency);

        // Fill in observed state sequence
        // NOTE: This code is very similar to the implementations in
        // set_observed_state_sequence_from_data and get_observed_state_from_data
        this->n_timesteps = (this->train_end_epoch - this->train_start_epoch) / this->modeling_frequency + 1 ;
    }

    this->observed_state_sequence.clear();

    // Access (concept is a vertex in the CAG)
    // [ timestep ][ concept ][ indicator ][ observation ]
    this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

    for (int ts = 0; ts < this->n_timesteps; ts++) {
        this->observed_state_sequence[ts] = vector<vector<vector<double>>>(num_verts);

        for (int v = 0; v < num_verts; v++) {
            Node& n = (*this)[v];
            this->observed_state_sequence[ts][v] = vector<vector<double>>(n.indicators.size());

            for (int i = 0; i < n.indicators.size(); i++) {
                this->observed_state_sequence[ts][v][i] = vector<double>();

                long epoch = this->train_start_epoch + ts * this->modeling_frequency;
                pair<multimap<long, double>::iterator,
                     multimap<long, double>::iterator> obs =
                    concept_indicator_data[v][i].equal_range(epoch);

                for(auto it = obs.first; it != obs.second; it++) {
                    this->observed_state_sequence[ts][v][i].push_back(it->second);
                }
            }
        }
    }
}

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      create-experiment (private)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

std::pair<int, int> AnalysisGraph::timestamp_to_year_month(long timestamp) {
    // The HMI uses milliseconds. So they multiply time-stamps by 1000.
    // Before converting them back to year and month, we have to divide
    // by 1000.
    timestamp /=  1000;

    // Convert the time-step to year and month.
    // We are converting it according to GMT.
    struct tm *ptm = gmtime(&timestamp);
    int year = 1900 + ptm->tm_year;
    int month = 1 + ptm->tm_mon;

    return make_pair(year, month);
}

std::pair<int, int> AnalysisGraph::calculate_end_year_month(int start_year,
                                                            int start_month,
                                                            int num_timesteps) {
    int end_year = start_year + (start_month + num_timesteps -1) / 12;
    int end_month = (start_month + num_timesteps -1) % 12;

    if (end_month == 0) {
        end_month = 12;
        end_year--;
    }

    return make_pair(end_year, end_month);
}

double AnalysisGraph::calculate_prediction_timestep_length(int start_year,
                                                           int start_month,
                                                           int end_year,
                                                           int end_month,
                                                           int pred_timesteps) {

    /*
     * We calculate the number of training time steps that fits within the
     * provided start time point and the end time point (both ends inclusive)
     * and then divide that number by the number of requested prediction time
     * steps.
     *
     * Example:
     * pred_timesteps : 1 2  3 4  5 6  7 8  9 10  11 12  13 14  15 16
     * train_timesteps:  1    2    3    4    5      6      7      8
     *
     *            (# of training time steps between  *  (training time step
     *                 prediction start and end)             duration)
     * Δt_pred  = -----------------------------------------------------------
     *                             prediction time steps
     *
     *            (# of training time steps between  *  Δt_train
     *                 prediction start and end)
     * Δt_pred  = -----------------------------------------------------------
     *                             prediction time steps
     *
     * Δt_train = 1 (we set it as this for convenience)
     *
     *            (# of training time steps between prediction start and end)
     * Δt_pred  = -----------------------------------------------------------
     *                             prediction time steps
     *
     * Δt_pred  = 8/16 = 0.5
     *
     * In effect for each training time step we produce two prediction points.
     */
    // Calculate the number of training time steps between start and end of the
    // prediction time points (# training time steps)
    int num_training_time_steps = this->calculate_num_timesteps(start_year,
                                            start_month, end_year, end_month);

    return (double)num_training_time_steps / (double)pred_timesteps;
}

void AnalysisGraph::extract_projection_constraints(
                            const nlohmann::json &projection_constraints, long skip_steps) {
    for (auto constraint : projection_constraints) {
        if (constraint["concept"].is_null()) {continue;}
        string concept_name = constraint["concept"].get<string>();

        for (auto values : constraint["values"]) {
            // We need both the step and the value to proceed. Thus checking
            // again to reduce bookkeeping.
            if (values["step"].is_null()) {continue;}
            if (values["value"].is_null()) {continue;}
            int step     = values["step"].get<int>();
            double ind_value = values["value"].get<double>();

            // NOTE: Delphi is capable of attaching multiple indicators to a
            //       single concept. Since we are constraining the latent state,
            //       we can constrain (or intervene) based on only one of those
            //       indicators. By passing an empty indicator name, we choose
            //       to constrain the first indicator attached to a concept
            //       which should be present irrespective of whether this
            //       concept has one or more indicators attached to it.
            //
            // Requested prediction stat epoch is earlier than the training
            // start epoch. The first requested prediction epoch after the
            // training start epoch is skip_steps after the requested prediction
            // start epoch and the constraints within those skiped epochs
            // cannot be applied and hence are been ignored.
            if(step >= skip_steps){
                this->add_constraint(step-skip_steps, concept_name, "", ind_value);
            }
        }
    }
}

FormattedProjectionResult
AnalysisGraph::run_causemos_projection_experiment_from_json_dict(const nlohmann::json &json_data,
                                                                 int burn,
                                                                 int res) {

    if (json_data["experimentParam"].is_null()) {
        throw BadCausemosInputException("Experiment parameters null");
    }

    auto projection_parameters = json_data["experimentParam"];

    if (projection_parameters["startTime"].is_null()) {
        throw BadCausemosInputException("Projection start time null");
    }

    long proj_start_epoch = projection_parameters["startTime"].get<long>();

    if (projection_parameters["endTime"].is_null()) {
        throw BadCausemosInputException("Projection end time null");
    }

    long proj_end_epoch = projection_parameters["endTime"].get<long>();

    if (projection_parameters["numTimesteps"].is_null()) {
        throw BadCausemosInputException("Projection number of time steps null");
    }

    this->pred_timesteps = projection_parameters["numTimesteps"].get<int>();

    if(proj_start_epoch > proj_end_epoch) {
        throw BadCausemosInputException("Projection end epoch is before projection start epoch");
    }

    // Align prediction epochs with modeling frequency
    // 1 trainig timesteps = modeling_frequency epochs           (1)
    // 1 prediction timesteps = projection_frequencey epochs     (2)
    // [ (2) / (1) ] * 1 training timesteps gives
    // 1 prediction timestep = ( projection_frequency / modeling_frequency ) training timesteps
    double projecting_freq = (proj_end_epoch - proj_start_epoch) / (this->pred_timesteps - 1);
    this->delta_t = projecting_freq / this->modeling_frequency;

    // To help clamping we predict one additional timestep
    this->pred_timesteps++;
    long epochs_till_proj_start = proj_start_epoch - this->train_start_epoch;
    double init_pred_step =
        epochs_till_proj_start /  this->modeling_frequency - this->delta_t;

    long epochs_till_proj_end = proj_end_epoch - this->train_start_epoch;
    double last_pred_step = epochs_till_proj_end /  this->modeling_frequency;

    int skip_steps = 0;

    if(init_pred_step < 0) {
        skip_steps = ceil(abs(init_pred_step)/this->delta_t);
        init_pred_step += skip_steps;
        this->pred_timesteps -= skip_steps;
    }

    if(init_pred_step > last_pred_step) {
      throw BadCausemosInputException("Projection end epoch is before projection start epoch");
    }

    if (this->train_start_epoch > this->train_end_epoch) {
        // No training data has been provided => Cannot train
        //                                    => Cannot project
        throw BadCausemosInputException("No training data");
    }

    if (!this->trained) {
        this->run_train_model(res, burn);
    }

    this->extract_projection_constraints(projection_parameters["constraints"], skip_steps);

    this->generate_latent_state_sequences(init_pred_step);
    this->generate_observed_state_sequences();
    return this->format_projection_result();
}

FormattedProjectionResult AnalysisGraph::format_projection_result() {
  // Access
  // [ vertex_name ][ timestep ][ sample ]
  FormattedProjectionResult result;

  for (auto [vert_name, vert_id] : this->name_to_vertex) {
    result[vert_name] =
        vector<vector<double>>(this->pred_timesteps, vector<double>(this->res));
    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      for (int samp = 0; samp < this->res; samp++) {
        result[vert_name][ts][samp] =
            this->predicted_observed_state_sequences[samp][ts][vert_id][0];
      }
      ranges::sort(result[vert_name][ts]);
    }
  }

  return result;
}

void AnalysisGraph::sample_transition_matrix_collection_from_prior() {
  this->transition_matrix_collection.clear();
  this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);

  for (int sample = 0; sample < this->res; sample++) {
    for (auto e : this->edges()) {
      this->graph[e].theta = this->graph[e].kde.resample(
          1, this->rand_num_generator, this->uni_dist, this->norm_dist)[0];
    }

    // Create this->A_original based on the sampled β and remember it
    this->set_transition_matrix_from_betas();
    this->transition_matrix_collection[sample] = this->A_original;
  }
}

/*
============================================================================
Public: Integration with Uncharted's CauseMos interface
============================================================================
*/

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                          create-model (public)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void AnalysisGraph::from_causemos_json_dict(const nlohmann::json &json_data) {

  this->causemos_call = true;

  // TODO: If model id is not present, we might want to not create the model
  // and send a failure response. At the moment we just create a blank model,
  // which could lead to future bugs that are hard to debug.
  if (json_data["id"].is_null()){return;}
  this->id = json_data["id"].get<string>();

  auto statements = json_data["statements"];

  for (auto stmt : statements) {
    /*
    auto evidence = stmt["evidence"];

    if (evidence.is_null()) {
      continue;
    }
    */

    auto subj = stmt["subj"];
    auto obj = stmt["obj"];

    if (subj.is_null() or obj.is_null()) {
      continue;
    }

    auto subj_db_ref = subj["db_refs"];
    auto obj_db_ref = obj["db_refs"];

    if (subj_db_ref.is_null() or obj_db_ref.is_null()) {
      continue;
    }

    auto subj_concept_json = subj_db_ref["concept"];
    auto obj_concept_json = obj_db_ref["concept"];

    if (subj_concept_json.is_null() or obj_concept_json.is_null()) {
      continue;
    }

    string subj_name = subj_concept_json.get<string>();
    string obj_name = obj_concept_json.get<string>();

    auto subj_delta = stmt["subj_delta"];
    auto obj_delta = stmt["obj_delta"];

    auto subj_polarity_json = subj_delta["polarity"];
    auto obj_polarity_json = obj_delta["polarity"];

    // We set polarities to 1 (positive) by default if they are not specified.
    int subj_polarity_val = 1;
    int obj_polarity_val = 1;
    if (!subj_polarity_json.is_null()) {
      subj_polarity_val = subj_polarity_json.get<int>();
    }

    if (!obj_polarity_json.is_null()) {
      obj_polarity_val = obj_polarity_json.get<int>();
    }

    auto subj_adjectives = subj_delta["adjectives"];
    auto obj_adjectives = obj_delta["adjectives"];

    vector<int> subj_polarity = vector<int>{subj_polarity_val};
    vector<int> obj_polarity = vector<int>{obj_polarity_val};

    vector<string> subj_adjective = vector<string>{"None"};
    vector<string> obj_adjective = vector<string>{"None"};

    if(!subj_adjectives.is_null()){
        if(subj_adjectives.size() > 0){
            subj_adjective = subj_adjectives.get<vector<string>>();
            subj_polarity = vector<int>(subj_adjective.size(), subj_polarity_val);
        }
    }

    if(!obj_adjectives.is_null()){
        if(obj_adjectives.size() > 0){
            obj_adjective = obj_adjectives.get<vector<string>>();
            obj_polarity = vector<int>(obj_adjectives.size(), obj_polarity_val);
        }
    }

    auto causal_fragments =
        CausalFragmentCollection({subj_adjective, subj_polarity, subj_name},
                       {obj_adjective, obj_polarity, obj_name});
    this->add_edge(causal_fragments);
  }

  if (json_data["conceptIndicators"].is_null()) {
      // No indicator data provided.
      // TODO: What is the best action here?
      //throw runtime_error("No indicator information provided");
      // Maybe this is acceptable since there is another call: edit-indicators,
      // which is not yet implemented. An analyst can create a CAG structure
      // without any indicators and then later attach indicators one by one.
  } else {
      this->set_observed_state_sequence_from_json_dict(json_data["conceptIndicators"]);
  }

  this->initialize_random_number_generator();
  this->construct_theta_pdfs();
}

AnalysisGraph AnalysisGraph::from_causemos_json_string(string json_string, size_t res) {
  AnalysisGraph G;
  G.set_res(res);

  auto json_data = nlohmann::json::parse(json_string);
  G.from_causemos_json_dict(json_data);
  return G;
}

AnalysisGraph AnalysisGraph::from_causemos_json_file(string filename, size_t res) {
  AnalysisGraph G;
  G.set_res(res);

  auto json_data = load_json(filename);
  G.from_causemos_json_dict(json_data);
  return G;
}

/**
 * Generate the response for the create model request from the HMI.
 * For now we always return success. We need to update this by conveying
 * errors into this response.
 */
string AnalysisGraph::generate_create_model_response() {
    using nlohmann::json, ranges::max;

    json j;
    j["status"] = "success";
    j["relations"] = {};
    j["conceptIndicators"] = {};
    vector<double> all_weights = {};

    for (auto e : this->edges()) {
        json edge_json = {{"source", this->source(e).name},
                          {"target", this->target(e).name},
                          {"weight", 0.5}};

        j["relations"].push_back(edge_json);
    }

    int num_verts = this->num_vertices();
    for (int v = 0; v < num_verts; v++) {
        Node& n = (*this)[v];
        j["conceptIndicators"][n.name] = {{"initialValue", nullptr},
                                          {"scalingFactor", nullptr},
                                          {"scalingBias", nullptr}};
    }
    return j.dump();
}

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                       create-experiment (public)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

FormattedProjectionResult
AnalysisGraph::run_causemos_projection_experiment_from_json_string(string json_string,
                                                                   int burn,
                                                                   int res) {
    using namespace fmt::literals;
    using nlohmann::json;


    // During the create-model call we called construct_theta_pdfs() and
    // serialized them to json. When we recreate the model we load them. So to
    // prevent Delphi putting cycle to compute them again when we call
    // initialize_parameters() below, let us inform Delphi that this is a
    // CauseMos call.
    // NOTE: In hindsight, I might be able to prevent Delphi calling
    //       initialize_parameters() within train_model() when we train due to
    //       a CauseMos call. I need to revise it to see whether anything is
    //       called out of order.
    this->causemos_call = true;

    auto json_data = nlohmann::json::parse(json_string);

    try {
        return run_causemos_projection_experiment_from_json_dict(json_data, burn, res);
    }
    catch (BadCausemosInputException& e) {
        cout << e.what() << endl;
        // Just a dummy empty prediction to signal that there is an error in
        // projection parameters.
        return FormattedProjectionResult();
    }
}

FormattedProjectionResult
AnalysisGraph::run_causemos_projection_experiment_from_json_file(string filename,
                                                                 int burn,
                                                                 int res) {
    using namespace fmt::literals;
    using nlohmann::json;


    // During the create-model call we called construct_theta_pdfs() and
    // serialized them to json. When we recreate the model we load them. So to
    // prevent Delphi putting cycle to compute them again when we call
    // initialize_parameters() below, let us inform Delphi that this is a
    // CauseMos call.
    // NOTE: In hindsight, I might be able to prevent Delphi calling
    //       initialize_parameters() within train_model() when we train due to
    //       a CauseMos call. I need to revise it to see whether anything is
    //       called out of order.
    this->causemos_call = true;

    auto json_data = load_json(filename);

    try {
        return run_causemos_projection_experiment_from_json_dict(json_data, burn, res);
    }
    catch (BadCausemosInputException& e) {
        cout << e.what() << endl;
        // Just a dummy empty prediction to signal that there is an error in
        // projection parameters.
        return FormattedProjectionResult();
    }
}
