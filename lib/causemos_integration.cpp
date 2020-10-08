// CauseMos integration methods

#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include <fstream>
#include <range/v3/all.hpp>
#include <time.h>
#include <limits.h>
//#include <uuid/uuid.h>// for uuid
//#include <sys/wait.h> // for fork
//#include <unistd.h>   // for fork
#include "dbg.h"

using namespace std;
using namespace delphi::utils;
using namespace fmt::literals;

// Just for debugging. Remove later
using fmt::print;

/*
============================================================================
Private: Integration with Uncharted's CauseMos interface
============================================================================
*/

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            create-model
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
        ConceptIndicatorDates &concept_indicator_dates,
        int &start_year, int &start_month,
        int &end_year, int &end_month) {

    int num_verts = this->num_vertices();

    start_year = INT_MAX;
    start_month = 13;
    end_year = 0;
    end_month = 0;

    time_t timestamp;
    int year;
    int month;
    struct tm *ptm;

    for (int v = 0; v < num_verts; v++) {
        Node& n = (*this)[v];
        // At the moment we are only attaching one indicator per node
        // when Analysis graph is called through CauseMos
        string indicator_name = "Qualitative measure of {}"_format(n.name);
        string indicator_source = "Delphi";

        if (json_indicators[n.name].is_null()) {
            // In this case we do not have any observation data to train the model
            n.add_indicator(indicator_name, indicator_source);
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
            n.add_indicator(indicator_name, indicator_source);
            n.get_indicator(indicator_name).set_mean(1.0);
            continue;
        }

        indicator_name = indicator["name"].get<string>();

        int ind_idx = n.add_indicator(indicator_name, indicator_source);

        if (ind_idx == -1) {
            // This is a duplicate indicator, and will not be added again.
            // TODO: Decide how to handle this situation and inform the user at
            // the CauseMos HMI end.
            continue;
        }

        // [ <year, month> --→ observation ]
        multimap<pair<int, int>, double> indicator_data;

        // Accumulate which dates data points are available for
        // The idea is to use this to assess the frequency of the data. Either
        // yearly or monthly.
        set<pair<int, int>> dates;

        dbg("---------------------------");
        dbg(indicator_name);

        for (auto& data_point : indicator["values"]) {
            if (data_point["value"].is_null()) {
                // This is a missing data point
                continue;
            }

            double observation = data_point["value"].get<double>();

            if (data_point["timestamp"].is_null()) {continue;}

            // The HMI uses milliseconds. So they multiply time-stamps by 1000.
            // Before converting them back to year and month, we have to divide
            // by 1000.
            timestamp = data_point["timestamp"].get<long>() / 1000;

            // Convert the time-step to year and month.
            // We are converting it according to GMT.
            ptm = gmtime(&timestamp);
            year = 1900 + ptm->tm_year;
            month = 1 + ptm->tm_mon;

            pair<int, int> year_month = make_pair(year, month);

            // Keep track of multiple observations for each year-month
            indicator_data.insert(make_pair(year_month, observation));

            // Record the dates where observations are available for this
            // indicator. This data is used to assess the observation
            // frequency.
            // At the moment Delphi assumes a monthly observation frequency.
            // This part is added thinking that we might be able to relax that
            // constrain in the future. At the moment, this information is not
            // used in the modeling process.
            dates.insert(year_month);

            // Find the start year and month of observations. When observation
            // sequences are not aligned:
            // start year month => earliest observation among all the
            // observation sequences.
            if (start_year > year) {
                start_year = year;
                start_month = month;
            } else if (start_year == year && start_month > month) {
                start_month = month;
            }

            // Find the end year and month of observations. When observation
            // sequences are not aligned:
            // end year month => latest observation among all the observation
            // sequences.
            if (end_year < year) {
                end_year = year;
                end_month = month;
            } else if (end_year == year && end_month < month) {
                end_month = month;
            }
        }

        // Add this indicator observations to the concept. The data structure
        // has provision to assign multiple indicator observation sequences for
        // a single concept.
        concept_indicator_data[v].push_back(indicator_data);

        // To assess the frequency of the data
        vector<pair<int, int>> date_sorted;
        for (auto ym : dates) {
            date_sorted.push_back(ym);
        }
        sort(date_sorted.begin(), date_sorted.end());
        concept_indicator_dates[v] = date_sorted;

        // Aggregation function
        // NOTE: This portion is not aligned with a single indicator having
        // multiple observations per time point. At the moment we assume single
        // observation per indicator per time point.

        // "last" is the default function specified in the specification.
        string func = "last";

        if (!indicator["func"].is_null()) {
            func = indicator["func"].get<string>();
        }

        n.get_indicator(indicator_name).set_aggregation_method(func);
    }
}

/** Infer the least common observation frequency for all the
 * observation sequences so that they are time aligned starting from the
 * start_year and start_month.
 * At the moment we do not use the information we gather in this method as
 * the rest of the code by default models at a monthly frequency. The
 * advantage of modeling at the least common observation frequency is less
 * missing data points.
 *
 * Check method declaration in AnalysisGraph.hpp for a detailed comment.
 */
void AnalysisGraph::infer_modeling_frequency(
                        const ConceptIndicatorDates &concept_indicator_dates,
                        int &shortest_gap,
                        int &longest_gap,
                        int &frequent_gap,
                        int &highest_frequency) {

    unordered_map<int, int> gap_frequencies;
    unordered_map<int, int>::iterator itr;

    for (vector<pair<int, int>> ind_dates : concept_indicator_dates) {
        // Compute number of months between data points
        for (int i = 0; i < ind_dates.size() - 1;  i++) {
            int y1 = ind_dates[i].first;
            int m1 = ind_dates[i].second;
            int y2 = ind_dates[i+1].first;
            int m2 = ind_dates[i+1].second;

            int months_between = (y2 - y1) * 12 + (m2 - m1);

            // Check whether two adjacent data points with months_between
            // months in between is already found.
            itr = gap_frequencies.find(months_between);

            if (itr != gap_frequencies.end()) {
                // There were previous adjacent pairs of data points with
                // months_between months in between. Now we have found one more
                // so increase the number of data points at this frequency.
                itr->second++;
            } else {
                // This is the first data point that is months_between months
                // away from its previous data point. Start recording this new
                // frequency.
                gap_frequencies.insert(make_pair(months_between, 1));
            }
        }
    }

    // Find the smallest gap and most frequent gap
    shortest_gap = INT_MAX;
    longest_gap = 0;
    frequent_gap = 0;
    highest_frequency = 0;

    for (itr = gap_frequencies.begin(); itr != gap_frequencies.end(); itr++) {
        if (shortest_gap > itr->first) {
            shortest_gap = itr->first;
        }

        if (longest_gap < itr->first) {
            longest_gap = itr->first;
        }

        if (highest_frequency < itr->second) {
            highest_frequency = itr->second;
            frequent_gap = itr->first;
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
    // [ concept ][ indicator ][ <year, month> --→ observation ]
    ConceptIndicatorData concept_indicator_data(num_verts);

    // Keeps the sequence of dates for which data points are available
    // Data points are sorted according to dates
    // Access:
    // [ concept ][ indicator ][<year, month>]
    ConceptIndicatorDates concept_indicator_dates(num_verts);

    int start_year = INT_MAX;
    int start_month = 13;
    int end_year = 0;
    int end_month = 0;

    this->extract_concept_indicator_mapping_and_observations_from_json(
            json_indicators, concept_indicator_data, concept_indicator_dates,
                                start_year, start_month, end_year, end_month);

    this->training_range = make_pair(make_pair(start_year, start_month),
                                            make_pair(end_year, end_month));

    dbg(start_year);
    dbg(start_month);
    dbg(end_year);
    dbg(end_month);

    // Decide the data frequency.
    int shortest_gap = INT_MAX;
    int longest_gap = 0;
    int frequent_gap = 0;
    int highest_frequency = 0;

    this->infer_modeling_frequency(concept_indicator_dates,
                  shortest_gap, longest_gap, frequent_gap, highest_frequency);

    dbg(shortest_gap);
    dbg(longest_gap);
    dbg(frequent_gap);
    dbg(highest_frequency);

    // Fill in observed state sequence
    // NOTE: This code is very similar to the implementations in
    // set_observed_state_sequence_from_data and get_observed_state_from_data
    this->n_timesteps = this->calculate_num_timesteps(
            start_year, start_month, end_year, end_month);

    this->observed_state_sequence.clear();

    // Access (concept is a vertex in the CAG)
    // [ timestep ][ concept ][ indicator ][ observation ]
    this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

    int year = start_year;
    int month = start_month;

    for (int ts = 0; ts < this->n_timesteps; ts++) {
        this->observed_state_sequence[ts] = vector<vector<vector<double>>>(num_verts);

        for (int v = 0; v < num_verts; v++) {
            Node& n = (*this)[v];
            this->observed_state_sequence[ts][v] = vector<vector<double>>(n.indicators.size());

            for (int i = 0; i < n.indicators.size(); i++) {
                this->observed_state_sequence[ts][v][i] = vector<double>();

                pair<int, int> year_month = make_pair(year, month);
                pair<multimap<pair<int, int>, double>::iterator,
                     multimap<pair<int, int>, double>::iterator> obs =
                    concept_indicator_data[v][i].equal_range(year_month);

                for(auto it = obs.first; it != obs.second; it++) {
                    this->observed_state_sequence[ts][v][i].push_back(it->second);
                }
            }
        }

        if (month == 12) {
            year++;
            month = 1;
        }
        else {
            month++;
        }
    }

    this->to_png("CAG_from_json.png");
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

/*
============================================================================
Public: Integration with Uncharted's CauseMos interface
============================================================================
*/

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            create-model
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
    auto evidence = stmt["evidence"];

    if (evidence.is_null()) {
      continue;
    }

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
    int subj_polarity = 1;
    int obj_polarity = 1;
    if (!subj_polarity_json.is_null()) {
      subj_polarity = subj_polarity_json.get<int>();
    }

    if (!obj_polarity_json.is_null()) {
      obj_polarity = obj_polarity_json.get<int>();
    }

    auto subj_adjectives = subj_delta["adjectives"];
    auto obj_adjectives = obj_delta["adjectives"];
    auto subj_adjective =
        (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
            ? subj_adjectives[0]
            : "None";
    auto obj_adjective =
        (!obj_adjectives.is_null() and obj_adjectives.size() > 0)
            ? obj_adjectives[0]
            : "None";

    string subj_adj_str = subj_adjective.get<string>();
    string obj_adj_str = obj_adjective.get<string>();

    auto causal_fragment =
        CausalFragment({subj_adj_str, subj_polarity, subj_name},
                       {obj_adj_str, obj_polarity, obj_name});
    this->add_edge(causal_fragment);
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

AnalysisGraph AnalysisGraph::from_causemos_json_string(string json_string) {
  AnalysisGraph G;

  auto json_data = nlohmann::json::parse(json_string);
  G.from_causemos_json_dict(json_data);
  return G;
}

AnalysisGraph AnalysisGraph::from_causemos_json_file(string filename) {
  AnalysisGraph G;

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
        int n_samples = DEFAULT_N_SAMPLES;

        // TODO: This variable is not used
        vector<double> sampled_thetas = this->edge(e).kde.resample(
                n_samples, this->rand_num_generator, this->uni_dist, this->norm_dist);
        double weight = abs(median(this->edge(e).kde.dataset));
        all_weights.push_back(weight);

        json edge_json = {{"source", this->source(e).name},
                          {"target", this->target(e).name},
                          {"weight", weight}};

        j["relations"].push_back(edge_json);
    }

    double max_weight = max(all_weights);

    // Divide the weights by the max weight so they all lie between 0-1
    for (auto& relation : j["relations"]) {
        relation["weight"] = relation["weight"].get<double>() / max_weight;
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

/*
 * TODO: Remove this method
 * generate_create_model_response would replace this method
 */
string AnalysisGraph::get_edge_weights_for_causemos_viz() {
  using nlohmann::json, ranges::max;
  json j;
  j["relations"] = {};
  vector<double> all_weights = {};
  for (auto e : this->edges()) {
    int n_samples = DEFAULT_N_SAMPLES;
    // TODO: This variable is not used
    vector<double> sampled_thetas = this->edge(e).kde.resample(
        n_samples, this->rand_num_generator, this->uni_dist, this->norm_dist);
    double weight = abs(median(this->edge(e).kde.dataset));
    all_weights.push_back(weight);
    json edge_json = {{"source", this->source(e).name},
                      {"target", this->target(e).name},
                      {"weight", weight}};
    j["relations"].push_back(edge_json);
  }
  double max_weight = max(all_weights);
  // Divide the weights by the max weight so they all lie between 0-1
  for (auto& relation : j["relations"]) {
    relation["weight"] = relation["weight"].get<double>() / max_weight;
  }
  return j.dump();
}

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                          create-experiment
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void AnalysisGraph::create_causemos_experiment_from_json_string(
                                                std::string json_string) {

  auto json_data = nlohmann::json::parse(json_string);
  create_causemos_experiment_from_json_dict(json_data);
}

void AnalysisGraph::create_causemos_experiment_from_json_file(string filename) {

  auto json_data = load_json(filename);
  create_causemos_experiment_from_json_dict(json_data);
}

void AnalysisGraph::create_causemos_experiment_from_json_dict(
                                            const nlohmann::json &json_data) {
  string experiment_type = json_data["experimentType"].get<string>();

  // Decide the type of the experiment to run
  if (experiment_type.compare("PROJECTION") == 0)
  {
     this->run_causemose_projection_experiment(json_data["experimentParam"]);
  }
}

std::pair<int, int> AnalysisGraph::timestamp_to_year_month(long timestamp) {
    //struct tm *ptm;
    //
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
                            const nlohmann::json &projection_constraints) {
    // During the create-model call we called construct_theta_pdfs() and
    // serialized them to json. When we recreate the model we load them. So to
    // prevent Delphi putting cycle to compute them again when we call
    // initialize_parameters() below, let us inform Delphi that this is a
    // CauseMos call.
    // NOTE: In hind site, I might be able to prevent Delphi calling
    //       initialize_parameters() within train_model() when we train due to
    //       a CauseMos call. I need revise it to see whether anything is
    //       called out of order.
    this->causemos_call = true;

    // We need indicator means (calling them mean is incorrect. However since
    // we have been doing it so far for consistency that is carried forward
    // here. The proper way to call this is staling factor). So set indicator
    // means:
    this->initialize_parameters();

    for (auto constraint : projection_constraints) {
        if (constraint["concept"].is_null()) {continue;}
        string concept_name = constraint["concept"].get<string>();
        dbg(concept_name);

        // Check whether this concept is in the CAG
        if (!in(this->name_to_vertex, concept_name)) {
            print("Concept \"{0}\" not in CAG!\n", concept_name);
            continue;
        }

        int concept_id = this->name_to_vertex.at(concept_name);

        // Get the first indicator of this node
        // NOTE: Delphi is capable of attaching multiple indicators to a single
        //       concept. Since we are constraining the latent state, We can
        //       constrain (or intervene) based on only one of those
        //       indicators. We choose to constrain the first indicator
        //       attached to a concept which should be present irrespective of
        //       whether this concept has one or more indicators attached to
        //       it.
        const int ind_id = 0;
        Indicator& ind =  (*this)[concept_id].indicators[ind_id];

        // Allocate the One off constraints data structure to be populated with
        // constraints
        this->one_off_constraints.clear();
        for (auto values : constraint["values"]) {
            if (values["step"].is_null()) {continue;}
            int step     = values["step"].get<int>();

            this->one_off_constraints[step] = vector<pair<int, double>>();

            dbg(step);
        }

        for (auto values : constraint["values"]) {
            // We need both the step and the value to proceed. Thus checking
            // again to reduce bookkeeping.
            if (values["step"].is_null()) {continue;}
            if (values["value"].is_null()) {continue;}
            int step     = values["step"].get<int>();
            double ind_value = values["value"].get<double>();

            // We have to clamp the latent state value corresponding to this
            // indicator such that the probability where the emission Gaussian
            // emitting the requested indicator value is the highest. For a
            // Gaussian emission function, the highest probable value is its
            // mean. So we have to set:
            //      μ = ind_value
            //      latent_clam_value * scaling_factor = ind_value
            //      latent_clamp_value = ind_value / scaling_factor
            // NOTE: In our code we incorrectly call scaling_factor as
            //       indicator mean. To avoid confusion here, I am using the
            //       correct terminology.
            //       (Gosh, when we have incorrect terminology and we know it
            //       and we have not fixed it, I have to type a lot of
            //       comments)
            double latent_clamp_value = ind_value / ind.get_mean();
            dbg(ind_value);
            dbg(ind.get_mean());
            dbg(latent_clamp_value);

            this->one_off_constraints[step].push_back(
                                    make_pair(concept_id, latent_clamp_value));
        }
    }
}

Prediction AnalysisGraph::run_causemose_projection_experiment(std::string json_string) {
    using namespace fmt::literals;
    using nlohmann::json;

    auto json_data = nlohmann::json::parse(json_string);
    auto projection_parameters = json_data["experimentParam"];

    // Just a dummy empty prediction to signal that there is an error in
    // projection parameters.
    Prediction null_prediction = Prediction();

    dbg("running exp");
    time_t timestamp;
    pair<int, int> year_month;

    if (projection_parameters["startTime"].is_null()) {return null_prediction;}
    timestamp = projection_parameters["startTime"].get<long>();

    year_month = this->timestamp_to_year_month(timestamp);
    int pred_start_year = year_month.first;
    int pred_start_month = year_month.second;

    dbg(pred_start_year);
    dbg(pred_start_month);

    if (projection_parameters["endTime"].is_null()) {return null_prediction;}
    timestamp = projection_parameters["endTime"].get<long>();

    year_month = this->timestamp_to_year_month(timestamp);
    int pred_end_year_given = year_month.first;
    int pred_end_month_given = year_month.second;

    dbg(pred_end_year_given);
    dbg(pred_end_month_given);

    if (projection_parameters["numTimesteps"].is_null()) {return null_prediction;}
    int num_timesteps = projection_parameters["numTimesteps"].get<int>();

    // Calculate end_year, end_month assuming that each time step is a month.
    // In other words, we are calculating the end_year, end_month assuming that
    // the duration of a prediction time step = the duration of a training time
    // step.
    // Yet another explanation is we are assuming that both training and
    // prediction frequencies are the same.
    // NOTE: We are calculating this because:
    //       Earlier both CauseMos and Delphi used (year, month) as the time
    //       stamp interval. So it is somewhat hard coded into Delphi.
    //       Uncharted suddenly changed the protocol and stated to use Posix
    //       time stamps to index data. So we are using a hack hear. For the
    //       moment we are disregarding the CauseMos provided end_year and
    //       end_month (end_year_given and end_month_given) above and compute
    //       the end_year, end_month to match up with the number of prediction
    //       time steps CauseMos is requesting for. Then we are feeding these
    //       calculated end_year, end_month to Delphi prediction so that Delphi
    //       generates the desired number of prediction points. One caveat is
    //       that now Delphi is predicting on a monthly basis. We can do better
    //       by making Delphi predict at a different frequency than the
    //       training frequency by setting Δt appropriately.
    year_month = calculate_end_year_month(pred_start_year, pred_start_month,
                                          num_timesteps);
    int pred_end_year_calculated = year_month.first;
    int pred_end_month_calculated = year_month.second;

    dbg(pred_end_year_calculated);
    dbg(pred_end_month_calculated);

    // NOTE: At the moment we are assuming that delta_t for prediction is also
    // 1. This is an effort to do otherwise which might make things better in
    // the long run. This is commented because, we have to check whether this
    // is mathematically sound and this is the correct way to handle the
    // situation.
    //this->delta_t = calculate_prediction_timestep_length(start_year, start_month,
    //                                                     end_year_given,
    //                                                     end_month_given,
    //                                                     num_timesteps);

    this->extract_projection_constraints(projection_parameters["constraints"]);

    int train_start_year = this->training_range.first.first;
    int train_start_month = this->training_range.first.second;
    int train_end_year = this->training_range.second.first;
    int train_end_month = this->training_range.second.second;

    this->train_model(train_start_year, train_start_month,
                            train_end_year, train_end_month);
    return this->generate_prediction(pred_start_year, pred_start_month,
                                     pred_end_year_calculated,
                                     pred_end_month_calculated);

    // We do not need to create multiple processes at Delphi end as Flask is
    // handling that.
    /*
    // Generate the experiment UUID
    char ch[37];
    memset(ch, 0, 37);
    uuid_t uuid;
    uuid_generate(uuid);
    uuid_unparse(uuid, ch);
    string* projection_experiment_uuid = new string(ch);
    dbg(*projection_experiment_uuid);

    // Create create-experiment::projection response
    json create_projection_response;
    create_projection_response["experimentId"] = *projection_experiment_uuid;
    free(projection_experiment_uuid);

    dbg(create_projection_response.dump(4));

    //this->train_model(train_start_year, train_start_month,
    //                            train_end_year, train_end_month, 20, 20);
    //return(create_projection_response.dump());

    // Create a parent and child processes
    // parent will return to the caller and the child will train the model
    pid_t pid = fork();

    if (pid > 0) {
        print("Parent process. Child is {0}\n", pid);
        int stat;
        wait(NULL);
        if (WIFEXITED(stat)) {
            printf("Exit status: %d\n", WEXITSTATUS(stat));
        } else if (WIFSIGNALED(stat)) {
            psignal(WTERMSIG(stat), "Exit signal");
        }
        return(create_projection_response.dump());
    } else {
        print("Child process. I got pid as {0}\n", pid);
        this->train_model(train_start_year, train_start_month,
                                train_end_year, train_end_month);
        this->generate_prediction(pred_start_year, pred_start_month,
                                  pred_end_year_calculated,
                                  pred_end_month_calculated);
        dbg("Training completed");
        return("Child returns\n");
    }
    */
}

FormattedProjectionResult
AnalysisGraph::generate_causemos_projection(string json_projection) {
  auto json_data = nlohmann::json::parse(json_projection);
  this->initialize_random_number_generator();
  this->find_all_paths();
  this->sample_transition_matrix_collection_from_prior();

  auto start_time = json_data["startTime"];
  int start_year = start_time["year"].get<int>();
  int start_month = start_time["month"].get<int>();

  int time_steps = json_data["timeStepsInMonths"].get<int>();

  // This calculation adds one more month to the time steps
  // To avoid that we need to check some corner cases
  int end_year = start_year + (start_month + time_steps) / 12;
  int end_month = (start_month + time_steps) % 12;

  // Create the perturbed initial latent state
  this->set_default_initial_state();

  auto perturbations = json_data["perturbations"];

  for (auto pert : perturbations) {
    string concept = pert["concept"].get<string>();
    double value = pert["value"].get<double>();

    this->s0(2 * this->name_to_vertex.at(concept) + 1) = value;
  }

  this->trained = true;
  this->run_model(start_year, start_month, end_year, end_month, true);
  this->trained = false;

  return this->format_projection_result();
}
