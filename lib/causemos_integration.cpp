// CauseMos integration methods

#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include <fstream>
#include <range/v3/all.hpp>
#include <time.h>
#include <limits.h>
#include "dbg.h"

using namespace std;
using namespace delphi::utils;
using namespace fmt::literals;

// Just for debugging. Remove later
using fmt::print;

void AnalysisGraph::from_causemos_json_dict(const nlohmann::json &json_data) {

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
        // TODO: How does this "and" in the condition work??
        (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
            ? subj_adjectives[0]
            : "None";
    auto obj_adjective =
        (obj_adjectives.size() > 0) ? obj_adjectives[0] : "None";

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
      throw runtime_error("No indicator information provided");
  }

  this->set_observed_state_sequence_from_json_dict(json_data["conceptIndicators"]);

  this->initialize_random_number_generator();
  this->construct_theta_pdfs();
}


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

        // Calculate aggregate from the values given
        // NOTE: This variable assumes that there is a single observation
        // per indicator per time point.
        // If we are updating to single indicator having multiple
        // observations per time point, we have to reconsider this.
        vector<double> values = {};

        for (auto& data_point : indicator["values"]) {
            if (data_point["value"].is_null()) {
                // This is a missing data point
                continue;
            }

            double observation = data_point["value"].get<double>();

            values.push_back(observation);

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
        double aggregated_value = 0.0;

        if (!indicator["func"].is_null()) {
            func = indicator["func"].get<string>();
        }

        if (func == "max") {
            aggregated_value = ranges::max(values);
        }
        else if (func == "min") {
            aggregated_value = ranges::min(values);
        }
        else if (func == "mean") {
            aggregated_value = mean(values);
        }
        else if (func == "median") {
            aggregated_value = median(values);
        }
        else if (func == "first") {
            aggregated_value = values[0];
        }
        else if (func == "last") {
            aggregated_value = values.back();
        }
        else {
            // Since we set func = "last" as the default value, this error will
            // never happen. We can remove this part.
            throw runtime_error(
                    "Invalid value of \"func\": {}. It must be one of [max|min|mean|median|first|last]"_format(
                        func));
        }

        n.get_indicator(indicator_name).set_aggregation_method(func);
        n.get_indicator(indicator_name).set_mean(aggregated_value);

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
void AnalysisGraph::infer_least_common_observation_frequency(
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

    dbg(start_year);
    dbg(start_month);
    dbg(end_year);
    dbg(end_month);

    // Decide the data frequency.
    int shortest_gap = INT_MAX;
    int longest_gap = 0;
    int frequent_gap = 0;
    int highest_frequency = 0;

    this->infer_least_common_observation_frequency(concept_indicator_dates,
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

                    //print("{}: {}: {}-{} --> {}\n", n.name, n.indicators[i].get_name(), year, month, this->observed_state_sequence[ts][v][i].back());
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
