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

AnalysisGraph AnalysisGraph::from_causemos_json_dict(nlohmann::json json_data) {
  AnalysisGraph G;

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
    G.add_edge(causal_fragment);
  }

  G.set_observed_state_sequence_from_json_dict(json_data);

  dbg("Done with json");
  G.initialize_random_number_generator();
  dbg("starting theta pdfs");
  G.construct_theta_pdfs();
  dbg("theta pdfs done");
  return G;
}


void
AnalysisGraph::set_observed_state_sequence_from_json_dict(
        nlohmann::json json_data) {
    using ranges::to;
    using ranges::views::transform;

    int num_verts = this->num_vertices();

    // This is a multimap to keep provision to have multiple observations per
    // time point per indicator.
    // Access (concept is a vertex in the CAG)
    // [ concept ][ indicator ][ <year, month> --→ observation ]
    vector<vector<multimap<pair<int, int>, double>>> concept_indicator_data(num_verts);

    // Keeps the sequence of dates for which data points are available
    // Data points are sorted according to dates
    // Access:
    // [ concept ][ indicator ][<year, month>]
    vector<vector<pair<int, int>>> concept_indicator_dates(num_verts);

    time_t timestamp;
    int year;
    int month;
    int start_year = INT_MAX;
    int start_month = 13;
    int end_year = 0;
    int end_month = 0;
    struct tm *ptm;

    for (int v = 0; v < num_verts; v++) {
        Node& n = (*this)[v];
        // At the moment we are only attaching one indicator per node
        // when Analysis graph is called through CauseMos
        string indicator_name = "Qualitative measure of {}"_format(n.name);
        string indicator_source = "Delphi";

        if (json_data["conceptIndicators"][n.name].is_null()) {
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
        // according to whatever the updated json file  format we come up with.
        auto indicator = json_data["conceptIndicators"][n.name];

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
        vector<double> values = {};

        for (auto& data_point : indicator["values"]) {
            if (data_point["value"].is_null()) {
                // This is a missing data point
                continue;
            }

            double observation = data_point["value"].get<double>();

            // NOTE: This variable assumes that there is a single observation
            // per indicator per time point.
            // If we are updating to single indicator having multiple
            // observations per time point, we have to reconsider this.
            values.push_back(observation);

            if (data_point["timestamp"].is_null()) {continue;}
            timestamp = data_point["timestamp"].get<long>() / 1000;

            ptm = gmtime(&timestamp);
            year = 1900 + ptm->tm_year;
            month = 1 + ptm->tm_mon;

            pair<int, int> year_month = make_pair(year, month);
            indicator_data.insert(make_pair(year_month, observation));

            dates.insert(year_month);

            if (start_year > year) {
                start_year = year;
                start_month = month;
            } else if (start_year == year && start_month > month) {
                start_month = month;
            }

            if (end_year < year) {
                end_year = year;
                end_month = month;
            } else if (end_year == year && end_month < month) {
                end_month = month;
            }
        }

        concept_indicator_data[v].push_back(indicator_data);

        // Assess the frequency of the data
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

    dbg(start_year);
    dbg(start_month);
    dbg(end_year);
    dbg(end_month);

    // Decide the data frequency.
    // TODO: Move this to a separate method
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

            itr = gap_frequencies.find(months_between);

            if (itr != gap_frequencies.end()) {
                itr->second++;
            } else {
                gap_frequencies.insert(make_pair(months_between, 1));
            }
        }
    }
    // Find the smallest gap and most frequent gap
    int shortest_gap = INT_MAX;
    int longest_gap = 0;
    int frequent_gap = 0;
    int highest_frequency = 0;
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
    // NOTE:
    // shortest_gap = longest_gap = 1  ⇒ monthly with no missing data
    // shortest_gap = longest_gap = 12 ⇒ yearly with no missing data
    // shortest_gap = longest_gap ≠ 1 or 12  ⇒ no missing data odd frequency
    // shortest_gap = 1 < longest_gap ⇒ monthly with missing data
    //      frequent_gap = 1 ⇒ little missing data
    //      frequent_gap > 1 ⇒ lot of missing data
    // 1 < shortest_gap < longest_gap
    //      best frequency to model at is the greatest common divisor of all
    //      gaps. For example if we see gaps 4, 6, 10 then gcd(4, 6, 10) = 2
    //      and modeling at a frequency of 2 months starting from the start
    //      date would allow us to capture all the observation sequences while
    //      aligning them with each other.
    // TODO: At the moment, by default we are modeling at monthly frequency. We
    // can and might need to make the program adapt to best frequency present
    // in the training data.
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

    year = start_year;
    month = start_month;

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
  auto json_data = nlohmann::json::parse(json_string);
  return AnalysisGraph::from_causemos_json_dict(json_data);
}

AnalysisGraph AnalysisGraph::from_causemos_json_file(string filename) {
  auto json_data = load_json(filename);
  return AnalysisGraph::from_causemos_json_dict(json_data);
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
