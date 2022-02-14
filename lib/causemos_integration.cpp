// CauseMos integration methods

#include "AnalysisGraph.hpp"
#include "ExperimentStatus.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include <fstream>
#include <range/v3/all.hpp>
#include <time.h>
#include <limits.h>

using namespace std;
using namespace delphi::utils;
using namespace fmt::literals;
using fmt::print;

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

    DataAggregationLevel first_resolution = DataAggregationLevel::NONE;

    for (int v = 0; v < num_verts; v++) {
        Node& n = (*this)[v];
        // At the moment we are only attaching one indicator per node
        // when Analysis graph is called through CauseMos
        string indicator_name = "Qualitative measure of {}"_format(n.name);
        string indicator_source = "Delphi";

        if (!json_indicators.contains(n.name)) {
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

        if (!indicator["minValue"].is_null()) {
            n.has_min = true;
            n.min_val_obs = indicator["minValue"].get<double>();
        }
        if (!indicator["maxValue"].is_null()) {
            n.has_max = true;
            n.max_val_obs = indicator["maxValue"].get<double>();
        }

        if (!indicator["period"].is_null()) {
            n.period = indicator["period"].get<int>();
        }

        if (!indicator["resolution"].is_null()) {
            string resolution = indicator["resolution"].get<string>();
            if (resolution.compare("month") == 0) {
                n.agg_level = DataAggregationLevel::MONTHLY;
            }
            else if (resolution.compare("year") == 0) {
                n.agg_level = DataAggregationLevel::YEARLY;
            }
        }

        if (first_resolution == DataAggregationLevel::NONE) {
            first_resolution = n.agg_level;
            this->model_data_agg_level = n.agg_level;
        }
        else if (first_resolution != n.agg_level) {
            cout << "\n* * * WARNING: Mixed resolution observations! * * *\n";
            cout << "\t" << n.name << " has resolution " <<
                (n.agg_level == DataAggregationLevel::YEARLY ? "yearly" : "monthly") << endl;
            cout << "\tDelphi is currently not designed to model mixed resolution data\n";
            cout << "\tThe predictions will be unintuitive\n\n";
        }

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
        concept_indicator_epochs[v] = vector<long>(epochs.begin(), epochs.end());
    }
}

double AnalysisGraph::epoch_to_timestep(long epoch, long train_start_epoch,
                                        long modeling_frequency) {
  return (epoch - train_start_epoch) / double(modeling_frequency);
}

/** Infer the best sampling period to align observations to be used as the
 * modeling frequency from all the observation sequences.
 *
 * We consider the sequence of epochs where observations are available and
 * then the gaps in epochs between adjacent observations. We take the most
 * frequent gap as the modeling frequency. When more than one gap is most
 * frequent, we take the smallest such gap.
 */
void AnalysisGraph::infer_modeling_period(
                        const ConceptIndicatorEpochs & concept_indicator_observation_timesteps,
                        long &shortest_gap,
                        long &longest_gap,
                        long &frequent_gap,
                        int &highest_frequency) {

    unordered_set<long> observation_timesteps_all;

    for (vector<long> ind_observation_timesteps : concept_indicator_observation_timesteps) {
        observation_timesteps_all.insert(ind_observation_timesteps.begin(),
                                         ind_observation_timesteps.end());
    }

    this->observation_timesteps_sorted = vector<long>(observation_timesteps_all.begin(),
                                                      observation_timesteps_all.end());
    sort(this->observation_timesteps_sorted.begin(),
         this->observation_timesteps_sorted.end());

    vector<long> observation_timestep_gaps(this->observation_timesteps_sorted.size());
    adjacent_difference(this->observation_timesteps_sorted.begin(),
                        this->observation_timesteps_sorted.end(),
                        observation_timestep_gaps.begin());

    // Compute number of epochs between data points
    unordered_map<long, int> gap_frequencies;
    unordered_map<long, int>::iterator itr;

    for (int gap = 1; gap < observation_timestep_gaps.size(); gap++) {
        // Check whether two adjacent data points with the same gap of
        // epochs in between is already found.
        itr = gap_frequencies.find(observation_timestep_gaps[gap]);

        if (itr != gap_frequencies.end()) {
            // There were previous adjacent pairs of data points with
            // gap epochs in between. Now we have found one more
            // so increase the number of data points at this frequency.
            itr->second++;
        } else {
            // This is the first data point that is gap epochs
            // away from its previous data point. Start recording this new
            // frequency.
            gap_frequencies.insert(make_pair(observation_timestep_gaps[gap], 1));
        }
    }

    // Find the smallest gap and most frequent gap
    shortest_gap = LONG_MAX;
    longest_gap = 0;
    frequent_gap = LONG_MAX;
    highest_frequency = 0;

    // Epochs statistics for individual indicator time series for debugging purposes.
    for(auto const& [gap, freq] : gap_frequencies){
        if (shortest_gap > gap) {
            shortest_gap = gap;
        }

        if (longest_gap < gap) {
            longest_gap = gap;
        }

        if (highest_frequency == freq) {
            // In case of multiple observation_timestep_gaps having the same highest frequency,
            // note down the shortest highest frequency gap
            if (frequent_gap > gap) {
              frequent_gap = gap;
            }
        } else if (highest_frequency < freq) {
          highest_frequency = freq;
          frequent_gap = gap;
        }
    }

    // num_modeling_timesteps_per_one_observation_timestep is the number of
    // observation timesteps taken as one modeling timestep.
    // Let us assume we have observations at observation timesteps
    // 0, 4, 10, 14, 22
    // Then, to advance the system from one observation timestep to the next
    // we have to use following observation timestep gaps:
    // 4, 6, 4, 8
    // If we use num_modeling_timesteps_per_one_observation_timestep = 1, we
    // have to calculate matrix exponentials for 4, 6 and 8 to advance the system.
    // However, if we instead use
    // num_modeling_timesteps_per_one_observation_timestep = gcd(4, 6, 8) = 2,
    // we could compute matrix exponentials for 2, 3, 4.
    // But, then if we later (during projection) have to advance the system by 1
    // observation timestep, we have to compute a matrix exponential for 0.5.
    // If we use
    // num_modeling_timesteps_per_one_observation_timestep = min(4, 6, 8) = 4,
    // we have to compute matrix exponentials for 1, 1.5, 2 and to advance the
    // system by 1 observation timestep we have to compute a matrix exponential
    // for 0.25.
    // In general, to advance the system by t observation timesteps, we have to
    // compute a matrix exponential for
    // t/num_modeling_timesteps_per_one_observation_timestep.
    // For the current naive seasonal head node model to work,
    // num_modeling_timesteps_per_one_observation_timestep **MUST** be 1
    this->num_modeling_timesteps_per_one_observation_timestep = 1; // one timestep //frequent_gap;

    vector<double> modeling_timesteps(this->observation_timesteps_sorted.size());
    transform(this->observation_timesteps_sorted.begin(),
              this->observation_timesteps_sorted.end(),
              modeling_timesteps.begin(),
             [&](long observation_timestep) {
                 //return this->epoch_to_timestep(observation_timestep, this->train_start_epoch,
                 //                               this->num_modeling_timesteps_per_one_observation_timestep);
                 //return this->epoch_to_timestep(
                 //    observation_timestep, 0,
                 //                               this->num_modeling_timesteps_per_one_observation_timestep);
                 return observation_timestep / this->num_modeling_timesteps_per_one_observation_timestep;
              });

    this->modeling_timestep_gaps.clear();
    this->modeling_timestep_gaps = vector<double>(modeling_timesteps.size());
    adjacent_difference(modeling_timesteps.begin(),
                        modeling_timesteps.end(),
                        this->modeling_timestep_gaps.begin());
}


void AnalysisGraph::infer_concept_period(const ConceptIndicatorEpochs &concept_indicator_epochs) {
  double milliseconds_per_day = 24 * 60 * 60 * 1000.0;
  int min_days_global = INT32_MAX;
  int start_day = INT32_MAX;
  this->num_modeling_timesteps_per_one_observation_timestep = 1;

  for (int concept_id = 0; concept_id < concept_indicator_epochs.size();
       concept_id++) {
    vector<long> ind_epochs = concept_indicator_epochs[concept_id];
    sort(ind_epochs.begin(), ind_epochs.end());

    vector<long> ind_days(ind_epochs.size());
    transform(ind_epochs.begin(), ind_epochs.end(), ind_days.begin(),
              [&](long epoch) {
                return round(epoch / milliseconds_per_day);
              });

    vector<tuple<int, int, int>> year_month_dates(ind_epochs.size());
    transform(ind_epochs.begin(), ind_epochs.end(),
        year_month_dates.begin(),
              [&](long epoch) {
                return this->epoch_to_year_month_date(epoch);
              });

    vector<int> gaps_in_months(year_month_dates.size() - 1);

    int shortest_monthly_gap_ind = INT32_MAX;
    for (int ts = 0; ts < year_month_dates.size() - 1; ts++) {
      int observation_timesteps = delphi::utils::observation_timesteps_between(
            year_month_dates[ts], year_month_dates[ts + 1]);
      gaps_in_months[ts] = observation_timesteps;

      if (observation_timesteps < shortest_monthly_gap_ind) {
        shortest_monthly_gap_ind = observation_timesteps;
      }
    }

//    vector<long> gaps(ind_epochs.size());
//    adjacent_difference(ind_epochs.begin(), ind_epochs.end(), gaps.begin());
//
//    vector<int> days_between(gaps.size() - 1);
//    transform(gaps.begin() + 1, gaps.end(), days_between.begin(),
//              [&](long gap) {
//                return round(gap / milliseconds_per_day);
//              });
    vector<int> days_between(ind_days.size());
    adjacent_difference(ind_days.begin(), ind_days.end(), days_between.begin());

    int min_days = INT32_MAX;
//    for (int days : days_between) {
//      if (days < min_days) {
//        min_days = days;
//      }
//    }
    for (int idx = 1; idx < days_between.size(); idx++) {
      int days = days_between[idx];
      if (days < min_days) {
        min_days = days;
      }
    }

    if (ind_days[0] < start_day) {
      start_day = ind_days[0];
    }

    if (min_days < min_days_global) {
      min_days_global = min_days;
    }

    int period = 1;
    if (min_days == 1) {
      // Daily
      period = 365;
    } else if (min_days == 7) {
      // Weekly
      period = 52;
    } else if (28 <= min_days && min_days <= 31) {
      // Monthly
      period = 12;
    } else if (59 <= min_days && min_days <= 62) {
      // 2 Months
      period = 6;
    } else if (89 <= min_days && min_days <= 92) {
      // 3 Months
      period = 4;
    } else if (120 <= min_days && min_days <= 123) {
      // 4 Months
      period = 3;
    } else if (181 <= min_days && min_days <= 184) {
      // 6 Months
      period = 2;
    }
    /*
    else if (365 <= min_days && min_days <= 366) {
      // Yearly
    }
    else if (730 <= min_days && min_days <= 731) {
      // 2 Years
    } else if (1095 <= min_days && min_days <= 1096) {
      // 3 Years
    } else if (1460 <= min_days && min_days <= 1461) {
      // 4 Years
    } else if (1825 <= min_days && min_days <= 1827) {
      // 5 Years
    }
    */
    this->graph[concept_id].period = period;
    this->num_modeling_timesteps_per_one_observation_timestep = lcm(this->num_modeling_timesteps_per_one_observation_timestep, period);

    for (auto yyyy_mm_dd : year_month_dates) {
      cout << "(" << get<0>(yyyy_mm_dd) << "-" << get<1>(yyyy_mm_dd) << "-" << get<2>(yyyy_mm_dd) << "), ";
    }
    cout << endl;
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
    long shortest_gap = LONG_MAX;
    long longest_gap = 0;
    long frequent_gap = LONG_MAX;
    int highest_frequency = 0;

    this->n_timesteps = 0;

    unordered_map<int, unordered_set<long>> observation_timestep_to_epochs;

    if (this->train_start_epoch <= this->train_end_epoch) {
        // Convert epochs to observation timesteps making train_start_epoch = 0
        tuple<int, int, int> train_start_date =
            this->epoch_to_year_month_date(this->train_start_epoch);
        tuple<int, int, int> train_end_date =
            this->epoch_to_year_month_date(this->train_end_epoch);

        for (int v = 0; v < num_verts; v++) {
            Node& n = (*this)[v];
            // int shortest_monthly_gap_ind = INT32_MAX;
            for (int obs = 0; obs < concept_indicator_epochs[v].size(); obs++) {
                long epoch = concept_indicator_epochs[v][obs];
                tuple<int, int, int> obs_date =
                    this->epoch_to_year_month_date(epoch);
                int observation_timestep = delphi::utils::observation_timesteps_between(
                    train_start_date, obs_date, n.agg_level);

                // We reuse this variable to store concept_indicator_timesteps
                concept_indicator_epochs[v][obs] = observation_timestep;

                observation_timestep_to_epochs[observation_timestep].insert(epoch);

                /*
                // TODO: There are cases this period estimation goes wrong. Need revision.
                // e.g. observation months: 1, 3, 5, 8, 10, 12
                // The good solution is to get the gcd of all the observation gaps for a indicator
                if (obs > 0) {
                    // TODO: Since we don't sort concept_indicator_epochs[v] this calculation is wrong
                    int gap = concept_indicator_epochs[v][obs] -
                              concept_indicator_epochs[v][obs - 1];
                    if (gap < shortest_monthly_gap_ind) {
                        shortest_monthly_gap_ind = gap;
                    }
                }
                */
            }

            /*
            int period = 1;
            if (shortest_monthly_gap_ind == 1) {
              // Monthly
              period = 12;
            } else if (shortest_monthly_gap_ind == 2) {
              // 2 Months
              period = 6;
            } else if (shortest_monthly_gap_ind == 3) {
              // 3 Months
              period = 4;
            } else if (shortest_monthly_gap_ind == 4) {
              // 4 Months
              period = 3;
            } else if (shortest_monthly_gap_ind == 6) {
              // 6 Months
              period = 2;
            }
            this->graph[v].period = period;
            */
        }
//        cout << "Inferring periods\n";
//        this->infer_concept_period(concept_indicator_epochs);

        // Some training data has been provided
        this->infer_modeling_period(concept_indicator_epochs, shortest_gap,
                                    longest_gap, frequent_gap, highest_frequency);

        this->n_timesteps = this->observation_timesteps_sorted.size();
    }

    this->observed_state_sequence.clear();

    // Access (concept is a vertex in the CAG)
    // [ timestep ][ concept ][ indicator ][ observation ]
    this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);

    // Fill in observed state sequence
    // NOTE: This code is very similar to the implementations in
    // set_observed_state_sequence_from_data and get_observed_state_from_data
    for (int ts = 0; ts < this->n_timesteps; ts++) {
        this->observed_state_sequence[ts] = vector<vector<vector<double>>>(num_verts);

        for (int v = 0; v < num_verts; v++) {
            Node& n = (*this)[v];

            if (concept_indicator_data[v].empty()) {
                // This concept has no indicator specified in the create-model
                // call
                continue;
            }

            this->observed_state_sequence[ts][v] = vector<vector<double>>(n.indicators.size());

            for (int i = 0; i < n.indicators.size(); i++) {
                this->observed_state_sequence[ts][v][i] = vector<double>();

                for (long epochs_in_ts :
                     observation_timestep_to_epochs[this->observation_timesteps_sorted[ts]]) {
                    pair<multimap<long, double>::iterator,
                         multimap<long, double>::iterator>
                        obs = concept_indicator_data[v][i].equal_range(epochs_in_ts);

                    for (auto it = obs.first; it != obs.second; it++) {
                      this->observed_state_sequence[ts][v][i].push_back(
                          it->second);
                    }
                }
            }
        }
    }
}

void AnalysisGraph::from_causemos_json_dict(const nlohmann::json &json_data,
                                            double belief_score_cutoff,
                                            double grounding_score_cutoff
                                            ) {

  this->causemos_call = true;

  // TODO: If model id is not present, we might want to not create the model
  // and send a failure response. At the moment we just create a blank model,
  // which could lead to future bugs that are hard to debug.
  if (!json_data.contains("id")){return;}
  this->id = json_data["id"].get<string>();

  if (!json_data.contains("statements")) {return;}

  if(json_data.contains("experiment_id")){
    this->experiment_id = json_data["experiment_id"].get<string>();
  }

  auto statements = json_data["statements"];

  for (auto stmt : statements) {
    if (!stmt.contains("belief") or
        stmt["belief"].get<double>() < belief_score_cutoff) {
      continue;
    }

    auto evidence = stmt["evidence"];

    if (evidence.is_null()) {
      continue;
    }

    auto subj = stmt["subj"];
    auto obj = stmt["obj"];

    if (subj.is_null() or obj.is_null()) {
      continue;
    }

    auto subj_score_json = subj["concept_score"];
    auto obj_score_json = obj["concept_score"];

    if (subj_score_json.is_null() or obj_score_json.is_null()) {
      continue;
    }

    double subj_score = subj_score_json.get<double>();
    double obj_score = obj_score_json.get<double>();

    if (subj_score < grounding_score_cutoff or
        obj_score < grounding_score_cutoff) {
      continue;
    }

    auto subj_concept_json = subj["concept"];
    auto obj_concept_json = obj["concept"];

    if (subj_concept_json.is_null() or obj_concept_json.is_null()) {
      continue;
    }

    string subj_name = subj_concept_json.get<string>();
    string obj_name = obj_concept_json.get<string>();

    if (subj_name.compare(obj_name) == 0) { // Guard against self loops
      // Add the nodes to the graph if they are not in it already
      continue;
    }

    this->add_node(subj_name);
    this->add_node(obj_name);

    // Add the edge to the graph if it is not in it already
    for (auto evid : evidence) {
      if (!evid.contains("evidence_context")) {
        continue;
      }

      auto evid_context = evid["evidence_context"];

      auto subj_polarity_json = evid_context["subj_polarity"];
      auto obj_polarity_json = evid_context["obj_polarity"];

      // We set polarities to 1 (positive) by default if they are not specified.
      int subj_polarity_val = 1;
      int obj_polarity_val = 1;
      if (!subj_polarity_json.is_null()) {
        subj_polarity_val = subj_polarity_json.get<int>();
      }

      if (!obj_polarity_json.is_null()) {
        obj_polarity_val = obj_polarity_json.get<int>();
      }

      auto subj_adjectives_json = evid_context["subj_adjectives"];
      auto obj_adjectives_json = evid_context["obj_adjectives"];

      vector<string> subj_adjectives{"None"};
      vector<string> obj_adjectives{"None"};

      if(!subj_adjectives_json.is_null()){
        if(subj_adjectives_json.size() > 0){
          subj_adjectives = subj_adjectives_json.get<vector<string>>();
        }
      }

      if(!obj_adjectives_json.is_null()){
        if(obj_adjectives_json.size() > 0){
          obj_adjectives = obj_adjectives_json.get<vector<string>>();
        }
      }

      auto causal_fragment =
          CausalFragment({subj_adjectives[0], subj_polarity_val, subj_name},
                         {obj_adjectives[0], obj_polarity_val, obj_name});
      this->add_edge(causal_fragment);
    }
  }

  if (!json_data.contains("conceptIndicators")) {
    // No indicator data provided.
    // TODO: What is the best action here?
    //throw runtime_error("No indicator information provided");
    // Maybe this is acceptable since there is another call: edit-indicators,
    // which is not yet implemented. An analyst can create a CAG structure
    // without any indicators and then later attach indicators one by one.
  } else {
    this->set_observed_state_sequence_from_json_dict(json_data["conceptIndicators"]);
  }
}

            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      create-experiment (private)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

std::tuple<int, int, int>
AnalysisGraph::epoch_to_year_month_date(long epoch) {
    // The HMI uses milliseconds. So they multiply time-stamps by 1000.
    // Before converting them back to year and month, we have to divide
    // by 1000.
    epoch /=  1000;

    // Convert the time-step to year and month.
    // We are converting it according to GMT.
    struct tm *ptm = gmtime(&epoch);
    int year = 1900 + ptm->tm_year;
    int month = 1 + ptm->tm_mon;
    int date = ptm->tm_mday;

    if (date > 1) {
      print("* * * * *   WARNING: Observation timestamp {0}-{1}-{2} does not adhere to the protocol!\n", year, month, date);
    }

    return make_tuple(year, month, date);
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
AnalysisGraph::run_causemos_projection_experiment_from_json_dict(
                                              const nlohmann::json &json_data) {

    ExperimentStatus es(this->id, this->experiment_id);

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

    int skip_steps = 0;

    if (this->observed_state_sequence.empty()) {
      // No training data has been provided
      // "Train" (more like derive) a model based on prior distributions
      cout << "\nNOTE:\n\t\"Training\" a model, without any training observations, using only prior distributions!\n";
      this->train_start_epoch = -1;
      this->pred_start_timestep = 0;
      this->delta_t = 1;
      this->pred_timesteps++;
    } else {
      /*
      this->pred_start_timestep = this->epoch_to_timestep(
          proj_start_epoch, this->train_start_epoch, this->num_modeling_timesteps_per_one_observation_timestep);
      double pred_end_timestep = this->epoch_to_timestep(
          proj_end_epoch, this->train_start_epoch, this->num_modeling_timesteps_per_one_observation_timestep);
      this->delta_t = (pred_end_timestep - this->pred_start_timestep) /
                      (this->pred_timesteps - 1.0);
      */
      tuple<int, int, int> train_start_date =
          this->epoch_to_year_month_date(this->train_start_epoch);
      tuple<int, int, int> pred_start_date =
          this->epoch_to_year_month_date(proj_start_epoch);
      this->pred_start_timestep = delphi::utils::observation_timesteps_between(
          train_start_date, pred_start_date, this->model_data_agg_level);
      this->delta_t = 1;

      tuple<int, int, int> pred_end_date =
          this->epoch_to_year_month_date(proj_end_epoch);
      //int num_pred_months = delphi::utils::observation_timesteps_between(pred_start_date, pred_end_date);
      //cout << "(" << get<0>(pred_end_date) << "-" << get<1>(pred_end_date) << "-" << get<2>(pred_end_date) << "), ";
      //cout << "(" << get<0>(pred_start_date) << "-" << get<1>(pred_start_date) << "-" << get<2>(pred_start_date) << "), ";

      // To help clamping we predict one additional timestep
      this->pred_timesteps++;
      this->pred_start_timestep -= this->delta_t;

      // Prevent predicting for timesteps earlier than training start time
      if (this->pred_start_timestep < 0) {
        cout << "\nNOTE:\n\t\"Predicting\" in the past\n";
        /*
        skip_steps = ceil(abs(this->pred_start_timestep) / this->delta_t);
        this->pred_start_timestep += skip_steps;
        this->pred_timesteps -= skip_steps;
        */
      }

      /*
      if (this->pred_start_timestep > pred_end_timestep) {
        throw BadCausemosInputException(
            "Projection end epoch is before projection start epoch");
      }
       */
    }

    this->extract_projection_constraints(projection_parameters["constraints"], skip_steps);

    this->generate_latent_state_sequences(this->pred_start_timestep);
    this->generate_observed_state_sequences();
    return this->format_projection_result();
}

FormattedProjectionResult AnalysisGraph::format_projection_result() {
  // Access
  // [ vertex_name ][ timestep ][ sample ]
  FormattedProjectionResult result;

  // To facilitate clamping derivative we start predicting one timestep before
  // the requested prediction start timestep. We are removing that timestep when
  // returning the results.
  for (auto [vert_name, vert_id] : this->name_to_vertex) {
    result[vert_name] =
        vector<vector<double>>(this->pred_timesteps - 1, vector<double>(this->res));
    for (int ts = 0; ts < this->pred_timesteps - 1; ts++) {
      for (int samp = 0; samp < this->res; samp++) {
        result[vert_name][ts][samp] =
            this->predicted_observed_state_sequences[samp][ts + 1][vert_id][0];
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
      this->graph[e].set_theta(this->graph[e].kde.resample(
          1, this->rand_num_generator, this->uni_dist, this->norm_dist)[0]);
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

AnalysisGraph AnalysisGraph::from_causemos_json_string(string json_string,
                                                       double belief_score_cutoff,
                                                       double grounding_score_cutoff,
                                                       int kde_kernels
                                                       ) {
  AnalysisGraph G;
  G.n_kde_kernels = kde_kernels;

  auto json_data = nlohmann::json::parse(json_string);
  G.from_causemos_json_dict(json_data, belief_score_cutoff, grounding_score_cutoff);
  return G;
}

AnalysisGraph AnalysisGraph::from_causemos_json_file(string filename,
                                                     double belief_score_cutoff,
                                                     double grounding_score_cutoff,
                                                     int kde_kernels
                                                     ) {
  AnalysisGraph G;
  G.n_kde_kernels = kde_kernels;

  auto json_data = load_json(filename);
  G.from_causemos_json_dict(json_data, belief_score_cutoff, grounding_score_cutoff);
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
    j["status"] = this->trained? "ready" : "training";
    j["relations"] = {};
    j["conceptIndicators"] = {};

    for (auto e : this->edges()) {
        /*
        vector<double> weights;
        if (this->trained) {
            weights = vector<double>(this->graph[e].sampled_thetas.size());
            transform(this->graph[e].sampled_thetas.begin(),
                      this->graph[e].sampled_thetas.end(),
                      weights.begin(),
                      [&](double theta) {
                        return (double)tan(theta);
                      });
        } else {
            weights = vector<double>{0.5};
        }
         */

        json edge_json = {{"source", this->source(e).name},
                          {"target", this->target(e).name},
                          {"weights", this->trained
                          ? vector<double>{((this->graph[e].is_frozen()
                                             ? this->graph[e].get_theta()
                                             : mean(this->graph[e].sampled_thetas))
                                                         + M_PI_2) * M_2_PI - 1
                                          }
                          : this->graph[e].is_frozen()
                                ? vector<double>{(this->graph[e].get_theta()
                                                         + M_PI_2) * M_2_PI - 1}
                                : vector<double>{0.5}}};

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
AnalysisGraph::run_causemos_projection_experiment_from_json_string(
                                                           string json_string) {
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
        return run_causemos_projection_experiment_from_json_dict(json_data);
    }
    catch (BadCausemosInputException& e) {
        cout << e.what() << endl;
        // Just a dummy empty prediction to signal that there is an error in
        // projection parameters.
        return FormattedProjectionResult();
    }
}

FormattedProjectionResult
AnalysisGraph::run_causemos_projection_experiment_from_json_file(
                                                              string filename) {
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
        return run_causemos_projection_experiment_from_json_dict(json_data);
    }
    catch (BadCausemosInputException& e) {
        cout << e.what() << endl;
        // Just a dummy empty prediction to signal that there is an error in
        // projection parameters.
        return FormattedProjectionResult();
    }
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                  edit-weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
   *
   * @param source Source concept name
   * @param target Target concept name
   * @param scaled_weight A value in the range [0, 1]. Delphi edge weights are
   *               angles in the range [-π/2, π/2]. Values in the range ]0, π/2[
   *               represents positive polarities and values in the range
   *               ]-π/2, 0[ represents negative polarities.
   * @param polarity Polarity of the edge. Should be either 1 or -1.
   * @return 0 freezing the edge is successful
   *         1 scaled_weight outside accepted range
   *         2 Source concept does not exist
   *         4 Target concept does not exist
   *         8 Edge does not exist
 */
unsigned short AnalysisGraph::freeze_edge_weight(std::string source_name,
                                       std::string target_name,
                                       double scaled_weight,
                                       int polarity) {
    if (scaled_weight < 0 || scaled_weight > 1) {
        return 1;
    }

    int source_id = -1;
    int target_id = -1;

    try {
        source_id = this->name_to_vertex.at(source_name);
    } catch(const out_of_range &e) {
        // Source concept does not exist
        return 2;
    }

    try {
        target_id = this->name_to_vertex.at(target_name);
    } catch(const out_of_range &e) {
        // Target concept does not exist
        return 4;
    }

    pair<EdgeDescriptor, bool> edg = boost::edge(source_id, target_id,
                                                         this->graph);

    if (!edg.second) {
        // There is no edge from source concept to target concept
        return 8;
    }

    double theta = polarity / abs(polarity) * scaled_weight * M_PI_2;

    this->graph[edg.first].set_theta(theta);
    this->graph[edg.first].freeze();

    this->trained = false;

    return 0;
}
