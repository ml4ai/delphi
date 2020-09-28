#include "AnalysisGraph.hpp"
#include "data.hpp"
#include <sqlite3.h>
#include <range/v3/all.hpp>

using namespace std;
using namespace delphi::utils;
using fmt::print;
using namespace fmt::literals;



/*
 ============================================================================
 Public: Indicator Manipulation
 ============================================================================
*/

void AnalysisGraph::set_indicator(string concept,
                                  string indicator,
                                  string source) {
  if (in(this->indicators_in_CAG, indicator)) {
    print("{0} already exists in Causal Analysis Graph, Indicator {0} was "
          "not added to Concept {1}.",
          indicator,
          concept);
    return;
  }
  (*this)[concept].add_indicator(indicator, source);
  this->indicators_in_CAG.insert(indicator);
}

void AnalysisGraph::delete_indicator(string concept, string indicator) {
  (*this)[concept].delete_indicator(indicator);
  this->indicators_in_CAG.erase(indicator);
}

void AnalysisGraph::delete_all_indicators(string concept) {
  (*this)[concept].clear_indicators();
}

void AnalysisGraph::map_concepts_to_indicators(int n_indicators,
                                               string country) {
  sqlite3* db = nullptr;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc != SQLITE_OK) {
    throw runtime_error(
        "Could not open db. Do you have the DELPHI_DB "
        "environment correctly set to point to the Delphi database?");
  }

  sqlite3_stmt* stmt = nullptr;
  string query_base = "select Indicator from concept_to_indicator_mapping ";
  string query;

  // Check if there are any data values for an indicator for this country.
  auto has_data = [&](string indicator) {
    query =
        "select `Value` from indicator where `Variable` like '{0}' and `Country` like '{1}'"_format(
            indicator, country);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return rc;
  };

  auto get_indicator_source = [&](string indicator) {
    query =
        "select `Source` from indicator where `Variable` like '{0}' and `Country` like '{1}' limit 1"_format(
            indicator, country);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    string source =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_finalize(stmt);
    stmt = nullptr;
    return source;
  };

  for (Node& node : this->nodes()) {
    node.clear_indicators(); // Clear pre-existing attached indicators

    query = "{0} where `Concept` like '{1}' order by `Score` desc"_format(
        query_base, node.name);
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);

    vector<string> matches = {};
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      matches.push_back(
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))));
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;

    string ind_name, ind_source;

    for (int i = 0; i < n_indicators; i++) {
      bool at_least_one_indicator_found = false;
      for (string indicator : matches) {
        if (!in(this->indicators_in_CAG, indicator) and has_data(indicator)) {
          node.add_indicator(indicator, get_indicator_source(indicator));
          this->indicators_in_CAG.insert(indicator);
          at_least_one_indicator_found = true;
          break;
        }
      }
      if (!at_least_one_indicator_found) {
        print("No suitable indicators found for concept '{0}' for country "
              "'{1}', please select "
              "one manually.\n",
              node.name,
              country);
      }
    }
  }
  rc = sqlite3_finalize(stmt);
  rc = sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
}

void AnalysisGraph::initialize_parameters(string country,
                                          string state,
                                          string county,
                                          int year,
                                          int month,
                                          map<string, string> units) {
  int num_verts = this->num_vertices();
  vector<double> mean_sequence;
  vector<double> std_sequence;
  vector<int> ts_sequence;

  for (int v = 0; v < num_verts; v++) {
      Node &n = (*this)[v];

      for (int i = 0; i < n.indicators.size(); i++) {
          Indicator &ind = n.indicators[i];

          // The (v, i) pair uniquely identifies an indicator. It is the i th
          // indicator of v th concept.
          // For each indicator, there could be multiple observations per
          // time step. Aggregate those observations to create a single
          // observation time series for the indicator (v, i).
          // We also ignore the missing values.
          mean_sequence.clear();
          std_sequence.clear();
          ts_sequence.clear();
          for (int ts = 0; ts < this->n_timesteps; ts++) {
              vector<double> &obs_at_ts = this->observed_state_sequence[ts][v][i];

              if (!obs_at_ts.empty()) {
                  // There is an observation for indicator (v, i) at ts.
                  ts_sequence.push_back(ts);
                  mean_sequence.push_back(delphi::utils::mean(obs_at_ts));
                  std_sequence.push_back(delphi::utils::standard_deviation(mean_sequence.back(), obs_at_ts));
              } //else {
                // This is a missing observation
                //}
          }

          // Set the indicator standard deviation
          // NOTE: This is what we have been doing earlier for delphi local
          // run:
          //stdev = 0.1 * abs(indicator.get_mean());
          //stdev = stdev == 0 ? 1 : stdev;
          //indicator.set_stdev(stdev);
          // Surprisingly, I did not see indicator standard deviation being set
          // when delphi is called from CauseMos HMI and in the Indicator class
          // the default standard deviation was set to be 0. This could be an
          // omission. I updated Indicator class so that the default indicator
          // mean is 1.
          double max_std = ranges::max(std_sequence);
          if(!isnan(max_std)) {
              // For indicator (v, i), at least one time step had
              // more than one observation.
              // We can use that to assign the indicator standard deviation.
              // TODO: Is that a good idea?
              ind.set_stdev(max_std);
          }   // else {
              // All the time steps had either 0 or 1 observation.
              // In this case, indicator standard deviation defaults to 1
              // TODO: Is this a good idea?
              //}

          // Set the indicator mean
          string aggregation_method = ind.get_aggregation_method();

          // TODO: Instead of comparing text, it would be better to define an
          // enumerated type say AggMethod and use it. Such an enumerated type
          // needs to be shared between AnalysisGraph and Indicator classes.
          if (aggregation_method.compare("first") != 0) {
              if (ts_sequence[0] == 0) {
                  // The first observation is not missing
                  ind.set_mean(mean_sequence[0]);
              } else {
                  // First observation is missing
                  // TODO: Decide what to do
                  // I feel getting a decaying weighted average of existing
                  // observations with earlier the observation, higher the
                  // weight is a good idea. TODO: If so how to calculate
                  // weights? We need i < j => w_i > w_j and sum of w's = 1.
                  // For the moment as a placeholder, average of the earliest
                  // available observation is used to set the indicator mean.
                  ind.set_mean(mean_sequence[0]);
              }
          }
          else if (aggregation_method.compare("last") != 0) {
              int last_obs_idx = this->n_timesteps - 1;
              if (ts_sequence.back() == last_obs_idx) {
                  // The first observation is not missing
                  ind.set_mean(mean_sequence.back());
              } else {
                  // First observation is missing
                  // TODO: Similar to "first" decide what to do.
                  // For the moment the observation closest to the final
                  // time step is used to set the indicator mean.
                  ind.set_mean(mean_sequence.back());
              }
          }
          else if (aggregation_method.compare("min") != 0) {
              ind.set_mean(ranges::min(mean_sequence));
          }
          else if (aggregation_method.compare("max") != 0) {
              ind.set_mean(ranges::max(mean_sequence));
          }
          else if (aggregation_method.compare("mean") != 0) {
              ind.set_mean(delphi::utils::mean(mean_sequence));
          }
          else if (aggregation_method.compare("median") != 0) {
              ind.set_mean(delphi::utils::median(mean_sequence));
          }
      }
  }
}
