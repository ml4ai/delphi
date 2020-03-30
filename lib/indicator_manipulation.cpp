#include "AnalysisGraph.hpp"
#include "data.hpp"
#include "libpq-fe.h"


using namespace std;
using namespace delphi::utils;
using fmt::print;
using namespace fmt::literals;



/*
 ============================================================================
 Public: Indicator Manipulation
 ============================================================================
*/

static void exit_nicely(PGconn *conn){
  PQfinish(conn);
  exit(1);
}

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
  const char   *conninfo;
  PGconn       *conn;
  PGresult     *res;
  conninfo = "dbname = delphi";
  int          nFields;

  /* Make a connection to the database */
  conn = PQconnectdb(conninfo);
  /* Check to see that the backend connection was successfully made */
  if (PQstatus(conn) != CONNECTION_OK){
    cout << "Connection to database failed: " << PQerrorMessage(conn) << endl;
    exit_nicely(conn);
  }
  cout << "Connection successful!!!!!!!!!!!" << endl; // todo


  string query_base = "select \"Indicator\" from concept_to_indicator_mapping ";
  string query;
  // Check if there are any data values for an indicator for this country.
  auto has_data = [&](string indicator) {
    query =
        "select `Value` from indicator where `Variable` like '{0}' and `Country` like '{1}'"_format(
            indicator, country);

    res = PQexec(conn, query.c_str());
    bool rc = (PQresultStatus(res) == PGRES_COMMAND_OK) ? true : false;
    PQclear(res);
    return rc;
  };

  auto get_indicator_source = [&](string indicator) {
    query =
        "select `Source` from indicator where `Variable` like '{0}' and `Country` like '{1}' limit 1"_format(
            indicator, country);
    res = PQexec(conn, query.c_str());
    string source =
        string(reinterpret_cast<const char*>(PQgetvalue(res, 0, 0)));
    PQclear(res);
    return source;
  };

  for (Node& node : this->nodes()) {
    node.clear_indicators(); // Clear pre-existing attached indicators

    query = "{0} where `Concept` like '{1}' order by `Score` desc"_format(
        query_base, node.name);
    res = PQexec(conn, query.c_str());

    
    nFields = PQnfields(res);
    vector<string> matches = {};
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      for (int i = 0; i < PQntuples(res); i++)
      {
          for (int j = 0; j < nFields; j++)
              matches.push_back(string(reinterpret_cast<const char*>(PQgetvalue(res, i, j)))); // todo
      }
    }
    PQclear(res);

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
              "one manually.",
              node.name,
              country);
      }
    }
  }
  PQfinish(conn);
}

void AnalysisGraph::parameterize(string country,
                                 string state,
                                 string county,
                                 int year,
                                 int month,
                                 map<string, string> units) {
  double stdev, mean;
  for (Node& node : this->nodes()) {
    for (Indicator& indicator : node.indicators) {
      if (in(units, indicator.name)) {
        indicator.set_unit(units[indicator.name]);
      }
      else {
        indicator.set_default_unit();
      }
      vector<double> data = get_data_value(indicator.name,
                                           country,
                                           state,
                                           county,
                                           year,
                                           month,
                                           indicator.unit,
                                           this->data_heuristic);

      mean = data.empty() ? 0 : delphi::utils::mean(data);
      indicator.set_mean(mean);
      stdev = 0.1 * abs(indicator.get_mean());
      stdev = stdev == 0 ? 1 : stdev;
      indicator.set_stdev(stdev);
    }
  }
}
