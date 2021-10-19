#include "data.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include <sqlite3.h>
#include <chrono>
#include <range/v3/all.hpp>
#include <thread>

using namespace std;

vector<double> get_observations_for(string indicator,
                                    string country,
                                    string state,
                                    string county,
                                    int year,
                                    int month,
                                    string unit,
                                    bool use_heuristic) {
  using fmt::print;
  using namespace fmt::literals;

  // TODO: Repeated code block
  // An exact copy of the method AnalysisGraph::open_delphi_db()
  // defined in database.cpp
  // vvvvvvvvvvvvvvvvvvvvvvvvv
  char* pPath;
  pPath = getenv ("DELPHI_DB");
  if (pPath == NULL) {
    cout << "\n\nERROR: DELPHI_DB environment variable containing the path to delphi.db is not set!\n\n";
    exit(1);
  }

  sqlite3* db = nullptr;
  if (sqlite3_open_v2(getenv("DELPHI_DB"), &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
    cout << "\n\nERROR: delphi.db does not exist at " << pPath << endl;
    cout << sqlite3_errmsg(db) << endl;
    exit(1);
  }
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // TODO: End repeated code block

  vector<double> observations = {};
  int rc;
  sqlite3_stmt* stmt = nullptr;
  string check_q;

  string query =
      "select Unit, Value from indicator where `Variable` like '{}'"_format(
          indicator);

  if (!country.empty()) {
    check_q = "{0} and `Country` is '{1}'"_format(query, country);

    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
      rc = sqlite3_step(stmt);
      if (rc == SQLITE_ROW) {
        query = check_q;
      }
      else {
        print("Could not find data for country {0}. Averaging data over all "
              "countries for given axes (Default Setting)\n",
              country);
      }
      sqlite3_finalize(stmt);
      stmt = nullptr;
    }
  }

  if (!state.empty()) {
    check_q = "{0} and `State` is '{1}'"_format(query, state);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      print("Could not find data for state {0}. Only obtaining data "
            "of the country level (Default Setting)\n",
            state);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  if (!county.empty()) {
    check_q = "{0} and `County` is '{1}'"_format(query, county);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      print("Could not find data for county {0}. Only obtaining data "
            "of the state level (Default Setting)\n",
            county);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  if (!unit.empty()) {
    check_q = "{0} and `Unit` is '{1}'"_format(query, unit);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      sqlite3_finalize(stmt);
      stmt = nullptr;
      print("Could not find data for unit {0}. Using first unit in "
            "alphabetical order (Default Setting)\n",
            unit);

      vector<string> units;

      rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
      while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        string ind_unit =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        units.push_back(ind_unit);
      }
      sqlite3_finalize(stmt);
      stmt = nullptr;
      if (!units.empty()) {
        ranges::sort(units);
        query = "{0} and `Unit` is '{1}'"_format(query, units.front());
      }
      else {
        print("No units found for indicator {0}", indicator);
      }
    }
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  if (!(year == -1)) {
    check_q = "{0} and `Year` is '{1}'"_format(query, year);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      print("Could not find data for year {0}. Aggregating data "
            "over all years (Default Setting)\n",
            year);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  if (month != 0) {
    check_q = "{0} and `Month` is '{1}'"_format(query, month);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      print("Could not find data for month {0}. Aggregating data "
            "over all months (Default Setting)\n",
            month);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  double observation;

  rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    observation = sqlite3_column_double(stmt, 1);
    observations.push_back(observation);
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  if (observations.empty() and use_heuristic) {
    string final_query =
        "{0} and `Year` is '{1}' and `Month` is '0'"_format(query, year);
    sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      observation = sqlite3_column_double(stmt, 1);
      // TODO: This math is only valid if the observation we query is an annual
      // aggregate. For example if it is an yearly sample or an yearly average
      // this is not correct. We need a more intelligent way to handle this
      // situation.
      observation = observation / 12;
      observations.push_back(observation);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  rc = sqlite3_finalize(stmt);
  rc = sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
  return observations;
}
