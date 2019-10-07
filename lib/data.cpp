#include "data.hpp"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <fmt/format.h>
#include <sqlite3.h>

using namespace std;

vector<double> get_data_value(string indicator,
                              string country,
                              string state,
                              string county,
                              int year,
                              int month,
                              string unit,
                              bool use_heuristic) {
  using fmt::print;
  using namespace fmt::literals;
  using spdlog::debug, spdlog::error, spdlog::info;

  sqlite3* db;

  vector<double> vals = {};
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc) {
    throw("Could not open db. Do you have the DELPHI_DB "
          "environment correctly set to point to the Delphi database?");
  }

  sqlite3_stmt* stmt;

  string query =
      "select Unit, Value from indicator where `Variable` like '{}'"_format(
          indicator);

  string check_q;

  if (!country.empty()) {
    check_q = "{0} and `Country` is '{1}'"_format(query, country);
    sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for country {}. Averaging data over all "
            "countries for given axes (Default Setting)\n",
            country);
    }
  }
  else {
    query = "{} and `Country` is 'None'"_format(query);
  }

  if (!state.empty()) {
    check_q = "{0} and `State` is '{1}'"_format(query, state);
    sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for state {}. Only obtaining data "
            "of the country level (Default Setting)\n",
            state);
    }
  }

  if (!county.empty()) {
    check_q = "{0} and `County` is '{1}'"_format(query, county);
    sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for county {}. Only obtaining data "
            "of the state level (Default Setting)\n",
            county);
    }
  }

  if (!unit.empty()) {
    check_q = "{0} and `Unit` is '{1}'"_format(query, unit);
    sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for unit {}. Using first unit in "
            "alphabetical order (Default Setting)\n",
            unit);

      vector<string> units;

      sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
      while (sqlite3_step(stmt) == SQLITE_ROW) {
        string ind_unit =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        units.push_back(ind_unit);
      }
      if (!units.empty()) {
        sort(units.begin(), units.end());
        query = "{0} and `Unit` is '{1}'"_format(query, units.front());
      }
      else {
        error("No units found for indicator {}", indicator);
      }
    }
  }

  string final_query =
      "{0} and `Year` is '{1}' and `Month` is '{2}'"_format(query, year, month);

  sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

  double value;

  while (sqlite3_step(stmt) == SQLITE_ROW) {
    value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }

  if (vals.empty() and use_heuristic) {
    final_query =
        "{0} and `Year` is '{1}' and `Month` is '0'"_format(query, year);
    sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
      value = sqlite3_column_double(stmt, 1);
      value = value / 12;
      vals.push_back(value);
    }
  }

  if ((rc = sqlite3_finalize(stmt)) == SQLITE_OK) {
    sqlite3_close(db);
  }
  return vals;
}
