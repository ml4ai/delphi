#include "data.hpp"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <fmt/format.h>
#include <sqlite3.h>
#include <chrono>
#include <thread>

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

  int rc;
  rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc != SQLITE_OK) {
    throw runtime_error("Could not open db. Do you have the DELPHI_DB "
        "environment correctly set to point to the Delphi database?");
  }

  sqlite3_stmt* stmt;

  string query =
      "select Unit, Value from indicator where `Variable` like '{}'"_format(
          indicator);

  string check_q;

  if (!country.empty()) {
    check_q = "{0} and `Country` is '{1}'"_format(query, country);

    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
      rc = sqlite3_step(stmt);
      if (rc == SQLITE_ROW) {
        query = check_q;
      }
      else {
        debug("Could not find data for country {}. Averaging data over all "
              "countries for given axes (Default Setting)\n",
              country);
      }
      sqlite3_reset(stmt);
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
      debug("Could not find data for state {}. Only obtaining data "
            "of the country level (Default Setting)\n",
            state);
    }
    sqlite3_reset(stmt);
  }

  if (!county.empty()) {
    check_q = "{0} and `County` is '{1}'"_format(query, county);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for county {}. Only obtaining data "
            "of the state level (Default Setting)\n",
            county);
    }
    sqlite3_reset(stmt);
  }

  if (!unit.empty()) {
    check_q = "{0} and `Unit` is '{1}'"_format(query, unit);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      sqlite3_reset(stmt);
      debug("Could not find data for unit {}. Using first unit in "
            "alphabetical order (Default Setting)\n",
            unit);

      vector<string> units;

      rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
      while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        string ind_unit =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        units.push_back(ind_unit);
      }
      sqlite3_reset(stmt);
      if (!units.empty()) {
        sort(units.begin(), units.end());
        query = "{0} and `Unit` is '{1}'"_format(query, units.front());
      }
      else {
        error("No units found for indicator {}", indicator);
      }
    }
  }
  sqlite3_reset(stmt);

  if (!(year == -1)) {
    check_q = "{0} and `Year` is '{1}'"_format(query, year);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for year {}. Aggregating data "
            "over all years (Default Setting)\n",
            county);
    }
    sqlite3_reset(stmt);
  }

  if (!(month == 0)) {
    check_q = "{0} and `Year` is '{1}'"_format(query, month);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      debug("Could not find data for month {}. Aggregating data "
            "over all months (Default Setting)\n",
            county);
    }
    sqlite3_reset(stmt);
  }

  double value;

  rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }
  sqlite3_reset(stmt);

  if (vals.empty() and use_heuristic) {
    string final_query =
        "{0} and `Year` is '{1}' and `Month` is '0'"_format(query, year);
    sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      value = sqlite3_column_double(stmt, 1);
      value = value / 12;
      vals.push_back(value);
    }
    sqlite3_reset(stmt);
  }

  rc = sqlite3_finalize(stmt);
  rc = sqlite3_close(db);
  return vals;
}
