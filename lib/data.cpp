#include "data.hpp"
#include <sqlite3.h>


std::vector<double> get_data_value(std::string indicator,
                      std::string country,
                      std::string state,
                      std::string county,
                      int year,
                      int month,
                      std::string unit,
                      bool use_heuristic) {
  using namespace std;
  using fmt::print;
  using namespace fmt::literals;
  using spdlog::debug, spdlog::error, spdlog::info;

  sqlite3* db;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);

  if (rc) {
    error("Could not open db");
    return vector<double>();
  }

  sqlite3_stmt* stmt;

  string query =
      "select Unit, Value from indicator where `Variable` like '{}'"_format(
          indicator);

  string check_q;

  if (!country.empty()) {
    check_q = "{0} and `Country` is '{1}'"_format(query, country);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      query = "{} and `Country` is 'None'"_format(query);
      debug("Could not find data for country {}. Averaging data over all "
            "countries for given axes (Default Setting)\n",
            country);
    }
    sqlite3_finalize(stmt);
  }
  else {
    query = "{} and `Country` is 'None'"_format(query);
  }

  if (!state.empty()) {
    check_q = "{0} and `State` is '{1}'"_format(query, state);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      query = "{} and `State` is 'None'"_format(query);
      debug("Could not find data for state {}. Only obtaining data "
            "of the country level (Default Setting)\n",
            state);
    }
    sqlite3_finalize(stmt);
  }
  else {
    query = "{} and `State` is 'None'"_format(query);
  }

  if (!county.empty()) {
    check_q = "{0} and `County` is '{1}'"_format(query, county);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
    }
    else {
      query = "{} and `County` is 'None'"_format(query);
      debug("Could not find data for county {}. Only obtaining data "
            "of the state level (Default Setting)\n",
            county);
    }
    sqlite3_finalize(stmt);
  }
  else {
    query = "{} and `County` is 'None'"_format(query);
  }

  if (!unit.empty()) {
    check_q = "{0} and `Unit` is '{1}'"_format(query, unit);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
      sqlite3_finalize(stmt);
    }
    else {
      debug("Could not find data for unit {}. Using first unit in "
            "alphabetical order (Default Setting)\n",
            unit);
      sqlite3_finalize(stmt);

      vector<string> units;

      rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
      while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        string ind_unit =
            string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
        units.push_back(ind_unit);
      }
      if (!units.empty()) {
        sort(units.begin(), units.end());
        query = query + " and `Unit` is '" + units.front() + "'";
        sqlite3_finalize(stmt);
      }
      else {
        error("data::get_data_value()\n"
              "\tIndicator: No data exists for indicator {}",
              indicator);

        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return vector<double>();
      }
    }
  }
  else {
    vector<string> units;
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      string ind_unit =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
      units.push_back(ind_unit);
    }
    if (!units.empty()) {
      sort(units.begin(), units.end());
      query = "{0} and `Unit` is '{1}'"_format(query, units.front());
      sqlite3_finalize(stmt);
    }
    else {
      error("data::get_data_value()\n"
            "\tIndicator: No data exists for {}",
            indicator);

      sqlite3_finalize(stmt);
      sqlite3_close(db);
      return vector<double>();
    }
  }

  string final_query =
      "{0} and `Year` is '{1}' and `Month` is '{2}'"_format(query, year, month);

  rc = sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

  double value;
  vector<double> vals;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }

  if (!vals.empty()) {
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return vals;
  }

  sqlite3_finalize(stmt);
  if (!use_heuristic) {
    sqlite3_close(db);
    return vector<double>();
  }

  final_query = "{0} and `Year` is '{1}' and `Month` is '0'"_format(query, year);
  rc = sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    value = sqlite3_column_double(stmt, 1);
    vals.push_back(value/12);
  }

  if (!vals.empty()) {
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return vals;
  }

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return vector<double>();
}
