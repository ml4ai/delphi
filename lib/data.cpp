#include "data.hpp"
#include "utils.hpp"
#include <fmt/format.h>
#include "libpq-fe.h"
#include <chrono>
#include <range/v3/all.hpp>
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

  const char   *conninfo;
  PGconn       *conn;
  PGresult     *res;
  conninfo = "dbname = delphi";
  int          nFields;

  vector<double> vals = {};

  conn = PQconnectdb(conninfo);
  if (PQstatus(conn) != CONNECTION_OK) {
    throw runtime_error("Could not open db. Do you have the DELPHI_DB "
        "environment correctly set to point to the Delphi database?");
  }
  cout << "Connection successful!!!!!!!!!!!" << endl; // todo

  string query =
      "select Unit, Value from indicator where `Variable` like '{}'"_format(
          indicator);

  string check_q;

  if (!country.empty()) {
    check_q = "{0} and `Country` is '{1}'"_format(query, country);

    res = PQexec(conn, check_q.c_str());
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      query = check_q;
    }
    else {
      print("Could not find data for country {}. Averaging data over all "
            "countries for given axes (Default Setting)\n",
            country);
    }
    PQclear(res);
  }

  if (!state.empty()) {
    check_q = "{0} and `State` is '{1}'"_format(query, state);
    res = PQexec(conn, check_q.c_str());
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      query = check_q;
    }
    else {
      print("Could not find data for state {}. Only obtaining data "
            "of the country level (Default Setting)\n",
            state);
    }
    PQclear(res);
  }

  if (!county.empty()) {
    check_q = "{0} and `County` is '{1}'"_format(query, county);
    res = PQexec(conn, check_q.c_str());
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      query = check_q;
    }
    else {
      print("Could not find data for county {}. Only obtaining data "
            "of the state level (Default Setting)\n",
            county);
    }
    PQclear(res);
  }

  if (!unit.empty()) {
    check_q = "{0} and `Unit` is '{1}'"_format(query, unit);
    res = PQexec(conn, check_q.c_str());
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      query = check_q;
    }
    else {
      PQclear(res);
      print("Could not find data for unit {}. Using first unit in "
            "alphabetical order (Default Setting)\n",
            unit);

      vector<string> units;
      nFields = PQnfields(res);
      res = PQexec(conn, query.c_str());
      if (PQresultStatus(res) == PGRES_COMMAND_OK) {
        for (i = 0; i < PQntuples(res); i++)
        {
          for (j = 0; j < nFields; j++)
          {
            string ind_unit =
                string(reinterpret_cast<const char*>(PQgetvalue(res, i, j)));
            units.push_back(ind_unit);
          }
        }
      }
      PQclear(res);
      if (!units.empty()) {
        ranges::sort(units);
        query = "{0} and `Unit` is '{1}'"_format(query, units.front());
      }
      else {
        print("No units found for indicator {}", indicator);
      }
    }
  }
  PQclear(res);

  if (!(year == -1)) {
    check_q = "{0} and `Year` is '{1}'"_format(query, year);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      print("Could not find data for year {}. Aggregating data "
            "over all years (Default Setting)\n",
            county);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  if (!(month == 0)) {
    check_q = "{0} and `Year` is '{1}'"_format(query, month);
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      query = check_q;
    }
    else {
      print("Could not find data for month {}. Aggregating data "
            "over all months (Default Setting)\n",
            county);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  double value;

  rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  if (vals.empty() and use_heuristic) {
    string final_query =
        "{0} and `Year` is '{1}' and `Month` is '0'"_format(query, year);
    sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      value = sqlite3_column_double(stmt, 1);
      value = value / 12;
      vals.push_back(value);
    }
    sqlite3_finalize(stmt);
    stmt = nullptr;
  }

  rc = sqlite3_finalize(stmt);
  rc = sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
  return vals;
}
