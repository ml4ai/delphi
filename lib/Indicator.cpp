#include "libpq-fe.h"
#include <fmt/format.h>
#include <string>
#include <vector>
#include <iostream>
#include "Indicator.hpp"
#include <range/v3/all.hpp>

using namespace std;
namespace rs = ranges;

void Indicator::set_default_unit() {
  const char   *conninfo;
  PGconn       *conn;
  PGresult     *res;
  conninfo = "dbname = delphi";
  int          nFields;

  conn = PQconnectdb(conninfo);

  if (PQstatus(conn) != CONNECTION_OK) {
    throw runtime_error("Could not open db. Do you have the DELPHI_DB "
        "environment correctly set to point to the Delphi database?");
  }
  cout << "Connection successful!!!!!!!!!!!" << endl; // todo

  vector<string> units;
  string query =
      "select Unit from indicator where `Variable` like '" + name + "'";
  res = PQexec(conn, query.c_str());

  if (PQresultStatus(res) == PGRES_COMMAND_OK) {
      for (int i = 0; i < PQntuples(res); i++)
      {
          string ind_unit =
              string(reinterpret_cast<const char*>(PQgetvalue(res, i, 0)));
          units.push_back(ind_unit);
      }
  }
  PQclear(res);

  if (!units.empty()) {
    rs::sort(units);
    this->unit = units.front();
  }
  else {
    cerr << "Error: Indicator::set_indicator_default_unit()\n"
         << "\tIndicator: No data exists for " << name << endl;
  }
  PQfinish(conn);
}
