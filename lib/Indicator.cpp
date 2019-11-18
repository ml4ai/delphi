#include <fmt/core.h>
#include <sqlite3.h>
#include <string>
#include <vector>
#include <iostream>
#include "Indicator.hpp"

using namespace std;

void Indicator::set_default_unit() {
  sqlite3* db;
  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc) {
    fmt::print("Could not open db");
    return;
  }
  vector<string> units;
  sqlite3_stmt* stmt;
  string query =
      "select Unit from indicator where `Variable` like '" + name + "'";
  rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string ind_unit =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    units.push_back(ind_unit);
  }
  if (!units.empty()) {
    sort(units.begin(), units.end());
    this->unit = units.front();
  }
  else {
    cerr << "Error: Indicator::set_indicator_default_unit()\n"
         << "\tIndicator: No data exists for " << name << endl;
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
}
