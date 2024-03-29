#include <fmt/format.h>
#include <sqlite3.h>
#include <string>
#include <vector>
#include <iostream>
#include "Indicator.hpp"
#include <range/v3/all.hpp>

using namespace std;
namespace rs = ranges;

void Indicator::set_default_unit() {
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

  vector<string> units;
  sqlite3_stmt* stmt;
  string query =
      "select Unit from indicator where `Variable` like '" + name + "'";
  int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    string ind_unit =
        string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    units.push_back(ind_unit);
  }
  if (!units.empty()) {
    rs::sort(units);
    this->unit = units.front();
  }
  else {
    cerr << "Error: Indicator::set_indicator_default_unit()\n"
         << "\tIndicator: No data exists for " << name << endl;
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
}
