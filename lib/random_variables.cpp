#include "random_variables.hpp"
#include <sqlite3.h>
#include <iostream>
#include <iterator>
#include <random>
#include <fmt/core.h>

double RV::sample() {
  std::vector<double> s;

  std::sample(this->dataset.begin(),
              this->dataset.end(),
              std::back_inserter(s),
              1,
              std::mt19937{std::random_device{}()});

  return s[0];
}
  void Indicator::set_default_unit() {
    sqlite3 *db;
    int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
    if (rc) {
      fmt::print("Could not open db");
      return;
    }
    std::vector<std::string> units;
    sqlite3_stmt *stmt;
    std::string query =
        "select Unit from indicator where `Variable` like '" + name + "'";
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      std::string ind_unit = std::string(
          reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
      units.push_back(ind_unit);
    }
    if (!units.empty()) {
      std::sort(units.begin(), units.end());
      this->unit = units.front();
    }
    else {
      std::cerr << "Error: Indicator::set_indicator_default_unit()\n"
                << "\tIndicator: No data exists for " << name << std::endl;
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
  }
