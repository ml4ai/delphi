#pragma once

double get_data_value(std::string indicator,
                      std::string country = "",
                      std::string state = "",
                      int year = 2012,
                      int month = 1,
                      std::string unit = "") {
  sqlite3 *db;
  int rc = sqlite3_open(std::getenv("DELPHI_DB"), &db);
  if (rc) {
    fmt::print("Could not open db");
    return 0.0;
  }
  sqlite3_stmt *stmt;

  std::string query =
      "select Unit,Value from indicator where `Variable` like '" + indicator +
      "'";
  std::string check_q;
  if (!country.empty()) {
    check_q = query + " and `Country` is '" + country + "'";
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
      sqlite3_finalize(stmt);
    }
    else {
      fmt::print("Could not find data for country {}. Averaging data over all "
                 "countries for given axes (Default Setting)\n",
                 country);
      sqlite3_finalize(stmt);
    }
  }
  if (!state.empty()) {
    check_q = query + " and `State` is '" + state + "'";
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
      sqlite3_finalize(stmt);
    }
    else {
      fmt::print("Could not find data for state {}. Averaging data over all "
                 "states for given axes (Default Setting)\n",
                 state);
      sqlite3_finalize(stmt);
    }
  }
  if (!unit.empty()) {
    check_q = query + " and `Unit` is '" + unit + "'";
    rc = sqlite3_prepare_v2(db, check_q.c_str(), -1, &stmt, NULL);
    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      query = check_q;
      sqlite3_finalize(stmt);
    }
    else {
      fmt::print("Could not find data for unit {}. Using first unit in "
                 "alphabetical order (Default Setting)\n",
                 unit);
      sqlite3_finalize(stmt);
      std::vector<std::string> units;
      rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
      while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        std::string ind_unit = std::string(
            reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
        units.push_back(ind_unit);
      }
      if (!units.empty()) {
        std::sort(units.begin(), units.end());
        query = query + " and `Unit` is '" + units.front() + "'";
        sqlite3_finalize(stmt);
      }
      else {
        std::cerr << "Error: data::get_data_value()\n"
                  << "\tIndicator: No data exists for " << indicator
                  << std::endl;

        // Should define an exception to throw in this situation
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 0.0;
      }
    }
  }
  else {
    std::vector<std::string> units;
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      std::string ind_unit = std::string(
          reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
      units.push_back(ind_unit);
    }
    if (!units.empty()) {
      std::sort(units.begin(), units.end());
      query = query + " and `Unit` is '" + units.front() + "'";
      sqlite3_finalize(stmt);
    }
    else {
      std::cerr << "Error: data::get_data_value()\n"
                << "\tIndicator: No data exists for " << indicator << std::endl;

      // Should define an exception to throw in this situation
      sqlite3_finalize(stmt);
      sqlite3_close(db);
      return 0.0;
    }
  }

  std::string final_query = query + " and `Year` is '" + std::to_string(year) +
                            "' and `Month` is '" + std::to_string(month) + "'";

  double sum;
  rc = sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);
  std::vector<double> vals;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    double value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }
  if (!vals.empty()) {
    sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return sum / vals.size();
  }
  else {
    fmt::print("No data found for {0},{1}. Averaging over all months for {1} "
               "(Default Setting)\n",
               month,
               year);
    sqlite3_finalize(stmt);
  }

  final_query = query + " and `Year` is '" + std::to_string(year) + "'";
  rc = sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    double value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }
  if (!vals.empty()) {
    sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return sum / vals.size();
  }
  else {
    fmt::print(
        "No data found for {}. Averaging over all years (Default Setting)\n",
        year);
    sqlite3_finalize(stmt);
  }

  final_query = query;
  rc = sqlite3_prepare_v2(db, final_query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    double value = sqlite3_column_double(stmt, 1);
    vals.push_back(value);
  }
  if (!vals.empty()) {
    sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return sum / vals.size();
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  std::cerr << "Error: data::get_data_value()\n"
            << "\tIndicator: No data exists for " << indicator << std::endl;
  // Should define an exception to throw in this situation
  return 0.0;
}
