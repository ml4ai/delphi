#include <sqlite3.h>
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

#define SHOW_LOGS

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

// set our data to the end state
void ModelStatus::finalize(string status) {
  progress = 1.0;
  json data;
  data[MODEL_ID] = model_id;
  data[PROGRESS] = progress;
  data[STATUS] = status;
  data[BUSY] = false;
  write_row(model_id, data);
}

// set our data to the start state
void ModelStatus::initialize() {
  progress = 0.0;
  json data;
  data[MODEL_ID] = model_id;
  data[PROGRESS] = progress;
  data[STATUS] = "Empty";
  data[BUSY] = false;
  write_row(model_id, data);
}

// called only at startup, any rows in the table with incomplete training
// are declared lost and get deleted.
void ModelStatus::prune_row(string id) {

  log_info("pruning row " + id);

  json row = read_row(id);
  string report = row.dump();

  if(row.empty()) {
    log_info(report + " FAIL (missing record, deleting row)");
    delete_row(id);
    return;
  }

  string dataString = row[COL_DATA];
  if(dataString.empty()) {
    log_info(report + " FAIL (missing raw data, deleting row)");
    delete_row(id);
    return;
  }

  json data = json::parse(dataString);
  if(data.empty()) {
    log_info(report + " FAIL (missing data, deleting row)");
    delete_row(id);
    return;
  }

  double row_progress = data.value(PROGRESS,0.0);
  if(row_progress < 1.0) {
    log_info(report + " FAIL (stale progress, deleting row)");
    delete_row(id);
    return;
  }

  bool busy = data.value(BUSY, true);
  if(busy) {
    log_info(report + " FAIL (stale lock, deleting row)");
    delete_row(id);
    return;
  }

  log_info(report + " PASS");
}

