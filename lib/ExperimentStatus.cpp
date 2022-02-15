#include <sqlite3.h>
#include "BaseStatus.hpp"
#include "DatabaseHelper.hpp"
#include "ExperimentStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>
#include "BaseStatus.hpp"

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;


// set our data to the start state
void ExperimentStatus::init_row() {
  progress = 0.0;
  json data;
  data[MODEL_ID] = model_id;
  data[EXPERIMENT_ID] = experiment_id;
  data[PROGRESS] = progress;
  data[STATUS] = "empty";
  data[BUSY] = false;
  write_row(experiment_id, data);
}

void ExperimentStatus::finalize(string status) {
  log_info("finalize " + experiment_id);
}


// experiment status rows are not archived
void ExperimentStatus::prune_row(string id) {

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
