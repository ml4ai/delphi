#include <sqlite3.h>
#include "AnalysisGraph.hpp"
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

/* write out the status as a string for the database */
json ExperimentStatus::compose_status() {
  json status;
  status[EXPERIMENT_ID] = experiment_id;
  status[PROGRESS] = progress;
  status[STATUS] = "not yet implemented!";
  return status;
}

/* write the current Model status to our table */
void ExperimentStatus::record_status() {
  logInfo("update_db()");
  set_status(experiment_id, compose_status());
}

