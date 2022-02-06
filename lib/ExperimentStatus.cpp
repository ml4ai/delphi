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

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

/* write out the status as a string for the database */
json ExperimentStatus::compose_status() {
  json status;
  if (ag != nullptr) {
    string experiment_id = ag->get_experiment_id();
    if(!experiment_id.empty()) {
      status[EXPERIMENT_ID] = experiment_id;
      status[PROGRESS] =
        delphi::utils::round_n(ag->get_experiment_progress(), 2);
    }
  }
  return status;
}

/* write the current Model status to our table */
#include "BaseStatus.hpp"
void ExperimentStatus::update_db() {
  logInfo("update_db()");
  string experiment_id = ag->get_experiment_id();
  set_status(experiment_id, compose_status());
}

/* Return the training progress for this model */
json ExperimentStatus::get_experiment_progress(string experiment_id) {

  json result = get_status(experiment_id);
  json ret;

  if(result.empty()) {
    ret[EXPERIMENT_ID] = experiment_id;
    ret[STATUS] = "Invalid experiment ID";
  } else {
    string statusString = result[STATUS];
    json status = json::parse(statusString);
    ret[PROGRESS] = status[PROGRESS];
  }

  return ret;
}
