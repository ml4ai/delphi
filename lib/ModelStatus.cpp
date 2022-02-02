#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "BaseStatus.hpp"
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;


/* write out the status as a string for the database */
json ModelStatus::compose_status() {
  json status;
  if (ag != nullptr) {
    string model_id = ag->id;
    if(!model_id.empty()) {
      status[COL_ID] = model_id;
      status[STATUS_PROGRESS] =
	delphi::utils::round_n(ag->get_training_progress(), 2);
      status[STATUS_TRAINED] = ag->get_trained();
//      status["stopped"] = ag->get_stopped(); 
//      status["log_likelihood"] = ag->get_log_likelihood();
//      status["log_likelihood_previous"] = ag->get_previous_log_likelihood();
//      status["log_likelihood_map"] = ag->get_log_likelihood_MAP();
    }
  }
  return status;
}

/* write the current Model status to our table */
void ModelStatus::update_db() {
  string model_id = ag->id;
  set_status(model_id, compose_status());
}

/* Return the training progress for this model */
json ModelStatus::get_training_progress_response(string modelId) {
  logInfo("get_training_progress_response");
  json result = get_status(modelId);
  json ret;
  ret[COL_ID] = modelId;

  if(result.empty()) {
    ret[COL_STATUS] = "Invalid model ID";
  } else {
    ret[STATUS_PROGRESS] = result[STATUS_PROGRESS];
  }

  logInfo("  " + ret.dump());
  return ret;
}

