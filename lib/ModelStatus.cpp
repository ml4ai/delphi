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
      status[MODEL_ID] = model_id;
      status[PROGRESS] =
	delphi::utils::round_n(ag->get_training_progress(), 2);
      status[TRAINED] = ag->get_trained();
//      status["stopped"] = ag->get_stopped(); 
//      status["log_likelihood"] = ag->get_log_likelihood();
//      status["log_likelihood_previous"] = ag->get_previous_log_likelihood();
//      status["log_likelihood_map"] = ag->get_log_likelihood_MAP();
    }
  }
  return status;
}

/* Compose the status for a new, untrained model */
void ModelStatus::set_initial_status(string modelId) {
  json status;
  status[MODEL_ID] = modelId;
  status[PROGRESS] = 0.0;
  set_status(modelId, status);
}

/* write the current Model status to our table */
void ModelStatus::update_db() {
  string model_id = ag->id;
  set_status(model_id, compose_status());
}

