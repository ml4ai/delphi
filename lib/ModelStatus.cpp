#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include "AnalysisGraph.hpp"
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

// Start the training process for a model. 
bool ModelStatus::start_training() {

  // Enter critical section. 
  sqlite3_mutex* mx = sqlite3_mutex_alloc(SQLITE_MUTEX_FAST);
  if(mx == nullptr) {
    log_error("Database error, cannot train model");
    return false;
  }
  sqlite3_mutex_enter(mx);

  // if there is no model with this ID in training, create one
  if(!is_training(model_id)) {
    state = "initiating";
    update_db();
    sqlite3_mutex_leave(mx);
    sqlite3_mutex_free(mx);
    return true;
  }

  // otherwise you will have to wait until it finishes
  sqlite3_mutex_leave(mx);
  sqlite3_mutex_free(mx);
  return false;
}

void ModelStatus::update_db() {
  json status;
  status[COL_ID] = model_id;
  status[PROGRESS] = progress;
  status[STATUS] = state;

  write_row(model_id, status);
}
