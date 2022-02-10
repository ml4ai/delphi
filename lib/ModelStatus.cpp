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

// Start the training process for a model. Make sure only one
// process does this for a given model ID
bool ModelStatus::start_training() {
  sqlite3_mutex* mx = sqlite3_mutex_alloc(SQLITE_MUTEX_FAST);
  if(mx == nullptr) {
    log_error("Could not create model status, database error");
    return false;
  }
  sqlite3_mutex_enter(mx);

  // if there is no model with this ID in training, create one
  if(!is_training(model_id)) {
    state = "Created";
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

// set our database status with local vars
void ModelStatus::update_db() {
  json status;
  status[COL_ID] = model_id;
  status[PROGRESS] = progress;
  status[STATUS] = state;

  write_row(model_id, status);
}

