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

// set our database data with local vars
void ExperimentStatus::update_db() {
  json data;
  data[MODEL_ID] = model_id;
  data[EXPERIMENT_ID] = experiment_id;
  data[PROGRESS] = progress;
  data[STATUS] = status;

  write_row(experiment_id, data);
}

// set our database data to the start stte
void ExperimentStatus::init_row() {
  json data;
  data[MODEL_ID] = model_id;
  data[EXPERIMENT_ID] = experiment_id;
  data[PROGRESS] = progress;
  data[STATUS] = "empty";
  data[BUSY] = false;

  write_row(experiment_id, data);
}

// set our local vars with database data
void ExperimentStatus::sync_to_db() {
  json data = get_data();
  if(!data.empty()) {
    model_id = data[MODEL_ID];
    progress = data[PROGRESS];
    status = data[STATUS];
  }
}
