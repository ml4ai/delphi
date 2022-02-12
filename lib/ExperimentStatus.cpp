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
