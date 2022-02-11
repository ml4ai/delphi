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

void ExperimentStatus::update_db() {
  json data;
  data[MODEL_ID] = model_id;
  data[EXPERIMENT_ID] = experiment_id;
  data[PROGRESS] = progress;
  data[BUSY] = busy;
  data[STATUS] = status;

  write_row(experiment_id, data);
}
