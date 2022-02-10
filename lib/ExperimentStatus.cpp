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

void ExperimentStatus::update_db() {
  json status;
  status[MODEL_ID] = model_id;
  status[EXPERIMENT_ID] = experiment_id;
  status[PROGRESS] = progress;
  status[STATUS] = state;

  write_row(experiment_id, status);
}
