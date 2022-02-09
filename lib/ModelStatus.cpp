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
  status[MODEL_ID] = model_id;
  status[PROGRESS] = delphi::utils::round_n(get_progress(), 2);
  return status;
}

/* write the current Model status to our table */
void ModelStatus::record_status() {
  set_status(model_id, compose_status());
}
