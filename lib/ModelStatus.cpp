#include <sqlite3.h>
#include "DatabaseHelper.hpp"
#include "ModelStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
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

ModelStatus::ModelStatus(string id) : BaseStatus(
      new Database(),
      "model_status",
      "ModelStatus"
    ), model_id(id) {
	sync_to_db();
}

ModelStatus::ModelStatus(string id, Database* database) : BaseStatus(
      database,
      "model_status",
      "ModelStatus"
    ), model_id(id) {
	sync_to_db();
}

// set our database data with local vars
void ModelStatus::update_db() {
  json data = get_data();
  data[PROGRESS] = delphi::utils::round_n(progress, 2);
  write_row(model_id, data);
}

// set our database data to the start stte
void ModelStatus::init_row() {
  json data;
  data[MODEL_ID] = model_id;
  data[PROGRESS] = progress;
  data[STATUS] = "empty";
  data[BUSY] = false;

  write_row(model_id, data);
}

// set our local vars with database data 
void ModelStatus::sync_to_db() {
  json data = get_data();
  if(!data.empty()) {
    progress = data[PROGRESS];
    status = data[STATUS];
    busy = data[BUSY];
  }
}
