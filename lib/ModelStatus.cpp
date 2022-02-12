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

// set our data to the start state
void ModelStatus::init_row() {
  progress = 0.0;

  json data;
  data[MODEL_ID] = model_id;
  data[PROGRESS] = progress;
  data[STATUS] = "Empty";
  data[BUSY] = false;
  write_row(model_id, data);
}
