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


// set our database status with local vars
void ModelStatus::update_db() {
  json status;
  status[COL_ID] = model_id;
  status[PROGRESS] = delphi::utils::round_n(progress, 2);
  status[STATUS] = status;
  write_row(model_id, status);
}
