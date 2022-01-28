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

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

/* write out the status as a string for the database */
json ExperimentStatus::compose_status() {
  json status;
  return status;
}

/* write the current Model status to our table */
#include "BaseStatus.hpp"
void ExperimentStatus::update_db() {
  logInfo("update_db()");
}

