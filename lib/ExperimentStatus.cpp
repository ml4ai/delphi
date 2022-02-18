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
void ExperimentStatus::enter_initial_state() {
  set_state(0.0, "Empty", false);
}

// set our data to begin preliminary data reading
void ExperimentStatus::enter_reading_state() {
  set_state(0.0, "Creating experiment", true);
}

// set our data to begin recording progress
void ExperimentStatus::enter_working_state() {
  set_state(0.0, "In progress", true);
  start_recording_progress();
}

// set our data to begin writing to database
void ExperimentStatus::enter_writing_state() {
  stop_recording_progress();
  set_state(0.99, "Writing to database", true);
}

// set our data to the end state
void ExperimentStatus::enter_finished_state() {
  set_state(1.0, "Complete", false);
}

// set the complete data for the database row
void ExperimentStatus::set_state(double progress, string status, bool busy) {
  this->progress = progress;
  json data;
  data[EXPERIMENT_ID] = experiment_id;
  data[MODEL_ID] = model_id;
  data[PROGRESS] = progress;
  data[STATUS] = status;
  data[BUSY] = busy;
  write_row(experiment_id, data);
}
