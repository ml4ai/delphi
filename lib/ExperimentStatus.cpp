#include "ExperimentStatus.hpp"
#include <nlohmann/json.hpp>

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

/*
sqlite> pragma table_info (causemosasyncexperimentresult);
0|id|VARCHAR|1||1
1|status|VARCHAR|0||0
2|experimentType|VARCHAR|0||0
3|results|TEXT|0||0
4|progress|VARCHAR|0||0
*/

// clean up database table of incomplete rows
void ExperimentStatus::initialize() {
  logger.info("initialize()");
}

// set our data to the start state
void ExperimentStatus::enter_initial_state() {
  logger.info("enter_initial_state()");
  set_state(0.0, "Empty", false);
}

// set our data to begin preliminary data reading
void ExperimentStatus::enter_reading_state() {
  logger.info("enter_reading_state()");
  set_state(0.0, "Creating experiment", true);
}

// set our data to begin recording progress
void ExperimentStatus::enter_working_state() {
  logger.info("enter_working_state()");
  set_state(0.0, "In progress", true);
  start_recording_progress();
}

// set our data to begin writing to database
void ExperimentStatus::enter_writing_state() {
  logger.info("enter_writing_state()");
  stop_recording_progress();
  set_state(0.99, "Writing to database", true);
}

// set our data to the end state
void ExperimentStatus::enter_finished_state(string status) {
  logger.info("enter_finished_state("
    + status
    + ")");
  set_state(1.0, status, false);
}
