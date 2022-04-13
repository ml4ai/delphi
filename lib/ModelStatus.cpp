#include "ModelStatus.hpp"
#include <nlohmann/json.hpp>

/*
sqlite> pragma table_info (delphimodel);
0|id|VARCHAR|1||1
1|model|VARCHAR|0||0
2|progress|VARCHAR|0||0
*/

// Check that the database table contains our row
void ModelStatus::enter_initial_state() {
  logger.info("enter_initial_state()");
  set_state(0.0, "Empty", false);
}

// set our data to begin preliminary data reading
void ModelStatus::enter_reading_state() {
  logger.info("enter_reading_state()");
  set_state(0.0, "Creating model", true);
}

// set our data to begin recording progress
void ModelStatus::enter_working_state() {
  logger.info("enter_working_state()");
  set_state(0.0, "Training", true);
  start_recording_progress();
}

// set our data to begin writing to database
void ModelStatus::enter_writing_state() {
  logger.info("enter_writing_state()");
  stop_recording_progress();
  set_state(0.99, "Writing to database", true);
}

// set our data to the end state
void ModelStatus::enter_finished_state(string status) {
  logger.info("enter_finished_state("
    + status
    + ")");
  set_state(1.0, status, false);
}
