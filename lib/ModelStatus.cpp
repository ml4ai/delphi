#include "ModelStatus.hpp"
#include <nlohmann/json.hpp>

// set our data to the start state
void ModelStatus::enter_initial_state() {
  set_state(0.0, "Empty", false);
}

// set our data to begin preliminary data reading
void ModelStatus::enter_reading_state() {
  set_state(0.0, "Creating model", true);
}

// set our data to begin recording progress
void ModelStatus::enter_working_state() {
  set_state(0.0, "Training", true);
  start_recording_progress();
}

// set our data to begin writing to database
void ModelStatus::enter_writing_state() {
  stop_recording_progress();
  set_state(0.99, "Writing to database", true);
}

// set our data to the end state
void ModelStatus::enter_finished_state(string status) {
  set_state(1.0, status, false);
}

// set the complete data for the database row
void ModelStatus::set_state(double progress, string status, bool busy) {
  set_progress(progress);
  nlohmann::json data;
  data[MODEL_ID] = model_id;
  data[PROGRESS] = progress;
  data[STATUS] = status;
  data[BUSY] = busy;
  write_data(data);
}

vector<json> ModelStatus::get_valid_rows() {
  log_info("get_valid_rows()");
  vector<json> rows;

  // get all the IDs from the model table
  string query = "SELECT id FROM " + MODEL_TABLE + ";";
  vector<string> ids = read_column_text(query);

  for(string id : ids) {
    log_info("id: " + id);
  }
  

  return rows;
}


