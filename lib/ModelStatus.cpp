#include "ModelStatus.hpp"
#include <nlohmann/json.hpp>

/*
sqlite> pragma table_info (delphimodel);
0|id|VARCHAR|1||1
1|model|VARCHAR|0||0
2|progress|VARCHAR|0||0
*/

// clean up database table of incomplete rows.  
// For each row:
//
// low-latency test:
// If we have a progress column {
//   If it shows incomplete training {
//     delete the row.
//   }
// }
// longer test if the low-latency test fails:
// else {
//   read the model 
//   get the value of the "trained" JSON field
//   If the value is "trained" {
//     add a completed progress column to the row, so the low-latency
//     test passes next time.
//   }
//   else {
//     delete the row
//   }
// }
void ModelStatus::initialize() {
  logger.info("initialize()");

  // find all the model IDs in table
  string id_query = "SELECT "
    + COL_ID
    + " FROM "
    + table_name
    + ";";

  logger.info("id_query: " + id_query);
  vector<string> id_vec = database->read_column_text(id_query);

  for(string id: id_vec) {
    if(validate_by_data(id)) {
      logger.info("validate_by_data(" + id + ") passed");
    } else {
      logger.info("validate_by_data(" + id + ") failed, validating by model");
      if(validate_by_model(id)) {
        logger.info("validate_by_model(" + id + ") passed");
	ModelStatus ms(id);
	ms.enter_finished_state("Trained");
      } else {
        logger.info("validate_by_model(" + id + ") failed");
	delete_row(id);
      }
    }
  }
}

// test for empty JSON, non-existent progress field, and progress != 0
bool ModelStatus::validate_by_data(string id) {
  logger.info("ModelStatus::validate_by_data(" + id + ")");
  json data = read_data(id);
  logger.info("data read: " + data.dump());
  if(data.contains(PROGRESS)) {
    return ((double)data[PROGRESS] == 1.0);
  }
  return false;
}

bool ModelStatus::validate_by_model(string id) {
  logger.info("ModelStatus::validate_by_model(" + id + ")");
  vector<string> model_strings =
     database-> read_column_text_query_where (table_name, COL_MODEL, COL_ID, id);
  for(string model_string : model_strings) {
    if(!model_string.empty()) {
      json model = json::parse(model_string);
      return ((bool)model.value("trained",false));
    }
  }
  return false;
}

void ModelStatus::delete_row(string id) {
  logger.info("ModelStatus::delete_row(" + id + ")");
  string delete_query = "DELETE FROM "
    + table_name
    + " WHERE "
    + COL_ID
    + " = '"
    + id
    + "';";
  bool deleted = database->insert(delete_query);
  if(deleted) {
    logger.info(table_name + "row with id '" + id + "' deleted");
  } else {
    logger.error(table_name + "row with id '" + id + "' could not be deleted");
  }
}

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
