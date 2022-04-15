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
//   If (it shows incomplete training) {
//     delete the row.
//   }
// }
// longer test:
// else {
//   read the model 
//   get the value of the "trained" JSON field
//   If (the value is "trained") {
//     add a completed progress column to the row
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
      logger.info("validate_by_id(" + id + ") passed");
    } else {
      logger.info("validate_by_id(" + id + ") failed, validating by model");
      if(validate_by_model(id)) {
        logger.info("validate_by_model(" + id + ") passed");
	logger.info("need to update progress to completed!");
      } else {
        logger.info("validate_by_failed(" + id + ") failed");
	logger.info("need to delete row!");
      }
    }
  }
}

// test for empty JSON, non-existent progress field, and progress != 0
bool ModelStatus::validate_by_data(string id) {
  json data = read_data(id);

  logger.info("ModelStatus::validate_by_data(" + id + ")");
  logger.info("data read: " + data.dump());

  double progress = 0.0;

  // FIXME try this on empty json and see if it still works.
  //
  // this would be better:
  // double progress = data.value("progress", 0.0)
  // return (progress == 1.0)
  //
  //
  // or even
  // return ((double)data.value("progress", 0.0) == 1.0)

  if(data.contains(PROGRESS)) {
    progress = data[PROGRESS];
  }

  return (progress == 1.0);
}

bool ModelStatus::validate_by_model(string id) {
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


/*
    // find progress data for each ID
    string data_query = "SELECT "
      + COL_DATA
      + " FROM "
      + MODEL_TABLE
      + " WHERE "
      + COL_ID
      + " = '"
      + id
      + "';";
    logger.info("data_query: " + data_query);
    vector<string> data_vec = database->read_column_text(data_query);
    if(data_vec.empty()) {
      // progress column not found, check model 
      logger.info("Did not find progress for " + id + ", checking model");

      // get the model for this ID
      string model_query = "SELECT "
        + COL_MODEL
        + " FROM "
        + MODEL_TABLE
        + " WHERE "
        + COL_ID
        + " = '"
        + id
        + "';";

      vector<string> model_vec = database->read_column_text(model_query);
      if(model_vec.empty()) {
        logger.error("No model found, deleting row with id " + id);
        delete_row(id);
      } else for (string model_str: model_vec) {
        // parse json for this model
        json model = json::parse(model_str);
        if(model.contains("trained")) {
          bool trained = model["trained"];
          if(trained) {
            // trained model, add complete progress record
            logger.info("Model is trained, adding progress record");
	    ModelStatus foo(id);
	    foo.enter_finished_state("trained");
          } else {
            // model not trained, delete row
            logger.info("Model training is incomplete, deleting");
            delete_row(id);
          }
        } else {
	  logger.info("'training' field not found in model JSON");
        }
      }
    } else for(string data_str: data_vec) {
      logger.info("Found progress for " + id + ": " + data_str);
      json data = json::parse(data_str);
      // if training progress is 1.0, the model row is OK
      double progress = data[PROGRESS];
      if(progress == 1.0) {
        logger.info("Model training is complete, keeping");
      } else {
        logger.info("Model training is incomplete, deleting");
        delete_row(id);
      }
    }
  }
}

*/

void ModelStatus::delete_row(string id) {
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
