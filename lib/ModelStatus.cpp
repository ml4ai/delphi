#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "ModelStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

// see DatabaseHelper.cpp
ModelStatus::ModelStatus(){

  int rc = sqlite3_open(getenv("DELPHI_DB"), &db);
  if (rc) {
    // Show an error message
    logError("Problem opening database:");
    logError(sqlite3_errmsg(db));

    // Close the connection
    sqlite3_close(db);
    db = nullptr;
    // Return an error
    throw "Could not open Training Status database\n";
  } else {
    init_db();
  }
}

ModelStatus::~ModelStatus(){
  stop_updating_db();
}

// Create a callback function, see DatabaseHelper.cpp
int ModelStatus::callback(
  void* NotUsed,
  int argc,
  char** argv,
  char** azColName
) {
  // Return successful
  return 0;
}

/* Start the thread that posts the status to the datbase */
void ModelStatus::scheduler()
{
  while(!ag->get_trained()){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      update_db();
    }
  }
}

/* Begin posting training status updates to the database on a regular interval */
void ModelStatus::start_updating_db(AnalysisGraph *ag){
  this->ag = ag;
  update_db();
  if(pThread == nullptr) {
    pThread = new thread(&ModelStatus::scheduler, this);
  }
}

/* Stop posting training status updates to the database */
void ModelStatus::stop_updating_db(){
    if (pThread != nullptr)
    {
        if(pThread->joinable()) {
            pThread->join();
	}
        delete pThread;
    }
    pThread = nullptr;
}

/* set the "stopped" field to true */
string ModelStatus::stop_training(string modelId){
  json status;
  status[COL_ID] = modelId;
  status[COL_JSON] = "Endpoint 'training-stop' not yet implemented.";
  return status.dump();
}

/* write out the status as a string for the database */
json ModelStatus::compose_status() {
  json status;
  if (ag != nullptr) {
    string model_id = ag->id;
    if(!model_id.empty()) {
      status[COL_ID] = model_id;
      status[STATUS_PROGRESS] =
	delphi::utils::round_n(ag->get_training_progress(), 2);
//      status[STATUS_TRAINED] = ag->get_trained();
//      status["stopped"] = ag->get_stopped(); 
//      status["log_likelihood"] = ag->get_log_likelihood();
//      status["log_likelihood_previous"] = ag->get_previous_log_likelihood();
//      status["log_likelihood_map"] = ag->get_log_likelihood_MAP();
    }
  }
  return status;
}

/* write the current status to our table */
void ModelStatus::update_db() {
  write_to_db(ag->id, compose_status());
}

/* write the model training status to the database */
void ModelStatus::write_to_db(string modelId, json status) {
  if(status.empty()) {
    logError("write_to_db with empty json");
  } else {

    string query = "INSERT OR REPLACE INTO " 
      + TABLE_NAME
      + " VALUES ('"
      + modelId
      + "', '"
      + status.dump() 
      +  "');";

    insert(query);
  }
}

/* Read the model training status from the database */
string ModelStatus::read_from_db(string modelId) {
  logInfo("read_from_db, model_id:");
  logInfo(modelId);

  json matches;
  sqlite3_stmt* stmt = nullptr;
  string query =
    "SELECT * from " + TABLE_NAME + " WHERE " + COL_ID + "='"+ modelId+ "'  LIMIT 1;";
  int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    matches[COL_ID] =
      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    matches[COL_JSON] =
      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  logInfo("read_from_db, results:");
  string dump = matches.dump();
  logInfo(dump);

  return dump;
}


/* create the table if we need it.  */
void ModelStatus::init_db() {
  string query = "CREATE TABLE IF NOT EXISTS " 
    + TABLE_NAME 
    + " (" + COL_ID + " TEXT PRIMARY KEY, " + COL_JSON + " TEXT NOT NULL);";

  insert(query);
}


int ModelStatus::insert(string query) {
  char* zErrMsg = 0;
  int rc = sqlite3_exec(db, query.c_str(), this->callback, 0, &zErrMsg);

  if(rc) {
    char buf[256];
    sprintf(buf, "Problem inserting query, error code: %d", rc);
    logError(buf);
    logError(query);
  }

  return rc;
}

/* Report a message to stdout */
void ModelStatus::logInfo(string text) {
  cout << "ModelStatus INFO: " << text << endl;
}

/* Report an error to stderr */
void ModelStatus::logError(string text) {
  cerr << "ModelStatus ERROR: " << text << endl;
}
