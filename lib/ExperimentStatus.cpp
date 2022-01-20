#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "ExperimentStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

// see DatabaseHelper.cpp
ExperimentStatus::ExperimentStatus(){

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

ExperimentStatus::~ExperimentStatus(){
  stop_updating_db();
}

// Create a callback function, see DatabaseHelper.cpp
int ExperimentStatus::callback(
  void* NotUsed,
  int argc,
  char** argv,
  char** azColName
) {
  // Return successful
  return 0;
}

/* Start the thread that posts the status to the datbase */
void ExperimentStatus::scheduler()
{
  while(!ag->get_trained()){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      update_db();
    }
  }
}

/* Begin posting training status updates to the database on a regular interval */
void ExperimentStatus::start_updating_db(AnalysisGraph *ag){
  this->ag = ag;
  update_db();
  if(pThread == nullptr) {
    pThread = new thread(&ExperimentStatus::scheduler, this);
  }
}

/* Stop posting training status updates to the database */
void ExperimentStatus::stop_updating_db(){
    if (pThread != nullptr)
    {
        if(pThread->joinable()) {
            pThread->join();
	}
        delete pThread;
    }
    pThread = nullptr;
}

/* write out the status as a string for the database */
json ExperimentStatus::compose_status() {
  json status;
  if (ag != nullptr) {
    string model_id = ag->id;
    if(!model_id.empty()) {
      status[COL_ID] = model_id;
    }
  }
  return status;
}

/* directly set the status */
void ExperimentStatus::set_status(
  string experimentId,
  string statusString
){
  string query = "INSERT OR REPLACE INTO " 
    + TABLE_NAME 
    + " VALUES ('" 
    + experimentId 
    + "', '" 
    + statusString 
    + "');";

  insert(query);
}


/* write the current status to our table */
void ExperimentStatus::update_db() {
  json status = compose_status();
  write_to_db(status);
}

/* write the model training status to the database */
void ExperimentStatus::write_to_db(json status) {
  if(status.empty()) {
    logError("write_to_db with empty json");
    return;
  } 
  string notFound = "";
  string id = status.value(COL_ID, notFound);
  if(id == notFound) {
    logError("write_to_db with no id in status JSON");
    return;
  }

  string statusString = status.dump();
  logInfo("write_to_db: " + statusString);

  string query = "INSERT OR REPLACE INTO " 
    + TABLE_NAME 
    + " VALUES ('" 
    + id 
    + "', '" 
    + statusString 
    + "');";

  insert(query);
}

/* Read the model training status from the database */
string ExperimentStatus::read_from_db(string experimentId) {
  logInfo("read_from_db, experimentId: " + experimentId);

  json matches;
  sqlite3_stmt* stmt = nullptr;
  string query = "SELECT * from " 
    + TABLE_NAME
    + " WHERE "
    + COL_ID
    + " = '"
    + experimentId
    + "'  LIMIT 1;";

  logInfo("read_from_db, query: " + query);

  int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    matches[COL_ID] =
      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    matches[COL_JSON] =
      string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  string statusString = matches.dump();
  logInfo("read_from_db, results: " + statusString);

  return statusString;
}


/* create the table if we need it.  */
void ExperimentStatus::init_db() {
  string query = "CREATE TABLE IF NOT EXISTS " 
    + TABLE_NAME 
    + " (" 
    + COL_ID + " TEXT PRIMARY KEY, " 
    + COL_JSON + " TEXT NOT NULL"
    + ");";

  insert(query);
}


int ExperimentStatus::insert(string query) {
  char* zErrMsg = 0;
  int rc = sqlite3_exec(db, query.c_str(), this->callback, 0, &zErrMsg);

  string text = "Problem inserting query, error code: " + rc;

  if(rc) {
    logError(text);
    logError(query);
  }

  return rc;
}

/* Report a message to stdout */
void ExperimentStatus::logInfo(string text) {
  cout << "ExperimentStatus INFO: " << text << endl;
}

/* Report an error to stderr */
void ExperimentStatus::logError(string text) {
  cerr << "ExperimentStatus ERROR: " << text << endl;
}
