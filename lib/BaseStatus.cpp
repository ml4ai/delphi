#include "BaseStatus.hpp"
#include "DatabaseHelper.hpp"
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <thread>
#include "utils.hpp"
#include <sys/time.h>
#include <chrono>

using namespace std;
using json = nlohmann::json;

BaseStatus::~BaseStatus() {
  // if we created a new Database, delete it
  if(delete_database) {
    delete(database);
  }
}

bool BaseStatus::insert_query(string query) {
  if(database->insert(query)) {
    return true;
  }
  log_error("Could not insert query: " + query);
  return false;
}

// Make a clean start with valid data
void BaseStatus::initialize() {

  // Drop our table if it exists
  string query = "SELECT name FROM sqlite_master WHERE type='table' AND name='"
    + table_name
    + "';";
  vector<string> results = database->read_column_text(query);
  for(string result : results) {
    query = "DROP TABLE " + result + ";";
    if(!insert_query(query)) {
      log_error("Could not initialize " + table_name);
      log_error("Query failed: " + query);
      return;
    }
  }

  // create new table 
  query = "CREATE TABLE "
    + table_name
    + " ("
    + COL_ID
    + " VARCHAR PRIMARY KEY, "
    + COL_DATA
    + " VARCHAR NOT NULL);";
  if(!insert_query(query)) {
    log_error("Could not initialize " + table_name);
    log_error("Query failed: " + query);
    return;
  }

  // add rows
  populate_table();

  log_info("Initialized " + table_name);
}


// add valid data to new table
void BaseStatus::insert_data(string id, json data){
  string query = "INSERT INTO "
    + table_name
    + " VALUES ('"
    + id
    + "', '"
    + data.dump()
    +  "');";
  insert_query(query);
}

// Start the thread that writes the data to the table
void BaseStatus::scheduler() {
  while(recording){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      json data = read_data();
      data[PROGRESS] = delphi::utils::round_n(progress, 2);
      write_data(data);
    }
  }
}

// Begin posting progress updates to the database
void BaseStatus::start_recording_progress(){
  recording = true;
  if(pThread == nullptr) {
    pThread = new thread(&BaseStatus::scheduler, this);
  }
}

// Stop posting progress updates to the database
void BaseStatus::stop_recording_progress(){
  recording = false;
  if (pThread != nullptr) {
    if(pThread->joinable()) {
      pThread->join();
    }
    delete pThread;
  }
  pThread = nullptr;
}

// return the data JSON for the primary key
json BaseStatus::read_data() {

  // get the entire database row 
  json row = database -> select_row(
    table_name,
    primary_key,
    COL_DATA
  );

  // If a row is returned by the query, return whatever data it has
  if(!row.empty() && row.contains(COL_DATA)) {
    string dataString = row.value(COL_DATA, "");
    json data = json::parse(dataString);

    if(data.contains(BUSY) 
      && data.contains(PROGRESS)
      && data.contains(STATUS)
    ) {
      return data;
    }
  }

  // otherwise report error and do not populate JSON
  log_error("read_data failed");

  json ret;
  return ret;
}

// write data at the primary key row, return true if successful.
bool BaseStatus::write_data(json data) {
  string dataString = data.dump();
  log_info("write_data, " + dataString);

  string query = "INSERT OR REPLACE INTO "
    + table_name
    + " VALUES ('"
    + primary_key
    + "', '"
    + dataString
    +  "');";

  return insert_query(query);
}
