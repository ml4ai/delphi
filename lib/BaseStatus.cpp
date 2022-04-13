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
  logger.error("Could not insert query: " + query);
  return false;
}

// Make a clean start with valid data
void BaseStatus::initialize() {
  logger.info("BaseStatus::initialize()");
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
  string label = "BaseStatus read_data() ";
  logger.info(label);

  // get the progress column from the database row
  string query = "SELECT "
    + COL_DATA
    + " FROM " 
    + table_name 
    + " WHERE "
    + COL_ID
    + " = '"
    + primary_key
    + "';";

  logger.info("BaseStatus query: " + query);

  vector<string> results = database->read_column_text(query);

  json data;

  for(string result: results) {
    data = json::parse(result);
  }

  logger.info("BaseStatus read: " + data.dump());

  return data;
}

// write data at the primary key row, return true if successful.
bool BaseStatus::write_data(json data) {
  string label = "BaseStatus write_data(json data) ";
  string data_string = data.dump();
  logger.info(label);
  logger.info("BaseStatus json data: " + data_string);

  string query = "UPDATE "
    + table_name
    + " SET "
    + COL_DATA
    + " = '"
    + data_string
    +  "' WHERE "
    + COL_ID 
    + " = '"
    + primary_key
    +  "';";

  logger.info("BaseStatus query: " + query);

  return insert_query(query);
}
