#include "BaseStatus.hpp"
#include "DatabaseHelper.hpp"
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <thread>
#include "utils.hpp"
#include <sys/time.h>
#include <chrono>

#define SHOW_INFO_LOGS   // define for cout debug messages
#define SHOW_ERROR_LOGS  // define for cout debug messages

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

  log_info("BaseStatus::initialize");

  // get rows for from table
//  vector<string> ids = database->read_column_text_query(table_name, COL_ID);
//  vector<json> rows;
//  for(string id : ids) {
//    rows.push_back(database->select_row(table_name, id, COL_DATA));
//  }

  // drop existing table to remove obsolete table versions
  string query = "DROP TABLE " + table_name + ";";
  if(!insert_query(query)) {
    return;
  }
 
  // create new table with latest definition
  query = "CREATE TABLE "
    + table_name
    + " ("
    + COL_ID
    + " VARCHAR PRIMARY KEY, "
    + COL_DATA
    + " VARCHAR NOT NULL);";
  if(!insert_query(query)) {
    return;
  }

  vector<json> rows = get_valid_rows();

  // add valid data to new table
  for(json row: rows) {
    string id = row.value(COL_ID,"");
    string dataString = row.value(COL_DATA, "");
    json data = json::parse(dataString);
    if(!data.empty()) {
      bool busy = data.value(BUSY, false);
      double progress = data.value(PROGRESS, 0.0);
      if(!busy && (progress == 1.0)) {
        query = "INSERT INTO "
          + table_name
          + " VALUES ('"
          + id
          + "', '"
          + dataString
          +  "');";
        if(!insert_query(query)) {
          return;
        }
        log_info("keeping " + dataString);
      } else {
        log_info("pruning " + dataString);
      }
    }
  }
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

// return current time like this:  2022-02-17 14:33:52:016
string BaseStatus::timestamp(){
  timeval curTime;
  gettimeofday(&curTime, NULL);
  int milli = curTime.tv_usec / 1000;
  char buffer [80];
  strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));
  char currentTime[84] = "";
  sprintf(currentTime, "%s:%03d", buffer, milli);
  return string(currentTime);
}

// Report a message to cout
void BaseStatus::log_info(string msg) {
#ifdef SHOW_INFO_LOGS
  cout << timestamp() << " " << class_name << " INFO: " << msg << endl;
#endif
}

// Report an error to cerr
void BaseStatus::log_error(string msg) {
#ifdef SHOW_ERROR_LOGS
  cout << timestamp() << " " << class_name << " ERROR: " << msg << endl;
#endif
}
