#include <sqlite3.h>
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

#define SHOW_LOGS   // define for cout debug messages

using namespace std;
using json = nlohmann::json;

BaseStatus::BaseStatus(
  Database* database,
  const string table_name,
  const string class_name
) : database(database),
    table_name(table_name),
    class_name(class_name) {
}


BaseStatus::~BaseStatus() {
}

/* Start the thread that writes the data to the table */
void BaseStatus::scheduler() {
  while(recording){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      write_progress();
    }
  }
}

/* Begin posting progress updates to the database on a regular interval */
void BaseStatus::begin_recording_progress(string status){
  set_status(status);
  progress = 0.0;
  write_progress();

  recording = true;
  if(pThread == nullptr) {
    pThread = new thread(&BaseStatus::scheduler, this);
  }
}

/* Stop posting progress updates to the database */
void BaseStatus::finish_recording_progress(string status){
  recording = false;
  if (pThread != nullptr) {
    if(pThread->joinable()) {
      pThread->join();
    }
    delete pThread;
  }
  pThread = nullptr;
  set_status(status);
}

/* purge table of partial records */
void BaseStatus::clean_db(){

/* create the table if we need it.  */
  string query = "CREATE TABLE IF NOT EXISTS "
    + table_name
    + " ("
    + COL_ID
    + " VARCHAR PRIMARY KEY, "
    + COL_DATA
    + " VARCHAR NOT NULL);";

  database->insert(query);

  // prune rows per table
  vector<string> ids = get_ids();
  for(string id : ids) {
    prune_row(id);
  }
}


// Attempt to lock this status by setting the 'busy' flag to true.
bool BaseStatus::lock() {

  // Enter critical section and get (or create) the row for this status
  if(get_data().empty()) {
    initialize();
  }
  json data = get_data();

  // exit critical section if the status is busy
  bool busy = data[BUSY];
  if(busy) {
    return false; 
  }
  
  // set the lock
  data[BUSY] = true;
  data[PROGRESS] = 0.0;
  write_row(get_id(), data);
  json check = get_data();
  bool locked = check[BUSY];

  // exit critical section with the lock state
  return locked; 
}

void BaseStatus::set_status(string status) {
  json data = get_data();
  data[STATUS] = status;
  write_row(get_id(), data);
}

// write our local progress var to the database
void BaseStatus::write_progress() {
  json data = get_data();
  data[PROGRESS] = delphi::utils::round_n(progress, 2);
  write_row(get_id(), data);
}


// report the current time
string BaseStatus::timestamp() {
    char timebuf[200];
    time_t t;
    struct tm *now;
    const char* fmt = "%F %T";
    t = time(NULL);
    now = gmtime(&t);
    if (now == NULL) {
      perror("gmtime error");
      exit(EXIT_FAILURE);
    }
    if (strftime(timebuf, sizeof(timebuf), fmt, now) == 0) {
      fprintf(stderr, "strftime returned 0");
      exit(EXIT_FAILURE);
    }
    return string(timebuf);
}

// return the data json for this ID
json BaseStatus::get_data() {
  json row = read_row(get_id());
  if(row.empty()) {
    return row;
  }
  string dataString = row[COL_DATA];
  json data = json::parse(dataString);
  return data;
}

// return the entire database row for this id
json BaseStatus::read_row(string id) {
  json row = database -> select_row(
    table_name,
    id,
    COL_DATA
  );
  return row;
}

// write this data to the table at the primary key
void BaseStatus::write_row(string id, json data) {
  string query = "INSERT OR REPLACE INTO "
    + table_name
    + " VALUES ('"
    + id
    + "', '"
    + data.dump()
    +  "');";
   database->insert(query);
}

// return all the primary keys in the table.
vector<string> BaseStatus::get_ids() {
  string query = "SELECT "
    + COL_ID
    + " from "
    + table_name
    +  ";";

  return database->read_column_text(query);
}

// delete the row with this primary key from the table
void BaseStatus::delete_row(string id) {
  string query = "DELETE FROM "
    + table_name
    + " WHERE (id = '"
    + id
    +  "');";
   database->insert(query);
}

/* Report a message to cout */
void BaseStatus::log_info(string msg) {
#ifdef SHOW_LOGS
  cout << timestamp() << " " << class_name << " INFO: " << msg << endl;
#endif
}

/* Report an error to cerr */
void BaseStatus::log_error(string msg) {
#ifdef SHOW_LOGS
  cout << timestamp() << " " << class_name << " ERROR: " << msg << endl;
#endif
}
