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

void BaseStatus::clean_db() {
  create_table();
  clean_table();
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

  json data = get_data();

  data[STATUS] = status;
  data[PROGRESS] = 1.0;
  write_row(get_id(), data);
}

/* create the table if we need it.  */
void BaseStatus::create_table() {
  string query = "CREATE TABLE IF NOT EXISTS "
    + table_name
    + " ("
    + COL_ID
    + " VARCHAR PRIMARY KEY, "
    + COL_DATA
    + " VARCHAR NOT NULL);";

  database->insert(query);
}

/* delete any entries with incomplete training status */
void BaseStatus::clean_table() {
  log_info("Cleaning table '" + table_name + "' of incomplete records");
  string query = "SELECT " 
    + COL_ID
    + " from "
    + table_name
    + ";";

  vector<string> ids = database->read_column_text(query);
  for(string id : ids) {
    clean_row(id);
  }

  log_info("Done.");
}

// called only at startup, any rows in the table with incomplete training
// are declared lost and get deleted.
void BaseStatus::clean_row(string id) {

  string report = "Inspecting " + table_name + " record '" + id + "': ";

  json row = read_row(id);
  if(row.empty()) {
    log_info(report + "FAIL (missing record, deleting row)");
    database->delete_rows(table_name, "id", id);
    return;
  }

  string dataString = row[COL_DATA];
  if(dataString.empty()) {
    log_info(report + "FAIL (missing raw data, deleting row)");
    database->delete_rows(table_name, "id", id);
    return;
  }

  json data = json::parse(dataString);
  if(data.empty()) {
    log_info(report + "FAIL (missing data, deleting row)");
    database->delete_rows(table_name, "id", id);
    return;
  }

  double row_progress = data.value(PROGRESS,0.0);
  if(row_progress < 1.0) {
    log_info(report + "FAIL (stale progress, deleting row)");
    database->delete_rows(table_name, "id", id);
    return;
  }

  bool busy = data.value(BUSY, true);
  if(busy) {
    log_info(report + "FAIL (stale lock, deleting row)");
    database->delete_rows(table_name, "id", id);
    return;
  }

  log_info(report + "PASS");
}

// Lock the database 
sqlite3_mutex* BaseStatus::enter_critical_section() {
  sqlite3_mutex* mx = sqlite3_mutex_alloc(SQLITE_MUTEX_RECURSIVE);
  if(mx == nullptr) {
    log_error("Could not create mutex, database error"); // error
    return mx;
  }
  // enter critical section (blocking method)
  sqlite3_mutex_enter(mx);
  return mx;
}

// Unlock the database
void BaseStatus::exit_critical_section(sqlite3_mutex* mx) {
  sqlite3_mutex_leave(mx);
  sqlite3_mutex_free(mx);
  mx = nullptr;
}

// Attempt to lock this status by setting the 'busy' flag to true.
bool BaseStatus::lock() {
  sqlite3_mutex* mx = enter_critical_section();
  if(mx == nullptr) {
    return false; // We did not get a mutex.  This should never happen
  }

  // Enter critical section and get (or create) the row for this status
  if(get_data().empty()) {
    init_row();
  }
  json data = get_data();

  // exit critical section if the status is busy
  bool busy = data[BUSY];
  if(busy) {
    exit_critical_section(mx);
    return false;  // This status is busy
  }
  
  // set the lock
  data[BUSY] = true;
  data[PROGRESS] = 0.0;
  write_row(get_id(), data);
  json check = get_data();
  bool locked = check[BUSY];

  // exit critical section with the lock state
  exit_critical_section(mx);
  return locked; 
}

// Attempt to unlock this status by setting the 'busy' flag to false.
bool BaseStatus::unlock() {
  sqlite3_mutex* mx = enter_critical_section();
  if(mx == nullptr) {
    return false; // error
  }

  // free the lock
  json data = get_data();
  data[BUSY] = false;
  write_row(get_id(), data);
  json check = get_data();
  bool locked = check[BUSY];

  exit_critical_section(mx);
  return !locked; 
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
