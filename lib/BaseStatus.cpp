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

//#define SHOW_LOGS   // define for cout debug messages

using namespace std;
using json = nlohmann::json;

void BaseStatus::clean_db() {
  create_table();
  clean_table();
}

/* Start the thread that writes the data to the table */
void BaseStatus::scheduler() {
  log_info("scheduler()");
  while(recording){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      update_db();
    }
  }
}

/* Begin posting progress updates to the database on a regular interval */
void BaseStatus::start_recording(){
  log_info("start_updating_db()");
  recording = true;
  update_db();
  if(pThread == nullptr) {
    pThread = new thread(&BaseStatus::scheduler, this);
  }
}

/* Stop posting progress updates to the database */
void BaseStatus::stop_recording(){
  log_info("stop_updating_db()");
  recording = false;
  update_db();
  if (pThread != nullptr) {
    if(pThread->joinable()) {
      pThread->join();
    }
    delete pThread;
  }
  pThread = nullptr;
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

  log_info(query);
  database->insert(query);
}

/* delete any entries with incomplete training status */
void BaseStatus::clean_table() {
  string query = "SELECT " 
    + COL_ID
    + " from "
    + table_name
    + ";";

  log_info(query);
  vector<string> ids = database->read_column_text(query);
  for(string id : ids) {
    clean_row(id);
  }
}

// called only at startup, any rows in the table with incomplete training
// are declared lost and get deleted.
void BaseStatus::clean_row(string id) {

  json row = read_row(id);
  log_info("clean_row(" + id + ") => " + row.dump());

  string dataString = row[COL_DATA];
  json data = json::parse(dataString);

  double row_progress = data[PROGRESS];
  bool busy = data[BUSY];
  string report = "Inspecting " + table_name + " record '" + id + "': ";
  if(row_progress < 1.0) {
    log_info(report + "FAIL (stale progress, deleting record)");
    database->delete_rows(table_name, "id", id);
  }
  else if(busy) {
    log_info(report + "FAIL (stale lock, deleting record)");
    database->delete_rows(table_name, "id", id);
  }
  else {
    log_info(report + "PASS");
  }
}

// return true if the progress exists and has not yet finished
bool BaseStatus::is_busy() {
  json data = get_data();
  if(data.empty()) 
    return false;
  else {
    bool busy = data[BUSY];
    return busy;
  }
}

// Attempt to lock this status
bool BaseStatus::lock() {
  sqlite3_mutex* mx = sqlite3_mutex_alloc(SQLITE_MUTEX_FAST);
  if(mx == nullptr) {
    log_error("Could not create mutex, database error");
    return false;  // could not lock
  }

  // enter critical section
  sqlite3_mutex_enter(mx);

  // exit critical section if the status is busy
  if(is_busy()) {
    sqlite3_mutex_leave(mx);
    sqlite3_mutex_free(mx);
    return false;  // already locked
  }

  // reset progress and lock this status
  progress = 0.0;
  busy = true;
  status = "locked";
  update_db();

  // exit critical section
  sqlite3_mutex_leave(mx);
  sqlite3_mutex_free(mx);
  return true; // success
}

void BaseStatus::set_status(string status) {
  json data = get_data();
  data[STATUS] = status;
  write_row(get_id(), data);
}

void BaseStatus::unlock() {
  json data = get_data();
  data[BUSY] = false;
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
  return database -> select_row(
    table_name,
    id,
    COL_DATA
  );
}

void BaseStatus::write_row(string id, json data) {
  log_info("write_row (" + id + ") " + data.dump());
  string query = "INSERT OR REPLACE INTO "
    + table_name
    + " VALUES ('"
    + id
    + "', '"
    + data.dump()
    +  "');";

   log_info(query);
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
