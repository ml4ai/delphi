#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

#define SHOW_LOGS

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

void BaseStatus::clean_db() {
  create_table();
  clean_table();
}

/* Start the thread that posts the status to the datbase */
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
    + " TEXT PRIMARY KEY, "
    + COL_STATUS
    + " TEXT NOT NULL);";

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
  string report = "Inspecting " + table_name + " record '" + id + "': ";

  json status = read_status(id);
  
  log_info("clean_row(" + id + ") => " + status.dump());


  float row_progress = status[PROGRESS];
  if(row_progress < 1.0) {
    log_info(report + "FAIL (stale progress, deleting record)");
    database->delete_rows(table_name, "id", id);
  }
  else {
    log_info(report + "PASS");
  }
}

bool BaseStatus::is_training(string id) {
  json status = read_status(id);
  if(status.empty()) 
    return false;
  else {
    float row_progress = status[PROGRESS];
    return (row_progress < 1.0);
  }
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

json BaseStatus::read_status(string id) {
  json ret = database -> select_row(
    table_name,
    id,
    COL_STATUS
  );
  return ret;
}

void BaseStatus::write_row(string id, json status) {
  string query = "INSERT OR REPLACE INTO "
    + table_name
    + " VALUES '"
    + id
    + "', '"
    + status.dump()
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
