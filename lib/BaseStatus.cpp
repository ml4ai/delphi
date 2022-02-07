#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

//#define SHOW_LOGS  // define this to see info and error messages

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

/* Start the thread that posts the status to the datbase */
void BaseStatus::scheduler() {
  logInfo("scheduler()");
  while(training){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      record_status();
    }
  }
}

/* Begin posting progress updates to the database on a regular interval */
void BaseStatus::start_recording_progress(){
  logInfo("start_updating_db()");
  training = true;
  record_status();
  if(pThread == nullptr) {
    pThread = new thread(&BaseStatus::scheduler, this);
  }
}

/* Stop posting progress updates to the database */
void BaseStatus::stop_recording_progress(){
  logInfo("stop_updating_db()");
  training = false;
  record_status();
  if (pThread != nullptr) {
    if(pThread->joinable()) {
      pThread->join();
    }
    delete pThread;
  }
  pThread = nullptr;
}

/* write the progress status to the database */
void BaseStatus::set_status(string id, json status) {
  string statusStr = status.dump();
  string info = "set_status(" + id + ", " + statusStr + ")";
  logInfo(info);

  string query = "INSERT OR REPLACE INTO " 
    + table_name
    + " VALUES ('"
    + id
    + "', '"
    + statusStr
    +  "');";

  insert(query);
}


bool BaseStatus::exists() {
  return exists(get_status());
}

bool BaseStatus::exists(json status) {
  return !status.empty();
}

bool BaseStatus::is_busy() {
  return is_busy(get_status());
}

bool BaseStatus::is_busy(json status) {
  if (status.empty()) return false;
  double progress = status[PROGRESS];
  return (progress < 1.0);
}

/* Return a JSON struct serialized from the 'status' query result row 
 * If the id is not found, return empty JSON */
json BaseStatus::get_status() {
  string id = get_id();
  string info = "get_status(" + id + ") => ";
   json queryResult = database->select_row(table_name, id, COL_STATUS);
  if(queryResult.empty()) {
    logError(info + " ID not found");
    return queryResult;
  }
  string statusString = queryResult[COL_STATUS];
  json status = json::parse(statusString);
  logInfo(info + status.dump());
  return status;
}

void BaseStatus::startup() {
  create_table();
  clean_table();
}

/* delete any entries with incomplete training status */
void BaseStatus::clean_table() {

  // get all the ids from our table
  string query = "SELECT " + COL_ID + " from " + table_name + ";";
  logInfo("Query: " + query);

  vector<string> ids = database->read_column_text(query);
  for(string id : ids) {
    clean_record(id);
  }
}


void BaseStatus::clean_record(string id) {
  vector<string> status_strings = 
    database->read_column_text_query_where(table_name, COL_STATUS, COL_ID, id);

  for(string status_string : status_strings) {
    json status = json::parse(status_string);
    string report = "Inspecting status record '" + id + "': ";

    double progress = status[PROGRESS];
    if(progress < 1.0) {
      logInfo(report + "FAIL  (stale progress, deleting record)");
      database->delete_rows(table_name, COL_ID, id);
    } 
    else {
      logInfo(report + "PASS");
    }
  }
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

  insert(query);
}

void BaseStatus::insert(string query) {
  string info = "insert('" + query + "')";
  logInfo(info);

  database->insert(query);
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


/* Report a message to stdout */
void BaseStatus::logInfo(string text) {
#ifdef SHOW_LOGS
  cout << timestamp() << " " << class_name << " INFO: " << text << endl;
#endif
}

/* Report an error to stderr */
void BaseStatus::logError(string text) {
#ifdef SHOW_LOGS
  cerr << timestamp() << " " << class_name << " ERROR: " << text << endl;
#endif
}
