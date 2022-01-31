#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

#define DEBUG  // define this to see info and error messages

using namespace std;
using namespace delphi::utils;
using json = nlohmann::json;

/* Start the thread that posts the status to the datbase */
void BaseStatus::scheduler() {
  logInfo("scheduler()");
  while(!done_updating_db()){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      update_db();
    }
  }
}

/* Begin posting progress updates to the database on a regular interval */
void BaseStatus::start_updating_db(AnalysisGraph *ag){
  logInfo("start_updating_db()");
  this->ag = ag;
  update_db();
  if(pThread == nullptr) {
    pThread = new thread(&BaseStatus::scheduler, this);
  }
}

/* Stop posting progress updates to the database */
void BaseStatus::stop_updating_db(){
  logInfo("stop_updating_db()");
  update_db();
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

/* Return a JSON struct serialized from the 'status' query result row */
json BaseStatus::get_status(string id) {
  string info = "get_status(" + id + ")";
  logInfo(info);
  json result = database->select_row(table_name, id, COL_STATUS);
  string statusString = result[COL_STATUS];
  json status = json::parse(statusString);
  logInfo("  " + status.dump());
  return status;
}

/* create the table if we need it.  */
void BaseStatus::init_db() {
  logInfo("init_db()");
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
#ifdef DEBUG
  cout << timestamp() << " " << class_name << " INFO: " << text << endl;
#endif
}

/* Report an error to stderr */
void BaseStatus::logError(string text) {
#ifdef DEBUG
  cerr << timestamp() << " " << class_name << " ERROR: " << text << endl;
#endif
}
