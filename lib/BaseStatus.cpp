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
      json data = get_data();
      data[PROGRESS] = delphi::utils::round_n(progress, 2);
      write_row(get_id(), data);
    }
  }
}

/* Begin posting progress updates to the database on a regular interval */
void BaseStatus::start_recording_progress(){
  recording = true;
  if(pThread == nullptr) {
    pThread = new thread(&BaseStatus::scheduler, this);
  }
}

/* Stop posting progress updates to the database */
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

  vector<string> ids = get_ids();
  log_info("Validating " + table_name + " table...");
  for(string id : ids) {
    prune_row(id);
  }
  log_info("Done.");
}

void BaseStatus::prune_row(string id) {

  json row = read_row(id);

  // row does not exist
  if(row.empty()) {
    log_info(" Pruning " + id + " FAIL (missing record, deleting row)");
    delete_row(id);
    return;
  }

  // No string in data column
  string dataString = row[COL_DATA];
  if(dataString.empty()) {
    log_info(" Pruning " + id + " FAIL (missing raw data, deleting row)");
    delete_row(id);
    return;
  }

  // No JSON from data string
  json data = json::parse(dataString);
  if(data.empty()) {
    log_info(" Pruning " + dataString + " FAIL (missing data, deleting row)");
    delete_row(id);
    return;
  }

  // progress below 1.0
  double row_progress = data.value(PROGRESS,0.0);
  if(row_progress < 1.0) {
    log_info(" Pruning " + dataString + " FAIL (stale progress, deleting row)");
    delete_row(id);
    return;
  }

  // busy flag set high
  bool busy = data.value(BUSY, true);
  if(busy) {
    log_info(" Pruning " + dataString + " FAIL (stale lock, deleting row)");
    delete_row(id);
    return;
  }

  log_info(" Keeping " + dataString);
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
