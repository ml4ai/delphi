#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DelphiConnect.hpp"
#include "TrainingStatus.hpp"
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using namespace DelphiConnect;
using json = nlohmann::json;


TrainingStatus::TrainingStatus(){
}

TrainingStatus::~TrainingStatus(){
  stop_monitoring();
}

void TrainingStatus::scheduler()
{
  while(!ag->get_trained()){
    this_thread::sleep_for(std::chrono::seconds(1));
    if(pThread != nullptr) {
      write_to_db();
    }
  }
}

void TrainingStatus::start_monitoring(AnalysisGraph *ag){
  this->ag = ag;

  if(pThread == nullptr) {
    pThread = new thread(&TrainingStatus::scheduler, this);
  }
}

void TrainingStatus::stop_monitoring(){
    if (pThread != nullptr)
    {
        if(pThread->joinable()) {
            pThread->join();
	}
        delete pThread;
    }
    pThread = nullptr;
}

bool TrainingStatus::write_to_db() {
  
  bool ret = true;
  cout << "TrainingStatus::write_to_db()" << endl;

  string model_id = ag->id;
  if (!model_id.empty()) {
    sqlite3_stmt* stmt = nullptr;
    sqlite3* db = delphiConnectReadWrite();
    if (db == nullptr) {
      cout << "\n\nERROR: opening delphi.db" << endl;
      ret = false;
    } else {

      char c1[100];
      sprintf(c1, "%5.3f", ag->get_training_progress());

      char c2[100];
      sprintf(c2, "%s", ag->get_trained()? "true": "false");

      char c3[100];
      sprintf(c3, "%s", ag->get_stopped()? "true": "false");

      char c4[100];
      sprintf(c4, "%f", ag->get_log_likelihood());

      char c5[100];
      sprintf(c5, "%f", ag->get_previous_log_likelihood());

      char c6[100];
      sprintf(c6, "%f", ag->get_log_likelihood_MAP());

      string query = "replace into " + table + " values ('" + 
	model_id + "', '" +
        c1 + "', '" +
        c2 + "', '" +
        c3 + "', '" +
        c4 + "', '" +
        c5 + "', '" +
        c6 + "');";

      cout << "QUERY: " << query << endl;
      int prep = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);

      if (prep != SQLITE_OK) {
        cout << "Could not prepare statement" << endl;
	cout << "Error code: " << prep << endl;
        cout << sqlite3_errmsg(db) << endl;
        ret = false;
      }
      int exec = sqlite3_step(stmt);
      if (exec != SQLITE_DONE) {
        cout << "Could not execute statement" << endl;
	cout << "Error code: " << exec << endl;
        cout << sqlite3_errmsg(db) << endl;
        ret = false;
      }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    stmt = nullptr;
    db = nullptr;
  }
  return ret;
}

/*
    Select/read all column and 1 rows of trainingprogress table
*/
json TrainingStatus::read_from_db(string modelId) {
  cout << "TrainingStatus::read_from_db()" << endl;

  json matches;
  sqlite3_stmt* stmt = nullptr;
  string query =
      "SELECT * from " + table + " WHERE id='" + modelId + "'  LIMIT 1;";
  cout << "QUERY: " << query << endl;
  sqlite3* db = delphiConnectReadOnly();
  int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  cout << "Processing result rows" << endl;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
	  cout << "SQLITE_ROW" << endl;
      matches[col0] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
      matches[col1] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
      matches[col2] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
      matches[col3] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
      matches[col4] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)));
      matches[col5] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
      matches[col6] =
          string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6)));
  }
  cout << "Done processing result rows" << endl;
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
  cout << "MATCHES: " << matches.dump() << endl;
  return matches;
}

// create the table if we need it.
void TrainingStatus::init_db() {
  string c0 = "id TEXT PRIMARY KEY, ";
  string c1 = "progress TEXT NOT NULL, ";
  string c2 = "trained TEXT NOT NULL, ";
  string c3 = "stopped TEXT NOT NULL, ";
  string c4 = "log_likelihood TEXT NOT NULL, ";
  string c5 = "log_likelihood_previous TEXT NOT NULL, ";
  string c6 = "log_likelihood_map TEXT NOT NULL";
  string cols = c0 + c1 + c2 + c3 + c4 + c5 + c6;
  string query = "CREATE TABLE IF NOT EXISTS " + table + " (" + cols + ");";

  cout << "DatabaseHelper.cpp.init_training_status" << endl;
  cout << "Query: " << query << endl;

  sqlite3* db = delphiConnectReadWrite();
  sqlite3_stmt* stmt = nullptr;
  int prep = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
  if(prep == SQLITE_OK) {
    int exec = sqlite3_step(stmt);
    if(exec == SQLITE_DONE) {
      cout << "Table created successfully" << endl;
    } else {
      cout << "Table was not created" << endl;
    }
  } else {
    cout << "statement preparation failed" << endl;
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  stmt = nullptr;
  db = nullptr;
}

