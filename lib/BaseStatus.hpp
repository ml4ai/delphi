#pragma once

#include "BaseStatus.hpp"
#include "DatabaseHelper.hpp"
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <thread>
#include "utils.hpp"

using namespace std;
using json = nlohmann::json;

class BaseStatus {

  private:
    string class_name = "N/A";  // extending class
    string table_name = "N/A";  // table of extending class
    string primary_key = "N/A"; // primary key of table
    Database* database = nullptr; // connection to Delphi DB
    std::thread *pThread = nullptr; // for timed updates to database
    bool recording = false;  // true if timed updates are happening
    double progress = 0.0; // training of model or completion of projection
    bool delete_database = false; // true if we created a new database
    void scheduler();
    bool insert_query(string query);
    string timestamp();

  protected:
    void set_progress(double p){ progress = p;}
    void start_recording_progress();
    void stop_recording_progress();
    bool write_data(json data);
    void log_info(string msg);
    void log_error(string msg);
    virtual vector<json> get_valid_rows() = 0;
    vector<string> get_ids();


  public:
    BaseStatus(
        const string class_name,
        const string table_name,
        const string primary_key
    ):database(new Database()),
      class_name(class_name),
      table_name(table_name),
      primary_key(primary_key),
      delete_database(true){}
    BaseStatus(
        const string class_name,
        const string table_name,
        const string primary_key,
	Database* database
    ):database(database),
      class_name(class_name),
      table_name(table_name),
      primary_key(primary_key){}
    ~BaseStatus();
    void initialize();
    json read_data();
    void increment_progress(double i) { progress += i;}

    // database table columns
    const string COL_ID = "id"; // database column, not exported
    const string COL_DATA = "data"; // database column, not exported

    // serialized JSON fields
    const string PROGRESS = "progressPercentage"; // double [0,1]
    const string STATUS = "status"; // string, current state
    const string BUSY = "busy"; // boolean, true if locked
};
