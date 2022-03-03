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
    string class_name = "N/A";
    string table_name = "N/A";
    string primary_key = "N/A";
    Database* database = nullptr;
    std::thread *pThread = nullptr;
    bool recording = false;
    double progress = 0.0;
    bool delete_database = false; 
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
