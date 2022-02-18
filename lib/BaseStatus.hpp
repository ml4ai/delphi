#pragma once

#include <sqlite3.h>
#include "DatabaseHelper.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

class BaseStatus {

  private:
    string class_name = "N/A";
    string table_name = "N/A";
    Database* database = nullptr;
    std::thread *pThread = nullptr;
    bool recording = false;
    string timestamp();
    void prune_row(string id);
    void scheduler();
    vector<string> get_ids();

  protected:
    void log_error(string msg);
    void log_info(string msg);
    virtual string get_id() = 0;
    double progress = 0.0;
    json read_row(string id);
    void write_row(string id, json data);
    void delete_row(string id);
    void start_recording_progress();
    void stop_recording_progress();

  public:
    BaseStatus(
      Database* database,
      const string table_name,
      const string class_name
    );
    ~BaseStatus();

    void clean_db();
    json get_data();
    void increment_progress(double i) { progress += i;}

    // database columns
    const string COL_ID = "id"; // database column, not exported
    const string COL_DATA = "data"; // database column, not exported

    // serialized JSON fields in the status text
    const string PROGRESS = "progressPercentage"; // double [0,1]
    const string STATUS = "status"; // string, a short description of current state
    const string BUSY = "busy"; // boolean, true if locked
};
