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
    Database* database = nullptr;
    std::thread *pThread = nullptr;
    bool recording = false;
    string timestamp();
    string class_name = "N/A";
    string table_name = "N/A";
    void write_progress();

  protected:
    void scheduler();
    void log_error(string msg);
    void log_info(string msg);
    virtual string get_id() = 0;
    double progress = 0.0;
    json read_row(string id);
    void write_row(string id, json data);
    virtual void prune_row(string id) = 0;
    void delete_row(string id);
    vector<string> get_ids();

  public:
    BaseStatus(
      Database* database,
      const string table_name,
      const string class_name
    );
    ~BaseStatus();

    void clean_db();
    json get_data();
    virtual void initialize() = 0;
    void set_progress(double p) { progress = p;}
    void increment_progress(double i) { progress += i;}
    bool lock();
    void set_status(string status);
    void set_data(json data){write_row(get_id(), data);}
    void begin_recording_progress(string status);
    void finish_recording_progress(string status);
    virtual void finalize(string status) = 0;

    // database columns
    const string COL_ID = "id"; // database column, not exported
    const string COL_DATA = "data"; // database column, not exported

    // serialized JSON fields in the status text
    const string PROGRESS = "progressPercentage"; // double [0,1]
    const string STATUS = "status"; // string, a short description of current state
    const string BUSY = "busy"; // boolean, true if locked
};
