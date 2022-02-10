#pragma once

#include <sqlite3.h>
#include "AnalysisGraph.hpp"
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
    void create_table();
    void clean_table();
    void clean_row(string id);

  protected:
    void scheduler();
    void log_error(string msg);
    void log_info(string msg);
    bool is_training(string id);
    void write_row(string id, json status);
    json read_status(string id);
    virtual void update_db() = 0;
    virtual string get_id() = 0;
    float progress = 0.0;
    string state = "not created";

  public:
    BaseStatus(
      Database* database,
      const string table_name,
      const string class_name
    ) :
      database(database),
      table_name(table_name),
      class_name(class_name){}
    ~BaseStatus(){}

    void clean_db();
    bool start_training();
    void start_recording();
    void stop_recording();
    void set_progress(float p) { progress = p;}
    void increment_progress(float i) { progress += i;}

    // serialized JSON fields in the status text
    const string COL_ID = "id"; // database column, not exported
    const string COL_STATUS = "status"; // database column, not exported
    const string PROGRESS = "progressPercentage"; // JSON field, API
    const string STATUS = "status"; // JSON field, ?
};
