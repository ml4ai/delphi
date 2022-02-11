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
    void create_table();
    void clean_table();
    void clean_row(string id);
    json read_row(string id);

  protected:
    void scheduler();
    void log_error(string msg);
    void log_info(string msg);
    bool is_busy(); 
    void write_row(string id, json status);
    virtual void update_db() = 0;
    virtual string get_id() = 0;
    double progress = 0.0;
    bool busy = false;
    string status = "Not created"; // reading, training, writing, ready

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
    void set_progress(double p) { progress = p;}
    void increment_progress(double i) { progress += i;}
    json get_data();
    void start_recording();
    void stop_recording();
    bool lock();
    void unlock();
    void set_status(string status);

    // serialized JSON fields in the status text
    const string COL_ID = "id"; // database column, not exported
    const string COL_DATA = "data"; // database column, not exported
    const string PROGRESS = "progressPercentage"; // JSON field, API
    const string STATUS = "status"; // JSON field
    const string BUSY = "busy"; // JSON field, not exported
};
