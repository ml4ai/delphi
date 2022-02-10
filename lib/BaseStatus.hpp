#pragma once
#include "DatabaseHelper.hpp"
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

class BaseStatus {

  private:
    BaseStatus(){}
    Database* database = nullptr;
    void insert(string query);
    void create_table();
    void clean_table();
    void clean_record(string id);
    const string table_name = "N/A";
    const string class_name = "N/A";
    const string COL_ID = "id"; // arbitrary, not exported
    const string COL_STATUS = "status"; // arbitrary, not exported
    bool training = false; // true when updating database
    bool stopped = false; // by human intervention
    double progress = 0.0; // values = [0.0, 1.0]

  protected:
    virtual json compose_status() = 0;
    virtual void record_status() = 0;
    virtual string get_id() = 0;
    void scheduler();
    void logError(string message);
    void logInfo(string message);
    void set_status(string id, json status);
    std::thread *pThread = nullptr;
    string timestamp();
    BaseStatus(
      Database* database,
      const string table_name,
      const string class_name
    ) : 
      database(database),
      table_name(table_name),
      class_name(class_name){}
    ~BaseStatus(){}

  public:
    void startup();
    json get_status();
    void start_recording_progress();
    void stop_recording_progress();
    void set_progress(double p) {progress = p;}
    void increment_progress(double i) {progress += i;}
    double get_progress(){return progress;}
    void set_initial_status(){record_status();}
    void logMessage(string message);
    bool is_busy(json status);
    bool is_busy();

    const string PROGRESS = "progressPercentage"; // API
    const string STATUS = "status"; // API
    const string STOPPED = "stopped"; // arbitrary, not exported
};
