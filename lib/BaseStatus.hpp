#pragma once
#include "DatabaseHelper.hpp"
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

class BaseStatus {

  private:
    Database* database = nullptr;
    void insert(string query);
    string table_name = "N/A";
    string class_name = "N/A";

  protected:
    virtual json compose_status() = 0;
    virtual bool done_updating_db() = 0;

    void scheduler();
    void logInfo(string message);
    void logError(string message);
    AnalysisGraph* ag = nullptr;
    std::thread *pThread = nullptr;
    string timestamp();
    BaseStatus(
      Database* database,
      string table_name,
      string class_name
    ) : 
      database(database),
      table_name(table_name),
      class_name(class_name){}
    ~BaseStatus(){}

  public:
    virtual void update_db() = 0;
    void init_db();
    json get_status(string id);
    void set_status(string id, json status);
    void start_updating_db(AnalysisGraph* ag);
    void stop_updating_db();

    const string COL_ID = "id"; // arbitrary
    const string COL_STATUS = "status"; // arbitrary
    const string STATUS_PROGRESS = "progressPercentage"; // API
    const string STATUS_STATUS = "status"; // API
};
