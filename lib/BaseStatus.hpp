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
    const string table_name = "N/A";
    const string class_name = "N/A";
    const string COL_ID = "id"; // arbitrary, not exported
    const string COL_STATUS = "status"; // arbitrary, not exported

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
      const string table_name,
      const string class_name
    ) : 
      database(database),
      table_name(table_name),
      class_name(class_name){}
    ~BaseStatus(){}

  public:
    virtual void set_initial_status(string id) = 0;
    virtual void update_db() = 0;
    void init_db();
    json get_status(string id);
    void set_status(string id, json status);
    void start_updating_db(AnalysisGraph* ag);
    void stop_updating_db();
    bool is_busy(json status);
    bool is_busy(string id);
    bool exists(string id);
    bool exists(json status);

    const string PROGRESS = "progressPercentage"; // API
    const string STATUS = "status"; // API
};
