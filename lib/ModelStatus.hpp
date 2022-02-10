#pragma once

#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include "utils.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ModelStatus : public BaseStatus {

  private:
    string model_id = "N/A";

  protected:
    void update_db();

  public:
    ModelStatus(string id) : BaseStatus(
      new Database(),
      "model_status",
      "ModelStatus"
    ), model_id(id) {log_info("ModelStatus created for " + id);}
    ModelStatus(string id, Database* database) : BaseStatus(
      database,
      "model_status",
      "ModelStatus"
    ), model_id(id) {log_info("ModelStatus created for " + id);}
    ~ModelStatus(){log_info("ModelStatus destroyed for " + model_id);}
    bool start_training();
    void record_status();
    json get_status(){return read_status(model_id);}

    // serialized JSON fields in the status text
    const string MODEL_ID = "id";  // API
    const string NODES = "nodes"; // API
    const string EDGES = "edges"; // API
};
