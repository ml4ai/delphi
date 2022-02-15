#pragma once

#include <sqlite3.h>
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
    void prune_row(string id);

  protected:
    void initialize();
    void clean_table();

  public:
    ModelStatus(string id) : BaseStatus(
      new Database(),
      "model_status",
      "ModelStatus"
    ), model_id(id) {}

    ModelStatus(string id, Database* database) : BaseStatus(
      database,
      "model_status",
      "ModelStatus"
    ), model_id(id) {}

    ~ModelStatus(){}

    string get_id(){return model_id;}
    void finalize(string status);

    // serialized JSON fields in the status text
    const string MODEL_ID = "id"; // API
    const string NODES = "nodes"; // API
    const string EDGES = "edges"; // API
};
