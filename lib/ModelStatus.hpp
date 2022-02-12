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

  protected:
    void update_db();
    void sync_to_db();
    void init_row();

  public:
    ModelStatus(string id);
    ModelStatus(string id, Database* database);
    ~ModelStatus(){}

    string get_id(){return model_id;}

    // serialized JSON fields in the status text
    const string MODEL_ID = "id"; // API
    const string NODES = "nodes"; // API
    const string EDGES = "edges"; // API
};
