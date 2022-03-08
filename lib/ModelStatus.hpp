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
    void set_state(double progress, string status, bool busy);
    vector<json> get_valid_rows();


  public:
    ModelStatus(string id) : BaseStatus(
      "model_status",
      "ModelStatus",
      model_id
    ), model_id(id) {}

    ModelStatus(string id, Database* database) : BaseStatus(
      "model_status",
      "ModelStatus",
      model_id,
      database
    ), model_id(id) {}

    ~ModelStatus(){}

    string get_id(){return model_id;}
    void enter_initial_state();
    void enter_reading_state();
    void enter_working_state();
    void enter_writing_state();
    void enter_finished_state(string status);

    // serialized JSON fields in the status text
    const string MODEL_ID = "id"; // API
    const string NODES = "nodes"; // API
    const string EDGES = "edges"; // API

    // delphi model table
    const string MODEL_TABLE = "delphimodel";
};
