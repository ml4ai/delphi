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
    Logger logger = Logger("ModelStatus");
    bool validate_by_data(string id);
    bool validate_by_model(string id);
    void delete_row(string id);

  public:
    ModelStatus(string model_id) : BaseStatus(
      "ModelStatus",
      "delphimodel",
      model_id
    ), model_id(model_id) {}

    ModelStatus(string model_id, Database* database) : BaseStatus(
      "ModelStatus",
      "delphimodel",
      model_id,
      database
    ), model_id(model_id) {}

    ~ModelStatus(){}
    void initialize();



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
    const string TRAINED = "trained"; // API

    // delphi model table
    const string COL_MODEL = "model";
};
