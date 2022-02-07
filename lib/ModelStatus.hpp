#pragma once
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include <thread>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ModelStatus : public BaseStatus {

  private:
    string model_id = "model_id_not_set";

  protected:
    json compose_status();
    void record_status();
    string get_id(){return model_id;}

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
    static string get_model_id_field_name(){return "id";} // API

    // serialized JSON fields in the status text
    const string MODEL_ID = get_model_id_field_name(); 
    const string NODES = "nodes"; // API
    const string EDGES = "edges"; // API
};
