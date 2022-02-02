#pragma once
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ModelStatus : public BaseStatus {

  private:
    void scheduler();

  protected:
    bool done_updating_db(){return ag->get_trained();}
    json compose_status();

  public:
    ModelStatus() : BaseStatus(
      new Database(), 
      "model_status",
      "ModelStatus"
    ) {}
    ModelStatus(Database* database) : BaseStatus(
      database, 
      "model_status",
      "ModelStatus"
    ) {}
    ~ModelStatus(){}
    void update_db();
    json get_training_progress_response(string modelId);

    const string TABLE_NAME = "model_status"; // arbitrary

    // serialized JSON fields in the status text
    const string STATUS_MODEL_ID = "model_id"; // arbitrary
    const string STATUS_PROGRESS = "progressPercentage"; // API
    const string STATUS_NODES = "nodes"; // API
    const string STATUS_EDGES = "edges"; // API
    const string STATUS_STATUS = "status";  // API
    const string STATUS_TRAINED = "trained";  // arbitrary
};
