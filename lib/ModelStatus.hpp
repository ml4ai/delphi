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
    double progress = 0.0;  // Range is [0.0, 1.0]
    bool trained = false;
    bool stopped = false;
    string model_id = "N/A";
    void scheduler();

  protected:
    bool done_updating_db(){return trained || stopped ;}
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
    void set_initial_status(string modelId = "N/A");
    void begin_recording_training();
    void set_progress(double p = 0.0) {progress = p;}
    double get_progress() {return progress;}

    // serialized JSON fields in the status text
    const string MODEL_ID = "id"; // API
    const string NODES = "nodes"; // API
    const string EDGES = "edges"; // API
    const string TRAINED = "trained";  // arbitrary
};
