#pragma once
#include "DatabaseHelper.hpp"
#include "BaseStatus.hpp"
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ExperimentStatus : public BaseStatus {

  private:
    void scheduler();
    bool done_updating();

  protected:
    bool done_updating_db(){return true;}
    json compose_status();

  public:
    ExperimentStatus() : BaseStatus(
      new Database(), 
      "experiment_status",
      "ExperimentStatus"
    ){}
    ExperimentStatus(Database* database) : BaseStatus(
      database, 
      "experiment_status",
      "ExperimentStatus"
    ){}
    ~ExperimentStatus(){}
    void update_db();
    json get_experiment_progress(string experimentId);

    const string CONSTRAINTS = "constraints"; // API
    const string END_TIME = "endTime"; // API
    const string EXPERIMENT_ID = "experimentId"; // API
    const string EXPERIMENT_PARAM = "experimentParam"; // API
    const string EXPERIMENT_TYPE = "experimentType"; // API
    const string MODEL_ID = "modelId"; // API
    const string NUM_TIME_STEPS = "numTimesteps"; // API
    const string RESULTS = "results"; // API
    const string START_TIME = "startTime"; // API
};
