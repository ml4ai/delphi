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
    string experiment_id = "experiment_id_not_set";

  protected:
    json compose_status();
    void record_status();
    string get_id(){return experiment_id;}

  public:
    ExperimentStatus(string id) : BaseStatus(
      new Database(), 
      "experiment_status",
      "ExperimentStatus"
    ), experiment_id(id){}
    ExperimentStatus(string id, Database* database) : BaseStatus(
      database, 
      "experiment_status",
      "ExperimentStatus"
    ), experiment_id(id){}
    ~ExperimentStatus(){}

    static string get_experiment_id_field_name(){return "experimentId";} // API
    static string get_model_id_field_name(){return "modelId";} // API
    const string CONSTRAINTS = "constraints"; // API
    const string END_TIME = "endTime"; // API
    const string EXPERIMENT_ID = get_experiment_id_field_name();
    const string EXPERIMENT_PARAM = "experimentParam"; // API
    const string EXPERIMENT_TYPE = "experimentType"; // API
    const string MODEL_ID = get_model_id_field_name();
    const string NUM_TIME_STEPS = "numTimesteps"; // API
    const string RESULTS = "results"; // API
    const string START_TIME = "startTime"; // API
};
