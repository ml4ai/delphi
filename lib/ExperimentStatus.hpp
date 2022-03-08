#pragma once

#include "BaseStatus.hpp"
#include <thread>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ExperimentStatus : public BaseStatus {

  private:
    void scheduler();
    string experiment_id = "N/A";
    string model_id = "N/A";

  protected:
    void set_state(double progress, string status, bool busy);
    vector<json> get_valid_rows();

  public:
    ExperimentStatus(
        string experiment_id,
        string model_id 
    ) : BaseStatus(
      "ExperimentStatus",
      "experiment_status",
      experiment_id
    ), experiment_id(experiment_id), model_id(model_id){}

    ExperimentStatus(
        string experiment_id,
        string model_id,
        Database* database
    ) : BaseStatus(
      "ExperimentStatus",
      "experiment_status",
      experiment_id,
      database
    ), experiment_id(experiment_id), model_id(model_id){}

    ~ExperimentStatus(){}

    string get_id(){ return experiment_id;}
    void enter_initial_state();
    void enter_reading_state();
    void enter_working_state();
    void enter_writing_state();
    void enter_finished_state(string status);

    const string CONSTRAINTS = "constraints"; // API
    const string END_TIME = "endTime"; // API
    const string EXPERIMENT_ID = "experimentId"; // API
    const string EXPERIMENT_PARAM = "experimentParam"; // API
    const string EXPERIMENT_TYPE = "experimentType"; // API
    const string MODEL_ID = "modelId"; // API
    const string NUM_TIME_STEPS = "numTimesteps"; // API
    const string RESULTS = "results"; // API
    const string START_TIME = "startTime"; // API

    const string EXPERIMENT_TABLE = "causemosasyncexperimentresult";
};
