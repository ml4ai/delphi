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
    Logger logger = Logger("ExperimentStatus");

  public:
    ExperimentStatus(
        string experiment_id,
        string model_id 
    ) : BaseStatus(
      "ExperimentStatus",
      "causemosasyncexperimentresult",
      experiment_id
    ), experiment_id(experiment_id), model_id(model_id){}

    ExperimentStatus(
        string experiment_id,
        string model_id,
        Database* database
    ) : BaseStatus(
      "ExperimentStatus",
      "causemosasyncexperimentresult",
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
    const string NUM_TIME_STEPS = "numTimesteps"; // API
    const string RESULTS = "results"; // API
    const string START_TIME = "startTime"; // API
    const string COMPLETED = "completed"; // API

    const string EXPERIMENT_TABLE = "causemosasyncexperimentresult";
    const string COL_STATUS = "status";
};
