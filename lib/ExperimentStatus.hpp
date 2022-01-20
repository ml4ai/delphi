#pragma once
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ExperimentStatus {

    public:
    ExperimentStatus();
    ~ExperimentStatus();
    void init_db();
    void update_db();
    void start_updating_db(AnalysisGraph* ag);
    void stop_updating_db();
    void set_status(string experimentId, string statusString);
    string stop_training(string modelId);
    string read_from_db(string modelId);

    const string TABLE_NAME = "experiment_status";
    const string COL_ID = "id";  // Experiment ID, primary key
    const string COL_JSON = "json"; // serialized JSON status string
    const string STATUS_EXPERIMENT_ID = "experimentId";
    const string STATUS_MODEL_ID = "modelId";
    const string STATUS_TYPE = "experimentType";
    const string STATUS_STATUS = "status"; 
    const string STATUS_PROGRESS = "progressPercentage";


    private:
    AnalysisGraph* ag =nullptr;
    std::thread *pThread = nullptr;
    sqlite3* db = nullptr;
    void scheduler();
    void write_to_db(json status);
    json compose_status();
    int insert(string query);
    void logInfo(string message);
    void logError(string message);
    static int callback(
        void* NotUsed,
        int argc,
        char** argv,
        char** azColName
    );

};
