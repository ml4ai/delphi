#pragma once
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

// maintain a table just for Model training 
class ModelStatus {

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

    public:
    ModelStatus();
    ~ModelStatus();
    void init_db();
    void update_db();
    void start_updating_db(AnalysisGraph* ag);
    void stop_updating_db();
    string stop_training(string modelId);
    string read_from_db(string modelId);

    const string TABLE_NAME = "model_status";
    const string COL_ID = "id";    // primary key
    const string COL_STATUS = "status"; // serialized JSON string
    const string STATUS_TRAINED = "trained";
    const string STATUS_PROGRESS_PERCENTAGE = "progressPercentage";
};
