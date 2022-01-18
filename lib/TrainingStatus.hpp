#pragma once
#include <thread>
#include <nlohmann/json.hpp>


class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

// maintain a table just for training status
// column "model_id" is the model ID as primary key
// column "status" is a JSON string 

class TrainingStatus {

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
  TrainingStatus();
  ~TrainingStatus();
  void init_db();
  void update_db();
  void start_updating_db(AnalysisGraph* ag);
  void stop_updating_db();
  string stop_training(string modelId);
  string read_from_db(string modelId);

  const string COL_MODEL_ID = "id";
  const string COL_MODEL_STATUS = "status";
  const string TABLE_NAME = "training_status";
};
