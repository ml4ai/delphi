#pragma once
#include <thread>
#include "DatabaseHelper.hpp"
#include <nlohmann/json.hpp>


class AnalysisGraph;

using namespace std;
using json = nlohmann::json;

class TrainingStatus {

  private:
  AnalysisGraph* ag =nullptr;
  std::thread *pThread = nullptr;
  Database* database = nullptr;
  void scheduler();
  void write_to_db(json status);
  json compose_status();

  public:
  const string table = "training_status";
  TrainingStatus();
  ~TrainingStatus();
  void init_db();
  void rebuild_db();
  void update_db();
  void start_updating_db(AnalysisGraph* ag);
  void stop_updating_db();
  string stop_training(string modelId);
  string read_from_db(string modelId);
};
