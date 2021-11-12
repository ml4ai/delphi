#pragma once
#include <thread>
#include "DatabaseHelper.hpp"

class AnalysisGraph;

using namespace std;

class TrainingStatus {

  private:
  AnalysisGraph* ag =nullptr;
  std::thread *pThread = nullptr;
  Database* database = nullptr;
  void scheduler();

  public:
  const string table = "training_status";
  TrainingStatus();
  ~TrainingStatus();
  void write_to_db();
  void init_db();
  void start_monitoring(AnalysisGraph* ag);
  void stop_monitoring();
  string read_from_db(string modelId);
};
