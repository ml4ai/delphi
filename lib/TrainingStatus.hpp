#pragma once
#include <thread>
#include <nlohmann/json.hpp>

class AnalysisGraph;

using namespace std;

class TrainingStatus {

  private:
  AnalysisGraph* ag =nullptr;
  std::thread *pThread = nullptr;
  void scheduler();

  public:
  const string table = "training_status";
  const int nCols = 7;
  const string col0 = "id";
  const string col1 = "progress";
  const string col2 = "trained";
  const string col3 = "stopped";
  const string col4 = "log_likelihood";
  const string col5 = "log_likelihood_previous";
  const string col6 = "log_likelihood_map";
  TrainingStatus();
  ~TrainingStatus();
  bool write_to_db();
  void init_db();
  void start_monitoring(AnalysisGraph* ag);
  void stop_monitoring();
  string read_from_db(string modelId);
  nlohmann::json select_training_status(string modelId);
};
