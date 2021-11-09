#pragma once

#include <sqlite3.h>
#include <nlohmann/json.hpp>

class AnalysisGraph;

class TrainingStatus {

  private:
  AnalysisGraph* ag = nullptr;
  std::string modelId = "not_set";
  float training_progress = 0.0;
  void update();
  void scheduler();
  TrainingStatus();

//  double log_likelihood = 0.0;
//  double previous_log_likelihood = 0.0;
//  bool trained = false;

  void startScheduler();

  public:
  TrainingStatus(AnalysisGraph * ag);

  ~TrainingStatus();

  void setModelId(std::string _modelId);
  std::string to_json_string();

};
