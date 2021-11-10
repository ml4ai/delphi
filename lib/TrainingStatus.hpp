#pragma once
#include <thread>

class AnalysisGraph;

class TrainingStatus {

  private:
  AnalysisGraph* ag = nullptr;
  std::thread *pThread = nullptr;
  void scheduler();
  void startScheduler();

  public:
  TrainingStatus(AnalysisGraph * ag);
  ~TrainingStatus();
};
