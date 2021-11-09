#pragma once

class AnalysisGraph;

class TrainingStatus {

  private:
  AnalysisGraph* ag = nullptr;
  TrainingStatus();
  void scheduler();
  void startScheduler();

  public:
  TrainingStatus(AnalysisGraph * ag);
  ~TrainingStatus();
};
