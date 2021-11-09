#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "TrainingStatus.hpp"
#include <thread>
#include <chrono>

using namespace std;

void TrainingStatus::scheduler()
{
  while(!ag->get_trained()){
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ag->write_training_status_to_db();
  }
}

void TrainingStatus::startScheduler(){
  std::thread thr(&TrainingStatus::scheduler, this);
  thr.detach();
}

TrainingStatus::TrainingStatus(AnalysisGraph* ag){
  this->ag = ag;
  startScheduler();
}

// private
TrainingStatus::TrainingStatus(){
}

TrainingStatus::~TrainingStatus(){
  ag->write_training_status_to_db(); // destructor called after all training is done
}
