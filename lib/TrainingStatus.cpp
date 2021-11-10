#include <sqlite3.h>
#include "AnalysisGraph.hpp"
#include "TrainingStatus.hpp"
#include <thread>
#include <chrono>

using namespace std;

void TrainingStatus::scheduler()
{
  while(!ag->get_trained()){
    this_thread::sleep_for(std::chrono::seconds(1));
    ag->write_training_status_to_db();
  }
}

TrainingStatus::TrainingStatus(AnalysisGraph* ag){
  this->ag = ag;

  if(pThread == nullptr) {
    pThread = new thread(&TrainingStatus::scheduler, this);
  }
}

TrainingStatus::~TrainingStatus(){
    if (pThread != nullptr)
    {
        if(pThread->joinable()) {
            pThread->join();
	}
        delete pThread;
    }
}
