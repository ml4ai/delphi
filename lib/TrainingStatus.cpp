#include <sqlite3.h>
#include <nlohmann/json.hpp>

#include "AnalysisGraph.hpp"
#include "TrainingStatus.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <boost/date_time.hpp>

namespace bpt = boost::posix_time;
using namespace std;


void TrainingStatus::update()
{

  float progress = ag->get_training_progress();
    std::cout << "Training progress: " << progress << " at " << bpt::microsec_clock::local_time().time_of_day() << endl;
}


void TrainingStatus::scheduler()
{

    while(!ag->get_trained()) {
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

TrainingStatus::TrainingStatus(){
	// private
}

TrainingStatus::~TrainingStatus(){
}

void TrainingStatus::setModelId(std::string modelId){
  this->modelId = modelId;
}


/*
  std::string modelId = "not_set";

  double log_likelihood = 0.0;
  double previous_log_likelihood = 0.0;
  float training_progress = 0.0;

  */

std::string TrainingStatus::to_json_string() {
  using nlohmann::json;

  json j;
  j["modelId"] = this->modelId;
//  j["status"] = this->trained? "ready" : "training";
  j["training_progress"] = this->training_progress;
//  j["previous_log_likelihood"] = this->previous_log_likelihood;
//  j["log_likelihood"] = this->log_likelihood;

  return j.dump();
}

